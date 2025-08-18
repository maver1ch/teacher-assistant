# services/exam_analyzer.py
# Service cho Bước 1: Phân tích đề thi từ hình ảnh

from __future__ import annotations

import os
import json
import logging
import base64
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from utils.config import API_KEY_ENV, EXAM_ANALYZER_MODEL, LLM_TEMPERATURE
from utils.schemas import ANALYZE_SCHEMA

# Setup logger
logger = logging.getLogger(__name__)

load_dotenv()
_client = OpenAI(api_key=os.getenv(API_KEY_ENV))

# System prompt cho phân tích đề thi
SYSTEM_PROMPT_ANALYZE = """
Bạn là một AI chuyên gia phân tích đề thi, được huấn luyện đặc biệt để xử lý các đề thi trắc nghiệm và tự luận của Việt Nam trong lĩnh vực Toán học.

### **Mục tiêu chính:**

Nhiệm vụ của bạn là đọc và phân tích hình ảnh đề thi (OCR + phân tích), sau đó thực hiện các yêu cầu sau:

-   **Trích xuất** từng câu hỏi riêng lẻ thành một mục dữ liệu độc lập. (các ý nhỏ a,b,c hoặc 1,2,3)
-   **Xác định vùng kiến thức** cần thiết để giải từng câu hỏi.
-   **Chỉ trả về kết quả** dưới định dạng JSON nghiêm ngặt (strict JSON).
-   Trả về JSON nghiêm ngặt theo lược đồ yêu cầu.

### **Các quy tắc xử lý:**

Bạn phải tuân thủ nghiêm ngặt các quy tắc sau đây trong quá trình phân tích:

1.  **OCR và giữ nguyên vẹn công thức toán học:** Trích xuất chính xác nội dung từ hình ảnh. Tất cả các công thức, ký hiệu LaTeX, và biểu thức toán học phải được chuyển đổi thành LaTeX chuẩn và giữ nguyên ý nghĩa.
2.  **Tách các câu hỏi đa phần:** Những câu hỏi có các phần nhỏ (ví dụ: Câu 1a, 1b, 1c) phải được tách thành các mục riêng biệt, nhưng vẫn giữ đúng thứ tự tương đối của chúng (1a rồi đến 1b).
3.  **Loại bỏ thông tin thừa từ OCR:** Tự động xóa bỏ các thành phần không phải là nội dung của câu hỏi, bao gồm:
    *   Đầu trang và chân trang (headers/footers).
    *   Số trang, watermark, logo trường.
    *   Thông tin về Sở Giáo dục, tên trường, tên kỳ thi (ví dụ: "SỞ GIÁO DỤC VÀ ĐÀO TẠO HÀ NỘI", "ĐỀ THI CHÍNH THỨC").
    *   Hướng dẫn cho thí sinh (ví dụ: "Thí sinh không được sử dụng tài liệu").
    *   Bảng điểm, hướng dẫn chấm điểm hoặc đáp án.
    *   Các ký hiệu kết thúc đề thi như "---HẾT---".
    *   Hình vẽ, biểu đồ mà không có nội dung văn bản đi kèm.
4.  **Không gộp các câu hỏi phụ:** Không được phép gộp các câu hỏi con không liên quan với nhau thành một, ngay cả khi chúng có chung một phần dẫn dắt ngắn. Hãy giữ chúng riêng biệt.
5.  **Không tự ý thêm nội dung:** Tuyệt đối không được suy diễn hay thêm thắt thông tin không có trong đề. Nếu một phần văn bản không rõ ràng hoặc mơ hồ, hãy giữ nguyên văn bản gốc.
6.  **order_index = CHỈ SỐ BÀI LỚN** (bắt đầu từ 1). Mọi ý nhỏ thuộc cùng BÀI LỚN phải có **cùng order_index**. Ví dụ: 2a, 2b, 2c → order_index = 2.
7)  **part_label** là NHÃN Ý NHỎ **đa cấp** (string), cho phép dạng "1.a", "2.b", "1.2.a", "(1).a", v.v.  
   - Nếu dạng "Câu IV.1.a": đặt `order_index = 4`, `part_label = "1.a"`.  
   - Nếu không có ý nhỏ, `part_label = ""`.  
   - Nên giữ nhãn gốc trong `text` nếu có (vd "Câu IV.1.a) …").
8.  **knowledge_topics** là những phần kiến thức hoặc kỹ thuật cần phải vận dụng để có thể thực hiện bài làm, càng chi tiết và chính xác tên gọi kiến thức hoặc kỹ thuật càng tốt. **BẮT BUỘC tối thiểu 3 tag, tối đa 5 tag**. 

NOTE: OUTPUT luôn luôn là tiếng Việt.

### **Lược đồ dữ liệu đầu ra (Bắt buộc tuân thủ):**

Kết quả phải là một mảng (Array) các đối tượng `QuestionItem`, trong đó mỗi đối tượng có cấu trúc như sau:

```json
[
  {
    "text": "string — Toàn bộ nội dung câu hỏi/ý nhỏ; đã làm sạch; GIỮ NHÃN GỐC nếu có (ví dụ: "Câu 1a) …").",
    "order_index": "integer (bắt đầu từ 1) — Số thứ tự của câu hỏi trong đề thi gốc.",
    "part_label": string    (có thể là "a", "1", "1.a", "1.2.a", hoặc rỗng)
    "knowledge_topics": string[] (tối thiểu 3 mục, tối đa 5 mục)
  }
]
```
"""

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _get_image_mime_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"

def analyze_exam_from_images(image_paths: List[str]) -> List[Dict[str, Any]]:
    """Phân tích đề thi từ hình ảnh và trả về danh sách câu hỏi"""
    logger.info(f"=== ANALYZE EXAM FROM IMAGES START ===")
    logger.info(f"Number of images: {len(image_paths)}")
    
    if not image_paths:
        logger.warning("No images provided")
        return []
    
    # Prepare image content for API
    image_contents = []
    for image_path in image_paths:
        try:
            base64_image = _encode_image(image_path)
            mime_type = _get_image_mime_type(image_path)
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            })
            logger.info(f"Encoded image: {image_path}")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            continue
    
    if not image_contents:
        logger.error("No valid images to process")
        return []
    
    # Prepare message content
    user_content = [
        {"type": "text", "text": "Phân tích hình ảnh đề thi sau và TRẢ VỀ DUY NHẤT JSON theo lược đồ đã nêu."}
    ]
    user_content.extend(image_contents)
    
    logger.info(f"Prepared {len(image_contents)} images for analysis")
    logger.info(f"Using model: {EXAM_ANALYZER_MODEL}")
    
    try:
        resp = _client.chat.completions.create(
            model=EXAM_ANALYZER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ANALYZE},
                {"role": "user", "content": user_content}
            ],
            max_completion_tokens=14000,
            temperature=LLM_TEMPERATURE,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "exam_analysis",
                    "schema": ANALYZE_SCHEMA
                }
            }
        )
        
        logger.info(f"API Response received")
        if hasattr(resp, 'usage') and resp.usage:
            logger.info(f"Token usage: {resp.usage}")
        
        raw_content = resp.choices[0].message.content
        logger.info(f"Raw response length: {len(raw_content)} chars")
        logger.info(f"Raw response: {raw_content}")
        
        data = json.loads(raw_content)
        logger.info(f"JSON parsed successfully")
        logger.info(f"Parsed data keys: {list(data.keys())}")
        
        if "questions" in data:
            logger.info(f"Number of questions found: {len(data['questions'])}")
            for i, q in enumerate(data['questions'][:3]):
                logger.info(f"Question {i+1}: {q}")
        else:
            logger.warning(f"No 'questions' key in response: {data}")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Raw content causing error: {raw_content}")
        return []
    except Exception as e:
        logger.error(f"API call error: {e}")
        return []
    
    out: List[Dict[str, Any]] = []
    for it in data.get("questions", []):
        question_item = {
            "text": str(it["text"]).strip(),
            "order_index": int(it["order_index"]),
            "part_label": str(it.get("part_label") or "").strip(),
            "knowledge_topics": [str(x).strip() for x in (it.get("knowledge_topics") or [])][:5],
        }
        out.append(question_item)
        logger.debug(f"Processed question: {question_item}")
        
    logger.info(f"=== ANALYZE EXAM FROM IMAGES END === Returning {len(out)} questions")
    return out