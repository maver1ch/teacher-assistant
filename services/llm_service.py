#llm_service.py

from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

# Setup logger
logger = logging.getLogger(__name__)

# ---------- Constants
API_KEY_ENV = "OPENAI_API_KEY"
MODEL_NAME = "o4-mini-2025-04-16"
TEMPERATURE = 0.1

load_dotenv()
_client = OpenAI(api_key=os.getenv(API_KEY_ENV))

# ---------- Fixed System Prompt (adapted from user's instruction)
SYSTEM_PROMPT_ANALYZE = """
Bạn là một AI chuyên gia phân tích đề thi, được huấn luyện đặc biệt để xử lý các đề thi trắc nghiệm và tự luận của Việt Nam trong lĩnh vực Toán học.

### **Mục tiêu chính:**

Nhiệm vụ của bạn là đọc và phân tích một văn bản đề thi, sau đó thực hiện các yêu cầu sau:

-   **Trích xuất** từng câu hỏi riêng lẻ thành một mục dữ liệu độc lập. (các ý nhỏ a,b,c hoặc 1,2,3)
-   **Ước tính độ khó** của mỗi câu hỏi theo thang điểm 10 (trong đó 1 là rất dễ và 10 là cực khó, vận dụng cao trở lên).
-   **Chỉ trả về kết quả** dưới định dạng JSON nghiêm ngặt (strict JSON).
-   Trả về JSON nghiêm ngặt theo lược đồ yêu cầu.

### **Các quy tắc xử lý:**

Bạn phải tuân thủ nghiêm ngặt các quy tắc sau đây trong quá trình phân tích:

1.  **Giữ nguyên vẹn công thức toán học:** Tất cả các công thức, ký hiệu LaTeX, và biểu thức toán học phải được giữ nguyên văn, không được thay đổi hay chuyển đổi.
2.  **Tách các câu hỏi đa phần:** Những câu hỏi có các phần nhỏ (ví dụ: Câu 1a, 1b, 1c) phải được tách thành các mục riêng biệt, nhưng vẫn giữ đúng thứ tự tương đối của chúng (1a rồi đến 1b).
3.  **Loại bỏ thông tin thừa:** Tự động xóa bỏ các thành phần không phải là nội dung của câu hỏi, bao gồm:
    *   Đầu trang và chân trang (headers/footers).
    *   Số trang.
    *   Thông tin về Sở Giáo dục, tên trường, tên kỳ thi (ví dụ: "SỞ GIÁO DỤC VÀ ĐÀO TẠO HÀ NỘI", "ĐỀ THI CHÍNH THỨC").
    *   Hướng dẫn cho thí sinh (ví dụ: "Thí sinh không được sử dụng tài liệu").
    *   Bảng điểm, hướng dẫn chấm điểm hoặc đáp án.
    *   Các ký hiệu kết thúc đề thi như "---HẾT---".
4.  **Không gộp các câu hỏi phụ:** Không được phép gộp các câu hỏi con không liên quan với nhau thành một, ngay cả khi chúng có chung một phần dẫn dắt ngắn. Hãy giữ chúng riêng biệt.
5.  **Không tự ý thêm nội dung:** Tuyệt đối không được suy diễn hay thêm thắt thông tin không có trong đề. Nếu một phần văn bản không rõ ràng hoặc mơ hồ, hãy giữ nguyên văn bản gốc.
6.  **order_index = CHỈ SỐ BÀI LỚN** (bắt đầu từ 1). Mọi ý nhỏ thuộc cùng BÀI LỚN phải có **cùng order_index**. Ví dụ: 2a, 2b, 2c → order_index = 2.
7)  **part_label** là NHÃN Ý NHỎ **đa cấp** (string), cho phép dạng “1.a”, “2.b”, “1.2.a”, “(1).a”, v.v.  
   - Nếu dạng “Câu IV.1.a”: đặt `order_index = 4`, `part_label = "1.a"`.  
   - Nếu không có ý nhỏ, `part_label = ""`.  
   - Nên giữ nhãn gốc trong `text` nếu có (vd “Câu IV.1.a) …”).
8.  **knowledge_topics** là những phần kiến thức hoặc kỹ thuật cần phải vận dụng để có thể thực hiện bài làm, càng chi tiết và chính xác tên gọi kiến thức hoặc kỹ thuật càng tốt. 

### **Hệ thống đánh giá độ khó:**

Sử dụng thang điểm từ 1 đến 10 dựa trên các tiêu chí sau, tương ứng với 4 mức độ phân loại trong các kỳ thi của Việt Nam:

-   **Mức 1-3 (Nhận biết):** Các câu hỏi yêu cầu nhớ lại kiến thức cơ bản, áp dụng trực tiếp một công thức hoặc định nghĩa. Thường chỉ cần một bước tính toán hoặc suy luận đơn giản.
-   **Mức 4-5 (Thông hiểu ):** Các câu hỏi đòi hỏi sự hiểu biết sâu hơn về khái niệm, có khả năng diễn giải và áp dụng kiến thức vào các tình huống quen thuộc. Thường yêu cầu nhiều bước suy luận và tính toán theo một quy trình chuẩn.
-   **Mức 6-8 (Vận dụng thấp):** Các câu hỏi phức tạp, đòi hỏi khả năng phân tích, tổng hợp kiến thức từ nhiều chuyên đề khác nhau. Thường có các yếu tố gây nhiễu hoặc các ràng buộc ẩn, cần tư duy sáng tạo để giải quyết.
-   **Mức 9-10 (Vận dụng cao / Cấp độ thi chuyên):** Những câu hỏi cực khó, đòi hỏi khả năng chứng minh, suy luận toán học sâu sắc, hoặc sử dụng các phương pháp giải quyết vấn đề không theo khuôn mẫu. Đây là những câu hỏi dùng để phân loại học sinh giỏi.

### **Lược đồ dữ liệu đầu ra (Bắt buộc tuân thủ):**

Kết quả phải là một mảng (Array) các đối tượng `QuestionItem`, trong đó mỗi đối tượng có cấu trúc như sau:

```json
[
  {
    "text": "string — Toàn bộ nội dung câu hỏi/ý nhỏ; đã làm sạch; GIỮ NHÃN GỐC nếu có (ví dụ: “Câu 1a) …”).",
    "difficulty": "integer (từ 1 đến 10) — Mức độ khó của câu hỏi được ước tính.",
    "order_index": "integer (bắt đầu từ 1) — Số thứ tự của câu hỏi trong đề thi gốc.",
    "part_label": string    (có thể là “a”, “1”, “1.a”, “1.2.a”, hoặc rỗng)
    "knowledge_topics": string[] (tối đa 4 mục)
  }
]
```
"""

SYSTEM_PROMPT_SEGMENT = """
Bạn là AI trích xuất/đối sánh bài làm học sinh với danh sách câu hỏi đã cho.

NHIỆM VỤ
- Nhận vào (1) danh sách câu hỏi (rút gọn) và (2) toàn văn bài làm đã OCR.
- Cắt bài làm thành các đoạn tương ứng từng câu hỏi.
- Trả về JSON nghiêm ngặt: mỗi đoạn gắn đúng question_id, giữ nguyên LaTeX ($/$$) và thứ tự xuất hiện.

QUY TẮC
1) Dùng ngữ nghĩa (từ khóa, kiến thức) để khớp; KHÔNG phụ thuộc ký hiệu đánh số trong bài làm.
2) Nếu không thấy phần trả lời cho một câu → tạo item với answer_text = "".
3) Cho phép gộp nhiều đoạn của cùng một câu thành một chuỗi liên tục (giữ thứ tự).
4) Không thêm bớt nội dung ngoài bài làm.

LƯỢC ĐỒ JSON (STRICT)
- Kết quả chính: items: Array<AnswerItem>
- Mỗi AnswerItem:
  - question_id: integer
  - order_index: integer  (BÀI LỚN của câu hỏi)
  - part_label: string    (vd "a"/"b"/"c" hoặc "")
  - position: integer     (thứ tự xuất hiện đoạn trong bài làm, bắt đầu từ 1)
  - answer_text: string   (đoạn trả lời, giữ nguyên LaTeX)
"""

@dataclass
class QuestionLite:
    question_id: int
    order_index: int
    part_label: str
    text_short: str
    keywords: List[str]

# ---------- JSON Schemas for OpenAI
ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "difficulty": {"type": "integer"},
                    "order_index": {"type": "integer"},
                    "part_label": {"type": "string"},
                    "knowledge_topics": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["text", "difficulty", "order_index", "part_label", "knowledge_topics"]
            }
        }
    },
    "required": ["questions"]
}

SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "integer"},
                    "order_index": {"type": "integer"},
                    "part_label": {"type": "string"},
                    "position": {"type": "integer"},
                    "answer_text": {"type": "string"}
                },
                "required": ["question_id", "order_index", "part_label", "position", "answer_text"]
            }
        }
    },
    "required": ["items"]
}

# ---------- Public APIs
def analyze_exam(exam_text: str) -> List[Dict[str, Any]]:
    logger.info(f"=== ANALYZE EXAM START ===")
    logger.info(f"Input text length: {len(exam_text)} chars")
    logger.info(f"Input text preview: {exam_text[:200]}...")
    
    prompt = (
        "Phân tích văn bản đề thi sau và TRẢ VỀ DUY NHẤT JSON theo lược đồ đã nêu.\n\n"
        f"{exam_text.strip()}"
    )
    
    logger.info(f"Prompt length: {len(prompt)} chars")
    logger.info(f"Using model: {MODEL_NAME}")
    
    try:
        resp = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ANALYZE},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=14000,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "exam_analysis",
                    "schema": ANALYZE_SCHEMA
                }
            },
            #reasoning_effort="low"
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
            for i, q in enumerate(data['questions'][:3]):  # Log first 3 questions
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
            "difficulty": int(it["difficulty"]),
            "order_index": int(it["order_index"]),
            "part_label": str(it.get("part_label") or "").strip(),
            "knowledge_topics": [str(x).strip() for x in (it.get("knowledge_topics") or [])][:4],
        }
        out.append(question_item)
        logger.debug(f"Processed question: {question_item}")
        
    logger.info(f"=== ANALYZE EXAM END === Returning {len(out)} questions")
    return out

def segment_submission(exam_outline: List[QuestionLite], submission_text: str) -> Dict[str, Any]:
    logger.info("=== SEGMENT SUBMISSION START ===")
    
    if not submission_text or not submission_text.strip():
        logger.warning("Submission text is empty. Returning empty segment list.")
        return {"items": []}

    outline_min = [
        {
            "question_id": q.question_id,
            "order_index": q.order_index,
            "part_label": q.part_label,
            "text_short": q.text_short[:200],
            "keywords": q.keywords[:5],
        }
        for q in exam_outline
    ]

    user_msg = (
        "Dưới đây là (1) danh sách câu hỏi rút gọn và (2) toàn văn bài làm. "
        "Hãy cắt bài làm thành các phần tương ứng và trả về JSON theo lược đồ.\n\n"
        f"(1) OUTLINE:\n{json.dumps(outline_min, ensure_ascii=False)}\n\n"
        "(2) SUBMISSION:\n" + submission_text.strip()
    )
    
    raw_content = "" # Khởi tạo biến để truy cập được trong khối except
    try:
        resp = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SEGMENT},
                {"role": "user", "content": user_msg}
            ],
            max_completion_tokens=14000,
            #temperature=TEMPERATURE,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "submission_segmentation",
                    "schema": SEGMENT_SCHEMA
                }
            }
        )
        
        raw_content = resp.choices[0].message.content
        logger.info(f"API response for segmentation received. Length: {len(raw_content)} chars.")
        
        # 1. KIỂM TRA CHUỖI RỖNG: Nếu rỗng, trả về dictionary rỗng hợp lệ
        if not raw_content or not raw_content.strip():
            logger.warning("API returned an empty string, possibly due to content filtering. Returning a valid empty dict.")
            return {"items": []}
            
        # 2. PARSE JSON: Nếu không rỗng, tiến hành parse
        return json.loads(raw_content)

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError during segmentation: {e}")
        logger.error(f"Raw content that caused the error: {raw_content}")
        # Trả về dictionary rỗng hợp lệ khi JSON không đúng định dạng
        return {"items": []}
    except Exception as e:
        logger.error(f"An unexpected error occurred during segmentation API call: {e}")
        # Trả về dictionary rỗng hợp lệ cho mọi lỗi khác
        return {"items": []}