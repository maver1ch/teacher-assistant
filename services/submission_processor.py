# services/submission_processor.py
# Service cho Bước 3: Xử lý và phân đoạn bài làm từ hình ảnh

from __future__ import annotations

import os
import json
import logging
import base64
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from utils.config import API_KEY_ENV, SEGMENT_MODEL, LLM_TEMPERATURE
from utils.schemas import SEGMENT_SCHEMA

# Setup logger
logger = logging.getLogger(__name__)

load_dotenv()
_client = OpenAI(api_key=os.getenv(API_KEY_ENV))

# System prompt cho phân đoạn bài làm
SYSTEM_PROMPT_SEGMENT = """
Bạn là AI chuyên trích xuất nội dung bài làm học sinh từ skeleton có sẵn.

NHIỆM VỤ
- Nhận vào: (1) SKELETON có sẵn order_index/part_label/question_id và (2) toàn văn bài làm
- Chỉ tìm và điền answer_text cho từng item trong skeleton
- KHÔNG thay đổi order_index, part_label, question_id, position

QUY TẮC QUAN TRỌNG
1) Với mỗi item trong skeleton, tìm phần trả lời tương ứng trong bài làm
2) Kết hợp Dùng ngữ nghĩa (từ khóa, kiến thức) để khớp + ký hiệu đánh số để xác định (Ví dụ Bài 1.a, Bài 2.3 hoặc Bài 4.1.a, ...)
3) Nếu tìm thấy → điền vào answer_text (giữ nguyên LaTeX $/$$)
4) Nếu KHÔNG tìm thấy (học sinh không làm ý đó) → để answer_text = ""
5) KHÔNG tạo item mới, KHÔNG xóa item khỏi skeleton
6) Cho phép gộp nhiều đoạn của cùng câu thành chuỗi liên tục

LƯỢC ĐỒ JSON (STRICT)
- Input skeleton giữ nguyên structure
- Chỉ fill answer_text cho từng item
- Kết quả: {"items": [skeleton đã điền answer_text]}

IMPORTANT NOTE: OUTPUT luôn luôn là tiếng Việt (Ví dụ Vi-ét không phải Viéte)
VÍ DỤ:
Input skeleton: [{"question_id": 1, "order_index": 1, "part_label": "a", "position": 1, "answer_text": ""}]
Output: {"items": [{"question_id": 1, "order_index": 1, "part_label": "a", "position": 1, "answer_text": "x = 5 vì..."}]}
"""

def create_submission_skeleton(questions: List) -> List[Dict[str, Any]]:
    """Create pre-populated skeleton with fixed order_index/part_label"""
    skeleton = []
    for q in questions:
        skeleton.append({
            "question_id": q.id,
            "order_index": q.order_index,
            "part_label": q.part_label or "",
            "position": len(skeleton) + 1,
            "answer_text": ""  # Empty - to be filled by LLM
        })
    return skeleton

def segment_submission(questions: List, submission_text: str) -> Dict[str, Any]:
    """Phân đoạn bài làm từ text (legacy function)"""
    logger.info("=== SEGMENT SUBMISSION START ===")
    
    if not submission_text or not submission_text.strip():
        logger.warning("Submission text is empty. Returning empty segment list.")
        return {"items": []}

    # Create skeleton with pre-populated structure
    skeleton = create_submission_skeleton(questions)
    
    logger.info(f"Created skeleton with {len(skeleton)} items")
    
    user_msg = (
        "Dưới đây là (1) SKELETON có sẵn cấu trúc và (2) toàn văn bài làm. "
        "Hãy điền answer_text cho từng item trong skeleton và trả về JSON. Nếu như không tìm được câu tương ứng (tức là học sinh không làm bài thì phần answer_text để rỗng). \n\n"
        f"(1) SKELETON:\n{json.dumps(skeleton, ensure_ascii=False)}\n\n"
        "(2) SUBMISSION:\n" + submission_text.strip()
    )
            
    raw_content = ""
    try:
        resp = _client.chat.completions.create(
            model=SEGMENT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SEGMENT},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=10000,
            temperature=LLM_TEMPERATURE,
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
        
        if not raw_content or not raw_content.strip():
            logger.warning("API returned an empty string, possibly due to content filtering. Returning a valid empty dict.")
            return {"items": []}
            
        return json.loads(raw_content)

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError during segmentation: {e}")
        logger.error(f"Raw content that caused the error: {raw_content}")
        return {"items": []}
    except Exception as e:
        logger.error(f"An unexpected error occurred during segmentation API call: {e}")
        return {"items": []}

def segment_submission_from_images(questions: List, image_paths: List[str]) -> Dict[str, Any]:
    """
    Phân đoạn bài làm học sinh từ hình ảnh sử dụng GPT-4.1-mini vision model.
    
    Args:
        questions: List các Question objects từ database
        image_paths: List đường dẫn đến các file ảnh
        
    Returns:
        Dict với key "items" chứa list các segmented items
    """
    logger.info("=== SEGMENT SUBMISSION FROM IMAGES START ===")
    
    if not image_paths:
        logger.warning("No image paths provided. Returning empty segment list.")
        return {"items": []}

    # Create skeleton with pre-populated structure
    skeleton = create_submission_skeleton(questions)
    
    logger.info(f"Created skeleton with {len(skeleton)} items for {len(image_paths)} images")
    
    # Encode images to base64
    encoded_images = []
    for path in image_paths:
        try:
            with open(path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append(encoded)
        except Exception as e:
            logger.error(f"Failed to encode image {path}: {e}")
            continue
    
    if not encoded_images:
        logger.error("No images could be encoded successfully")
        return {"items": []}

    # Prepare messages for vision model
    content = [
        {
            "type": "text", 
            "text": (
                "Dưới đây là (1) SKELETON có sẵn cấu trúc và (2) hình ảnh bài làm học sinh. "
                "Hãy phân tích hình ảnh, trích xuất nội dung bài làm, sau đó điền answer_text cho từng item trong skeleton và trả về JSON. "
                "Nếu như không tìm được câu tương ứng (tức là học sinh không làm bài thì phần answer_text để rỗng). "
                "Giữ nguyên LaTeX format (sử dụng $ và $$).\n\n"
                f"(1) SKELETON:\n{json.dumps(skeleton, ensure_ascii=False)}"
            )
        }
    ]
    
    # Add images to content
    for i, encoded_img in enumerate(encoded_images):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_img}",
                "detail": "high"
            }
        })

    raw_content = ""
    try:
        resp = _client.chat.completions.create(
            model=SEGMENT_MODEL,  # gpt-4.1-mini
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SEGMENT},
                {"role": "user", "content": content}
            ],
            max_tokens=10000,
            temperature=LLM_TEMPERATURE,
            response_format={
                "type": "json_object"
            }
        )

        raw_content = resp.choices[0].message.content.strip()
        logger.info(f"Raw API response: {raw_content[:200]}...")

        if not raw_content:
            logger.warning("Empty response from vision model")
            return {"items": []}

        parsed = json.loads(raw_content)
        
        if "items" not in parsed:
            logger.warning("No 'items' key in vision model response")
            return {"items": []}

        items = parsed["items"]
        logger.info(f"Vision model returned {len(items)} segmented items")
        
        # Validate and clean items
        valid_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            
            # Required fields validation
            required_fields = ["question_id", "order_index", "position"]
            if not all(field in item for field in required_fields):
                logger.warning(f"Skipping item with missing fields: {item}")
                continue
                
            # Type conversion and cleaning
            try:
                clean_item = {
                    "question_id": int(item["question_id"]),
                    "order_index": int(item["order_index"]),
                    "part_label": str(item.get("part_label", "")).strip(),
                    "position": int(item["position"]),
                    "answer_text": str(item.get("answer_text", "")).strip()
                }
                valid_items.append(clean_item)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid item due to type conversion error: {item}, error: {e}")
                continue
        
        logger.info(f"Successfully processed {len(valid_items)} valid segmented items from vision model")
        return {"items": valid_items}

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in vision segmentation: {e}")
        logger.error(f"Raw content that failed to parse: {raw_content}")
        return {"items": []}
    except Exception as e:
        logger.error(f"Unexpected error in vision segmentation: {e}")
        return {"items": []}