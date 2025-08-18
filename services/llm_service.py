#llm_service.py

from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from utils.prompts import SYSTEM_PROMPT_ANALYZE, SYSTEM_PROMPT_SEGMENT
from utils.config import API_KEY_ENV, MODEL_NAME, SEGMENT_MODEL, LLM_TEMPERATURE
from utils.schemas import ANALYZE_SCHEMA, SEGMENT_SCHEMA
from utils.data_models import QuestionLite

# Setup logger
logger = logging.getLogger(__name__)

load_dotenv()
_client = OpenAI(api_key=os.getenv(API_KEY_ENV))

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

# ---------- JSON Schemas imported from utils/schemas.py

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

def segment_submission(questions: List, submission_text: str) -> Dict[str, Any]:
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
            
    raw_content = "" # Khởi tạo biến để truy cập được trong khối except
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