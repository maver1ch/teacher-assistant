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
from utils.prompts import ANALYZE_SYSTEM_PROMPT
from utils.llm_logger import log_llm_call

# Setup logger
logger = logging.getLogger(__name__)

load_dotenv()
_client = OpenAI(api_key=os.getenv(API_KEY_ENV))


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
                {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
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
        log_llm_call(response=resp, model_name=EXAM_ANALYZER_MODEL, service_name="exam_analysis")
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