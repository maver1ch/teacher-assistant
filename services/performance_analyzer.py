# services/performance_analyzer.py
import json
import logging
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from database.db_manager import db
from database.models import Grading
from utils.config import GROUPING_MODEL
from utils.schemas import PERFORMANCE_ANALYSIS_SCHEMA
from utils.llm_logger import log_llm_call
from utils.constants import (
    PERFORMANCE_ANALYSIS_USER_PROMPT_TEMPLATE,
    SERVICE_PERFORMANCE_ANALYSIS,
    INVALID_KNOWLEDGE_GAPS,
    INVALID_CALCULATION_ERRORS,
    ERROR_PERFORMANCE_ANALYSIS_FAILED,
    SUCCESS_PERFORMANCE_ANALYSIS
)
from utils.prompts import PERFORMANCE_ANALYSIS_SYSTEM_PROMPT

# Initialize OpenAI client
load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def analyze_submission_performance(submission_id: int) -> Dict[str, List[Dict[str, Any]]]:
    """Analyzes all grading results for a submission and groups common mistakes."""
    logger.info(f"=== PERFORMANCE ANALYSIS START === Submission ID: {submission_id}")
    logger.info("Performing fresh analysis from latest grading data...")
    with db.get_session() as session:
        gradings = session.query(Grading).filter(Grading.submission_id == submission_id).all()
    
    logger.info(f"Found {len(gradings)} grading records from database")
    if not gradings:
        logger.info("No gradings found, returning empty result")
        return {"knowledge_summary": [], "error_summary": []}

    # Prepare data for the LLM
    error_data, skipped_count = _prepare_analysis_data(gradings)
            
    logger.info(f"Data preparation complete: {len(error_data)} valid items, {skipped_count} items skipped")
    if not error_data:
        logger.info("No valid error data found after filtering, returning empty result")
        return {"knowledge_summary": [], "error_summary": []}

    # Log the input data that will be sent to LLM
    logger.info("=== INPUT DATA FOR LLM ===")
    for i, item in enumerate(error_data):
        logger.info(f"Item {i+1}: {item['question_label']}")
        logger.info(f"  Knowledge gaps: {item['knowledge_gaps']}")
        logger.info(f"  Calculation errors: {item['calculation_logic_errors']}")
    logger.info("=== END INPUT DATA ===")
    print(f"\n[PERFORMANCE ANALYZER] Input data summary: {len(error_data)} items prepared for LLM analysis")
    print(f"Input data: {json.dumps(error_data, ensure_ascii=False, indent=2)}")

    try:
        analysis_items = _call_performance_analysis_llm(error_data)
        logger.info(f"LLM returned {len(analysis_items)} analysis groups")
        
        # Lưu kết quả vào database
        if analysis_items:
            logger.info(f"Saving {len(analysis_items)} analysis items to database")
            db.save_performance_analysis(submission_id, analysis_items)
        else:
            logger.info("No analysis items to save")
        
        # Chuyển đổi format về dạng cũ cho backward compatibility
        knowledge_summary = []
        error_summary = []
        
        for item in analysis_items:
            analysis_item = {
                "group_name": item.get("group", ""),
                "description": item.get("description", ""),
                "related_questions": item.get("questions", [])
            }
            
            if item.get("type") == "knowledge":
                knowledge_summary.append(analysis_item)
            elif item.get("type") == "error":
                error_summary.append(analysis_item)
        
        logger.info(f"=== PERFORMANCE ANALYSIS COMPLETE === Final result: {len(knowledge_summary)} knowledge + {len(error_summary)} error groups")
        print(f"[PERFORMANCE ANALYZER] Analysis complete: {len(knowledge_summary)} knowledge groups, {len(error_summary)} error groups")
        return {"knowledge_summary": knowledge_summary, "error_summary": error_summary}
        
    except Exception as e:
        error_msg = ERROR_PERFORMANCE_ANALYSIS_FAILED.format(str(e))
        logger.error(error_msg)
        print(f"[PERFORMANCE ANALYZER ERROR] {error_msg}")
        return {"knowledge_summary": [], "error_summary": []}


# =====================
# Helper Functions
# =====================

def _prepare_analysis_data(gradings: List[Grading]) -> tuple[List[Dict[str, Any]], int]:
    """Prepare grading data for LLM analysis"""
    error_data = []
    skipped_count = 0
    
    for g in gradings:
        # Skip items that were not done or had system errors
        gaps = json.loads(g.knowledge_gaps or "[]")
        errors = json.loads(g.calculation_logic_errors or "[]")
        
        is_valid_gap = any(gap not in INVALID_KNOWLEDGE_GAPS for gap in gaps)
        is_valid_error = any(error not in INVALID_CALCULATION_ERRORS for error in errors)
        
        if is_valid_gap or is_valid_error:
            error_data.append({
                "question_label": g.question_label or "N/A",
                "knowledge_gaps": gaps if is_valid_gap else [],
                "calculation_logic_errors": errors if is_valid_error else []
            })
        else:
            skipped_count += 1
    
    return error_data, skipped_count


def _call_performance_analysis_llm(error_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Call LLM for performance analysis"""
    logger.info(f"Calling LLM with model: {GROUPING_MODEL}")
    print(f"\n[PERFORMANCE ANALYZER] Calling LLM for analysis...")
    
    # Tách ra 2 danh sách riêng biệt
    knowledge_gaps_list = []
    calculation_errors_list = []
    
    for item in error_data:
        question_label = item["question_label"]
        
        # Thêm knowledge gaps
        for gap in item["knowledge_gaps"]:
            knowledge_gaps_list.append(f"Câu {question_label}: {gap}")
        
        # Thêm calculation errors  
        for error in item["calculation_logic_errors"]:
            calculation_errors_list.append(f"Câu {question_label}: {error}")
    
    user_prompt = PERFORMANCE_ANALYSIS_USER_PROMPT_TEMPLATE.format(
        json.dumps(knowledge_gaps_list, ensure_ascii=False, indent=2),
        json.dumps(calculation_errors_list, ensure_ascii=False, indent=2)
    )
    
    try:
        resp = _client.chat.completions.create(
            model=GROUPING_MODEL,
            messages=[
                {"role": "system", "content": PERFORMANCE_ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "performance_analysis",
                    "schema": PERFORMANCE_ANALYSIS_SCHEMA
                }
            }
        )
        log_llm_call(response=resp, model_name=GROUPING_MODEL, service_name=SERVICE_PERFORMANCE_ANALYSIS)
        analysis_result = json.loads(resp.choices[0].message.content)
        return analysis_result.get("analysis", [])
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return []