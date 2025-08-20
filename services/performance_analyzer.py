# services/performance_analyzer.py
import os
import json
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from database.db_manager import db
from database.models import Grading
from utils.config import GROUPING_MODEL
from utils.schemas import PERFORMANCE_ANALYSIS_SCHEMA
from utils.llm_logger import log_llm_call

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

load_dotenv()
_client = OpenAI(api_key=os.getenv(os.getenv("OPENAI_API_KEY")))

SYSTEM_PROMPT_ANALYZE_PERFORMANCE = """
Bạn là một chuyên gia phân tích giáo dục, có khả năng nhận diện các mẫu hình sai sót và lỗ hổng kiến thức từ kết quả chấm bài của học sinh.

NHIỆM VỤ:
- Nhận vào một danh sách các lỗi sai và lỗ hổng kiến thức từ bài làm của học sinh.
- Phân tích và nhóm (grouping) các vấn đề này thành các hạng mục có ý nghĩa với **PHẠM VI RỘNG**.
- Với mỗi nhóm, đưa ra một tên gọi **TỔNG QUÁT**, một mô tả về bản chất vấn đề, và liệt kê các câu hỏi liên quan.
- Trả về kết quả dưới dạng JSON nghiêm ngặt theo schema đã cho.

NGUYÊN TẮC GROUPING RỘNG:
1. **Tìm kiếm sự tương đồng**: Nhóm các lỗi có cùng **LĨNH VỰC KIẾN THỨC** hoặc **LOẠI SAI SỐT**.
2. **Ưu tiên group names rộng và có ý nghĩa**:
   - ✅ GOOD: "Vấn đề về hàm số và đồ thị"
   - ❌ BAD: "Không tìm được giao điểm" 
3. **Mỗi group phải cover nhiều lỗi**: Một group tối thiểu nên chứa 2+ lỗi tương tự.
4. **Hierarchy thinking**: Tư duy theo thứ bậc từ cụ thể → chung:
   - Lỗi cụ thể: "Tính sai (-2)×3"
   - Lỗi loại: "Lỗi tính toán cơ bản"  
   - Vấn đề tổng quát: "Kỹ năng tính toán và cẩn thận"

VÍ DỤ GROUPING TỐT:
- "Lỗ hổng kiến thức đại số cơ bản" (thay vì "Không nhớ công thức")
- "Vấn đề về phương trình và bất phương trình" (thay vì "Giải sai phương trình")
- "Kỹ năng tính toán và độ chính xác" (thay vì "Tính sai")
"""

def analyze_submission_performance(submission_id: int) -> Dict[str, List[Dict[str, Any]]]:
    """Analyzes all grading results for a submission and groups common mistakes."""
    logger.info(f"=== PERFORMANCE ANALYSIS START === Submission ID: {submission_id}")
    
    # Kiểm tra xem đã có analysis lưu sẵn chưa
    saved_analysis = db.get_performance_analysis(submission_id)
    if saved_analysis:
        logger.info(f"Found existing analysis in database: {len(saved_analysis)} items")
        # Chuyển đổi format từ database về format cũ
        knowledge_summary = []
        error_summary = []
        
        for item in saved_analysis:
            analysis_item = {
                "group_name": item["group"],
                "description": item["description"],
                "related_questions": item["questions"]
            }
            
            if item["type"] == "knowledge":
                knowledge_summary.append(analysis_item)
            elif item["type"] == "error":
                error_summary.append(analysis_item)
        
        logger.info(f"Returning cached analysis: {len(knowledge_summary)} knowledge + {len(error_summary)} error groups")
        return {"knowledge_summary": knowledge_summary, "error_summary": error_summary}
    
    # Nếu chưa có, thực hiện phân tích mới
    logger.info("No cached analysis found, performing new analysis...")
    with db.get_session() as session:
        gradings = session.query(Grading).filter(Grading.submission_id == submission_id).all()
    
    logger.info(f"Found {len(gradings)} grading records from database")
    if not gradings:
        logger.info("No gradings found, returning empty result")
        return {"knowledge_summary": [], "error_summary": []}

    # Prepare data for the LLM
    error_data = []
    skipped_count = 0
    for g in gradings:
        # Skip items that were not done or had system errors
        gaps = json.loads(g.knowledge_gaps or "[]")
        errors = json.loads(g.calculation_logic_errors or "[]")
        
        is_valid_gap = any(gap not in ["Chưa làm", "Không thể phân tích do lỗi hệ thống"] for gap in gaps)
        is_valid_error = any(error not in ["Chưa làm", "Không có"] for error in errors)
        
        if is_valid_gap or is_valid_error:
            error_data.append({
                "question_label": g.question_label or "N/A",
                "knowledge_gaps": gaps if is_valid_gap else [],
                "calculation_logic_errors": errors if is_valid_error else []
            })
        else:
            skipped_count += 1
            
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

    user_prompt = (
        "Dưới đây là danh sách chi tiết các lỗi từ bài làm của một học sinh. "
        "Hãy phân tích và nhóm chúng lại theo hướng dẫn.\n\n"
        f"{json.dumps(error_data, ensure_ascii=False, indent=2)}"
    )

    try:
        logger.info(f"Calling LLM with model: {GROUPING_MODEL}")
        print(f"\n[PERFORMANCE ANALYZER] Calling LLM for analysis...")
        
        resp = _client.chat.completions.create(
            model=GROUPING_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ANALYZE_PERFORMANCE},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "performance_analysis",
                    "schema": PERFORMANCE_ANALYSIS_SCHEMA
                }
            }
        )
        log_llm_call(response=resp, model_name=GROUPING_MODEL, service_name="performance_analysis")
        
        # Parse response
        analysis_result = json.loads(resp.choices[0].message.content)
        analysis_items = analysis_result.get("analysis", [])
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
        error_msg = f"Error during performance analysis: {e}"
        logger.error(error_msg)
        print(f"[PERFORMANCE ANALYZER ERROR] {error_msg}")
        return {"knowledge_summary": [], "error_summary": []}