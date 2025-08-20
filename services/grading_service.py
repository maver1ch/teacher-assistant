from __future__ import annotations

import os
import json
from typing import List, Dict, Optional, Tuple, Any

from dotenv import load_dotenv
from openai import OpenAI

from utils.prompts import GRADING_SYSTEM_PROMPT
from utils.config import MODEL_GRADING_ADVANCED
from utils.schemas import GRADING_SCHEMA
from utils.data_models import GradingResult
from utils.llm_logger import log_llm_call
from utils.constants import (
    GRADING_USER_PROMPT_TEMPLATE,
    SERVICE_GRADING_COMPARISON_ADVANCED,
    DEFAULT_MISSING_ANSWER,
    ERROR_SOLUTION_NOT_FOUND
)

# Database
from database.db_manager import db
from database.models import Question, Submission, SubmissionItem, Grading
from services.solution_service import get_solution_by_question
from services.grading.report_builder import report_builder
from services.grading.statistics_calculator import stats_calculator

# Initialize OpenAI client
load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =====================
# Data structures imported from utils/data_models.py
# =====================

# =====================
# Public API
# =====================

def grade_with_solution_comparison(question_id: int, student_answer: str, difficulty: int = None) -> Dict[str, Any]:
    """Grade by comparing student answer with standard solution and rubric"""
    solution = get_solution_by_question(question_id)
    if not solution:
        raise ValueError(ERROR_SOLUTION_NOT_FOUND.format(question_id))
    
    payload = {
        "final_answer": solution["final_answer"],             # Đáp án chuẩn
        "reasoning_approach": solution["reasoning_approach"], # BAREM chấm điểm
        "student_answer": student_answer                      # Bài làm học sinh
    }
    
    # Always use GPT-5 mini with low reasoning effort for all grading
    return _call_grading_ai(payload)

def _call_grading_ai(payload: Dict) -> Dict[str, Any]:
    """Call OpenAI GPT-5 mini with reasoning to grade with solution comparison"""
    user_content = GRADING_USER_PROMPT_TEMPLATE.format(
        payload['final_answer'],
        payload['reasoning_approach'],
        payload['student_answer']
    )
    
    try:
        resp = _client.chat.completions.create(
            model=MODEL_GRADING_ADVANCED,
            messages=[
                {"role": "system", "content": GRADING_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "grading_comparison_result",
                    "schema": GRADING_SCHEMA
                }
            }
        )
        log_llm_call(response=resp, model_name=MODEL_GRADING_ADVANCED, service_name=SERVICE_GRADING_COMPARISON_ADVANCED)
        return json.loads(resp.choices[0].message.content)
        
    except Exception as e:
        return {
            "knowledge_gaps": ["Không thể phân tích do lỗi hệ thống"],
            "calculation_logic_errors": [],
            "is_correct": False
        }

def grade_submission(submission_id: int) -> List[GradingResult]:
    """Grade all matched (question, answer) pairs for a submission.
    Why: single entry point for app.py.
    """
    submission = _get_submission(submission_id)
    if not submission:
        return []

    exam_id = submission.exam_id
    questions = db.get_questions_by_exam(exam_id)
    items = db.get_submission_items(submission_id)

    # Build maps for matching and context
    q_map_qa, mismatches = _match_pairs(questions, items)

    results: List[GradingResult] = []

    # Process by order_index groups to preserve dependency chain
    for order_index in sorted(q_map_qa.keys()):
        pairs = q_map_qa[order_index]
        # Sort by student's position (already preserved in _match_pairs)
        context_stack: List[Tuple[Question, SubmissionItem]] = []
        
        for q, a in pairs:
            if not (a.answer_text or "").strip():
                # Create grading result for missing answer but don't save to DB
                question_label = f"{q.order_index}{q.part_label or ''}"
                results.append(
                    GradingResult(
                        submission_id=submission_id,
                        question_id=q.id,
                        order_index=q.order_index,
                        part_label=(getattr(q, "part_label", None) or ""),
                        knowledge_gaps=DEFAULT_MISSING_ANSWER,
                        calculation_logic_errors=DEFAULT_MISSING_ANSWER,
                        is_correct=False,
                    )
                )
                continue

            # Sử dụng solution comparison grading mới với difficulty
            question_label = f"{q.order_index}{q.part_label or ''}"
            grading_data = grade_with_solution_comparison(q.id, a.answer_text, q.difficulty)
            
            results.append(
                GradingResult(
                    submission_id=submission_id,
                    question_id=q.id,
                    order_index=q.order_index,
                    part_label=(getattr(q, "part_label", None) or ""),
                    knowledge_gaps=grading_data["knowledge_gaps"],
                    calculation_logic_errors=grading_data["calculation_logic_errors"],
                    is_correct=grading_data["is_correct"],
                )
            )
            # Extend context after grading current item
            context_stack.append((q, a))

    # Note: mismatches intentionally ignored here (teacher fixes in Step 3)
    return results


# Delegated to extracted modules
def calculate_statistics(compact_data: List[Dict]) -> Dict[str, Any]:
    """Calculate statistics from grading data - DEPRECATED, use stats_calculator"""
    return stats_calculator.calculate_basic_statistics(compact_data)

def build_final_report(submission_id: int) -> str:
    """Build a student-friendly Markdown summary - DEPRECATED, use report_builder"""
    return report_builder.build_report(submission_id)

def save_grading_results(results: List[GradingResult]) -> bool:
    """Save grading results to database"""
    if not results:
        print("No grading results to save")
        return False
    
    try:
        print(f"Saving {len(results)} grading results to database...")
        for i, result in enumerate(results):
            question_label = f"{result.order_index}{result.part_label}"
            print(f"Saving result {i+1}/{len(results)}: Question {question_label}")
            
            if result.knowledge_gaps == ["Chưa làm"]:
                # Handle missing answers
                _create_missing_grading_from_result(result)
            else:
                # Handle normal grading results
                _save_grading_new(
                    result.submission_id,
                    result.question_id,
                    question_label,
                    result.knowledge_gaps,
                    result.calculation_logic_errors,
                    result.is_correct
                )
        print("All grading results saved successfully!")
        return True
    except Exception as e:
        print(f"Error saving grading results: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_or_generate_report(submission_id: int) -> str:
    """Get saved report from DB, or generate new one if not exists"""
    return report_builder.get_or_generate_report(submission_id)

# =====================
# Internals
# =====================

def _get_submission(submission_id: int) -> Optional[Submission]:
    with db.get_session() as session:
        return session.query(Submission).filter(Submission.id == submission_id).first()


def _match_pairs(questions: List[Question], items: List[SubmissionItem]):
    """Return mapping: order_index -> list of (Question, SubmissionItem), mismatches list.
    Why: priority by explicit question_id; else by (order_index, part_label).
    """
    # Build lookup by (order_index, part_label)
    q_lookup: Dict[Tuple[int, str], Question] = {}
    for q in questions:
        key = (q.order_index, getattr(q, "part_label", None) or "")
        q_lookup[key] = q

    # Group items by order_index in order of appearance (position)
    pairs_by_order: Dict[int, List[Tuple[Question, SubmissionItem]]] = {}
    mismatches: List[SubmissionItem] = []

    for a in items:
        q: Optional[Question] = None
        if getattr(a, "question_id", None):
            # Explicit mapping from Step 3
            q = next((x for x in questions if x.id == a.question_id), None)
        else:
            key = (a.order_index, getattr(a, "part_label", None) or "")
            q = q_lookup.get(key)

        if q is None:
            mismatches.append(a)
            continue

        pairs_by_order.setdefault(q.order_index, []).append((q, a))

    return pairs_by_order, mismatches

def _save_grading_new(submission_id: int, question_id: int, question_label: str, knowledge_gaps: List[str], 
                     calculation_logic_errors: List[str], is_correct: bool):
    knowledge_gaps_json = json.dumps(knowledge_gaps, ensure_ascii=False)
    calculation_errors_json = json.dumps(calculation_logic_errors, ensure_ascii=False)
    
    with db.get_session() as session:
        row = (
            session.query(Grading)
            .filter(Grading.submission_id == submission_id, Grading.question_id == question_id)
            .first()
        )
        if not row:
            row = Grading(
                submission_id=submission_id,
                question_id=question_id,
                question_label=question_label, 
                knowledge_gaps=knowledge_gaps_json,
                calculation_logic_errors=calculation_errors_json,
                is_correct=1 if is_correct else 0,
                final_score=None,
            )
            session.add(row)
        else:
            row.question_label = question_label 
            row.knowledge_gaps = knowledge_gaps_json
            row.calculation_logic_errors = calculation_errors_json
        session.commit()

def _create_missing_grading(question: Question, submission_id: int):
    """Tạo grading record cho câu học sinh không làm."""
    question_label = f"{question.order_index}{question.part_label or ''}"
    # Lưu grading record với thông báo "Chưa làm bài"
    _save_grading_new(
        submission_id=submission_id,
        question_label=question_label,
        question_id=question.id,
        knowledge_gaps=DEFAULT_MISSING_ANSWER,
        calculation_logic_errors=DEFAULT_MISSING_ANSWER,
        is_correct=False
    )

def _create_missing_grading_from_result(result: GradingResult):
    """Create missing grading record from GradingResult"""
    question_label = f"{result.order_index}{result.part_label}"
    _save_grading_new(
        submission_id=result.submission_id,
        question_id=result.question_id,
        question_label=question_label,
        knowledge_gaps=result.knowledge_gaps,
        calculation_logic_errors=result.calculation_logic_errors,
        is_correct=result.is_correct
    )

def _safe_json_loads(s: Optional[str]):
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []