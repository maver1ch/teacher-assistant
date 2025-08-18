from __future__ import annotations

import os
import json
from typing import List, Dict, Optional, Tuple, Any

from dotenv import load_dotenv
from openai import OpenAI
from utils.prompts import GRADING_SYSTEM_PROMPT
from utils.config import MODEL_GRADING, COMMENT_MODEL, GRADING_TEMPERATURE, CTX_MAX_CHARS_QUESTION, CTX_MAX_CHARS_ANSWER
from utils.schemas import GRADING_SCHEMA
from utils.data_models import GradingResult

# Database
from database.db_manager import db
from database.models import Question, Submission, SubmissionItem, Grading, QuestionSolution
from services.solution_service import get_solution_by_question

# =====================
# Client bootstrap
# =====================
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("Missing OPENAI_API_KEY")
_client = OpenAI(api_key=_api_key)


# =====================
# Data structures imported from utils/data_models.py
# =====================

# =====================
# Public API
# =====================

def grade_with_solution_comparison(question_id: int, student_answer: str) -> Dict[str, Any]:
    """Grade by comparing student answer with standard solution and rubric"""
    solution = get_solution_by_question(question_id)
    if not solution:
        raise ValueError(f"Không tìm thấy lời giải chuẩn cho question_id {question_id}")
    
    payload = {
        "solution_text": solution["solution_text"],          # Hướng logic giải
        "final_answer": solution["final_answer"],            # Đáp án chuẩn
        "reasoning_approach": solution["reasoning_approach"], # BAREM chấm điểm
        "student_answer": student_answer                     # Bài làm học sinh
    }
    
    return _call_grading_ai(payload)

def _call_grading_ai(payload: Dict) -> Dict[str, Any]:
    """Call OpenAI to grade with solution comparison"""
    user_content = (
        "So sánh bài làm học sinh với lời giải chuẩn và barem chấm điểm:\n\n"
        f"**LỜI GIẢI CHUẨN:**\n{payload['solution_text']}\n\n"
        f"**ĐÁP ÁN CHUẨN:**\n{payload['final_answer']}\n\n"
        f"**BAREM CHẤM ĐIỂM:**\n{payload['reasoning_approach']}\n\n"
        f"**BÀI LÀM HỌC SINH:**\n{payload['student_answer']}\n\n"
        "Hãy phân tích và đánh giá theo 5 yếu tố đã nêu trong system prompt."
    )
    
    try:
        resp = _client.chat.completions.create(
            model=MODEL_GRADING,
            messages=[
                {"role": "system", "content": GRADING_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=GRADING_TEMPERATURE,
            response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "grading_comparison_result",
                    "schema": GRADING_SCHEMA
                }
            }
            # Không dùng reasoning_effort (no reasoning theo yêu cầu)
        )
        
        return json.loads(resp.choices[0].message.content)
        
    except Exception as e:
        return {
            "knowledge_gaps": ["Không thể phân tích do lỗi hệ thống"],
            "calculation_logic_errors": [],
            "knowledge_gap_tag": ["hệ thống"],
            "error_tag": ["lỗi hệ thống"],
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
                # Tạo grading record cho câu không làm với knowledge_topics từ question
                _create_missing_grading(q, submission_id)
                continue
            
            ctx = _build_context(order_index, context_stack)
            reasoning_effort = _get_reasoning_effort(q.difficulty)
            payload = _make_payload(q, a, ctx)

            # Sử dụng solution comparison grading mới
            grading_data = grade_with_solution_comparison(q.id, a.answer_text)
            
            _save_grading_new(
                submission_id, 
                q.id, 
                grading_data["knowledge_gaps"],
                grading_data["calculation_logic_errors"], 
                grading_data["knowledge_gap_tag"],
                grading_data["error_tag"],
                grading_data["is_correct"]
            )
            
            results.append(
                GradingResult(
                    submission_id=submission_id,
                    question_id=q.id,
                    order_index=q.order_index,
                    part_label=(getattr(q, "part_label", None) or ""),
                    knowledge_gaps=grading_data["knowledge_gaps"],
                    calculation_logic_errors=grading_data["calculation_logic_errors"],
                    knowledge_gap_tag=grading_data["knowledge_gap_tag"],
                    error_tag=grading_data["error_tag"],
                    is_correct=grading_data["is_correct"],
                )
            )
            # Extend context after grading current item
            context_stack.append((q, a))

    # Note: mismatches intentionally ignored here (teacher fixes in Step 3)
    return results


def build_final_report(submission_id: int) -> str:
    """Build a student-friendly Markdown summary from existing gradings and save to DB.
    """
    with db.get_session() as session:
        grades = (
            session.query(Grading, Question)
            .join(Question, Grading.question_id == Question.id)
            .filter(Grading.submission_id == submission_id)
            .order_by(Question.order_index, Question.id)
            .all()
        )

    # Prepare compact input for LLM
    compact = []
    for g, q in grades:
        compact.append({
            "order_index": q.order_index,
            "part_label": getattr(q, "part_label", None) or "",
            "question_text": (q.question_text or "")[:CTX_MAX_CHARS_QUESTION],
            "knowledge_gaps": _safe_json_loads(g.knowledge_gaps) or [],
            "calculation_logic_errors": _safe_json_loads(g.calculation_logic_errors) or [],
            "is_correct": bool(g.is_correct),
        })

    system = (
        "Bạn là trợ lý sư phạm. Hãy tạo báo cáo ngắn gọn cho học sinh dưới dạng Markdown:\n"
        "1) Bảng tóm tắt theo từng câu: Câu X → Trạng thái (✓/✗) → Lỗ hổng kiến thức và lỗi tính toán (nếu có)\n"
        "2) Tổng kết: Danh sách kiến thức cần ôn tập (gộp từ tất cả knowledge_gaps)\n"
        "KHÔNG viết phần mở đầu tổng quan dài dòng. Tập trung vào thông tin cần thiết.\n"
        "Chỉ trả về Markdown, không kèm JSON."
    )

    user = (
        "Dưới đây là danh sách kết quả chấm theo từng ý. Hãy biên tập thành báo cáo tổng hợp cho học sinh.\n\n"
        + json.dumps(compact, ensure_ascii=False)
    )

    try:
        resp = _client.chat.completions.create(
            model=MODEL_GRADING,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.2
        )
        report_content = resp.choices[0].message.content
        
        # Save report to database
        try:
            db.save_submission_report(submission_id, report_content)
        except Exception as e:
            print(f"Warning: Could not save report to DB: {e}")
        
        return report_content
    except Exception:
        return "Không thể tạo báo cáo do lỗi hệ thống."

def get_or_generate_report(submission_id: int) -> str:
    """Get saved report from DB, or generate new one if not exists"""
    # Try to get saved report first
    saved_report = db.get_latest_report(submission_id)
    if saved_report:
        return saved_report.report_content
    
    # Generate new report if not found
    return build_final_report(submission_id)


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


def _build_context(order_index: int, context_stack: List[Tuple[Question, SubmissionItem]]):
    """Context from previous parts in the same order_index (question+answer)."""
    ctx = []
    for q, a in context_stack:
        if q.order_index != order_index:
            continue
        ctx.append({
            "part_label": getattr(q, "part_label", None) or "",
            "question_text": (q.question_text or "")[:CTX_MAX_CHARS_QUESTION],
            "student_answer": (a.answer_text or "")[:CTX_MAX_CHARS_ANSWER],
        })
    return ctx


def _make_payload(q: Question, a: SubmissionItem, ctx) -> Dict:
    return {
        "current": {
            "order_index": q.order_index,
            "part_label": getattr(q, "part_label", None) or "",
            "question_text": (q.question_text or "")[:CTX_MAX_CHARS_QUESTION],
            "student_answer": (a.answer_text or "")[:CTX_MAX_CHARS_ANSWER],
        },
        "context_previous": ctx,
    }


def _get_reasoning_effort(difficulty: Optional[int]) -> str:
    if difficulty is None:
        return "medium"
    try:
        d = int(difficulty)
    except Exception:
        return "medium"
    if d < 5:
        return "low"
    elif 6 <= d <= 8:
        return "medium"
    else:
        return "high"


def _call_llm_json_with_openai(payload: Dict, reasoning_effort: str) -> Dict:
    system = GRADING_SYSTEM_PROMPT
    user = (
        "Chấm ý hiện tại dựa trên đề, câu trả lời của học sinh và ngữ cảnh các ý trước đó.\n"
        "Trả về JSON đúng schema đã định.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    try:
        resp = _client.chat.completions.create(
            model=MODEL_GRADING,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=GRADING_TEMPERATURE,
            response_format={
                "type": "json_schema", 
                "json_schema": {
                    "name": "grading_result",
                    "schema": GRADING_SCHEMA
                }
            },
            reasoning_effort=reasoning_effort
        )
        
        return json.loads(resp.choices[0].message.content)
        
    except Exception:
        return {"nhan_xet": "Không thể chấm do lỗi hệ thống", "kien_thuc_hong": []}



def _save_grading_new(submission_id: int, question_id: int, knowledge_gaps: List[str], 
                     calculation_logic_errors: List[str], knowledge_gap_tag: List[str], 
                     error_tag: List[str], is_correct: bool):
    knowledge_gaps_json = json.dumps(knowledge_gaps, ensure_ascii=False)
    calculation_errors_json = json.dumps(calculation_logic_errors, ensure_ascii=False)
    knowledge_gap_tag_json = json.dumps(knowledge_gap_tag, ensure_ascii=False)
    error_tag_json = json.dumps(error_tag, ensure_ascii=False)
    
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
                knowledge_gaps=knowledge_gaps_json,
                calculation_logic_errors=calculation_errors_json,
                knowledge_gap_tag=knowledge_gap_tag_json,
                error_tag=error_tag_json,
                is_correct=1 if is_correct else 0,
                final_score=None,
            )
            session.add(row)
        else:
            row.knowledge_gaps = knowledge_gaps_json
            row.calculation_logic_errors = calculation_errors_json
            row.knowledge_gap_tag = knowledge_gap_tag_json
            row.error_tag = error_tag_json
            row.is_correct = 1 if is_correct else 0
            row.final_score = None
        session.commit()


def _create_missing_grading(question: Question, submission_id: int):
    """Tạo grading record cho câu học sinh không làm, sử dụng knowledge_topics từ question"""
    # Parse knowledge_topics từ question (JSON string)
    knowledge_topics = _safe_json_loads(question.knowledge_topics)
    
    # Tạo tags từ knowledge_topics
    knowledge_gap_tags = [topic.replace(" ", "_").lower() for topic in knowledge_topics[:3]]
    
    # Lưu grading record với knowledge_gaps = knowledge_topics
    _save_grading_new(
        submission_id=submission_id,
        question_id=question.id,
        knowledge_gaps=knowledge_topics,  # Sử dụng knowledge_topics từ question
        calculation_logic_errors=[],      # Rỗng vì không có tính toán
        knowledge_gap_tag=knowledge_gap_tags,  # Tags từ knowledge_topics
        error_tag=["không làm"],          # Tag cho việc không làm bài
        is_correct=False                  # Không đúng vì không làm
    )


def _safe_json_loads(s: Optional[str]):
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []
