from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Database
from database.db_manager import db
from database.models import Question, Submission, SubmissionItem, Grading, QuestionSolution
from services.solution_service import get_solution_by_question

# =====================
# Constants (single source of truth)
# =====================
MODEL_GRADING = "gpt-4.1-mini-2025-04-14"
COMMENT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1
CTX_MAX_CHARS_QUESTION = 1200
CTX_MAX_CHARS_ANSWER = 1200

GRADING_SYSTEM_PROMPT = """
Bạn là giáo viên Toán chuyên nghiệp tại Việt Nam với 15 năm kinh nghiệm chấm thi. 
Nhiệm vụ: So sánh bài làm học sinh với lời giải chuẩn và barem chấm điểm để đưa ra đánh giá chính xác.

### INPUT BẠN NHẬN ĐƯỢC:
1. **solution_text**: Hướng logic giải chuẩn của câu hỏi
2. **final_answer**: Đáp án chính xác cuối cùng  
3. **reasoning_approach**: Barem chấm điểm - các tiêu chí đánh giá
4. **student_answer**: Bài làm thực tế của học sinh

### NHIỆM VỤ PHÂN TÍCH:

#### A) **Lỗ hổng kiến thức** (knowledge_gaps):
- Xác định kiến thức nào học sinh chưa nắm vững
- VD: "Chưa biết điều kiện xác định phân thức", "Không hiểu định lý Pythagore"
- Chỉ liệt kê những gì THỰC SỰ THIẾU trong bài làm
- Mỗi mục ≤ 20 từ, tối đa 5 mục

#### B) **Lỗi tính toán & logic** (calculation_logic_errors):
- Những sai sót cụ thể trong quá trình giải
- VD: "Tính sai (-3)² = -9", "Quên đổi dấu khi chuyển vế", "Kết luận sai từ điều kiện đúng"
- Ghi rõ CHỖ SAI và SỬA THÀNH GÌ
- Mỗi mục ≤ 25 từ, tối đa 5 mục

#### C) **Nhận xét tổng quan** (llm_feedback):
- Đánh giá chi tiết bài làm (120-180 từ)
- Điểm mạnh, điểm yếu, cách cải thiện
- Dùng Markdown với **in đậm** cho lỗi quan trọng
- So sánh với lời giải chuẩn và barem

#### D) **Đánh giá kết quả** (is_correct):
- `true`: Từ đầu đến cuối hoàn toàn chính xác, logic rõ ràng, đáp án đúng
- `false`: Có ít nhất 1 trong số: kết quả sai, tự luận sai, thiếu bước quan trọng theo barem

### QUY TẮC CHẤM NGHIÊM NGẶT:
- **Đúng hoàn toàn** mới được `is_correct = true`
- **Có đáp án đúng** nhưng **thiếu giải thích theo barem** → `false` 
- **Logic đúng** nhưng **tính toán sai** → `false`
- **Kết quả đúng** nhưng **cách làm sai** → `false`
- **Thiếu bước quan trọng** theo barem → `false`

### NGUYÊN TẮC SO SÁNH:
- Dựa vào **reasoning_approach** (barem) để đánh giá từng bước
- Đối chiếu **final_answer** với kết quả học sinh
- Sử dụng **solution_text** làm chuẩn logic giải
- Chỉ chấm theo những gì có trong barem, không tự ý thêm yêu cầu

### OUTPUT FORMAT:
Chỉ trả về JSON nghiêm ngặt theo schema, không thêm text nào khác.

"""

# JSON Schema for OpenAI
GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "knowledge_gaps": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "Các lỗ hổng kiến thức cụ thể"
        },
        "calculation_logic_errors": {
            "type": "array",
            "items": {"type": "string"}, 
            "description": "Lỗi tính toán và logic cụ thể"
        },
        "llm_feedback": {
            "type": "string",
            "description": "Nhận xét chi tiết 120-180 từ"
        },
        "is_correct": {
            "type": "boolean",
            "description": "true nếu hoàn toàn đúng, false nếu có lỗi"
        }
    },
    "required": ["knowledge_gaps", "calculation_logic_errors", "llm_feedback", "is_correct"]
}

# =====================
# Client bootstrap
# =====================
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("Missing OPENAI_API_KEY")
_client = OpenAI(api_key=_api_key)


# =====================
# Data structures
# =====================
@dataclass
class GradingResult:
    submission_id: int
    question_id: int
    order_index: int
    part_label: str
    knowledge_gaps: List[str]
    calculation_logic_errors: List[str]
    llm_feedback: str
    is_correct: bool


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
        "Hãy phân tích và đánh giá theo 3 yếu tố đã nêu trong system prompt."
    )
    
    try:
        resp = _client.chat.completions.create(
            model=MODEL_GRADING,
            messages=[
                {"role": "system", "content": GRADING_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=TEMPERATURE,
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
            "llm_feedback": f"Lỗi khi chấm bài: {str(e)}",
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
                grading_data["llm_feedback"],
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
                    llm_feedback=grading_data["llm_feedback"],
                    is_correct=grading_data["is_correct"],
                )
            )
            # Extend context after grading current item
            context_stack.append((q, a))

    # Note: mismatches intentionally ignored here (teacher fixes in Step 3)
    return results


def build_final_report(submission_id: int) -> str:
    """Build a student-friendly Markdown summary from existing gradings.
    Why: keep DB schema unchanged; UI can display or download.
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
            "llm_feedback": (g.llm_feedback or ""),
            "is_correct": bool(g.is_correct),
        })

    system = (
        "Bạn là trợ lý sư phạm. Hãy biên tập báo cáo tổng hợp dễ hiểu cho học sinh, ưu tiên dưới dạng Markdown notion:\n"
        "1) Mở đầu: Tổng quan kết quả (số câu đúng/tổng số câu).\n"
        "2) Theo từng ý: Trạng thái đúng/sai, tóm tắt lỗi chính và cách khắc phục.\n"
        "3) Phần tổng kết: Nhóm lỗ hổng kiến thức và lỗi tính toán thường gặp.\n"
        "4) Đề xuất 3-5 hạng mục ôn tập cụ thể.\n"
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
        return resp.choices[0].message.content
    except Exception:
        return "Không thể tạo báo cáo do lỗi hệ thống."


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
            temperature=TEMPERATURE,
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
                     calculation_logic_errors: List[str], llm_feedback: str, is_correct: bool):
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
                knowledge_gaps=knowledge_gaps_json,
                calculation_logic_errors=calculation_errors_json,
                llm_feedback=llm_feedback,
                is_correct=1 if is_correct else 0,
                final_score=None,
            )
            session.add(row)
        else:
            row.knowledge_gaps = knowledge_gaps_json
            row.calculation_logic_errors = calculation_errors_json
            row.llm_feedback = llm_feedback
            row.is_correct = 1 if is_correct else 0
            row.final_score = None
        session.commit()


def _safe_json_loads(s: Optional[str]):
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []
