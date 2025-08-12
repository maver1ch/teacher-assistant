from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Database
from database.db_manager import db
from database.models import Question, Submission, SubmissionItem, Grading

# =====================
# Constants (single source of truth)
# =====================
MODEL_GRADING = "gemini-2.5-pro"
MAX_OUTPUT_TOKENS = 8192  # Why: keep at model max to reduce truncation risk
THINK_BUDGET_LOW = 1500   # difficulty 1-3
THINK_BUDGET_MED = 2500   # difficulty 4-6
THINK_BUDGET_HIGH = 4000  # difficulty 7-8
RESUME_MAX_ROUNDS = 2     # Why: avoid infinite loops
CTX_MAX_CHARS_QUESTION = 1200
CTX_MAX_CHARS_ANSWER = 1200
REPORT_THINK_BUDGET = 2500

GRADING_SYSTEM_PROMPT = """
Bạn là **giáo viên Toán kinh nghiệm tại Việt Nam, thường xuyên chấm các bài thi Toán 9 vào cấp ba** . Chấm **từng ý nhỏ** của một **Bài/Câu** theo đúng tinh thần đề thi Việt Nam:

* Phản hồi **ngắn gọn, chuẩn mực sư phạm**, chỉ ra lỗi/thiếu bước trọng yếu; **bôi đậm** phần sai/lệch bằng Markdown.
* Không chép lại toàn bộ đề; tránh diễn giải lan man.

## 1) Quy tắc chung theo dạng bài. Tôi có thể liệt kê ra một vài ví dụ:

* **Giải phương trình / hệ phương trình / bất phương trình**: nêu thiếu *điều kiện xác định*, sai *biến đổi đồng nhất*, nghiệm **ngoài điều kiện**; khuyến nghị bước rút gọn, thử lại nghiệm.
* **Hàm số bậc hai & đồ thị**: kiểm tra cách tìm **đỉnh, trục đối xứng, giao trục**, xác định **tham số** từ điều kiện hình học; nhắc **đơn vị** nếu có thực tế.
* **Hình học phẳng (tam giác, đường tròn, tiếp tuyến, tứ giác nội tiếp)**: kiểm tra **giải thích lập luận** (định lý góc nội tiếp, tiếp tuyến–bán kính, đồng dạng, hệ thức lượng, hệ quả Thales/Pythagore); nếu ý sau dùng kết quả ý trước phải viện dẫn rõ.
* **Bài toán thực tế / mô hình hóa**: kiểm tra **ẩn, điều kiện ràng buộc, phương trình/ hệ thức** đúng ngữ cảnh; **đơn vị** và **kết luận**.
* Nếu câu trả lời chỉ ghi **đáp án cuối** (kiểu trắc nghiệm), hãy nêu thiếu **lập luận tối thiểu**.

## 2) Cách dùng ngữ cảnh

* Chỉ dùng `context_previous` thuộc cùng `order_index` (ví dụ chấm “1c” thấy được “1a, 1b”: đề rút gọn + bài làm HS).
* Khi ý hiện tại **phụ thuộc** kết quả trước: nếu HS **không viện dẫn** hoặc **dùng sai**, hãy **bôi đậm** phần đó.

## 3) Phong cách nhận xét

* Tối đa \~6 gạch đầu dòng; mỗi gạch ≤ 25 từ; tổng **≤ \~180 từ**.
* Viết rõ cái sai (**bôi đậm**), cái thiếu, và **gợi ý sửa** (ngắn).
* Nếu bài khó đọc/thiếu dữ kiện (do ảnh/không có hình vẽ), hãy nêu rõ **giới hạn** khi chấm.

## 4) Đầu ra bắt buộc (strict JSON)

Chỉ trả về JSON đúng **schema**:

```json
{
  "nhan_xet": "string (Markdown, có **bôi đậm** lỗi sai)",
  "kien_thuc_hong": ["string", "string", "..."]
}
```

* `kien_thuc_hong`: liệt kê ngắn gọn (≤ 16 từ/điểm), ví dụ:

  * "Điều kiện xác định phân thức", "Hệ thức lượng trong tam giác vuông",
  * "Góc giữa tiếp tuyến và dây", "Lập ẩn & đơn vị bài toán thực tế".

---

# C) Định dạng input gửi vào LLM

Khi gọi `grade_item(...)`, truyền **user content** dạng JSON (string hóa) có ví dụ như sau:

```json
{
  "current": {
    "order_index": 1,
    "part_label": "c",
    "question_text": "Cho tam giác ABC vuông tại A, ... Chứng minh ...",
    "student_answer": "HS lập luận ... suy ra ..."
  },
  "context_previous": [
    { "part_label": "a", "question_text": "Chứng minh AB = AC", "student_answer": "..." },
    { "part_label": "b", "question_text": "Tính BC", "student_answer": "..." }
  ]
}
```

> Lưu ý khi build input: cắt gọn `question_text`/`student_answer` theo giới hạn đã đặt (để không chạm trần token); chỉ đưa **cùng `order_index`**.

LƯU Ý QUAN TRỌNG: KHÔNG ĐƯỢC PHÉP ĐƯA RA GỢI Ý, chỉ được chấm bài theo barem và logic suy luận trình tự hợp lý.

"""

# Strict JSON schema: only the required fields
JSON_SCHEMA_GRADING = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "nhan_xet": types.Schema(type=types.Type.STRING),
        "kien_thuc_hong": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING)
        ),
    },
    required=["nhan_xet", "kien_thuc_hong"],
)

# =====================
# Client bootstrap
# =====================
load_dotenv()
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("Missing GEMINI_API_KEY")
_client = genai.Client(api_key=_api_key)


# =====================
# Data structures
# =====================
@dataclass
class GradingResult:
    submission_id: int
    question_id: int
    order_index: int
    part_label: str
    nhan_xet: str
    kien_thuc_hong: List[str]


# =====================
# Public API
# =====================

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
            thinking_budget = _select_thinking_budget(q.difficulty)
            payload = _make_payload(q, a, ctx)

            data = _call_llm_json_with_resume(payload, thinking_budget)
            nhan_xet = data.get("nhan_xet", "")
            kien_thuc_hong = data.get("kien_thuc_hong", [])

            _save_grading(submission_id, q.id, nhan_xet, kien_thuc_hong)
            results.append(
                GradingResult(
                    submission_id=submission_id,
                    question_id=q.id,
                    order_index=q.order_index,
                    part_label=(getattr(q, "part_label", None) or ""),
                    nhan_xet=nhan_xet,
                    kien_thuc_hong=kien_thuc_hong,
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
            "nhan_xet": (g.feedback_text or ""),
            "kien_thuc_hong": _safe_json_loads(g.knowledge_gaps) or [],
        })

    system = (
        "Bạn là trợ lý sư phạm. Hãy biên tập báo cáo tổng hợp dễ hiểu cho học sinh:\n"
        "1) Mở đầu ngắn (2-4 câu) nêu điểm mạnh/yếu chính.\n"
        "2) Theo từng ý (theo thứ tự order_index → part_label): tóm tắt lỗi chính (đậm chỗ sai) và cách khắc phục.\n"
        "3) Gom nhóm các 'kien_thuc_hong' toàn bài (loại trùng), đề xuất 3-5 hạng mục ôn tập.\n"
        "Chỉ trả về Markdown, không kèm JSON."
    )

    cfg = _md_config(REPORT_THINK_BUDGET)

    user = (
        "Dưới đây là danh sách kết quả chấm theo từng ý. Hãy biên tập thành báo cáo tổng hợp cho học sinh.\n\n"
        + json.dumps(compact, ensure_ascii=False)
    )

    cfg.system_instruction = system  # put system into config (stable across SDKs)
    resp = _client.models.generate_content(
        model=MODEL_GRADING,
        contents=user,  # just the user content as string
        config=cfg,
    )

    # Safe extraction without assuming .text always exists
    text = getattr(resp, "text", None)
    if not text:
        # Fallback: try candidates/parts
        try:
            cand = resp.candidates[0]
            parts = getattr(cand.content, "parts", [])
            text = "".join([getattr(p, "text", "") for p in parts])
        except Exception:
            text = ""
    return text or ""


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


def _select_thinking_budget(difficulty: Optional[int]) -> Optional[int]:
    if difficulty is None:
        return THINK_BUDGET_MED
    try:
        d = int(difficulty)
    except Exception:
        return THINK_BUDGET_MED
    if 1 <= d <= 3:
        return THINK_BUDGET_LOW
    if 4 <= d <= 6:
        return THINK_BUDGET_MED
    if 7 <= d <= 8:
        return THINK_BUDGET_HIGH
    # 9-10: no thinking_budget
    return None


def _call_llm_json_with_resume(payload: Dict, thinking_budget: Optional[int]) -> Dict:
    cfg = _json_config(thinking_budget)

    system = GRADING_SYSTEM_PROMPT
    user = (
        "Chấm ý hiện tại dựa trên đề, câu trả lời của học sinh và ngữ cảnh các ý trước đó.\n"
        "Trả về JSON đúng schema đã định.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    cfg.system_instruction = system
    resp = _client.models.generate_content(
        model=MODEL_GRADING,
        contents=user,
        config=cfg,
    )

    data = _extract_parsed_or_text_json(resp)
    if data is not None:
        return data

    # Resume path if cut or invalid JSON
    partial = _extract_text(resp)

    rounds = 0
    while rounds < RESUME_MAX_ROUNDS:
        rounds += 1
        resume_prompt = (
            "Here is the partial JSON you returned. Return only the remaining JSON tokens to complete a valid JSON. "
            "Do not repeat existing keys/values. Strict JSON only.\n\n"
            + (partial or "")
        )

        # Prefer streaming for resume to collect chunks
        stream_cfg = _json_config(thinking_budget)

        stream_cfg.system_instruction = system
        stream = _client.models.generate_content_stream(
            model=MODEL_GRADING,
            contents=resume_prompt,
            config=stream_cfg,
        )
        chunks = []
        try:
            for evt in stream:
                t = getattr(evt, "text", None)
                if t:
                    chunks.append(t)
        except Exception:
            pass

        stitched = (partial or "") + ("".join(chunks) if chunks else "")
        try:
            return json.loads(stitched)
        except Exception:
            partial = stitched  # try again if allowed

    # Last attempt: mini-fix
    return _minifix_json(partial)

def _safe_build_config(base: dict, thinking_budget: Optional[int]):
    # Try new-style thinking config if SDK hỗ trợ; fallback nếu không
    if thinking_budget is not None and hasattr(types, "ThinkingConfig"):
        try:
            base["thinking"] = types.ThinkingConfig(budget_tokens=thinking_budget)
        except Exception:
            pass
    try:
        return types.GenerateContentConfig(**base)
    except Exception:
        base.pop("thinking", None)
        return types.GenerateContentConfig(**base)

def _json_config(thinking_budget: Optional[int]):
    base = dict(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=JSON_SCHEMA_GRADING,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return _safe_build_config(base, thinking_budget)

def _md_config(thinking_budget: Optional[int]):
    base = dict(
        temperature=0.2,
        response_mime_type="text/plain",
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return _safe_build_config(base, thinking_budget)

def _extract_parsed_or_text_json(resp) -> Optional[Dict]:
    # Prefer structured .parsed if available
    data = getattr(resp, "parsed", None)
    if data is not None:
        return data
    # Fallback: try resp.text
    t = getattr(resp, "text", None)
    if t:
        try:
            return json.loads(t)
        except Exception:
            return None
    # Last: try candidates/parts concatenation
    t = _extract_text(resp)
    if t:
        try:
            return json.loads(t)
        except Exception:
            return None
    return None


def _extract_text(resp) -> str:
    # Why: avoid relying on .text when there is no Part
    try:
        cand = resp.candidates[0]
        parts = getattr(cand.content, "parts", [])
        return "".join([getattr(p, "text", "") for p in parts])
    except Exception:
        return ""


def _minifix_json(s: Optional[str]) -> Dict:
    # Why: minimal repair only for truncated tail
    if not s:
        return {"nhan_xet": "", "kien_thuc_hong": []}
    s2 = s.strip()
    if not s2.endswith("}"):
        s2 += "}"
    try:
        obj = json.loads(s2)
        if isinstance(obj, dict):
            obj.setdefault("nhan_xet", "")
            obj.setdefault("kien_thuc_hong", [])
            return obj
    except Exception:
        pass
    return {"nhan_xet": "", "kien_thuc_hong": []}


def _save_grading(submission_id: int, question_id: int, nhan_xet: str, kien_thuc_hong: List[str]):
    payload = json.dumps(kien_thuc_hong, ensure_ascii=False)
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
                feedback_text=nhan_xet,
                knowledge_gaps=payload,
                final_score=None,
            )
            session.add(row)
        else:
            row.feedback_text = nhan_xet
            row.knowledge_gaps = payload
            row.final_score = None
        session.commit()


def _safe_json_loads(s: Optional[str]):
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return []
