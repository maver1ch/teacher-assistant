import streamlit as st
import tempfile
import os
import logging
import pandas as pd
import re
import json

# ---------- Constants (single source of truth)
PAGE_TITLE = "Trợ lý Chấm bài"
PAGE_ICON = "📚"
LAYOUT = "wide"
EDITOR_HEIGHT = 420
DF_HEIGHT = 360

# ---------- Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Services & DB
from services.ocr_service import ocr
from services.llm_service import analyze_exam, segment_submission, QuestionLite
from database.db_manager import db
from database.models import Exam, Submission, Question, SubmissionItem
from services.grading_service import grade_submission, build_final_report

# ---------- App config
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# ---------- Session State
ss = st.session_state
ss.setdefault("ocr_text", "")
ss.setdefault("editor_text", "")
ss.setdefault("exam_id", None)
ss.setdefault("current_step", 1)
ss.setdefault("parsed_questions", [])
ss.setdefault("questions_from_db", [])
ss.setdefault("submission_text", "")
ss.setdefault("submission_name_guess", "")
ss.setdefault("submission_id", None)
ss.setdefault("segmented_items", [])
ss.setdefault("submission_editor_text", "")

# ---------- Helpers
def display_math_text(text: str):
    # Why: dùng 1 API thống nhất để tránh Streamlit auto-render lạ
    if text is None:
        return
    for raw in str(text).splitlines():
        s = str(raw).rstrip()
        if not s or s.strip().lower() == "none":
            st.markdown("&nbsp;")  # giữ khoảng trống nhẹ, không in 'None'
            continue
        # st.markdown render được cả thường lẫn LaTeX ($/$$)
        st.markdown(s)

def extract_student_name(txt: str) -> str:
    # Why: quick guess only; teacher can edit
    patterns = [
        r"(Họ\s* và \s*tên|Họ\s*tên|Họ\s*-\s*tên|Họ\s*&\s*tên)\s*[:\-]\s*(.+)",
        r"(Tên|Name)\s*[:\-]\s*(.+)",
    ]
    lines = [l.strip() for l in txt.splitlines()[:10] if l.strip()]
    for ln in lines:
        for pat in patterns:
            m = re.search(pat, ln, flags=re.IGNORECASE | re.UNICODE | re.VERBOSE)
            if m:
                val = m.group(len(m.groups()))
                val = re.split(r"(Lớp|Lop|Class)\s*[:\-]", val, flags=re.IGNORECASE)[0]
                return val.strip()[:60]
    return ""

# ---------- DB Helpers (pick từ DB khi nhảy bước)
def list_exams():
    try:
        with db.get_session() as session:
            rows = session.query(Exam).order_by(Exam.id.desc()).all()
            return [{"id": e.id, "name": getattr(e, "name", getattr(e, "title", f"Exam {e.id}"))} for e in rows]
    except Exception as ex:
        logger.exception("list_exams failed: %s", ex)
        return []

def list_submissions(exam_id: int):
    if not exam_id:
        return []
    try:
        with db.get_session() as session:
            rows = (
                session.query(Submission)
                .filter(Submission.exam_id == exam_id)
                .order_by(Submission.id.desc())
                .all()
            )
            return [{"id": s.id, "student_name": getattr(s, "student_name", f"Submission {s.id}")} for s in rows]
    except Exception as ex:
        logger.exception("list_submissions failed: %s", ex)
        return []

# ---------- Sidebar (Navigator + DB picker) — NO AUTO-APPLY ----------
with st.sidebar:
    st.header("📋 Điều hướng nhanh")

    step_labels = {
        1: "1️⃣ Upload & OCR đề",
        2: "2️⃣ Phân tích đề",
        3: "3️⃣ Upload bài làm",
        4: "4️⃣ Chấm bài",
        5: "5️⃣ Xuất báo cáo",
    }

    desired_step = st.selectbox(
        "🔀 Đi tới bước",
        options=[1, 2, 3, 4, 5],
        index=max(0, min(ss.current_step, 5) - 1),
        format_func=lambda x: step_labels[x],
        key="jump_step_select",
    )

    # Lấy danh sách từ DB nhưng KHÔNG áp dụng ngay lập tức
    pending_exam_id = None
    pending_submission_id = None

    with st.expander("🔗 Chọn dữ liệu từ DB (để nhảy thẳng)", expanded=(desired_step >= 4)):
        exams = list_exams()
        if exams:
            exam_options = [f'#{e["id"]} • {e["name"]}' for e in exams]
            # Nếu đã có ss.exam_id thì chọn đúng mục đó; nếu chưa có thì vẫn chọn mục đầu (chỉ pending)
            default_exam_idx = next((i for i, e in enumerate(exams) if e["id"] == ss.exam_id), 0)
            chosen_exam_label = st.selectbox("Đề thi (Exam)", exam_options, index=default_exam_idx, key="pick_exam")
            pending_exam_id = exams[exam_options.index(chosen_exam_label)]["id"]
        else:
            st.info("Chưa có Exam trong DB.")

        if pending_exam_id or ss.exam_id:
            # Xem submissions theo exam đang được CHỌN trong selectbox (pending hoặc ss.exam_id nếu có)
            exam_for_subs = pending_exam_id or ss.exam_id
            subs = list_submissions(exam_for_subs)
            if subs:
                sub_options = [f'#{s["id"]} • {s["student_name"]}' for s in subs]
                # Nếu đã có ss.submission_id thì giữ chọn; nếu chưa có thì đang pending ở mục đầu
                default_sub_idx = next((i for i, s in enumerate(subs) if s["id"] == ss.submission_id), 0)
                chosen_sub_label = st.selectbox("Bài làm (Submission)", sub_options, index=default_sub_idx, key="pick_sub")
                pending_submission_id = subs[sub_options.index(chosen_sub_label)]["id"]
            else:
                st.info("Exam này chưa có Submission.")

    # Chỉ khi bấm nút này mới ÁP DỤNG lựa chọn + NHẢY BƯỚC
    if st.button("⏩ Đi đến bước đã chọn", use_container_width=True):
        ok = True
        # Với step >=2 phải có exam (đang có sẵn hoặc pending)
        if desired_step >= 2 and not (ss.exam_id or pending_exam_id):
            st.warning("🔔 Cần chọn Exam trước (trong 'Chọn dữ liệu từ DB').")
            ok = False
        # Với step >=4 phải có submission (đang có sẵn hoặc pending)
        if desired_step >= 4 and not (ss.submission_id or pending_submission_id):
            st.warning("🔔 Cần chọn Submission cho Exam đã chọn.")
            ok = False

        if ok:
            # Áp dụng các lựa chọn pending (nếu có)
            if pending_exam_id:
                ss.exam_id = pending_exam_id
            if pending_submission_id:
                ss.submission_id = pending_submission_id

            ss.current_step = desired_step
            st.rerun()

    st.divider()
    st.caption(f"Step: {ss.current_step} • Exam: {ss.exam_id or '-'} • Submission: {ss.submission_id or '-'}")

# ====================== STEP 1 ======================
if ss.current_step == 1:
    st.header("Bước 1: Upload và OCR đề bài")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("📤 Upload ảnh đề bài")
        exam_name = st.text_input("Tên đề bài:", placeholder="VD: Đề thi giữa kỳ I Toán 12")
        uploaded_files = st.file_uploader(
            "Chọn ảnh (có thể nhiều ảnh)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
        )

        if uploaded_files and exam_name and st.button("🔍 Bắt đầu OCR", type="primary", key="start_ocr_exam"):
            with st.spinner("Đang OCR đề..."):
                temp_paths = []
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(f.getbuffer())
                        temp_paths.append(tmp.name)

                ocr_result = ocr.ocr_multiple_images(temp_paths) if len(temp_paths) > 1 else ocr.ocr_single_image(temp_paths[0])

                for p in temp_paths:
                    os.unlink(p)

                ss.ocr_text = ocr_result
                ss.editor_text = ocr_result
                st.success(f"✅ OCR hoàn thành ({len(uploaded_files)} ảnh).")

    with right:
        st.subheader("🗒️ Trạng thái")
        if ss.ocr_text:
            st.success("Đã có nội dung OCR. Kéo xuống để chỉnh sửa & xem trước.")
        else:
            st.info("Chưa có nội dung.")

    st.divider()

    if ss.ocr_text:
        st.subheader("✏️ Chỉnh sửa & 👀 Xem trước (real-time)")
        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            st.markdown("**Editor**")
            ss.editor_text = st.text_area(
                "Nội dung đề (LaTeX dùng $/$$):", value=ss.editor_text, height=EDITOR_HEIGHT,
                help="Inline: $x^2$, Display: $$\\frac{a}{b}$$", key="editor_area"
            )
            b1, b2, _ = st.columns([1, 1, 3])
            with b1:
                if st.button("💾 Lưu (preview)"):
                    ss.ocr_text = ss.editor_text
                    st.success("Đã lưu bản nháp.")
            with b2:
                if st.button("✅ Xác nhận & Tiếp tục", type="primary"):
                    if not exam_name:
                        st.error("Vui lòng nhập Tên đề bài.")
                    else:
                        exam_id = db.create_exam(exam_name)
                        ss.exam_id = exam_id
                        ss.ocr_text = ss.editor_text
                        ss.current_step = 2
                        st.rerun()

        with c2:
            st.markdown("**Preview (real-time)**")
            display_math_text(ss.editor_text)

# ====================== STEP 2 ======================
elif ss.current_step == 2 and ss.exam_id:
    st.header("Bước 2: Phân tích đề bài (LLM)")
    st.info(f"📌 Đề đã lưu • ID: {ss.exam_id}")

    st.markdown("**Nội dung đề:**")
    display_math_text(ss.ocr_text)

    st.divider()
    st.subheader("🔎 Phân tích & Trích xuất câu hỏi")

    cA, cB = st.columns([1, 1])
    with cA:
        if st.button("🚀 Phân tích đề (Gemini)", use_container_width=True):
            with st.spinner("Đang phân tích..."):
                parsed = analyze_exam(ss.ocr_text)
                ss.parsed_questions = [
                    {
                        "order_index": int(p["order_index"]),                       # BÀI LỚN
                        "part_label": str(p.get("part_label") or ""),
                        "text": str(p["text"]).strip(),
                        "difficulty": int(p["difficulty"]),
                        "knowledge_topics": [str(x).strip() for x in (p.get("knowledge_topics") or [])][:4],
                    }
                    for p in parsed
                ]
                st.success(f"Đã phân tích: {len(ss.parsed_questions)} mục.")

    with cB:
        if ss.parsed_questions:
            df_prev = pd.DataFrame(ss.parsed_questions).sort_values(["order_index", "part_label"])
            st.dataframe(df_prev, use_container_width=True, height=DF_HEIGHT)

    if ss.parsed_questions:
        st.divider()
        bL, bR, bN = st.columns([1, 1, 1])
        with bL:
            if st.button("💾 Lưu Questions vào DB", type="primary", use_container_width=True):
                with db.get_session() as session:
                    for row in ss.parsed_questions:
                        q = Question(
                            exam_id=ss.exam_id,
                            question_text=row["text"],
                            difficulty=row["difficulty"],
                            order_index=row["order_index"],
                            part_label=row.get("part_label", ""),
                            knowledge_topics=json.dumps(row["knowledge_topics"], ensure_ascii=False),
                        )
                        session.add(q)
                    session.commit()
                st.success("Đã lưu danh sách câu hỏi.")
                ss.questions_from_db = db.get_questions_by_exam(ss.exam_id)

        with bR:
            if st.button("🗂️ Tải lại từ DB", use_container_width=True):
                ss.questions_from_db = db.get_questions_by_exam(ss.exam_id)

        with bN:
            if st.button("➡️ Tiếp tục Bước 3", use_container_width=True):
                ss.current_step = 3
                st.rerun()

        if ss.questions_from_db:
            df_db = pd.DataFrame(
                [
                    {
                        "order_index": q.order_index,
                        "part_label": getattr(q, "part_label", ""),
                        "difficulty": q.difficulty,
                        "knowledge_topics": q.knowledge_topics,
                        "text": q.question_text,
                    }
                    for q in ss.questions_from_db
                ]
            ).sort_values(["order_index", "part_label"])
            st.markdown("**Danh sách câu hỏi (DB):**")
            st.dataframe(df_db, use_container_width=True, height=DF_HEIGHT)

# ====================== STEP 3 ======================
elif ss.current_step == 3 and ss.exam_id:
    st.header("Bước 3: Upload và OCR bài làm học sinh")
    st.info(f"📌 Đề • ID: {ss.exam_id}")

    upl, act = st.columns([1, 1])
    with upl:
        st.subheader("📤 Upload ảnh bài làm (nhiều ảnh)")
        submission_files = st.file_uploader(
            "Chọn ảnh bài làm", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="submission_files"
        )

        if submission_files and st.button("🔍 OCR bài làm", type="primary", key="start_ocr_submission"):
            with st.spinner("Đang OCR bài làm..."):
                temp_paths = []
                for f in submission_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(f.getbuffer())
                        temp_paths.append(tmp.name)

                sub_text = ocr.ocr_submission_images(temp_paths)

                for p in temp_paths:
                    os.unlink(p)

                ss.submission_text = sub_text
                ss.submission_name_guess = extract_student_name(sub_text)
                ss.submission_editor_text = sub_text
                st.success(f"✅ OCR hoàn thành ({len(submission_files)} ảnh).")

    with act:
        st.subheader("🧑‍🎓 Thông tin & Lưu")
        student_name = st.text_input("Tên học sinh (có thể chỉnh):", value=ss.submission_name_guess or "")
        if ss.submission_text:
            if st.button("💾 Lưu bài làm vào DB", type="primary"):
                sub_id = db.create_submission(
                    exam_id=ss.exam_id,
                    student_name=student_name.strip() or "Chưa rõ",
                    original_text=ss.submission_text
                )
                ss.submission_id = sub_id
                st.success(f"Đã lưu bài làm • Submission ID: {sub_id}")

    st.divider()
    st.subheader("✏️ Chỉnh sửa bài làm & 👀 Xem trước (real-time)")
    if ss.submission_text:
        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            st.markdown("**Editor (bài làm học sinh)**")
            ss.submission_editor_text = st.text_area(
                "Nội dung (LaTeX dùng $/$$):",
                value=ss.submission_editor_text,
                height=EDITOR_HEIGHT,
                help="Inline: $x^2$, Display: $$\\frac{a}{b}$$",
                key="submission_editor_area"
            )
            if st.button("💾 Lưu (preview)", key="btn_save_submission_preview"):
                ss.submission_text = ss.submission_editor_text
                st.success("Đã lưu bản nháp bài làm.")

        with c2:
            st.markdown("**Preview (real-time)**")
            display_math_text(ss.submission_editor_text or ss.submission_text)
    else:
        st.info("Chưa có nội dung. Hãy upload ảnh và chạy OCR.")

    st.divider()
    st.subheader("✂️ Phân đoạn bài làm theo câu hỏi (LLM)")

    # Chuẩn bị outline rút gọn từ Questions trong DB
    if ss.submission_id:
        questions = db.get_questions_by_exam(ss.exam_id)
        outline = []

        # map part_label từ kết quả STEP 2 (nếu còn trong session)
        mp = {}
        if ss.get("parsed_questions"):
            for r in ss.parsed_questions:
                mp[(r["text"] or "")[:50]] = r.get("part_label", "")

        for q in questions:
            try:
                topics = json.loads(q.knowledge_topics or "[]")
            except Exception:
                topics = []
            key = (q.question_text or "")[:50]
            outline.append(
                QuestionLite(
                    question_id=q.id,
                    order_index=q.order_index,
                    part_label=mp.get(key, ""),        # ưu tiên part_label nếu có
                    text_short=(q.question_text or "")[:200],
                    keywords=list(topics)[:5],
                )
            )

        if st.button("🔧 Phân đoạn bằng Gemini", use_container_width=True):
            with st.spinner("Đang phân đoạn..."):
                data = segment_submission(outline, ss.submission_text)
                ss.segmented_items = data.get("items", [])
                st.success(f"Đã tách thành {len(ss.segmented_items)} đoạn.")

        if ss.segmented_items:
            df_seg = pd.DataFrame(ss.segmented_items).sort_values(["position"])
            st.dataframe(df_seg, use_container_width=True, height=DF_HEIGHT)

            if st.button("💾 Lưu chi tiết từng ý (submission_items)", type="primary", use_container_width=True):
                with db.get_session() as session:
                    for it in ss.segmented_items:
                        item = SubmissionItem(
                            submission_id=ss.submission_id,
                            question_id=int(it["question_id"]),
                            order_index=int(it["order_index"]),
                            part_label=str(it.get("part_label") or ""),
                            position=int(it.get("position") or 1),
                            answer_text=str(it.get("answer_text") or "").strip(),
                        )
                        session.add(item)
                    session.commit()
                st.success("Đã lưu các ý của bài làm vào submission_items.")

    # Nút chuyển bước 4
    if ss.submission_id:
        st.divider()
        if st.button("➡️ Tiếp tục Bước 4 (Chấm bài)", type="primary", use_container_width=True):
            ss.current_step = 4
            st.rerun()

# ====================== STEP 4 ======================
elif ss.current_step == 4 and ss.exam_id:
    st.header("Bước 4: Chấm bài")
    st.info(f"📌 Exam ID: {ss.exam_id}")

    # Nếu user nhảy thẳng vào Bước 4 mà chưa có submission_id → cho chọn
    if not ss.submission_id:
        st.warning("Bạn chưa chọn Submission. Hãy chọn bên Sidebar, hoặc ngay tại đây.")
        subs_inline = list_submissions(ss.exam_id)
        if subs_inline:
            opt_inline = [f'#{s["id"]} • {s["student_name"]}' for s in subs_inline]
            pick_inline = st.selectbox("Chọn Submission để chấm", opt_inline, key="pick_sub_inline")
            picked = subs_inline[opt_inline.index(pick_inline)]
            ss.submission_id = picked["id"]
            st.rerun()
        else:
            st.stop()

    st.success(f"🎯 Submission ID đang chấm: {ss.submission_id}")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("🧮 Chấm toàn bộ bài", use_container_width=True):
            results = grade_submission(int(ss.submission_id))
            if results:
                df = pd.DataFrame([{
                    "order_index": r.order_index,
                    "part_label": r.part_label,
                    "nhan_xet": r.nhan_xet,
                    "kien_thuc_hong": ", ".join(r.kien_thuc_hong),
                } for r in results])
                st.dataframe(df, use_container_width=True, height=DF_HEIGHT)
            else:
                st.info("Không có mục nào để chấm hoặc submission_id không hợp lệ.")

    with colB:
        if st.button("📝 Tạo bản chấm tổng hợp", use_container_width=True):
            report_md = build_final_report(int(ss.submission_id))
            if report_md.strip():
                st.markdown(report_md)
                st.download_button(
                    "⬇️ Tải báo cáo (.md)",
                    data=report_md,
                    file_name=f"grading_report_{int(ss.submission_id)}.md",
                    mime="text/markdown",
                )
            else:
                st.info("Chưa có dữ liệu chấm hoặc báo cáo rỗng.")

# ====================== STEP 5 (optional placeholder) ======================
elif ss.current_step == 5 and ss.exam_id:
    st.header("Bước 5: Xuất báo cáo")
    st.info("Bạn có thể tạo báo cáo ở Bước 4 (nút 'Tạo bản chấm tổng hợp').")
    st.warning("(Placeholder) Tuỳ ý mở rộng thêm các định dạng export khác: PDF/Docx,...")

st.divider()
st.caption("Teacher Assistant v1.0 - MVP")
