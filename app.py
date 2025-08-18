#app.py

import streamlit as st
import tempfile
import os
import logging
import pandas as pd
import re
import json

# ---------- Import config
from utils.config import PAGE_TITLE, PAGE_ICON, LAYOUT, EDITOR_HEIGHT, DF_HEIGHT

# ---------- Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Services & DB
from services.ocr_service import ocr
from services.llm_service import analyze_exam, segment_submission, QuestionLite
from database.db_manager import db
from database.models import Exam, Submission, Question, SubmissionItem
from services.grading_service import grade_submission, build_final_report, get_or_generate_report
from services.solution_service import create_and_save_solution, get_solution_by_question

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
ss.setdefault("selected_line", {})  # Track selected lines for different editors

# ---------- Helpers
def display_math_text(text: str, enable_line_click: bool = False, target_key: str = None):
    # Why: dùng 1 API thống nhất để tránh Streamlit auto-render lạ
    if text is None:
        return
    
    lines = str(text).splitlines()
    if not enable_line_click:
        # Original behavior for backward compatibility
        for raw in lines:
            s = str(raw).rstrip()
            if not s or s.strip().lower() == "none":
                st.markdown("&nbsp;")  # giữ khoảng trống nhẹ, không in 'None'
                continue
            st.markdown(s)
        return
    
    # Enhanced version with clickable lines and line numbers
    css_and_js = """
    <style>
    .line-container {
        display: flex;
        align-items: flex-start;
        margin: 2px 0;
        padding: 2px 5px;
        border-radius: 3px;
        transition: background-color 0.2s;
    }
    .line-container:hover {
        background-color: #f0f2f6;
        cursor: pointer;
    }
    .line-container.highlighted {
        background-color: #e8f4f8;
        border-left: 3px solid #0066cc;
    }
    .line-number {
        min-width: 30px;
        color: #666;
        font-size: 12px;
        font-family: monospace;
        padding-right: 10px;
        user-select: none;
        text-align: right;
    }
    .line-content {
        flex: 1;
        line-height: 1.4;
    }
    </style>
    
    <script>
    function scrollToEditorLine(lineNum, targetKey) {
        // Store selected line in session state for editor highlighting
        const event = new CustomEvent('lineSelected', {
            detail: { lineNumber: lineNum, targetKey: targetKey }
        });
        window.dispatchEvent(event);
        
        // Highlight the clicked line
        document.querySelectorAll('.line-container').forEach(el => {
            el.classList.remove('highlighted');
        });
        document.getElementById('line-' + targetKey + '-' + lineNum).classList.add('highlighted');
    }
    </script>
    """
    
    st.components.v1.html(css_and_js, height=0)
    
    html_content = ""
    for i, raw in enumerate(lines, 1):
        s = str(raw).rstrip()
        line_id = f"line-{target_key}-{i}"
        
        if not s or s.strip().lower() == "none":
            content = "&nbsp;"
        else:
            # Escape HTML but preserve LaTeX
            content = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
        html_content += f'''
        <div class="line-container" id="{line_id}" onclick="scrollToEditorLine({i}, '{target_key}')">
            <div class="line-number">{i}</div>
            <div class="line-content">{content}</div>
        </div>
        '''
    
    # Display the interactive content
    st.components.v1.html(html_content, height=len(lines) * 25 + 20)
    
    # Also render with regular markdown for LaTeX processing
    with st.expander("📄 Rendered LaTeX View", expanded=False):
        for raw in lines:
            s = str(raw).rstrip()
            if not s or s.strip().lower() == "none":
                st.markdown("&nbsp;")
                continue
            st.markdown(s)

def create_enhanced_text_area(label: str, value: str, height: int, key: str, target_key: str = None):
    """Enhanced text area with line highlighting support"""
    
    # Add JavaScript to listen for line selection events
    js_listener = f"""
    <script>
    function highlightEditorLine_{target_key}(lineNum) {{
        // This will be handled by Streamlit's text area component
        // For now, we'll store the line number in session state
        console.log('Selected line:', lineNum, 'for target:', '{target_key}');
    }}
    
    window.addEventListener('lineSelected', function(event) {{
        if (event.detail.targetKey === '{target_key}') {{
            highlightEditorLine_{target_key}(event.detail.lineNumber);
            
            // Try to scroll to the approximate line in the text area
            const textArea = document.querySelector('[data-testid="stTextArea"] textarea[aria-label*="{label}"]');
            if (textArea) {{
                const lines = textArea.value.split('\\n');
                const targetLine = event.detail.lineNumber - 1;
                if (targetLine >= 0 && targetLine < lines.length) {{
                    // Calculate approximate character position
                    let charPos = 0;
                    for (let i = 0; i < targetLine; i++) {{
                        charPos += lines[i].length + 1; // +1 for newline
                    }}
                    
                    // Set cursor position
                    textArea.focus();
                    textArea.setSelectionRange(charPos, charPos + lines[targetLine].length);
                    
                    // Scroll to make the line visible
                    const lineHeight = 20; // approximate line height
                    const scrollTop = targetLine * lineHeight - textArea.clientHeight / 2;
                    textArea.scrollTop = Math.max(0, scrollTop);
                }}
            }}
        }}
    }});
    </script>
    """
    
    if target_key:
        st.components.v1.html(js_listener, height=0)
    
    return st.text_area(
        label,
        value=value,
        height=height,
        key=key,
        help="Click on lines in the preview to navigate here"
    )

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
        3: "3️⃣ Tạo lời giải",
        4: "4️⃣ Upload bài làm",
        5: "5️⃣ Chấm bài",
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

    # Export CSV functionality
    st.markdown("---")
    st.markdown("**📊 Export dữ liệu**")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Export Gradings", use_container_width=True):
            try:
                from export_gradings import export_gradings_to_csv
                filename = export_gradings_to_csv()
                st.success(f"✅ Exported: {filename}")
                
                # Provide download button
                with open(filename, "rb") as file:
                    st.download_button(
                        "⬇️ Download CSV",
                        data=file,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
    
    with col2:
        if st.button("📈 Export Summary", use_container_width=True):
            try:
                from export_gradings import export_summary_by_student
                filename = export_summary_by_student()
                st.success(f"✅ Exported: {filename}")
                
                # Provide download button  
                with open(filename, "rb") as file:
                    st.download_button(
                        "⬇️ Download CSV",
                        data=file,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"❌ Export failed: {e}")

    # Chỉ khi bấm nút này mới ÁP DỤNG lựa chọn + NHẢY BƯỚC
    if st.button("⏩ Đi đến bước đã chọn", use_container_width=True):
        ok = True
        # Với step >=2 phải có exam (đang có sẵn hoặc pending)
        if desired_step >= 2 and not (ss.exam_id or pending_exam_id):
            st.warning("🔔 Cần chọn Exam trước (trong 'Chọn dữ liệu từ DB').")
            ok = False
        # Với step >=5 phải có submission (đang có sẵn hoặc pending)  
        if desired_step >= 5 and not (ss.submission_id or pending_submission_id):
            st.warning("🔔 Cần chọn Submission cho Exam đã chọn.")
            ok = False

        if ok:
            # Áp dụng các lựa chọn pending (nếu có)
            if pending_exam_id:
                ss.exam_id = pending_exam_id
            if pending_submission_id:
                ss.submission_id = pending_submission_id
                # Load submission original_text khi chọn submission
                submission = db.get_submission_by_id(pending_submission_id)
                if submission and submission.original_text:
                    ss.submission_text = submission.original_text
                    ss.submission_editor_text = submission.original_text

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
            ss.editor_text = create_enhanced_text_area(
                "Nội dung đề (LaTeX dùng $/$$):", 
                value=ss.editor_text, 
                height=EDITOR_HEIGHT,
                key="editor_area",
                target_key="editor"
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
                        exam_id = db.create_exam(exam_name, ss.editor_text)
                        ss.exam_id = exam_id
                        ss.ocr_text = ss.editor_text
                        ss.current_step = 2
                        st.rerun()

        with c2:
            st.markdown("**Preview (real-time)**")
            display_math_text(ss.editor_text, enable_line_click=True, target_key="editor")

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
            edited_df = st.data_editor(
                df_prev, 
                use_container_width=True, 
                height=DF_HEIGHT,
                column_config={
                    "order_index": st.column_config.NumberColumn("Order", disabled=False),
                    "part_label": st.column_config.TextColumn("Part", disabled=False),
                    "text": st.column_config.TextColumn("Text", disabled=False, width="large"),
                    "difficulty": st.column_config.NumberColumn("Difficulty", disabled=False, min_value=1, max_value=10),
                    "knowledge_topics": st.column_config.TextColumn("Knowledge Topics", disabled=False)
                },
                key="edit_parsed_questions"
            )
            # Update session state with edited data
            ss.parsed_questions = edited_df.to_dict('records')

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
            if st.button("➡️ Tiếp tục Bước 3 (Tạo lời giải)", use_container_width=True):
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
            edited_db_df = st.data_editor(
                df_db, 
                use_container_width=True, 
                height=DF_HEIGHT,
                column_config={
                    "order_index": st.column_config.NumberColumn("Order", disabled=False),
                    "part_label": st.column_config.TextColumn("Part", disabled=False),
                    "text": st.column_config.TextColumn("Text", disabled=False, width="large"),
                    "difficulty": st.column_config.NumberColumn("Difficulty", disabled=False, min_value=1, max_value=10),
                    "knowledge_topics": st.column_config.TextColumn("Knowledge Topics", disabled=False)
                },
                key="edit_db_questions"
            )

# ====================== STEP 3 (NEW): TẠO LỜI GIẢI ======================
elif ss.current_step == 3 and ss.exam_id:
    st.header("Bước 3: Tạo lời giải và barem chấm điểm")
    st.info(f"📌 Exam ID: {ss.exam_id}")

    questions = db.get_questions_by_exam(ss.exam_id)
    
    if questions:
        st.subheader("🧠 Tạo lời giải tự động")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Danh sách câu hỏi:**")
            question_options = [f"Câu {q.order_index}{q.part_label if q.part_label else ''}: {q.question_text[:50]}..." for q in questions]
            selected_idx = st.selectbox("Chọn câu hỏi để tạo lời giải:", range(len(questions)), format_func=lambda x: question_options[x])
            
            selected_question = questions[selected_idx]
            
            if st.button(f"🚀 Tạo lời giải cho câu {selected_question.order_index}{selected_question.part_label or ''}", use_container_width=True):
                with st.spinner("Đang tạo lời giải..."):
                    try:
                        solution_id = create_and_save_solution(selected_question.id)
                        st.success(f"✅ Đã tạo lời giải (ID: {solution_id})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Lỗi khi tạo lời giải: {str(e)}")
        
        with col2:
            existing_solution = get_solution_by_question(selected_question.id)
            if existing_solution:
                st.markdown(f"**Lời giải câu {existing_solution['order_index']}{existing_solution['part_label'] or ''}:**")
                
                with st.expander("📝 Hướng logic giải", expanded=True):
                    display_math_text(existing_solution["solution_text"])
                
                with st.expander("🎯 Đáp án cuối"):
                    display_math_text(existing_solution["final_answer"])
                    
                with st.expander("📋 Barem chấm điểm"):
                    display_math_text(existing_solution["reasoning_approach"])
                    
                st.caption(f"Tạo lúc: {existing_solution['created_at']}")
            else:
                st.info("Chưa có lời giải cho câu hỏi này.")
        
        st.divider()
        
        # Tạo lời giải cho tất cả câu hỏi
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("🔥 Tạo lời giải cho TẤT CẢ câu hỏi", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, q in enumerate(questions):
                    status_text.text(f"Đang xử lý câu {q.order_index}{q.part_label or ''}...")
                    try:
                        create_and_save_solution(q.id)
                        progress_bar.progress((i + 1) / len(questions))
                    except Exception as e:
                        st.warning(f"Lỗi câu {q.order_index}{q.part_label or ''}: {str(e)}")
                
                status_text.text("✅ Hoàn thành!")
                st.success(f"Đã tạo lời giải cho {len(questions)} câu hỏi.")
        
        with col_b:
            if st.button("➡️ Tiếp tục Bước 4 (Upload bài làm)", use_container_width=True):
                ss.current_step = 4
                st.rerun()
        
        st.divider()
        st.subheader("📊 Tổng quan lời giải đã tạo")
        
        # Hiển thị bảng tổng quan các solutions
        solutions_data = []
        for q in questions:
            sol = get_solution_by_question(q.id)
            if sol:
                solutions_data.append({
                    "Câu": f"{sol['order_index']}{sol['part_label'] or ''}",
                    "Nội dung": q.question_text[:80] + "..." if len(q.question_text) > 80 else q.question_text,
                    "Có lời giải": "✅",
                    "Độ khó": q.difficulty,
                    "Tạo lúc": sol['created_at'].strftime("%H:%M %d/%m") if hasattr(sol['created_at'], 'strftime') else str(sol['created_at'])
                })
            else:
                solutions_data.append({
                    "Câu": f"{q.order_index}{q.part_label if q.part_label else ''}",
                    "Nội dung": q.question_text[:80] + "..." if len(q.question_text) > 80 else q.question_text,
                    "Có lời giải": "❌",
                    "Độ khó": q.difficulty,
                    "Tạo lúc": "-"
                })
        
        if solutions_data:
            df_solutions = pd.DataFrame(solutions_data)
            edited_solutions_df = st.data_editor(
                df_solutions, 
                use_container_width=True, 
                height=300,
                column_config={
                    "Câu": st.column_config.TextColumn("Câu", disabled=False),
                    "Nội dung": st.column_config.TextColumn("Nội dung", disabled=False, width="large"),
                    "Có lời giải": st.column_config.TextColumn("Có lời giải", disabled=True),
                    "Độ khó": st.column_config.NumberColumn("Độ khó", disabled=False, min_value=1, max_value=10),
                    "Tạo lúc": st.column_config.TextColumn("Tạo lúc", disabled=True)
                },
                key="edit_solutions_overview"
            )
    else:
        st.warning("Không có câu hỏi nào. Vui lòng quay lại Bước 2 để phân tích đề.")

# ====================== STEP 4 (OLD STEP 3): UPLOAD BÀI LÀM ======================
elif ss.current_step == 4 and ss.exam_id:
    st.header("Bước 4: Upload và OCR bài làm học sinh")
    
    # Auto load submission text nếu đã chọn submission
    if ss.submission_id and not ss.submission_text:
        submission = db.get_submission_by_id(ss.submission_id)
        if submission and submission.original_text:
            ss.submission_text = submission.original_text
            ss.submission_editor_text = submission.original_text
            st.success(f"📁 Đã load bài làm từ DB (Submission #{ss.submission_id})")
    
    if ss.submission_id:
        st.info(f"📌 Đề • ID: {ss.exam_id} | 📝 Bài làm • ID: {ss.submission_id}")
    else:
        st.info(f"📌 Đề • ID: {ss.exam_id}")

    # Hiển thị bài làm đã load từ DB
    if ss.submission_text:
        st.subheader("📄 Bài làm học sinh")
        source_indicator = "📁 Đã load từ DB" if ss.submission_id else "🔍 Vừa OCR"
        st.caption(f"{source_indicator} • Độ dài: {len(ss.submission_text)} ký tự")
        
        with st.expander("👀 Xem nội dung bài làm", expanded=True):
            display_math_text(ss.submission_text)
        
        
        st.divider()

    # Simplified upload section
    if not ss.submission_text:
        st.subheader("📤 Upload và OCR bài làm mới")
    
    # Upload interface (always available)
    upl, act = st.columns([1, 1])
    with upl:
        st.markdown("**📷 Chọn ảnh bài làm**")
        submission_files = st.file_uploader(
            "Upload nhiều ảnh:", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="submission_files"
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
                st.rerun()

    with act:
        st.markdown("**👤 Thông tin học sinh**")
        student_name = st.text_input("Tên học sinh:", value=ss.submission_name_guess or "")
        if ss.submission_text:
            if st.button("💾 Lưu bài làm vào DB", type="primary"):
                sub_id = db.create_submission(
                    exam_id=ss.exam_id,
                    student_name=student_name.strip() or "Chưa rõ",
                    original_text=ss.submission_text
                )
                ss.submission_id = sub_id
                st.success(f"Đã lưu bài làm • Submission ID: {sub_id}")
                st.rerun()

    st.divider()
    st.subheader("✏️ Chỉnh sửa bài làm & 👀 Xem trước (real-time)")
    if ss.submission_text:
        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            st.markdown("**Editor (bài làm học sinh)**")
            ss.submission_editor_text = create_enhanced_text_area(
                "Nội dung (LaTeX dùng $/$$):",
                value=ss.submission_editor_text,
                height=EDITOR_HEIGHT,
                key="submission_editor_area",
                target_key="submission_editor"
            )
            if st.button("💾 Lưu (preview)", key="btn_save_submission_preview"):
                ss.submission_text = ss.submission_editor_text
                st.success("Đã lưu bản nháp bài làm.")

        with c2:
            st.markdown("**Preview (real-time)**")
            display_math_text(ss.submission_editor_text or ss.submission_text, enable_line_click=True, target_key="submission_editor")
    else:
        # Kiểm tra xem có submission được chọn từ sidebar không
        if ss.submission_id:
            submission = db.get_submission_by_id(ss.submission_id)
            if submission and submission.original_text:
                ss.submission_text = submission.original_text
                ss.submission_editor_text = submission.original_text
                st.success("📁 Đã load bài làm từ DB. Scroll xuống để xem.")
                st.rerun()
            else:
                st.info("Submission đã chọn nhưng chưa có OCR text. Hãy upload ảnh và chạy OCR.")
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

        if st.button("🔧 Phân đoạn bằng LLM (Skeleton approach)", use_container_width=True):
            with st.spinner("Đang phân đoạn với skeleton..."):
                data = segment_submission(questions, ss.submission_text)
                ss.segmented_items = data.get("items", [])
                st.success(f"Đã phân đoạn {len(ss.segmented_items)} items từ skeleton.")

        if ss.segmented_items:
            st.subheader("✏️ Xem và chỉnh sửa kết quả phân đoạn")
            
            # Editable dataframe với LaTeX preview
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("**📝 Chỉnh sửa answer_text:**")
                
                # Convert to editable DataFrame
                df_seg = pd.DataFrame(ss.segmented_items).sort_values(["position"])
                
                edited_df = st.data_editor(
                    df_seg,
                    column_config={
                        "question_id": st.column_config.NumberColumn("Question ID", disabled=True),
                        "order_index": st.column_config.NumberColumn("Order", disabled=True), 
                        "part_label": st.column_config.TextColumn("Part", disabled=True),
                        "position": st.column_config.NumberColumn("Pos", disabled=True),
                        "answer_text": st.column_config.TextColumn("Answer Text", width="large")
                    },
                    disabled=["question_id", "order_index", "part_label", "position"],
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                    key="editable_segments"
                )
                
                # Update session state với data đã edit
                ss.segmented_items = edited_df.to_dict('records')
            
            with col2:
                st.markdown("**🔍 LaTeX Preview:**")
                
                # Select row để preview
                selected_row = st.selectbox(
                    "Chọn row để preview:",
                    range(len(edited_df)),
                    format_func=lambda x: f"Row {x+1}: {edited_df.iloc[x]['order_index']}{edited_df.iloc[x]['part_label']}"
                )
                
                if selected_row is not None:
                    preview_text = edited_df.iloc[selected_row]['answer_text']
                    if preview_text and preview_text.strip():
                        st.markdown("**Preview:**")
                        with st.container():
                            display_math_text(preview_text)
                    else:
                        st.info("Answer text trống")

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
        if st.button("➡️ Tiếp tục Bước 5 (Chấm bài)", type="primary", use_container_width=True):
            ss.current_step = 5
            st.rerun()

# ====================== STEP 5 (OLD STEP 4): CHẤM BÀI ======================
elif ss.current_step == 5 and ss.exam_id:
    st.header("Bước 5: Chấm bài")
    st.info(f"📌 Exam ID: {ss.exam_id}")

    # Nếu user nhảy thẳng vào Bước 5 mà chưa có submission_id → cho chọn
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
        if st.button("🧮 Chấm toàn bộ bài (So sánh với lời giải chuẩn)", use_container_width=True):
            with st.spinner("Đang chấm bài với AI..."):
                results = grade_submission(int(ss.submission_id))
            if results:
                st.subheader("📊 Kết quả chấm chi tiết")
                
                # Tổng quan kết quả
                correct_count = sum(1 for r in results if r.is_correct)
                total_count = len(results)
                st.metric("Tổng quan", f"{correct_count}/{total_count} câu đúng", 
                         f"{correct_count/total_count*100:.1f}%" if total_count > 0 else "0%")
                
                # Hiển thị từng câu
                for r in results:
                    status_icon = "✅" if r.is_correct else "❌"
                    with st.expander(f"{status_icon} Câu {r.order_index}{r.part_label} - {'ĐÚNG' if r.is_correct else 'SAI'}"):
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("**🧠 Lỗ hổng kiến thức:**")
                            if r.knowledge_gaps:
                                for gap in r.knowledge_gaps:
                                    st.write(f"• {gap}")
                            else:
                                st.write("✅ Không có lỗ hổng kiến thức")
                            
                            st.markdown("**🏷️ Tags kiến thức:**")
                            if r.knowledge_gap_tag:
                                st.write(", ".join(r.knowledge_gap_tag))
                            else:
                                st.write("Không có tag")
                        
                        with col2:
                            st.markdown("**⚠️ Lỗi tính toán & logic:**")
                            if r.calculation_logic_errors:
                                for error in r.calculation_logic_errors:
                                    st.write(f"• {error}")
                            else:
                                st.write("✅ Không có lỗi tính toán/logic")
                            
                            st.markdown("**🏷️ Tags lỗi:**")
                            if r.error_tag:
                                st.write(", ".join(r.error_tag))
                            else:
                                st.write("Không có tag")
            else:
                st.info("Không có mục nào để chấm hoặc submission_id không hợp lệ.")

    with colB:
        # Hiển thị báo cáo đã lưu nếu có
        saved_report = db.get_latest_report(int(ss.submission_id))
        if saved_report:
            st.success(f"📄 Báo cáo đã lưu • {saved_report.created_at.strftime('%H:%M %d/%m/%Y')}")
            with st.expander("👀 Xem báo cáo đã lưu", expanded=True):
                st.markdown(saved_report.report_content)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Tạo lại báo cáo", use_container_width=True):
                    with st.spinner("Đang tạo báo cáo mới..."):
                        report_md = build_final_report(int(ss.submission_id))
                        if report_md.strip():
                            st.success("✅ Đã tạo báo cáo mới")
                            st.rerun()
            with col2:
                st.download_button(
                    "⬇️ Tải báo cáo (.md)",
                    data=saved_report.report_content,
                    file_name=f"grading_report_{int(ss.submission_id)}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        else:
            if st.button("📝 Tạo bản chấm tổng hợp", use_container_width=True):
                with st.spinner("Đang tạo báo cáo..."):
                    report_md = build_final_report(int(ss.submission_id))
                    if report_md.strip():
                        st.success("✅ Đã tạo và lưu báo cáo")
                        st.rerun()
                    else:
                        st.info("Chưa có dữ liệu chấm hoặc báo cáo rỗng.")


st.divider()
st.caption("Teacher Assistant v1.0 - MVP")