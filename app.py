#app.py
import streamlit as st
import tempfile
import os
import logging
import pandas as pd
import json

# ---------- Import config
from utils.config import PAGE_TITLE, PAGE_ICON, LAYOUT, EDITOR_HEIGHT, DF_HEIGHT

# ---------- Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Initialize LLM Logger (for API call tracking)
from utils.llm_logger import setup_llm_logger
setup_llm_logger()

# ---------- Services & DB
from services.performance_analyzer import analyze_submission_performance
from services.exam_analyzer import analyze_exam_from_images
from services.submission_processor import segment_submission_from_images
from database.db_manager import db
from database.models import Exam, Submission, Question, SubmissionItem, Grading
from services.grading_service import grade_submission, build_final_report, get_or_generate_report
from services.solution_service import create_and_save_solution, get_solution_by_question

# ---------- App config
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# ---------- Session State
ss = st.session_state
ss.setdefault("analysis_result", None)
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
ss.setdefault("password_correct", False) # Thêm session state cho mật khẩu
ss.setdefault("grading_results", None)  # Lưu kết quả chấm bài

# ---------- Helpers
def display_math_text(text: str, enable_line_click: bool = False, target_key: str = None, max_height: int = None):
    """Enhanced LaTeX display với preview cải thiện"""
    if text is None or text.strip() == "":
        st.info("Nội dung trống")
        return
    
    lines = str(text).splitlines()
    if not enable_line_click:
        # Enhanced display với LaTeX preview
        container = st.container()
        with container:
            if max_height:
                # Sử dụng scrollable container nếu có max_height
                st.markdown(f"""
                <div style="max-height: {max_height}px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa;">
                """, unsafe_allow_html=True)
            
            for raw in lines:
                s = str(raw).rstrip()
                if not s or s.strip().lower() == "none":
                    st.markdown("&nbsp;")
                    continue
                
                # Cải thiện LaTeX display
                if "$" in s:
                    st.markdown(s)  # Streamlit tự render LaTeX
                else:
                    st.markdown(s)
            
            if max_height:
                st.markdown("</div>", unsafe_allow_html=True)
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

# --- HÀM KIỂM TRA MẬT KHẨU ---
def check_password():
    """Returns `True` if the user had the correct password."""
    try:
        # Lấy khóa truy cập đã cấu hình trong Streamlit Secrets
        correct_password = st.secrets["ACCESS_KEY"]
    except KeyError:
        # Nếu không có khóa nào được cấu hình, cho phép truy cập (hữu ích khi chạy local)
        st.warning("🔑 Khóa truy cập chưa được thiết lập trong Secrets. Bỏ qua kiểm tra.")
        return True

    # Kiểm tra xem người dùng đã đăng nhập thành công trong session này chưa
    if ss.get("password_correct", False):
        return True

    # Hiển thị form nhập mật khẩu
    st.header("🔐 Yêu cầu Truy cập")
    st.write("Vui lòng nhập khóa truy cập để sử dụng ứng dụng.")
    with st.form("password_form"):
        password = st.text_input("Khóa Truy Cập", type="password")
        submitted = st.form_submit_button("Xác nhận")

        if submitted:
            if password == correct_password:
                # Nếu mật khẩu đúng, lưu trạng thái và chạy lại app
                ss["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Khóa truy cập không chính xác. Vui lòng thử lại.")
    return False

# --- BỌC TOÀN BỘ APP TRONG HÀM KIỂM TRA MẬT KHẨU ---
if check_password():
    # ---------- Sidebar (Navigator + DB picker) — NO AUTO-APPLY ----------
    with st.sidebar:
        st.header("📋 Điều hướng nhanh")

        step_labels = {
            1: "1️⃣ Upload & OCR đề", 
            3: "2️⃣ Tạo lời giải",
            4: "3️⃣ Upload bài làm", 
            5: "4️⃣ Chấm bài",
        }

        desired_step = st.selectbox(
            "🔀 Đi tới bước",
            options=[1, 3, 4, 5],
            index=max(0, [1, 3, 4, 5].index(ss.current_step) if ss.current_step in [1, 3, 4, 5] else 0),
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
                default_exam_idx = next((i for i, e in enumerate(exams) if e["id"] == ss.exam_id), 0)
                chosen_exam_label = st.selectbox("Đề thi (Exam)", exam_options, index=default_exam_idx, key="pick_exam")
                pending_exam_id = exams[exam_options.index(chosen_exam_label)]["id"]
                
                # Preview questions với LaTeX
                if st.checkbox("Xem trước câu hỏi", value=False, key="preview_questions"):
                    questions_preview = db.get_questions_with_preview(pending_exam_id)
                    if questions_preview:
                        for q in questions_preview[:3]:  # Chỉ show 3 câu đầu
                            with st.expander(f"Câu {q['order_index']}{q['part_label']}", expanded=False):
                                st.markdown(q['question_preview'])
                                difficulty_text = f"{q['difficulty']}/10" if q['difficulty'] > 0 else "Chưa đánh giá"
                                st.caption(f"Độ khó: {difficulty_text}")
                        if len(questions_preview) > 3:
                            st.caption(f"... và {len(questions_preview) - 3} câu khác")
            else:
                st.info("Chưa có Exam trong DB.")

            if pending_exam_id or ss.exam_id:
                exam_for_subs = pending_exam_id or ss.exam_id
                subs = list_submissions(exam_for_subs)
                if subs:
                    sub_options = [f'#{s["id"]} • {s["student_name"]}' for s in subs]
                    default_sub_idx = next((i for i, s in enumerate(subs) if s["id"] == ss.submission_id), 0)
                    chosen_sub_label = st.selectbox("Bài làm (Submission)", sub_options, index=default_sub_idx, key="pick_sub")
                    pending_submission_id = subs[sub_options.index(chosen_sub_label)]["id"]
                    
                    # Preview bài làm nếu có
                    if st.checkbox("Xem trước bài làm", value=False, key="preview_submission"):
                        submission = db.get_submission_by_id(pending_submission_id)
                        if submission and hasattr(submission, 'original_text') and submission.original_text:
                            st.text_area("", value=submission.original_text[:300] + "..." if len(submission.original_text) > 300 else submission.original_text, height=100, disabled=True, key="sidebar_preview")
                else:
                    st.info("Exam này chưa có Submission.")


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
        st.header("Bước 1: Upload ảnh đề & Phân tích")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("📤 Upload ảnh đề bài")
            exam_name = st.text_input("Tên đề bài:", placeholder="VD: Đề thi giữa kỳ I Toán 12")
            
            # --- NEW INPUTS ---
            grade_level = st.selectbox(
                "Lớp học (bắt buộc):", 
                options=[f"Lớp {i}" for i in range(6, 13)], 
                index=3  # Default to "Lớp 9"
            )
            exam_topic = st.text_input(
                "Chủ đề chính của đề (không bắt buộc):", 
                placeholder="VD: Hàm số và đồ thị, Phương trình lượng giác"
            )
            # --- END NEW INPUTS ---
            
            uploaded_files = st.file_uploader(
                "Chọn ảnh (có thể nhiều ảnh)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
            )

            if uploaded_files and exam_name and grade_level and st.button("🚀 Phân tích đề bài", type="primary", key="analyze_exam"):
                with st.spinner("Đang OCR và phân tích đề..."):
                    temp_paths = []
                    for f in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            tmp.write(f.getbuffer())
                            temp_paths.append(tmp.name)
                    
                    # Extract grade number from "Lớp X"
                    grade_number = grade_level.split(" ")[1]

                    parsed = analyze_exam_from_images(temp_paths, grade_number, exam_topic)

                    for p in temp_paths:
                        os.unlink(p)

                    ss.parsed_questions = [
                        {
                            "order_index": int(p["order_index"]),
                            "part_label": str(p.get("part_label") or ""),
                            "text": str(p["text"]).strip(),
                            "knowledge_topics": [str(x).strip() for x in (p.get("knowledge_topics") or [])][:4],
                        }
                        for p in parsed
                    ]
                    
                    if ss.parsed_questions:
                        # Update create_exam call
                        exam_id = db.create_exam(exam_name, grade_number, exam_topic)
                        ss.exam_id = exam_id
                        st.success(f"✅ Phân tích hoàn thành: {len(ss.parsed_questions)} câu hỏi.")
                    else:
                        st.error("Không thể phân tích đề bài. Vui lòng thử lại.")

        with right:
            st.subheader("🗒️ Trạng thái")
            if ss.parsed_questions:
                st.success(f"Đã phân tích {len(ss.parsed_questions)} câu hỏi.")
            else:
                st.info("Chưa có dữ liệu phân tích.")

        st.divider()

        if ss.parsed_questions:
            st.subheader("📋 Kết quả phân tích")
            df_data = []
            for i, q in enumerate(ss.parsed_questions):
                df_data.append({
                    "STT": i + 1,
                    "Bài": q["order_index"],
                    "Ý": q["part_label"] or "",
                    "Nội dung": q["text"],
                    "Kiến thức": " • ".join(q["knowledge_topics"])
                })
            
            # Editable dataframe
            import pandas as pd
            df_preview = pd.DataFrame(df_data)
            st.info("💡 **Hướng dẫn chỉnh sửa:** Kiến thức phân tách bằng dấu • và cần có **3-5 tags**. VD: Hàm số • Đạo hàm • Cực trị")
            edited_df = st.data_editor(
                df_preview,
                height=300,
                use_container_width=True,
                column_config={
                    "STT": st.column_config.NumberColumn("STT", disabled=True),
                    "Bài": st.column_config.NumberColumn("Bài", disabled=False),
                    "Ý": st.column_config.TextColumn("Ý", disabled=False),
                    "Nội dung": st.column_config.TextColumn("Nội dung", disabled=False, width="large"),
                    "Kiến thức": st.column_config.TextColumn("Kiến thức (cách nhau bằng •)", disabled=False, width="medium")
                },
                key="edit_parsed_preview"
            )
            
            # Update session state with edited data and validation
            validation_errors = []
            for i, row in edited_df.iterrows():
                if i < len(ss.parsed_questions):
                    ss.parsed_questions[i]["order_index"] = int(row["Bài"])
                    ss.parsed_questions[i]["part_label"] = str(row["Ý"])
                    ss.parsed_questions[i]["text"] = str(row["Nội dung"])
                    
                    # Parse and validate knowledge topics (3-5 tags)
                    knowledge_str = str(row["Kiến thức"]).strip()
                    knowledge_topics = [t.strip() for t in knowledge_str.split("•") if t.strip()]
                    
                    ss.parsed_questions[i]["knowledge_topics"] = knowledge_topics[:5]  # Trim to max 5
            
            # Display validation errors if any
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            
            if st.button("✅ Lưu câu hỏi & Tiếp tục", type="primary", disabled=len(validation_errors) > 0):
                for q in ss.parsed_questions:
                    db.create_question(
                        exam_id=ss.exam_id,
                        order_index=q["order_index"],
                        part_label=q["part_label"],
                        text=q["text"],
                        difficulty=0,  # Default 0, will be set when creating solution
                        knowledge_topics=q["knowledge_topics"]
                    )
                
                ss.current_step = 3
                st.rerun()

    # ====================== STEP 4 ======================
    elif ss.current_step == 4 and ss.exam_id:
        st.header("Bước 3: Upload và xử lý bài làm học sinh")
        
        if ss.submission_id:
            st.info(f"📌 Đề • ID: {ss.exam_id} | 📝 Bài làm • ID: {ss.submission_id}")
        else:
            st.info(f"📌 Đề • ID: {ss.exam_id}")

        st.subheader("📝 BÀI LÀM HỌC SINH")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Xem nội dung bài
            if ss.submission_id:
                submission = db.get_submission_by_id(ss.submission_id)
                if submission:
                    with st.expander("👀 Xem nội dung bài", expanded=False):
                        st.markdown(f"**Tên học sinh:** {submission.student_name}")
                        st.markdown(f"**Đã tạo:** {submission.created_at}")
                        if hasattr(submission, 'original_text') and submission.original_text:
                            st.caption("Nội dung OCR từ trước:")
                            st.text_area("", value=submission.original_text[:500] + "...", height=100, disabled=True)
            
            # Chọn ảnh bài làm
            st.markdown("**📷 Chọn ảnh bài làm**")
            submission_files = st.file_uploader(
                "Upload nhiều ảnh bài làm:", 
                type=["png", "jpg", "jpeg"], 
                accept_multiple_files=True, 
                key="submission_images"
            )
            
            if submission_files:
                st.success(f"Đã chọn {len(submission_files)} ảnh")
                
                # Preview ảnh đã upload
                if st.checkbox("Xem preview ảnh"):
                    cols = st.columns(min(len(submission_files), 3))
                    for i, img_file in enumerate(submission_files[:3]):
                        with cols[i]:
                            st.image(img_file, caption=f"Ảnh {i+1}", use_container_width=True)
                    if len(submission_files) > 3:
                        st.caption(f"... và {len(submission_files) - 3} ảnh khác")
        
        with col2:
            # Thông tin học sinh
            st.markdown("**👤 Thông tin học sinh**")
            student_name = st.text_input("Tên học sinh:", value="", key="student_name_input")
            
            # Nút xử lý
            if submission_files and student_name.strip():
                if st.button("🚀 Xử lý bài làm", type="primary", use_container_width=True):
                    with st.spinner("Đang xử lý hình ảnh và phân đoạn bài làm..."):
                        # Tạo temp files
                        temp_paths = []
                        for f in submission_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                                tmp.write(f.getbuffer())
                                temp_paths.append(tmp.name)
                        
                        try:
                            # Gọi function xử lý hình ảnh (sẽ tạo sau)
                            questions = db.get_questions_by_exam(ss.exam_id)
                            if questions:
                                from services.submission_processor import segment_submission_from_images
                                data = segment_submission_from_images(questions, temp_paths)
                                ss.segmented_items = data.get("items", [])
                                
                                # Lưu submission vào DB (không có original_text từ OCR)
                                sub_id = db.create_submission(
                                    exam_id=ss.exam_id,
                                    student_name=student_name.strip(),
                                    original_text=""  # Không lưu text vì xử lý trực tiếp từ hình
                                )
                                ss.submission_id = sub_id
                                
                                st.success(f"✅ Đã xử lý {len(submission_files)} ảnh • Phân đoạn {len(ss.segmented_items)} items • Submission ID: {sub_id}")
                            else:
                                st.error("Không tìm thấy câu hỏi cho đề này.")
                                
                        except Exception as e:
                            st.error(f"❌ Lỗi xử lý: {str(e)}")
                        finally:
                            # Cleanup temp files
                            for p in temp_paths:
                                try:
                                    os.unlink(p)
                                except:
                                    pass
                        
                    st.rerun()
            elif submission_files:
                st.warning("Vui lòng nhập tên học sinh")
            elif student_name.strip():
                st.warning("Vui lòng chọn ảnh bài làm")

        # Hiển thị kết quả phân đoạn
        if ss.segmented_items:
            st.divider()
            st.subheader("✏️ Kết quả phân đoạn bài làm")
            
            # Editable dataframe với LaTeX preview  
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("**📝 Chỉnh sửa kết quả phân đoạn:**")
                
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

            # The save and continue button is now combined
            if st.button("💾 Lưu & Tiếp tục Chấm bài", type="primary", use_container_width=True):
                with st.spinner("Đang lưu các câu trả lời..."):
                    with db.get_session() as session:
                        # Clear old items for this submission to prevent duplicates if user goes back and edits
                        session.query(SubmissionItem).filter(SubmissionItem.submission_id == ss.submission_id).delete()
                        
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
                    st.success("Đã lưu các ý của bài làm.")
                
                # Navigate to the next step
                ss.current_step = 5
                st.rerun()

        # Nút chuyển bước 5 (đã được gộp vào nút trên)
        if ss.submission_id and not ss.segmented_items:
            st.divider()
            if st.button("➡️ Tiếp tục Bước 4 (Chấm bài)", use_container_width=True):
                ss.current_step = 5
                st.rerun()

    # ====================== STEP 5 ======================
    elif ss.current_step == 5 and ss.exam_id:
        st.header("Bước 4: Chấm bài")
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

        # --- Lấy dữ liệu đã lưu từ DB lên trước ---
        saved_analysis = db.get_performance_analysis(int(ss.submission_id))

        colA, colB = st.columns([1, 1])
        with colA:
            st.subheader("⚙️ Chấm bài tự động")

            if st.button("🧮 Chấm toàn bộ bài", use_container_width=True, type="primary"):
                with st.spinner("Đang chấm bài ..."):
                    results = grade_submission(int(ss.submission_id))
                    ss.grading_results = results  # Lưu vào session state
                
            # Hiển thị kết quả từ session state thay vì biến local
            if ss.grading_results:
                st.subheader("📊 Kết quả chấm chi tiết")
                    
                submission_items = db.get_submission_items(int(ss.submission_id))
                answers_map = {item.question_id: item.answer_text for item in submission_items}

                correct_count = sum(1 for r in ss.grading_results if r.is_correct)
                total_count = len(ss.grading_results)
                st.metric("Tổng quan", f"{correct_count}/{total_count} câu đúng", 
                         f"{correct_count/total_count*100:.1f}%" if total_count > 0 else "0%")
                
                from services.solution_service import get_solution_by_question
                solutions_map = {}
                for r in ss.grading_results:
                    solution = get_solution_by_question(r.question_id)
                    solutions_map[r.question_id] = solution.get("reasoning_approach", "Chưa có barem chấm")
                
                table_data = []
                for r in ss.grading_results:
                    student_answer = answers_map.get(r.question_id, "Không làm")
                    status = "✅ ĐÚNG" if r.is_correct else "❌ SAI"
                    reasoning_approach = solutions_map.get(r.question_id, "Chưa có barem chấm")
                    
                    knowledge_gaps_text = "\n".join([f"• {gap}" for gap in r.knowledge_gaps]) if r.knowledge_gaps else "Không có"
                    errors_text = "\n".join([f"• {error}" for error in r.calculation_logic_errors]) if r.calculation_logic_errors else "Không có"
                    
                    table_data.append({
                        "Câu": f"{r.order_index}{r.part_label}",
                        "Barem chấm điểm": reasoning_approach,
                        "Kết quả": status,
                        "Bài làm học sinh": student_answer,
                        "Lỗ hổng kiến thức": knowledge_gaps_text,
                        "Lỗi tính toán/logic": errors_text
                    })
                    
                import pandas as pd
                df_results = pd.DataFrame(table_data)
                
                col_title, col_download1, col_download2, col_save = st.columns([2, 1, 1, 1])
                with col_title:
                    st.subheader("📋 Bảng kết quả tổng hợp")
                with col_download1:
                    csv_data = df_results.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="⬇️ Tải CSV",
                        data=csv_data,
                        file_name=f"grading_results_{ss.submission_id}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_download2:
                    # Excel download
                    import io
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df_results.to_excel(writer, index=False, sheet_name='Kết quả chấm bài')
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        label="📊 Tải Excel",
                        data=excel_data,
                        file_name=f"grading_results_{ss.submission_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                with col_save:
                    if st.button("💾 Lưu vào DB", use_container_width=True):
                        from services.grading_service import save_grading_results
                        if save_grading_results(ss.grading_results):
                            st.success("✅ Đã lưu kết quả!")
                        else:
                            st.error("❌ Lỗi khi lưu!")
                
                st.dataframe(
                    df_results,
                    use_container_width=True,
                    height=600,
                    column_config={
                        "Câu": st.column_config.TextColumn("Câu", width="small"),
                        "Barem chấm điểm": st.column_config.TextColumn("Barem chấm điểm", width="large"),
                        "Kết quả": st.column_config.TextColumn("Kết quả", width="small"),
                        "Bài làm học sinh": st.column_config.TextColumn("Bài làm học sinh", width="large"),
                        "Lỗ hổng kiến thức": st.column_config.TextColumn("Lỗ hổng kiến thức", width="medium"),
                        "Lỗi tính toán/logic": st.column_config.TextColumn("Lỗi tính toán/logic", width="medium")
                    }
                )

            else:
                st.info("Chưa chấm bài hoặc không có kết quả chấm.")

        with colB:
            saved_report = db.get_latest_report(int(ss.submission_id))
            if saved_report:
                st.success(f"📄 Báo cáo đã lưu • {saved_report.created_at.strftime('%H:%M %d/%m/%Y')}")
                with st.expander("👀 Xem báo cáo đã lưu", expanded=True):
                    st.markdown(saved_report.report_content)
            
            # --- Cụm nút hành động ---
            report_button_label = "🔄 Tạo lại báo cáo" if saved_report else "📝 Tạo bản chấm tổng hợp"
            if st.button(report_button_label, use_container_width=True):
                with st.spinner("Đang tạo báo cáo..."):
                    report_md = build_final_report(int(ss.submission_id))
                    if report_md.strip():
                        st.success("✅ Đã tạo và lưu báo cáo")
                        st.rerun()
                    else:
                        st.info("Chưa có dữ liệu chấm hoặc báo cáo rỗng.")
            
            analysis_button_label = "🔄 Phân tích lại Nhóm lỗi" if saved_analysis else "🔎 Phân tích Nhóm lỗi"
            if st.button(analysis_button_label, use_container_width=True):
                with st.spinner("Đang thực hiện phân tích chuyên sâu..."):
                    analyze_submission_performance(int(ss.submission_id))
                    st.success("✅ Phân tích hoàn tất và đã lưu!")
                    st.rerun()
            
            if saved_report:
                st.download_button(
                    "⬇️ Tải MD",
                    data=saved_report.report_content,
                    file_name=f"grading_report_{int(ss.submission_id)}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        # --- Hiển thị kết quả phân tích (ưu tiên từ DB) ---
        if saved_analysis:
            st.divider()
            st.subheader("🔍 Kết quả Phân tích Chuyên sâu (Đã lưu)")
            
            # Group analysis by type
            knowledge_summary = [item for item in saved_analysis if item["type"] == "knowledge"]
            error_summary = [item for item in saved_analysis if item["type"] == "error"]

            if not knowledge_summary and not error_summary:
                st.info("Không tìm thấy các nhóm lỗi hoặc lỗ hổng kiến thức nổi bật.")
            else:
                if knowledge_summary:
                    st.markdown("##### 🧠 Lỗ hổng kiến thức nổi bật")
                    for group in knowledge_summary:
                        with st.expander(f"**{group['group']}** (Câu: {', '.join(group['questions'])})"):
                            st.markdown(group['description'])
                
                if error_summary:
                    st.markdown("##### ✏️ Lỗi sai phổ biến")
                    for group in error_summary:
                        with st.expander(f"**{group['group']}** (Câu: {', '.join(group['questions'])})"):
                            st.markdown(group['description'])

    st.divider()
    st.caption("Teacher Assistant v1.0 - MVP")