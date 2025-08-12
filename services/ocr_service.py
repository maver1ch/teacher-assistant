# services/ocr_service.py
from __future__ import annotations

import os
import logging
from typing import List
from pathlib import Path

# Why: latest Google GenAI SDK supports system_instruction; keep config centralized
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
# -------------------- Constants (single source of truth)
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_OCR_MODEL = "gemini-1.5-pro"   # switch to *-flash nếu muốn tốc độ
TEMPERATURE = 0.0

# Heuristics for math wrapping
MATH_HINT_TOKENS = (
    "\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\log",
    "\\sin", "\\cos", "\\tan", "\\left", "\\right",
)
INLINE_SYMBOLS = ("^", "_")
DISPLAY_WRAP = "$$"
INLINE_WRAP = "$"

# -------------------- SYSTEM PROMPT (tiếng Việt)
SYSTEM_PROMPT_OCR = """
Bạn là tác nhân OCR tiếng Việt, chuyên xử lý đề thi Toán.

MỤC TIÊU
- Chép lại văn bản sạch (UTF-8).
- Với công thức toán, xuất LaTeX **hợp lệ** để render trực tiếp trong Markdown.
- Không trả về code fence, không thêm đánh dấu Markdown thừa.
- Trong bài có một vài hình học được vẽ (đối với các bài hình), nếu gặp thì hãy bỏ qua nó, không OCR.

QUY TẮC ĐỊNH DẠNG TOÁN
- Giữ nguyên ngữ nghĩa toán; không tự rút gọn hay biến đổi.
- Dùng LaTeX chuẩn: \\frac{a}{b}, \\sqrt{...}, mũ ^{...}, chỉ số _{...}.
- Ký hiệu: \\pi, \\alpha, \\beta, \\theta, ^{\\circ}, mũi tên \\Rightarrow/\\Longrightarrow...
- Ma trận/vec: dùng LaTeX chuẩn nếu có; nếu không chắc chắn, chép nguyên văn.

DELIMITER
- Dòng là công thức độc lập → bọc **$$...$$** (display).
- Công thức chen trong câu → bọc **$...$** (inline).
- Tuyệt đối không dùng ``` hoặc HTML.

LÀM SẠCH
- Loại bỏ header/footer, số trang, tên Sở/Trường/Kỳ thi, hướng dẫn thí sinh, thang điểm/đáp án, “---HẾT---”.
- Giữ xuống dòng/đoạn văn hợp lý; giữ dấu câu và khoảng trắng tự nhiên.
"""

# -------------------- Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Helpers
def _guess_mime(path: str) -> str:
    # Why: minimal mime detection without extra deps
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"

def _looks_like_formula(line: str) -> bool:
    # Why: lightweight rule to decide display-math wrapping
    s = line.strip()
    if not s:
        return False
    if "$" in s:
        return False
    if any(tok in s for tok in MATH_HINT_TOKENS):
        return True
    if any(sym in s for sym in INLINE_SYMBOLS) and "=" in s:
        return True
    letters = sum(ch.isalpha() for ch in s)
    mathsy = sum(ch in "=+-/*()[]{}<>\\" for ch in s)
    return mathsy >= letters and mathsy >= 3

def _ensure_latex_delimiters(text: str) -> str:
    # Why: normalize math lines so Markdown render ổn định
    lines = text.splitlines()
    out = []
    for ln in lines:
        s = ln.rstrip()
        if _looks_like_formula(s):
            out.append(f"{DISPLAY_WRAP}{s}{DISPLAY_WRAP}")
        else:
            out.append(s)
    return "\n".join(out)

# -------------------- Service
class OCRService:
    def __init__(self) -> None:
        api_key = os.getenv(GEMINI_API_KEY_ENV)
        self._client = genai.Client(api_key=api_key)

    def _gen_config(self) -> types.GenerateContentConfig:
        # Why: keep system_instruction & temperature consistent
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_OCR,
            temperature=TEMPERATURE,
        )

    def _ocr_single_image_with_msg(self, image_path: str, user_msg: str) -> str:
        mime = _guess_mime(image_path)
        with open(image_path, "rb") as f:
            img_part = types.Part.from_bytes(mime_type=mime, data=f.read())  # kw-only args

        resp = self._client.models.generate_content(
            model=GEMINI_OCR_MODEL,
            contents=[user_msg, img_part],
            config=self._gen_config(),
        )
        text = (resp.text or "").strip()
        return _ensure_latex_delimiters(text) if text else ""

    # --- OCR cho đề thi (giữ nguyên)
    def ocr_single_image(self, image_path: str) -> str:
        user_msg = "Hãy chép lại TRANG ĐỀ THI này. Tuân thủ nghiêm các quy tắc LaTeX và delimiter đã nêu."
        return self._ocr_single_image_with_msg(image_path, user_msg)

    def ocr_multiple_images(self, image_paths: List[str]) -> str:
        parts = [self.ocr_single_image(p) for p in image_paths]
        return "\n\n---\n\n".join(parts)

    # --- OCR cho bài làm học sinh (user message khác)
    def ocr_submission_images(self, image_paths: List[str]) -> str:
        user_msg = (
            "Đây là BÀI LÀM của học sinh. Hãy chép lại nguyên văn, tuân thủ quy tắc LaTeX và delimiter đã nêu. "
            "Nếu phát hiện dòng ghi tên học sinh (ví dụ: 'Họ và tên: ...', 'Họ tên: ...', 'Name: ...'), hãy GIỮ NGUYÊN dòng đó."
        )
        parts = [self._ocr_single_image_with_msg(p, user_msg) for p in image_paths]
        return "\n\n---\n\n".join(parts)

    def format_math_for_display(self, text: str) -> str:
        # Why: placeholder for future display tweaks
        return text

# Singleton
ocr = OCRService()