# services/ocr_service.py
from __future__ import annotations

import os
import logging
from typing import List
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
import base64
load_dotenv()
# -------------------- Constants (single source of truth)
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OCR_MODEL = "gpt-4.1-mini-2025-04-14"
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
def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _get_image_mime_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"

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
        api_key = os.getenv(OPENAI_API_KEY_ENV)
        try:
            self._client = OpenAI(api_key=api_key)
        except TypeError as e:
            # Handle proxy-related initialization issues
            logger.warning(f"OpenAI client initialization with basic params: {str(e)}")
            self._client = OpenAI(api_key=api_key)

    def _ocr_single_image_with_msg(self, image_path: str, user_msg: str) -> str:
        base64_image = _encode_image(image_path)
        mime_type = _get_image_mime_type(image_path)
        
        try:
            response = self._client.chat.completions.create(
                model=OCR_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_OCR
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": user_msg},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=4000
            )
            
            text = response.choices[0].message.content.strip()
            return _ensure_latex_delimiters(text) if text else ""
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {str(e)}")
            return ""

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