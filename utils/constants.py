"""
Application constants and hardcoded values.
Centralized location for all constant values used throughout the application.
"""

from typing import List

# =====================
# Grading Constants
# =====================

# Default responses for missing or error cases
DEFAULT_MISSING_ANSWER = ["Chưa làm"]
DEFAULT_SYSTEM_ERROR = ["Không thể phân tích do lỗi hệ thống"]
DEFAULT_NO_ERRORS = ["Không có"]

# Filter values for data processing
INVALID_KNOWLEDGE_GAPS = ["Chưa làm", "Không thể phân tích do lỗi hệ thống"]
INVALID_CALCULATION_ERRORS = ["Chưa làm", "Không có"]

# =====================
# LLM Service Names
# =====================

SERVICE_GRADING_COMPARISON_ADVANCED = "grading_comparison_advanced"
SERVICE_REPORT_GENERATION = "report_generation"
SERVICE_PERFORMANCE_ANALYSIS = "performance_analysis"
SERVICE_SOLUTION_GENERATION = "solution_generation"

# =====================
# Error Messages
# =====================

ERROR_MISSING_OPENAI_KEY = "Missing OPENAI_API_KEY environment variable"
ERROR_SOLUTION_NOT_FOUND = "Không tìm thấy lời giải chuẩn cho question_id {}"
ERROR_SYSTEM_GRADING_FAILED = "Không thể tạo báo cáo do lỗi hệ thống."
ERROR_PERFORMANCE_ANALYSIS_FAILED = "Error during performance analysis: {}"

# =====================
# Success Messages
# =====================

SUCCESS_GRADING_SAVED = "All grading results saved successfully!"
SUCCESS_PERFORMANCE_ANALYSIS = "Analysis complete: {} knowledge groups, {} error groups"

# =====================
# User Prompt Templates
# =====================

GRADING_USER_PROMPT_TEMPLATE = """So sánh bài làm học sinh với đáp án chuẩn và barem chấm điểm:

**ĐÁP ÁN CHUẨN:**
{}

**BAREM CHẤM ĐIỂM:**
{}

**BÀI LÀM HỌC SINH:**
{}

Hãy phân tích và đánh giá theo các yếu tố đã nêu trong system prompt."""

REPORT_USER_PROMPT_TEMPLATE = """Dưới đây là dữ liệu chấm bài chi tiết và phân tích hiệu suất của học sinh:

**GRADING DATA:**
{}

**PERFORMANCE ANALYSIS:**
{}

**STATISTICS:**
{}

Hãy tạo báo cáo tổng hợp theo cấu trúc trong system prompt, sử dụng performance analysis để tạo action plan hiệu quả."""

PERFORMANCE_ANALYSIS_USER_PROMPT_TEMPLATE = """Dưới đây là danh sách chi tiết các lỗi từ bài làm của một học sinh, được phân thành 2 loại:

**DANH SÁCH LỖ HỔNG KIẾN THỨC:**
{}

**DANH SÁCH LỖI TÍNH TOÁN & LOGIC:**
{}

Hãy phân tích và nhóm từng loại riêng biệt theo hướng dẫn. Tạo các nhóm cho knowledge_gaps và calculation_errors tách biệt."""

# =====================
# Default Configurations
# =====================

DEFAULT_TEMPERATURE_GRADING = 0.1
DEFAULT_TEMPERATURE_REPORT = 0.2
DEFAULT_MAX_KNOWLEDGE_GAPS = 3
DEFAULT_MAX_CALCULATION_ERRORS = 3
DEFAULT_MAX_CHARS_PER_ITEM = 25

# =====================
# Text Cleaning Functions
# =====================

def clean_text(text: str) -> str:
    """Remove NUL characters and clean text for JSON/API processing"""
    if not text:
        return ""
    return text.replace('\x00', '').replace('\0', '').strip()

def clean_dict_values(data: dict) -> dict:
    """Recursively clean all string values in a dictionary"""
    if not isinstance(data, dict):
        return data
    
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, str):
            cleaned[key] = clean_text(value)
        elif isinstance(value, dict):
            cleaned[key] = clean_dict_values(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_text(item) if isinstance(item, str) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned