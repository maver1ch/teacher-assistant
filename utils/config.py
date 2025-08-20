# ==================== APP CONFIGURATION ====================
PAGE_TITLE = "Trợ lý Chấm bài"
PAGE_ICON = "📚"
LAYOUT = "wide"
EDITOR_HEIGHT = 420
DF_HEIGHT = 360

# ==================== API CONFIGURATION ====================
API_KEY_ENV = "OPENAI_API_KEY"

# ==================== MODEL CONFIGURATION ====================
# LLM Service Models
EXAM_ANALYZER_MODEL = "gpt-4.1-mini"
SEGMENT_MODEL = "gpt-4.1-mini"
GROUPING_MODEL = "gpt-4.1-mini"

# Grading Service Models  
MODEL_GRADING = "gpt-4.1-mini"
MODEL_GRADING_ADVANCED = "gpt-5-mini"

# Solution Service Model
SOLUTION_MODEL_NAME = "gpt-5-mini"

# ==================== TEMPERATURE SETTINGS ====================
LLM_TEMPERATURE = 0.1
GRADING_TEMPERATURE = 0.1

# ==================== DATABASE CONFIGURATION ====================
DATABASE_PATH = "data/database.db"

MODEL_PRICING = {
    # gpt-4o-mini and its variants
    "gpt-4.1-mini":           {"input": 0.4, "output": 1.60},
    "gpt-5-mini":                {"input": 0.5, "output": 2.00},
    "gpt-4o-mini":            {"input": 0.15, "output": 0.60},
}