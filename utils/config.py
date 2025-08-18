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

# Grading Service Models  
MODEL_GRADING = "gpt-4.1-mini-2025-04-14"
COMMENT_MODEL = "gpt-4o-mini"

# Solution Service Model
SOLUTION_MODEL_NAME = "o4-mini"

# ==================== TEMPERATURE SETTINGS ====================
LLM_TEMPERATURE = 0.1
GRADING_TEMPERATURE = 0.1

# ==================== CONTEXT LIMITS ====================
CTX_MAX_CHARS_QUESTION = 800
CTX_MAX_CHARS_ANSWER = 800

# ==================== DATABASE CONFIGURATION ====================
DATABASE_PATH = "data/database.db"