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
MODEL_NAME = "o4-mini-2025-04-16"
SEGMENT_MODEL = "gpt-4.1-mini"

# Grading Service Models  
MODEL_GRADING = "gpt-4.1-mini-2025-04-14"
COMMENT_MODEL = "gpt-4o-mini"

# OCR Service Model
OCR_MODEL = "gpt-4.1-mini-2025-04-14"

# Solution Service Model
SOLUTION_MODEL_NAME = "o4-mini"

# ==================== TEMPERATURE SETTINGS ====================
LLM_TEMPERATURE = 0.1
OCR_TEMPERATURE = 0.0  
SOLUTION_TEMPERATURE = 0.1
GRADING_TEMPERATURE = 0.1

# ==================== CONTEXT LIMITS ====================
CTX_MAX_CHARS_QUESTION = 800
CTX_MAX_CHARS_ANSWER = 800

# ==================== DATABASE CONFIGURATION ====================
DATABASE_PATH = "data/database.db"

# ==================== OCR CONFIGURATION ====================

# Basic Math Operations
BASIC_MATH_TOKENS = (
    "\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\log", "\\ln", "\\exp",
    "\\prod", "\\coprod", "\\bigcup", "\\bigcap", "\\bigoplus", "\\bigotimes",
    "\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg"
)

# Trigonometric Functions
TRIG_TOKENS = (
    "\\sin", "\\cos", "\\tan", "\\cot", "\\sec", "\\csc",
    "\\arcsin", "\\arccos", "\\arctan", "\\arccot", "\\arcsec", "\\arccsc",
    "\\sinh", "\\cosh", "\\tanh", "\\coth"
)

# Greek Letters
GREEK_LETTERS = (
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\varepsilon",
    "\\zeta", "\\eta", "\\theta", "\\vartheta", "\\iota", "\\kappa",
    "\\lambda", "\\mu", "\\nu", "\\xi", "\\pi", "\\varpi", "\\rho", "\\varrho",
    "\\sigma", "\\varsigma", "\\tau", "\\upsilon", "\\phi", "\\varphi",
    "\\chi", "\\psi", "\\omega",
    "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi", "\\Pi", "\\Sigma",
    "\\Upsilon", "\\Phi", "\\Psi", "\\Omega"
)

# Calculus & Advanced Math
CALCULUS_TOKENS = (
    "\\partial", "\\nabla", "\\infty", "\\pm", "\\mp", "\\times", "\\div",
    "\\cdot", "\\ast", "\\star", "\\circ", "\\bullet", "\\oplus", "\\ominus",
    "\\otimes", "\\oslash", "\\odot", "\\dagger", "\\ddagger"
)

# Set Theory & Logic
SET_LOGIC_TOKENS = (
    "\\forall", "\\exists", "\\nexists", "\\in", "\\notin", "\\ni", "\\not\\ni",
    "\\subset", "\\supset", "\\subseteq", "\\supseteq", "\\subsetneq", "\\supsetneq",
    "\\cup", "\\cap", "\\setminus", "\\emptyset", "\\varnothing",
    "\\land", "\\lor", "\\lnot", "\\implies", "\\iff"
)

# Relations & Inequalities
RELATION_TOKENS = (
    "\\leq", "\\geq", "\\ll", "\\gg", "\\neq", "\\equiv", "\\approx", "\\cong",
    "\\sim", "\\simeq", "\\propto", "\\parallel", "\\perp", "\\mid", "\\nmid"
)

# Arrows
ARROW_TOKENS = (
    "\\leftarrow", "\\rightarrow", "\\leftrightarrow", "\\Leftarrow", "\\Rightarrow",
    "\\Leftrightarrow", "\\uparrow", "\\downarrow", "\\updownarrow",
    "\\nwarrow", "\\nearrow", "\\searrow", "\\swarrow", "\\mapsto", "\\longmapsto"
)

# Brackets & Delimiters
BRACKET_TOKENS = (
    "\\langle", "\\rangle", "\\lceil", "\\rceil", "\\lfloor", "\\rfloor",
    "\\lvert", "\\rvert", "\\lVert", "\\rVert"
)

# Functions & Operators
FUNCTION_TOKENS = (
    "\\max", "\\min", "\\sup", "\\inf", "\\arg", "\\ker", "\\deg", "\\gcd",
    "\\det", "\\dim", "\\hom", "\\end", "\\mod", "\\bmod", "\\pmod"
)

# Combine all tokens into comprehensive set for O(1) lookup
MATH_HINT_TOKENS = set(
    BASIC_MATH_TOKENS + TRIG_TOKENS + GREEK_LETTERS + CALCULUS_TOKENS +
    SET_LOGIC_TOKENS + RELATION_TOKENS + ARROW_TOKENS + BRACKET_TOKENS +
    FUNCTION_TOKENS
)

# Keep minimal set as fallback option
MINIMAL_MATH_TOKENS = {
    "\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\log",
    "\\sin", "\\cos", "\\tan", "\\left", "\\right"
}

# Configuration flag to switch between comprehensive and minimal mode
USE_COMPREHENSIVE_MATH_DETECTION = True

INLINE_SYMBOLS = ("^", "_")
DISPLAY_WRAP = "$$"
INLINE_WRAP = "$"