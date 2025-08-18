# utils/data_models.py
# Centralized dataclass models

from __future__ import annotations
from dataclasses import dataclass
from typing import List

# Data model for question lite (from llm_service.py)
@dataclass
class QuestionLite:
    question_id: int
    order_index: int
    part_label: str
    text_short: str
    keywords: List[str]

# Data model for grading results (from grading_service.py)
@dataclass
class GradingResult:
    submission_id: int
    question_id: int
    order_index: int
    part_label: str
    knowledge_gaps: List[str]
    calculation_logic_errors: List[str]
    knowledge_gap_tag: List[str]
    error_tag: List[str]
    is_correct: bool

# Data model for solution results (from solution_service.py)
@dataclass
class SolutionResult:
    order_index: int
    part_label: str
    solution_text: str
    final_answer: str
    reasoning_approach: str