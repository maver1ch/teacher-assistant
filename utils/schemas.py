# utils/schemas.py
# Centralized JSON schemas for OpenAI API calls

# Schema for exam analysis (from llm_service.py)
ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "order_index": {"type": "integer"},
                    "part_label": {"type": "string"},
                    "knowledge_topics": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["text", "order_index", "part_label", "knowledge_topics"]
            }
        }
    },
    "required": ["questions"]
}

# Schema for submission segmentation (from llm_service.py)  
SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "integer"},
                    "order_index": {"type": "integer"},
                    "part_label": {"type": "string"},
                    "position": {"type": "integer"},
                    "answer_text": {"type": "string"}
                },
                "required": ["question_id", "order_index", "part_label", "position", "answer_text"]
            }
        }
    },
    "required": ["items"]
}

# Schema for grading results (from grading_service.py)
GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "knowledge_gaps": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "Các lỗ hổng kiến thức cụ thể"
        },
        "calculation_logic_errors": {
            "type": "array",
            "items": {"type": "string"}, 
            "description": "Lỗi tính toán và logic cụ thể"
        },
        "knowledge_gap_tag": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tag keyword cho lỗ hổng kiến thức"
        },
        "error_tag": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tag keyword cho lỗi sai"
        },
        "is_correct": {
            "type": "boolean",
            "description": "true nếu hoàn toàn đúng, false nếu có lỗi"
        }
    },
    "required": ["knowledge_gaps", "calculation_logic_errors", "knowledge_gap_tag", "error_tag", "is_correct"]
}

# Schema for solution generation (from solution_service.py)
SOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "solution_text": {"type": "string"},
        "final_answer": {"type": "string"},
        "reasoning_approach": {"type": "string"},
        "difficulty": {"type": "integer", "minimum": 1, "maximum": 10}
    },
    "required": ["solution_text", "final_answer", "reasoning_approach", "difficulty"]
}