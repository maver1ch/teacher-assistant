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
        "is_correct": {
            "type": "boolean",
            "description": "True nếu đưa án đúng cùng lời giải hợp lí, false nếu sai đáp án hoặc lời giải có phần logic nghiêm trọng, lí luận bất hợp lí."
        }
    },
    "required": ["knowledge_gaps", "calculation_logic_errors", "is_correct"]
}

# Schema for solution generation (from solution_service.py)
SOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "reasoning_approach": {"type": "string"},
        "difficulty": {"type": "integer", "minimum": 1, "maximum": 10}
    },
    "required": ["final_answer", "reasoning_approach", "difficulty"]
}

PERFORMANCE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["knowledge", "error"],
                        "description": "Loại vấn đề: 'knowledge' cho lỗ hổng kiến thức, 'error' cho lỗi tính toán/logic."
                    },
                    "group": {
                        "type": "string",
                        "description": "Tên nhóm vấn đề (ngắn gọn)."
                    },
                    "description": {
                        "type": "string",
                        "description": "Mô tả chi tiết hơn về bản chất của nhóm vấn đề."
                    },
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Danh sách các câu hỏi liên quan (ví dụ: '1a', '3b')."
                    }
                },
                "required": ["type", "group", "description", "questions"]
            }
        }
    },
    "required": ["analysis"]
}