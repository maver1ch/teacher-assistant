from __future__ import annotations

import os
import json
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from utils.prompts import SOLUTION_SYSTEM_PROMPT
from utils.config import API_KEY_ENV, SOLUTION_MODEL_NAME, SOLUTION_TEMPERATURE
from utils.schemas import SOLUTION_SCHEMA
from utils.data_models import SolutionResult

from database.db_manager import db
from database.models import Question, QuestionSolution

load_dotenv()
_client = OpenAI(api_key=os.getenv(API_KEY_ENV))

def _get_reasoning_effort(difficulty: int) -> str:
    if difficulty < 5:
        return "low"
    elif 6 <= difficulty <= 8:
        return "medium" 
    else:
        return "high"

def _generate_solution_with_context(
    target_question: Question, 
    context_questions: list[Question]
) -> SolutionResult:
    """
    Gọi API OpenAI để giải một câu hỏi với context từ các câu hỏi liên quan.
    """
    
    context_str = ""
    if context_questions:
        context_parts = []
        for q in context_questions:
            # Sử dụng f-string an toàn hơn để tạo nhãn
            label = f" {q.part_label}" if q.part_label else ""
            context_parts.append(f"Câu {q.order_index}{label}: {q.question_text}")
        
        joined_context = "\n".join(context_parts)
        
        context_str = (
            "Để giải câu hỏi này, hãy xem xét bối cảnh từ các câu hỏi liên quan sau:\n"
            "--- BỐI CẢNH BẮT ĐẦU ---\n"
            f"{joined_context}\n"
            "Nếu như phần bối cảnh có liên quan đến bài hiện tại thì đọc kĩ."
            "Nếu như các phần trước tách biệt hoàn toàn và không liên quan đến phần này thì bỏ qua phần bối cảnh. Tập trung vào giải phần hiện tại."
            "--- BỐI CẢNH KẾT THÚC ---\n\n"
        )

    # Xây dựng prompt hoàn chỉnh
    prompt = (
        f"{context_str}"
        "Dựa vào bối cảnh trên (nếu có) và nội dung câu hỏi dưới đây, "
        "hãy tạo hướng logic giải bài và barem chấm điểm:\n\n"
        f"**Câu hỏi cần giải**: {target_question.question_text}\n"
        f"**Độ khó**: {target_question.difficulty}/10\n"
        f"**Kiến thức liên quan**: {target_question.knowledge_topics}\n\n"
        "Trả về JSON với 3 trường:\n"
        "- solution_text: Hướng logic giải (không chi tiết từng bước)\n"
        "- final_answer: Kết quả cuối cùng\n"
        "- reasoning_approach: Barem chấm điểm và tiêu chí đánh giá"
    )
    
    resp = _client.chat.completions.create(
        model=SOLUTION_MODEL_NAME,
        messages=[
            {"role": "system", "content": SOLUTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=SOLUTION_TEMPERATURE,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "question_solution",
                "schema": SOLUTION_SCHEMA
            }
        },
        # reasoning_effort=reasoning_effort # Tham số này không tồn tại, giữ ở dạng comment
    )
    
    data = json.loads(resp.choices[0].message.content)
    
    return SolutionResult(
        order_index=target_question.order_index,
        part_label=target_question.part_label or "",
        solution_text=data["solution_text"],
        final_answer=data["final_answer"], 
        reasoning_approach=data["reasoning_approach"]
    )

def create_and_save_solution(question_id: int) -> int:
    with db.get_session() as session:
        # 1. Lấy câu hỏi mục tiêu
        target_question = session.query(Question).filter(Question.id == question_id).first()
        if not target_question:
            raise ValueError(f"Question with id {question_id} not found")
        
        # 2. Lấy tất cả các câu hỏi có cùng order_index để làm context
        related_questions = session.query(Question).filter(
            Question.exam_id == target_question.exam_id, # Thêm điều kiện exam_id để chắc chắn
            Question.order_index == target_question.order_index
        ).order_by(Question.part_label).all()
        
        # 3. Tách ra những câu hỏi đứng trước để làm context
        context_questions = []
        for q in related_questions:
            if q.id == target_question.id:
                break # Dừng lại khi gặp câu hỏi hiện tại
            context_questions.append(q)
            
        # 4. Gọi hàm sinh lời giải với đầy đủ context
        solution_result = _generate_solution_with_context(target_question, context_questions)
        
        # 5. Lưu kết quả vào CSDL (giữ nguyên logic cũ)
        existing = session.query(QuestionSolution).filter(
            QuestionSolution.question_id == question_id
        ).first()
        
        if existing:
            existing.order_index = solution_result.order_index
            existing.part_label = solution_result.part_label
            existing.solution_text = solution_result.solution_text
            existing.final_answer = solution_result.final_answer
            existing.reasoning_approach = solution_result.reasoning_approach
            session.commit()
            return existing.id
        else:
            solution = QuestionSolution(
                question_id=question_id,
                order_index=solution_result.order_index,
                part_label=solution_result.part_label,
                solution_text=solution_result.solution_text,
                final_answer=solution_result.final_answer,
                reasoning_approach=solution_result.reasoning_approach
            )
            session.add(solution)
            session.commit()
            return solution.id

def get_solution_by_question(question_id: int) -> Dict[str, Any]:
    with db.get_session() as session:
        solution = session.query(QuestionSolution).filter(
            QuestionSolution.question_id == question_id
        ).first()
        
        if not solution:
            return {}
            
        return {
            "id": solution.id,
            "question_id": solution.question_id,
            "order_index": solution.order_index,
            "part_label": solution.part_label,
            "solution_text": solution.solution_text,
            "final_answer": solution.final_answer,
            "reasoning_approach": solution.reasoning_approach,
            "created_at": solution.created_at
        }