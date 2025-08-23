# database/db_manager.py
import os
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from database.models import Base, Exam, Question, Submission, Grading, SubmissionItem, QuestionSolution, SubmissionReport, PerformanceAnalysis
from utils.config import DATABASE_URL

def format_latex_preview(text: str, max_length: int = 100) -> str:
    """Helper function để format text cho LaTeX preview"""
    if not text:
        return ""
    
    # Truncate nếu quá dài
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    text = text.replace("\\", "\\\\")  # Escape backslashes
    text = text.replace("$$$", "$$")   # Normalize display math
    
    return text

class DatabaseManager:
    def __init__(self):
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable not set. Please create a .env file.")
        
        self.engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    # --- Minimal helpers (keep API small)
    def create_exam(self, name: str, grade_level: str, exam_topic: str) -> int:
        with self.get_session() as session:
            exam = Exam(name=name, grade_level=grade_level, exam_topic=exam_topic)
            session.add(exam)
            session.commit()
            return exam.id

    def create_submission(self, exam_id: int, student_name: str, original_text: str) -> int:
        with self.get_session() as session:
            sub = Submission(exam_id=exam_id, student_name=student_name, original_text=original_text)
            session.add(sub)
            session.commit()
            return sub.id

    def get_questions_by_exam(self, exam_id: int):
        with self.get_session() as session:
            return session.query(Question).filter(
                Question.exam_id == exam_id
            ).order_by(Question.order_index, Question.id).all()

    def get_submission_items(self, submission_id: int):
        with self.get_session() as session:
            return session.query(SubmissionItem).filter(
                SubmissionItem.submission_id == submission_id
            ).order_by(SubmissionItem.position).all()

    def get_submission_by_id(self, submission_id: int):
        with self.get_session() as session:
            return session.query(Submission).filter(
                Submission.id == submission_id
            ).first()

    def save_submission_report(self, submission_id: int, report_content: str) -> int:
        with self.get_session() as session:
            report = SubmissionReport(
                submission_id=submission_id,
                report_content=report_content
            )
            session.add(report)
            session.commit()
            return report.id

    def get_submission_report(self, submission_id: int):
        with self.get_session() as session:
            return session.query(SubmissionReport).filter(
                SubmissionReport.submission_id == submission_id
            ).order_by(SubmissionReport.created_at.desc()).first()

    def get_latest_report(self, submission_id: int):
        return self.get_submission_report(submission_id)

    def create_solution(self, question_id: int, final_answer: str, reasoning_approach: str) -> int:
        with self.get_session() as session:
            solution = QuestionSolution(
                question_id=question_id,
                final_answer=final_answer,
                reasoning_approach=reasoning_approach
            )
            session.add(solution)
            session.commit()
            return solution.id

    def get_solution_by_question(self, question_id: int):
        with self.get_session() as session:
            return session.query(QuestionSolution).filter(
                QuestionSolution.question_id == question_id
            ).first()

    def create_question(self, exam_id: int, order_index: int, part_label: str, text: str, difficulty: int, knowledge_topics: list) -> int:
        """Tạo câu hỏi mới trong database"""
        with self.get_session() as session:
            knowledge_topics_json = json.dumps(knowledge_topics, ensure_ascii=False)
            
            question = Question(
                exam_id=exam_id,
                question_text=text,
                difficulty=difficulty,
                order_index=order_index,
                part_label=part_label or "",
                knowledge_topics=knowledge_topics_json
            )
            
            session.add(question)
            session.commit()
            return question.id

    def get_question_by_text(self, exam_id: int, question_text: str):
        """Tìm question theo exam_id và text để lấy ID"""
        with self.get_session() as session:
            return session.query(Question).filter(
                Question.exam_id == exam_id,
                Question.question_text == question_text
            ).first()

    def get_questions_with_preview(self, exam_id: int):
        """Lấy danh sách questions với LaTeX preview"""
        with self.get_session() as session:
            questions = session.query(Question).filter(
                Question.exam_id == exam_id
            ).order_by(Question.order_index, Question.id).all()
            
            results = []
            for q in questions:
                results.append({
                    "id": q.id,
                    "order_index": q.order_index,
                    "part_label": q.part_label or "",
                    "question_preview": format_latex_preview(q.question_text, 120),
                    "difficulty": q.difficulty or 0,  # Default 0 if not set yet
                    "knowledge_topics": json.loads(q.knowledge_topics or "[]")
                })
            
            return results

    def save_performance_analysis(self, submission_id: int, analysis_data: list) -> list:
        """Lưu kết quả performance analysis vào database"""
        analysis_ids = []
        with self.get_session() as session:
            # Xóa analysis cũ nếu có
            session.query(PerformanceAnalysis).filter(
                PerformanceAnalysis.submission_id == submission_id
            ).delete()
            
            # Lưu analysis mới
            for item in analysis_data:
                analysis = PerformanceAnalysis(
                    submission_id=submission_id,
                    group_name=item.get("group", ""),
                    group_type=item.get("type", ""),
                    description=item.get("description", ""),
                    related_questions=json.dumps(item.get("questions", []), ensure_ascii=False)
                )
                session.add(analysis)
                session.flush()  # Get ID before commit
                analysis_ids.append(analysis.id)
            
            session.commit()
        return analysis_ids

    def get_performance_analysis(self, submission_id: int) -> list:
        """Lấy kết quả performance analysis đã lưu"""
        with self.get_session() as session:
            analyses = session.query(PerformanceAnalysis).filter(
                PerformanceAnalysis.submission_id == submission_id
            ).order_by(PerformanceAnalysis.created_at.desc()).all()
            
            results = []
            for analysis in analyses:
                results.append({
                    "id": analysis.id,
                    "group": analysis.group_name,
                    "type": analysis.group_type,
                    "description": analysis.description,
                    "questions": json.loads(analysis.related_questions or "[]"),
                    "created_at": analysis.created_at
                })
            
            return results

db = DatabaseManager()