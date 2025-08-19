# database/db_manager.py
import os
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base, Exam, Question, Submission, Grading, SubmissionItem, QuestionSolution, SubmissionReport
from utils.config import DATABASE_PATH

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
        os.makedirs("data", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{DATABASE_PATH}")
        Base.metadata.create_all(self.engine)  # create new tables if missing
        self._run_migrations()  # run any needed schema migrations
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    # --- Minimal helpers (keep API small)
    def create_exam(self, name: str) -> int:
        with self.get_session() as session:
            exam = Exam(name=name)
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

    def _run_migrations(self):
        """Run any needed database schema migrations"""
        try:
            with self.engine.begin() as conn:  # Use begin() for auto-commit
                # Migration 1: Remove original_text column from exams table if it exists
                try:
                    # Test if column exists by trying to select it
                    conn.execute("SELECT original_text FROM exams LIMIT 1")
                    # Column exists, drop it
                    print("Dropping original_text column from exams table...")
                    conn.execute("ALTER TABLE exams DROP COLUMN original_text")
                    print("Migration completed: original_text column removed from exams")
                except Exception:
                    # Column doesn't exist, continue
                    pass
                
                # Migration 2: Check if submission_reports table exists
                try:
                    conn.execute("SELECT COUNT(*) FROM submission_reports LIMIT 1")
                except Exception:
                    # Table doesn't exist, but SQLAlchemy will create it automatically
                    print("submission_reports table will be created by SQLAlchemy")
        except Exception as e:
            print(f"Migration warning: {e}")
            # Continue even if migration fails (table might not exist yet)

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

    def update_solution(self, solution_id: int, final_answer: str, reasoning_approach: str):
        with self.get_session() as session:
            solution = session.query(QuestionSolution).filter(QuestionSolution.id == solution_id).first()
            if solution:
                solution.final_answer = final_answer
                solution.reasoning_approach = reasoning_approach
                session.commit()

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

    def get_solution_preview(self, question_id: int) -> dict:
        """Lấy solution với LaTeX preview formatted"""
        solution = self.get_solution_by_question(question_id)
        if not solution:
            return {}
        
        return {
            "id": solution.id,
            "question_id": solution.question_id,
            "final_answer_preview": format_latex_preview(solution.final_answer, 80),
            "reasoning_preview": format_latex_preview(solution.reasoning_approach, 120),
            "created_at": solution.created_at
        }

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

db = DatabaseManager()