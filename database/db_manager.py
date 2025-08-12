# database/db_manager.py
import os
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base, Exam, Question, Submission, Grading, SubmissionItem

DATABASE_PATH = "data/database.db"

class DatabaseManager:
    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{DATABASE_PATH}")
        Base.metadata.create_all(self.engine)  # create new tables if missing
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

db = DatabaseManager()