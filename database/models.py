from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

# Why: single declarative base
Base = declarative_base()

class Exam(Base):
    __tablename__ = "exams"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    questions = relationship("Question", back_populates="exam", cascade="all, delete-orphan")
    submissions = relationship("Submission", back_populates="exam", cascade="all, delete-orphan")

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    question_text = Column(Text, nullable=False)
    difficulty = Column(Integer, default=1)
    order_index = Column(Integer, nullable=False)     # BÀI LỚN
    part_label = Column(String(32))                   # multi-level label, e.g. "1.a" or "IV.1.b"
    knowledge_topics = Column(Text, default="[]")     # JSON string

    exam = relationship("Exam", back_populates="questions")
    gradings = relationship("Grading", back_populates="question", cascade="all, delete-orphan")
    submission_items = relationship("SubmissionItem", back_populates="question", cascade="all, delete-orphan")
    solution = relationship("QuestionSolution", back_populates="question", uselist=False, cascade="all, delete-orphan")
    
class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    student_name = Column(String(255), nullable=False)
    original_text = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    exam = relationship("Exam", back_populates="submissions")
    gradings = relationship("Grading", back_populates="submission", cascade="all, delete-orphan")
    items = relationship("SubmissionItem", back_populates="submission", cascade="all, delete-orphan")

class SubmissionItem(Base):
    __tablename__ = "submission_items"
    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    order_index = Column(Integer, nullable=False)     # BÀI LỚN (duplicate for query convenience)
    part_label = Column(String(8))                    # "a"/"b"/"c" hoặc None
    position = Column(Integer, default=1)             # thứ tự xuất hiện trong bài làm
    answer_text = Column(Text)                        # đoạn trả lời

    submission = relationship("Submission", back_populates="items")
    question = relationship("Question", back_populates="submission_items")

class QuestionSolution(Base):
    __tablename__ = "question_solutions"
    id = Column(Integer, primary_key=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    order_index = Column(Integer, nullable=False)
    part_label = Column(String(32))
    solution_text = Column(Text, nullable=False)
    final_answer = Column(Text)
    reasoning_approach = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    question = relationship("Question", back_populates="solution")

class Grading(Base):
    __tablename__ = "gradings"
    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    
    # 3 yếu tố phân tích mới
    knowledge_gaps = Column(Text)           # Lỗ hổng kiến thức (JSON array)
    calculation_logic_errors = Column(Text) # Lỗi tính toán/logic (JSON array)  
    llm_feedback = Column(Text)             # Nhận xét của LLM
    
    # Đánh giá kết quả
    is_correct = Column(Integer, default=0) # 0=False, 1=True - đúng/sai
    final_score = Column(Float)             # Điểm số (nếu cần)
    
    created_at = Column(DateTime, default=datetime.now)

    submission = relationship("Submission", back_populates="gradings")
    question = relationship("Question", back_populates="gradings")