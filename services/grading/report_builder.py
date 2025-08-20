"""
Report builder module for generating student reports.
Handles report generation logic separated from main grading service.
"""

import json
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from utils.llm_logger import log_llm_call
from utils.constants import (
    REPORT_USER_PROMPT_TEMPLATE,
    SERVICE_REPORT_GENERATION,
    DEFAULT_TEMPERATURE_REPORT,
    ERROR_SYSTEM_GRADING_FAILED,
    INVALID_KNOWLEDGE_GAPS,
    INVALID_CALCULATION_ERRORS
)
from utils.prompts import REPORT_SYSTEM_PROMPT
from utils.config import MODEL_REPORT
from database.db_manager import db
from database.models import Grading, Question

# Initialize OpenAI client
load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ReportBuilder:
    """Handles building student reports from grading data"""
    
    def __init__(self):
        pass
    
    def build_report(self, submission_id: int) -> str:
        """
        Build a comprehensive student report from grading data and performance analysis
        
        Args:
            submission_id: ID of the submission to generate report for
            
        Returns:
            Markdown formatted report string
        """
        try:
            # Get grading data
            grading_data = self._get_grading_data(submission_id)
            if not grading_data:
                return "Không có dữ liệu chấm bài để tạo báo cáo."
            
            # Calculate statistics
            statistics = self._calculate_statistics(grading_data)
            
            # Get performance analysis
            performance_analysis = self._get_performance_analysis(submission_id)
            
            # Generate report
            report_content = self._generate_report_with_llm(
                grading_data, statistics, performance_analysis
            )
            
            # Save to database
            self._save_report(submission_id, report_content)
            
            return report_content
            
        except Exception as e:
            print(f"Error building report: {e}")
            return ERROR_SYSTEM_GRADING_FAILED
    
    def get_or_generate_report(self, submission_id: int) -> str:
        """Get existing report or generate new one"""
        saved_report = db.get_latest_report(submission_id)
        if saved_report:
            return saved_report.report_content
        return self.build_report(submission_id)
    
    def _get_grading_data(self, submission_id: int) -> List[Dict[str, Any]]:
        """Get and format grading data from database"""
        with db.get_session() as session:
            grades = (
                session.query(Grading, Question)
                .join(Question, Grading.question_id == Question.id)
                .filter(Grading.submission_id == submission_id)
                .order_by(Question.order_index, Question.id)
                .all()
            )
        
        grading_data = []
        for g, q in grades:
            question_label = f"{q.order_index}{getattr(q, 'part_label', None) or ''}"
            grading_data.append({
                "order_index": q.order_index,
                "part_label": getattr(q, "part_label", None) or "",
                "question_label": question_label,
                "knowledge_gaps": self._safe_json_loads(g.knowledge_gaps) or [],
                "calculation_logic_errors": self._safe_json_loads(g.calculation_logic_errors) or [],
                "is_correct": bool(g.is_correct),
            })
        
        return grading_data
    
    def _calculate_statistics(self, grading_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic statistics from grading data"""
        total_questions = len(grading_data)
        correct_answers = sum(1 for item in grading_data if item["is_correct"])
        accuracy_rate = correct_answers / total_questions if total_questions > 0 else 0
        
        knowledge_gap_count = 0
        calculation_error_count = 0
        
        for item in grading_data:
            # Filter out invalid entries
            gaps = [gap for gap in item["knowledge_gaps"] if gap not in INVALID_KNOWLEDGE_GAPS]
            errors = [error for error in item["calculation_logic_errors"] if error not in INVALID_CALCULATION_ERRORS]
            
            knowledge_gap_count += len(gaps)
            calculation_error_count += len(errors)
        
        return {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy_rate": round(accuracy_rate, 2),
            "knowledge_gap_count": knowledge_gap_count,
            "calculation_error_count": calculation_error_count
        }
    
    def _get_performance_analysis(self, submission_id: int) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance analysis data with fallback"""
        try:
            from services.performance_analyzer import analyze_submission_performance
            performance_data = analyze_submission_performance(submission_id)
            
            return {
                "knowledge_groups": performance_data.get("knowledge_summary", []),
                "error_groups": performance_data.get("error_summary", [])
            }
        except Exception as e:
            print(f"Warning: Could not get performance analysis: {e}")
            return {"knowledge_groups": [], "error_groups": []}
    
    def _generate_report_with_llm(
        self, 
        grading_data: List[Dict[str, Any]], 
        statistics: Dict[str, Any],
        performance_analysis: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Generate report using LLM"""
        user_prompt = REPORT_USER_PROMPT_TEMPLATE.format(
            json.dumps(grading_data, ensure_ascii=False, indent=2),
            json.dumps(performance_analysis, ensure_ascii=False, indent=2),
            json.dumps(statistics, ensure_ascii=False, indent=2)
        )
        
        try:
            resp = _client.chat.completions.create(
                model=MODEL_REPORT,
                messages=[
                    {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=DEFAULT_TEMPERATURE_REPORT
            )
            log_llm_call(response=resp, model_name=MODEL_REPORT, service_name=SERVICE_REPORT_GENERATION)
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Error generating report: {e}")
            return ERROR_SYSTEM_GRADING_FAILED
    
    def _save_report(self, submission_id: int, report_content: str):
        """Save report to database"""
        try:
            db.save_submission_report(submission_id, report_content)
        except Exception as e:
            print(f"Warning: Could not save report to DB: {e}")
    
    def _safe_json_loads(self, s: Optional[str]) -> List[str]:
        """Safely parse JSON string"""
        if not s:
            return []
        try:
            return json.loads(s)
        except Exception:
            return []


# Global instance
report_builder = ReportBuilder()