"""
Statistics calculator module for grading data analysis.
Handles all statistical computations separated from main grading logic.
"""

from typing import List, Dict, Any

from utils.constants import INVALID_KNOWLEDGE_GAPS, INVALID_CALCULATION_ERRORS


class StatisticsCalculator:
    """Handles statistical calculations for grading data"""
    
    @staticmethod
    def calculate_basic_statistics(grading_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate basic statistics from grading data
        
        Args:
            grading_data: List of grading items with is_correct, knowledge_gaps, calculation_logic_errors
            
        Returns:
            Dictionary with basic statistics
        """
        total_questions = len(grading_data)
        correct_answers = sum(1 for item in grading_data if item.get("is_correct", False))
        accuracy_rate = correct_answers / total_questions if total_questions > 0 else 0
        
        knowledge_gap_count = 0
        calculation_error_count = 0
        
        for item in grading_data:
            # Filter out invalid entries
            gaps = StatisticsCalculator._filter_valid_items(
                item.get("knowledge_gaps", []), 
                INVALID_KNOWLEDGE_GAPS
            )
            errors = StatisticsCalculator._filter_valid_items(
                item.get("calculation_logic_errors", []), 
                INVALID_CALCULATION_ERRORS
            )
            
            knowledge_gap_count += len(gaps)
            calculation_error_count += len(errors)
        
        return {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy_rate": round(accuracy_rate, 2),
            "knowledge_gap_count": knowledge_gap_count,
            "calculation_error_count": calculation_error_count
        }
    
    @staticmethod
    def calculate_detailed_statistics(grading_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate detailed statistics including breakdowns by categories
        
        Args:
            grading_data: List of grading items
            
        Returns:
            Dictionary with detailed statistics
        """
        basic_stats = StatisticsCalculator.calculate_basic_statistics(grading_data)
        
        # Additional detailed calculations
        missing_answers = sum(1 for item in grading_data 
                            if "Chưa làm" in item.get("knowledge_gaps", []))
        
        system_errors = sum(1 for item in grading_data 
                          if "Không thể phân tích do lỗi hệ thống" in item.get("knowledge_gaps", []))
        
        attempted_questions = basic_stats["total_questions"] - missing_answers
        
        return {
            **basic_stats,
            "missing_answers": missing_answers,
            "system_errors": system_errors,
            "attempted_questions": attempted_questions,
            "attempt_rate": round(attempted_questions / basic_stats["total_questions"], 2) 
                          if basic_stats["total_questions"] > 0 else 0
        }
    
    @staticmethod
    def get_performance_breakdown(grading_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Get breakdown of all knowledge gaps and calculation errors
        
        Args:
            grading_data: List of grading items
            
        Returns:
            Dictionary with lists of all gaps and errors
        """
        all_knowledge_gaps = []
        all_calculation_errors = []
        
        for item in grading_data:
            gaps = StatisticsCalculator._filter_valid_items(
                item.get("knowledge_gaps", []), 
                INVALID_KNOWLEDGE_GAPS
            )
            errors = StatisticsCalculator._filter_valid_items(
                item.get("calculation_logic_errors", []), 
                INVALID_CALCULATION_ERRORS
            )
            
            all_knowledge_gaps.extend(gaps)
            all_calculation_errors.extend(errors)
        
        return {
            "knowledge_gaps": all_knowledge_gaps,
            "calculation_errors": all_calculation_errors,
            "unique_knowledge_gaps": list(set(all_knowledge_gaps)),
            "unique_calculation_errors": list(set(all_calculation_errors))
        }
    
    @staticmethod
    def _filter_valid_items(items: List[str], invalid_items: List[str]) -> List[str]:
        """Filter out invalid items from a list"""
        return [item for item in items if item not in invalid_items]


# Global instance for convenience
stats_calculator = StatisticsCalculator()