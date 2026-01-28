"""
Test cases for benchmarking Direct vs Agent.
"""

from typing import List, Dict, Any
from pydantic import BaseModel


class SimpleResponse(BaseModel):
    """Simple response format for structured output tests."""
    answer: str
    confidence: float


class ComplexResponse(BaseModel):
    """Complex response format for advanced tests."""
    summary: str
    key_points: List[str]
    sentiment: str
    word_count: int


class MathResponse(BaseModel):
    """Math problem response format."""
    result: int
    explanation: str


class TestCases:
    """Collection of test cases for benchmarking."""
    
    @staticmethod
    def get_simple_text() -> Dict[str, Any]:
        """
        Simple text prompt - minimum complexity.
        
        Returns:
            Test case dictionary with description and expected format
        """
        return {
            "name": "Simple Text Query",
            "description": "What is 2 + 2?",
            "response_format": str,
            "attachments": None,
            "context": None
        }
    
    @staticmethod
    def get_simple_structured() -> Dict[str, Any]:
        """
        Simple structured output - Pydantic model response.
        
        Returns:
            Test case dictionary with Pydantic response format
        """
        return {
            "name": "Simple Structured Output",
            "description": "What is the capital of France? Provide your answer with confidence level (0.0 to 1.0).",
            "response_format": SimpleResponse,
            "attachments": None,
            "context": None
        }
    
    @staticmethod
    def get_math_problem() -> Dict[str, Any]:
        """
        Math problem with explanation.
        
        Returns:
            Test case for math reasoning
        """
        return {
            "name": "Math Problem",
            "description": "Calculate 213123 * 24 and explain your reasoning step by step.",
            "response_format": MathResponse,
            "attachments": None,
            "context": None
        }
    
    @staticmethod
    def get_text_analysis() -> Dict[str, Any]:
        """
        Text analysis with complex output.
        
        Returns:
            Test case for text analysis
        """
        text = """
        Artificial intelligence is transforming how businesses operate. 
        Companies are investing heavily in AI technologies to improve 
        efficiency, reduce costs, and enhance customer experiences. 
        However, there are also concerns about job displacement and 
        ethical implications of AI decision-making.
        """
        
        return {
            "name": "Text Analysis",
            "description": f"Analyze the following text and provide a summary, key points, sentiment, and word count:\n\n{text}",
            "response_format": ComplexResponse,
            "attachments": None,
            "context": None
        }
    
    @staticmethod
    def get_context_query() -> Dict[str, Any]:
        """
        Query with context information.
        
        Returns:
            Test case with context
        """
        context_info = """
        Company Name: TechCorp
        Industry: Software Development
        Employees: 500
        Revenue: $50M annually
        Founded: 2015
        """
        
        return {
            "name": "Context-based Query",
            "description": "Based on the company information provided, what is their approximate revenue per employee?",
            "response_format": str,
            "attachments": None,
            "context": [context_info]
        }
    
    @staticmethod
    def get_all_test_cases() -> List[Dict[str, Any]]:
        """
        Get all available test cases.
        
        Returns:
            List of all test case dictionaries
        """
        return [
            TestCases.get_simple_text(),
            TestCases.get_simple_structured(),
            TestCases.get_math_problem(),
            TestCases.get_text_analysis(),
            TestCases.get_context_query()
        ]
    
    @staticmethod
    def get_test_case_by_name(name: str) -> Dict[str, Any]:
        """
        Get a specific test case by name.
        
        Args:
            name: Name of the test case
            
        Returns:
            Test case dictionary
            
        Raises:
            ValueError: If test case name not found
        """
        all_cases = TestCases.get_all_test_cases()
        for case in all_cases:
            if case["name"] == name:
                return case
        
        available_names = [case["name"] for case in all_cases]
        raise ValueError(
            f"Test case '{name}' not found. Available: {', '.join(available_names)}"
        )

