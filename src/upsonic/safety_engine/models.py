"""
Data models for AI Safety Engine
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class PolicyInput(BaseModel):
    """Input data for policy processing"""
    input_texts: Optional[List[str]] = None
    input_images: Optional[List[str]] = None
    input_videos: Optional[List[str]] = None
    input_audio: Optional[List[str]] = None
    input_files: Optional[List[str]] = None
    extra_data: Optional[Dict[str, Any]] = None



class RuleOutput(BaseModel):
    """Result from rule processing"""
    confidence: float
    content_type: str
    details: str
    triggered_keywords: Optional[List[str]] = None



class PolicyOutput(BaseModel):
    """Result from policy execution"""
    output_texts: Optional[List[str]] = None
    output_images: Optional[List[str]] = None
    output_videos: Optional[List[str]] = None
    output_audio: Optional[List[str]] = None
    output_files: Optional[List[str]] = None
    action_output: Optional[Dict[str, Any]] = None
    transformation_map: Optional[Dict[int, Dict[str, str]]] = None


# Backward compatibility aliases
RuleInput = PolicyInput
ActionResult = PolicyOutput


ActionOutput = PolicyOutput