"""
Agent module for the Upsonic AI Agent Framework.

This module provides agent classes for executing tasks and managing AI interactions.
"""

from .agent import Agent  # Keep backward compatibility
from .base import BaseAgent
from .run_result import AgentRunResult, OutputDataT

__all__ = [
    'Agent',     # New primary agent class
    'BaseAgent', # Base class
    'AgentRunResult', # Result wrapper with message tracking
    'OutputDataT', # Type variable for output data
]
