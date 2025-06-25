# context/registry.py
from .strategies import (
    TaskStrategy,
    AgentStrategy,
    DefaultPromptStrategy,
    KnowledgeBaseStrategy,
)
REGISTRY = [
    TaskStrategy(),
    AgentStrategy(),
    DefaultPromptStrategy(),
    KnowledgeBaseStrategy(),
]
