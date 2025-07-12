"""
Strategy for processing KnowledgeBase context items.
"""

from typing import Any, Optional

from ..knowledge_base.knowledge_base import KnowledgeBase
from .base_strategy import ContextProcessingStrategy


class KnowledgeBaseContextStrategy(ContextProcessingStrategy):
    """Strategy for processing KnowledgeBase context items."""

    def can_process(self, context_item: Any) -> bool:
        """Check if this strategy can process the given context item."""
        return isinstance(context_item, KnowledgeBase)

    def process(self, context_item: Any) -> str:
        """Process the knowledge base context item."""
        if not self.can_process(context_item):
            raise ValueError(
                f"KnowledgeBaseContextStrategy cannot process {type(context_item)}"
            )

        kb = context_item
        return f"Knowledge Base: {kb.markdown()}\n"

    def get_section_name(self) -> str:
        """Get the XML section name for knowledge bases."""
        return "Knowledge Base"

    def validate(self, context_item: Any) -> Optional[str]:
        """Validate the knowledge base context item."""
        if not isinstance(context_item, KnowledgeBase):
            return f"Expected KnowledgeBase, got {type(context_item)}"

        if not hasattr(context_item, "markdown"):
            return "KnowledgeBase missing markdown method"

        return None
