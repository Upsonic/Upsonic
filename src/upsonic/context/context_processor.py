"""
Main context processor that uses strategy pattern for processing different context types.
"""

from typing import List, Any, Dict, Optional
from collections import defaultdict

from .base_strategy import ContextProcessingStrategy
from .task import TaskContextStrategy
from .agent import AgentContextStrategy
from .default_prompt import DefaultPromptContextStrategy, default_prompt
from .knowledge_base_strategy import KnowledgeBaseContextStrategy
from .exceptions import (
    ContextProcessingError,
    ContextValidationError,
    ContextStrategyError,
)


class ContextProcessor:
    """Main context processor that uses strategy pattern."""

    def __init__(self):
        """Initialize the context processor with default strategies."""
        self.strategies: List[ContextProcessingStrategy] = [
            TaskContextStrategy(),
            AgentContextStrategy(),
            DefaultPromptContextStrategy(),
            KnowledgeBaseContextStrategy(),
        ]

    def add_strategy(self, strategy: ContextProcessingStrategy) -> None:
        """
        Add a new processing strategy.

        Args:
            strategy: The strategy to add
        """
        self.strategies.append(strategy)

    def remove_strategy(self, strategy_type: type) -> bool:
        """
        Remove a strategy by type.

        Args:
            strategy_type: The type of strategy to remove

        Returns:
            True if strategy was removed, False if not found
        """
        for i, strategy in enumerate(self.strategies):
            if isinstance(strategy, strategy_type):
                del self.strategies[i]
                return True
        return False

    def process_context(self, context: Any, strict: bool = False) -> str:
        """
        Process context items and return formatted XML string.

        Args:
            context: Context items to process
            strict: If True, raise exceptions on errors instead of collecting them

        Returns:
            Formatted XML string with all context sections

        Raises:
            ContextProcessingError: If strict=True and processing fails
        """
        if context is None:
            context = []

        # Always add default prompt
        try:
            context.append(default_prompt())
        except Exception as e:
            error_msg = f"Failed to add default prompt: {str(e)}"
            if strict:
                raise ContextProcessingError(error_msg) from e

        # Group processed items by section
        sections: Dict[str, List[str]] = defaultdict(list)
        errors: List[str] = []

        for item in context:
            processed = False

            for strategy in self.strategies:
                if strategy.can_process(item):
                    # Validate before processing
                    validation_error = strategy.validate(item)
                    if validation_error:
                        error_msg = f"Validation error: {validation_error}"
                        if strict:
                            raise ContextValidationError(error_msg)
                        errors.append(error_msg)
                        continue

                    try:
                        processed_content = strategy.process(item)
                        section_name = strategy.get_section_name()
                        sections[section_name].append(processed_content)
                        processed = True
                        break
                    except Exception as e:
                        error_msg = (
                            f"Processing error for {type(item).__name__}: {str(e)}"
                        )
                        if strict:
                            raise ContextProcessingError(error_msg) from e
                        errors.append(error_msg)

            if not processed:
                error_msg = f"No strategy found for {type(item).__name__}"
                if strict:
                    raise ContextStrategyError(error_msg)
                errors.append(error_msg)

        # Build the final XML structure
        return self._build_xml_output(sections, errors)

    def _build_xml_output(
        self, sections: Dict[str, List[str]], errors: List[str]
    ) -> str:
        """
        Build the final XML output from processed sections.

        Args:
            sections: Dictionary of section names to content lists
            errors: List of processing errors

        Returns:
            Formatted XML string
        """
        total_context = "<Context>"

        # Add each section
        for section_name, content_list in sections.items():
            if content_list:  # Only add non-empty sections
                section_content = f"<{section_name}>"
                for content in content_list:
                    section_content += content
                section_content += f"</{section_name}>"
                total_context += section_content

        # Add errors as comments if any (for debugging)
        if errors:
            total_context += "\n<!-- Processing Errors:\n"
            for error in errors:
                total_context += f"  - {error}\n"
            total_context += "-->"

        total_context += "</Context>"

        return total_context

    def get_available_strategies(self) -> List[str]:
        """
        Get list of available strategy types.

        Returns:
            List of strategy class names
        """
        return [strategy.__class__.__name__ for strategy in self.strategies]
