from pydantic import BaseModel
from typing import Any, Optional

from .base_strategy import ContextProcessingStrategy


class DefaultPrompt(BaseModel):
    prompt: str


class DefaultPromptContextStrategy(ContextProcessingStrategy):
    """Strategy for processing DefaultPrompt context items."""

    def can_process(self, context_item: Any) -> bool:
        """Check if this strategy can process the given context item."""
        return isinstance(context_item, DefaultPrompt)

    def process(self, context_item: Any) -> str:
        """Process the default prompt context item."""
        if not self.can_process(context_item):
            raise ValueError(
                f"DefaultPromptContextStrategy cannot process {type(context_item)}"
            )

        prompt = context_item
        return f"Default Prompt: {prompt.prompt}\n"

    def get_section_name(self) -> str:
        """Get the XML section name for default prompts."""
        return "Default Prompt"

    def validate(self, context_item: Any) -> Optional[str]:
        """Validate the default prompt context item."""
        if not isinstance(context_item, DefaultPrompt):
            return f"Expected DefaultPrompt, got {type(context_item)}"

        if not hasattr(context_item, "prompt"):
            return "DefaultPrompt missing prompt attribute"

        return None


def default_prompt():
    """Factory function to create default prompt instance."""
    return DefaultPrompt(
        prompt="""
You are a helpful assistant that can answer questions and help with tasks. 
Please be logical, concise, and to the point. 
Your provider is Upsonic. 
Think in your backend and dont waste time to write to the answer. Write only what the user want.
                         
About the context: If there is an Task context user want you to know that. Use it to think in your backend.
                         """
    )
