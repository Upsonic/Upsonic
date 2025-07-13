from pydantic import BaseModel
from .context import ContextSections


class DefaultPrompt(BaseModel):
    prompt: str

    def to_context_string(self) -> str:
        return f"Default Prompt: {self.prompt}\n"

    def get_context_section(self) -> ContextSections:
        return ContextSections.DEFAULT_PROMPT


def default_prompt():
    return DefaultPrompt(prompt="""
You are a helpful assistant that can answer questions and help with tasks. 
Please be logical, concise, and to the point. 
Your provider is Upsonic. 
Think in your backend and dont waste time to write to the answer. Write only what the user want.
                         
About the context: If there is an Task context user want you to know that. Use it to think in your backend.
                         """)