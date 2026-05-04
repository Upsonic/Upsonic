import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from upsonic.agent.agent import Agent
    from upsonic.models import Model
else:
    Agent = "Agent"

load_dotenv()


class LLMManager:
    def __init__(
        self,
        default_model: Optional[str],
        agent: "Agent",
        requested_model: Optional[str] = None,
    ) -> None:
        self.agent: "Agent" = agent
        self.default_model: Optional[str] = default_model
        self.requested_model: Optional[str] = requested_model
        self.selected_model: Optional["Model"] = None

    def _model_set(self, model: Optional[str]) -> "Model":
        from upsonic.models import infer_model

        resolved: str
        if model is None:
            raw_key: str | None = os.getenv("LLM_MODEL_KEY")
            resolved = raw_key.split(":")[0] if raw_key else "openai/gpt-4o"
        else:
            resolved = model
        return infer_model(resolved)
        
    def get_model(self):
        return self.selected_model
    
    async def aprepare(self) -> None:
        """Prepare the LLM selection before the task execution."""
        if self.requested_model is None:
            self.selected_model = self._model_set(self.default_model)
        else:
            self.selected_model = self._model_set(self.requested_model)
    
    async def afinalize(self) -> None:
        """Finalize LLM resources after the task execution."""
        if self.selected_model is None:
            return
        self.agent.model = self.selected_model
        self.agent._agent_run_output.model_name = self.selected_model.model_name if self.selected_model else None
        self.agent._agent_run_output.model_provider = self.selected_model.system if self.selected_model else None
        self.agent._agent_run_output.model_provider_profile = self.selected_model.profile if self.selected_model else None
    
    def prepare(self) -> None:
        """Synchronous version of aprepare."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.aprepare())
    
    def finalize(self) -> None:
        """Synchronous version of afinalize."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.afinalize())
    
    @asynccontextmanager
    async def manage_llm(self):
        """
        Async context manager for LLM lifecycle.
        
        Note: This context manager is kept for backward compatibility.
        For step-based architecture, use aprepare() and afinalize() directly.
        """
        await self.aprepare()
        
        try:
            yield self
        finally:
            await self.afinalize()