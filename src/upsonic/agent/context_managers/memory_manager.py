from contextlib import asynccontextmanager
from upsonic.memory.memory import get_agent_memory, save_agent_memory


class MemoryManager:
    def __init__(self, agent, task):
        self.agent = agent
        self.task = task
        self.historical_messages = []
        self.historical_message_count = 0
        self.model_response = None
        
    def process_response(self, model_response):
        self.model_response = model_response
        return self.model_response

    def save_memory(self, answer):
        save_agent_memory(self.agent, answer)

    @asynccontextmanager
    async def manage_memory(self):
        self.historical_messages = get_agent_memory(self.agent) if self.agent.memory else []
        self.historical_message_count = len(self.historical_messages)

        try:
            yield self
        finally:
            # Automatically save memory if a response was captured and memory is enabled
            if self.agent.memory and self.model_response is not None:
                save_agent_memory(self.agent, self.model_response) 