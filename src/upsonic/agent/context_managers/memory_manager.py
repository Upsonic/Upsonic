from contextlib import asynccontextmanager
from upsonic.memory.memory import get_agent_memory, save_agent_memory


class MemoryHandler:
    def __init__(self, historical_messages, historical_message_count, agent):
        self.historical_messages = historical_messages
        self.historical_message_count = historical_message_count
        self.agent = agent
        self.model_response = None
        
    def process_response(self, model_response):
        self.model_response = model_response
        return self.model_response


class MemoryManager:
    def __init__(self, agent, task):
        self.agent = agent
        self.task = task

    def save_memory(self, answer):
        save_agent_memory(self.agent, answer)

    @asynccontextmanager
    async def manage_memory(self):
        historical_messages = get_agent_memory(self.agent) if self.agent.memory else []
        historical_message_count = len(historical_messages)
        
        memory_handler = MemoryHandler(historical_messages, historical_message_count, self.agent)

        try:
            yield memory_handler
        finally:
            # Automatically save memory if a response was captured and memory is enabled
            if self.agent.memory and memory_handler.model_response is not None:
                save_agent_memory(self.agent, memory_handler.model_response) 