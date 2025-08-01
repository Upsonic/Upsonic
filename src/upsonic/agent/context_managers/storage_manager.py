from contextlib import asynccontextmanager
from pydantic_ai.messages import ModelMessagesTypeAdapter
from upsonic.storage.session.sessions import AgentSession
import time


class StorageManager:
    def __init__(self, agent, task):
        self.agent = agent
        self.task = task
        self.message_history = []
        self.current_session = None
        self.model_response = None
        
    def get_message_history(self):
        """Get the loaded and limited message history"""
        return self.message_history
        
    def process_response(self, model_response):
        """Store the model response for later saving"""
        self.model_response = model_response
        return self.model_response

    @asynccontextmanager
    async def manage_storage(self):
        """Context manager for handling storage operations"""
        # Load conversation history from storage
        if self.agent.storage:
            self.current_session = self.agent.storage.read(session_id=self.agent.session_id)

            if self.agent.add_history_to_messages and self.current_session and self.current_session.memory:
                try:
                    # Load conversation history from storage (currently stored as single entry with all messages)
                    for history in self.current_session.memory:
                        self.message_history = ModelMessagesTypeAdapter.validate_python(history)
                    
                    # Apply num_history_runs limiting at the message level
                    self.message_history = self.agent._limit_message_history(self.message_history)
                    
                except Exception as e:
                    print(f"Warning: Could not validate stored history. Starting fresh. Error: {e}")
                    self.message_history = []

        try:
            yield self
        finally:
            # Save results back to storage
            if self.agent.storage and self.model_response:
                self._save_to_storage()

    def _save_to_storage(self):
        """Save the conversation results to storage"""
        updated_session_data = {}
        
        if self.current_session:
            updated_session_data = self.current_session.model_dump()
        else:
            updated_session_data['memory'] = []
            updated_session_data['extra_data'] = {}

        updated_session_data.setdefault('memory', [])
        updated_session_data.update({
            "session_id": self.agent.session_id,
            "agent_id": self.agent.agent_id,
            "user_id": self.agent.get_agent_id(),
            "updated_at": time.time(),
        })

        if self.model_response:
            from pydantic_core import to_jsonable_python
            # Save the full conversation history to storage
            all_messages_as_dicts = to_jsonable_python(self.model_response.all_messages())
            updated_session_data['memory'] = [all_messages_as_dicts]

        final_session = AgentSession.model_validate(updated_session_data)
        self.agent.storage.upsert(final_session) 