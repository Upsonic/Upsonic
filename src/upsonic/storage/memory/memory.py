import asyncio
from typing import Any, Dict, List, Optional
import json

from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic import BaseModel, Field

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import InteractionSession, UserProfile
from upsonic.storage.types import SessionId, UserId
from upsonic.models.model_registry import ModelNames
from upsonic.schemas import UserTraits


class Memory:
    """
    A comprehensive, configurable memory orchestrator for an AI agent.

    This class serves as a centralized module for managing different types of
    memory and respects the specific data formats and logic established in
    the original application design for handling chat history.
    """

    def __init__(
        self,
        storage: Storage,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        num_last_messages: Optional[int] = None,
        model: ModelNames | None = None,
        debug: bool = False
    ):
        self.storage = storage
        self.num_last_messages = num_last_messages
        self.full_session_memory_enabled = full_session_memory
        self.summary_memory_enabled = summary_memory
        self.user_analysis_memory_enabled = user_analysis_memory
        self.model = model
        self.debug = debug

        if self.full_session_memory_enabled or self.summary_memory_enabled:
            if not session_id:
                raise ValueError("`session_id` is required when full_session_memory or summary_memory is enabled.")
            self.session_id: Optional[SessionId] = SessionId(session_id)
        else:
            self.session_id = None
        if self.user_analysis_memory_enabled:
            if not user_id:
                raise ValueError("`user_id` is required when user_analysis_memory is enabled.")
            self.user_id: Optional[UserId] = UserId(user_id)
        elif user_id:
            self.user_id: Optional[UserId] = UserId(user_id)
        else:
            self.user_id = None

    async def prepare_inputs_for_task(self) -> Dict[str, Any]:
        """
        Gathers all relevant memory data before a task execution, correctly
        parsing and limiting the chat history.
        """
        prepared_data = {
            "message_history": [],
            "context_injection": "",
            "system_prompt_injection": ""
        }

        if self.user_analysis_memory_enabled and self.user_id:
            profile = await self.storage.read_async(self.user_id, UserProfile)
            if profile and profile.profile_data:
                profile_str = "\n".join(f"- {key}: {value}" for key, value in profile.profile_data.items())
                prepared_data["system_prompt_injection"] = f"<UserProfile>\n{profile_str}\n</UserProfile>"

        if self.session_id:
            session = await self.storage.read_async(self.session_id, InteractionSession)
            if session:
                if self.summary_memory_enabled and session.summary:
                    prepared_data["context_injection"] = f"<SessionSummary>\n{session.summary}\n</SessionSummary>"

                if self.full_session_memory_enabled and session.chat_history:
                    try:
                        raw_messages = session.chat_history[0] if session.chat_history else []
                        validated_history = ModelMessagesTypeAdapter.validate_python(raw_messages)
                        limited_history = self._limit_message_history(validated_history)
                        prepared_data["message_history"] = limited_history
                        
                    except Exception as e:
                        print(f"Warning: Could not validate or process stored chat history. Starting fresh. Error: {e}")
                        prepared_data["message_history"] = []
        return prepared_data

    async def update_memories_after_task(self, model_response) -> None:
        """
        Updates all relevant memories after a task has been completed, saving
        the chat history in the correct format.
        """
        await asyncio.gather(
            self._update_interaction_session(model_response),
            self._update_user_profile(model_response)
        )

    async def _update_interaction_session(self, model_response):
        """Helper to handle updating the InteractionSession object."""
        if not (self.full_session_memory_enabled or self.summary_memory_enabled) or not self.session_id:
            return

        session = await self.storage.read_async(self.session_id, InteractionSession)
        if not session:
            session = InteractionSession(session_id=self.session_id, user_id=self.user_id)
        
        if self.full_session_memory_enabled:
            from pydantic_core import to_jsonable_python
            all_messages = model_response.all_messages()
            all_messages_as_dicts = to_jsonable_python(all_messages)
            session.chat_history = [all_messages_as_dicts]

        if self.summary_memory_enabled:
            try:
                session.summary = await self._generate_new_summary(session.summary, model_response)
            except Exception as e:
                print(f"Warning: Failed to generate session summary: {e}")
        
        await self.storage.upsert_async(session)

    async def _update_user_profile(self, model_response):
        """Helper to handle updating the UserProfile object."""
        if not self.user_analysis_memory_enabled or not self.user_id:
            return
        
        profile = await self.storage.read_async(self.user_id, UserProfile)

        if not profile:
            profile = UserProfile(user_id=self.user_id)

        if self.user_analysis_memory_enabled:
            try:
                updated_traits = await self._analyze_interaction_for_traits(profile.profile_data, model_response)
                profile.profile_data.update(updated_traits)
            except Exception as e:
                print(f"Warning: Failed to analyze user profile: {e}")

        await self.storage.upsert_async(profile)


    def _limit_message_history(self, message_history: list) -> list:
        """
        Limit conversation history based on num_last_messages parameter.
        
        Args:
            message_history: List of messages from storage
            
        Returns:
            Limited message history with system prompt + last N conversation runs
        """
        if not self.num_last_messages or self.num_last_messages <= 0 or len(message_history) <= 1:
            return message_history
        
        # Separate system prompt (always first) from conversation messages
        system_message = message_history[0] if len(message_history) > 0 else None
        conversation_messages = message_history[1:] if len(message_history) > 1 else []
        
        # Group conversation messages into runs (request-response pairs)
        conversation_runs = []
        current_run = []
        
        for msg in conversation_messages:
            current_run.append(msg)
            if len(current_run) == 2:  # Complete run (request + response)
                conversation_runs.append(current_run)
                current_run = []
        
        # Handle incomplete run
        if current_run:
            conversation_runs.append(current_run)
        
        # Keep only the last num_last_messages
        if len(conversation_runs) > self.num_last_messages:
            kept_runs = conversation_runs[-self.num_last_messages:]
        else:
            kept_runs = conversation_runs
        
        # Flatten kept runs back to message list
        limited_conversation = []
        for run in kept_runs:
            limited_conversation.extend(run)
        
        # Rebuild with system message + limited conversation
        if system_message:
            return [system_message] + limited_conversation
        else:
            return limited_conversation
        

    async def _generate_new_summary(self, previous_summary: str, model_response) -> str:
        from upsonic.agent.agent import Direct
        from upsonic.tasks.tasks import Task
        from pydantic_core import to_jsonable_python

        last_turn = to_jsonable_python(model_response.all_messages()) # or new_message()
        
        summarizer = Direct("Summarizer", model=self.model, debug=self.debug)
        
        prompt = f"""
        Previous Summary: "{previous_summary or 'None'}"
        New Conversation Turn: {json.dumps(last_turn, indent=2)}
        Update the summary concisely based on the new turn.
        """
        task = Task(description=prompt, response_format=str)
        
        summary_response = await summarizer.do_async(task)
        return str(summary_response)

    async def _analyze_interaction_for_traits(self, current_profile: dict, model_response) -> dict:
        from upsonic.agent.agent import Direct
        from upsonic.tasks.tasks import Task

        last_user_message = model_response.all_messages()
        
        analyzer = Direct("User Trait Analyzer", model=self.model, debug=self.debug)
        
        prompt = f"""
        Analyze the user's last messages to update their profile.
        Current Profile: {json.dumps(current_profile, indent=2)}
        User's Message: "{last_user_message}"
        Extract their expertise, preferred tone, and interests. Only provide new or updated information.
        """
        task = Task(description=prompt, response_format=UserTraits)
        
        trait_response = await analyzer.do_async(task)
        if trait_response:
            result_dict = trait_response.model_dump(exclude_unset=True)
            return result_dict
        return {}