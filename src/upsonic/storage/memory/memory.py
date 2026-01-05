import asyncio
import uuid
from typing import Any, Dict, List, Optional, Type, Literal, Union
import json
import copy
import time

from upsonic.messages.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, SystemPromptPart, UserPromptPart
from pydantic import BaseModel, Field, create_model

from upsonic.storage.base import Storage
from upsonic.session.agent import AgentSession
from upsonic.schemas import UserTraits
from upsonic.models import Model
from upsonic.utils.printing import info_log


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
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        
        self.storage = storage
        self.num_last_messages = num_last_messages
        self.full_session_memory_enabled = full_session_memory
        self.summary_memory_enabled = summary_memory
        self.user_analysis_memory_enabled = user_analysis_memory
        self.model = model
        self.debug = debug
        self.debug_level = debug_level if debug else 1
        self.feed_tool_call_results = feed_tool_call_results

        self.profile_schema_model = user_profile_schema or UserTraits
        self.is_profile_dynamic = dynamic_user_profile
        self.user_memory_mode = user_memory_mode

        if self.is_profile_dynamic:
            if user_profile_schema:
                from upsonic.utils.printing import warning_log
                warning_log("`dynamic_user_profile` is True, so the provided `user_profile_schema` will be ignored.", "MemoryStorage")
            self.profile_schema_model = None
        else:
            self.profile_schema_model = user_profile_schema or UserTraits        

        # Auto-generate session_id if not provided
        if session_id:
            self.session_id: str = session_id
        else:
            self.session_id: str = str(uuid.uuid4())
            if self.debug:
                info_log(f"Auto-generated session_id: {self.session_id}", "Memory")
        
        # Auto-generate user_id if not provided  
        if user_id:
            self.user_id: str = user_id
        else:
            self.user_id: str = str(uuid.uuid4())
            if self.debug:
                info_log(f"Auto-generated user_id: {self.user_id}", "Memory")
        
        if self.debug:
            info_log(f"Memory initialized with configuration:", "Memory")
            info_log(f"  - Full Session Memory: {self.full_session_memory_enabled}", "Memory")
            info_log(f"  - Summary Memory: {self.summary_memory_enabled}", "Memory")
            info_log(f"  - User Analysis Memory: {self.user_analysis_memory_enabled}", "Memory")
            info_log(f"  - Session ID: {self.session_id}", "Memory")
            info_log(f"  - User ID: {self.user_id}", "Memory")
            info_log(f"  - Max Messages: {self.num_last_messages}", "Memory")
            info_log(f"  - Feed Tool Results: {self.feed_tool_call_results}", "Memory")
            info_log(f"  - User Memory Mode: {self.user_memory_mode}", "Memory")
            info_log(f"  - Dynamic Profile: {self.is_profile_dynamic}", "Memory")
            info_log(f"  - Model: {self.model}", "Memory")

    def _format_profile_data(self, profile_data: Dict[str, Any]) -> Optional[str]:
        """
        Formats user profile data into a readable string format.
        """
        if not profile_data:
            return None
        
        profile_items = []
        for key, value in profile_data.items():
            if value is None:
                continue
            
            if isinstance(value, (list, tuple)):
                if len(value) > 0:
                    value_str = ", ".join(str(item) for item in value)
                else:
                    continue
            elif isinstance(value, dict):
                if len(value) > 0:
                    value_str = json.dumps(value, ensure_ascii=False)
                else:
                    continue
            elif value == "" or (isinstance(value, str) and value.strip() == ""):
                continue
            else:
                value_str = str(value)
            
            profile_items.append(f"- {key}: {value_str}")
        
        if profile_items:
            return "\n".join(profile_items)
        return None

    async def prepare_inputs_for_task(
        self, 
        agent_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Gathers all relevant memory data before a task execution, correctly
        parsing and limiting the chat history.
        
        Args:
            agent_metadata: Optional metadata from the Agent to inject into prompts.
        """
        if self.debug:
            info_log("Preparing memory inputs for task...", "Memory")
        
        prepared_data = {
            "message_history": [],
            "context_injection": "",
            "system_prompt_injection": "",
            "metadata_injection": ""
        }

        # Load AgentSession
        session = await self.storage.read_async(self.session_id, AgentSession)
        if session:
            # User profile
            if self.user_analysis_memory_enabled and session.user_profile:
                if self.debug:
                    info_log(f"Profile found in session: keys={list(session.user_profile.keys())}", "Memory")
                profile_str = self._format_profile_data(session.user_profile)
                if profile_str:
                    prepared_data["system_prompt_injection"] = f"<UserProfile>\n{profile_str}\n</UserProfile>"
                    if self.debug:
                        info_log(f"Loaded user profile with {len(session.user_profile)} traits", "Memory")
            
            # Session summary
            if self.summary_memory_enabled and session.summary:
                prepared_data["context_injection"] = f"<SessionSummary>\n{session.summary}\n</SessionSummary>"
                if self.debug:
                    info_log(f"Loaded session summary ({len(session.summary)} chars)", "Memory")
            
            # Chat history from flattened messages
            if self.full_session_memory_enabled:
                try:
                    if session.messages:
                        # Use flattened view, apply num_last_messages limit
                        if self.num_last_messages:
                            messages = session.messages[-self.num_last_messages:]
                        else:
                            messages = session.messages
                        
                        # Filter tool messages if needed
                        if not self.feed_tool_call_results:
                            from upsonic.messages import ModelRequest, ModelResponse
                            filtered = []
                            tool_messages_removed = 0
                            for msg in messages:
                                should_filter = False
                                
                                # Filter out tool-return messages (user-side)
                                if isinstance(msg, ModelRequest):
                                    if hasattr(msg, 'parts'):
                                        for part in msg.parts:
                                            if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                                                should_filter = True
                                                tool_messages_removed += 1
                                                break
                                
                                # Filter out assistant messages with tool_calls (to avoid incomplete sequences)
                                elif isinstance(msg, ModelResponse):
                                    if hasattr(msg, 'parts'):
                                        for part in msg.parts:
                                            if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                                                should_filter = True
                                                tool_messages_removed += 1
                                                break
                                
                                if not should_filter:
                                    filtered.append(msg)
                            messages = filtered
                            if self.debug:
                                info_log(f"Filtered out {tool_messages_removed} tool-related messages (feed_tool_call_results=False)", "Memory")
                        elif self.debug:
                            # Count tool messages when not filtering (for debugging)
                            from upsonic.messages import ModelRequest
                            tool_count = 0
                            for msg in messages:
                                if isinstance(msg, ModelRequest) and hasattr(msg, 'parts'):
                                    for part in msg.parts:
                                        if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                                            tool_count += 1
                                            break
                            if tool_count > 0:
                                info_log(f"Including {tool_count} tool messages in history (feed_tool_call_results=True)", "Memory")
                        
                        if self.debug:
                            info_log(f"Loaded {len(messages)} messages from session.messages", "Memory")
                        prepared_data["message_history"] = messages
                    else:
                        prepared_data["message_history"] = []
                except Exception as e:
                    from upsonic.utils.printing import warning_log
                    warning_log(f"Could not load messages from session. Starting fresh. Error: {e}", "MemoryStorage")
                    prepared_data["message_history"] = []
            # Session metadata
            if session.metadata:
                metadata_parts = []
                for key, value in session.metadata.items():
                    metadata_parts.append(f"  {key}: {value}")
                if metadata_parts:
                    prepared_data["metadata_injection"] = (
                        "<SessionMetadata>\n" + "\n".join(metadata_parts) + "\n</SessionMetadata>"
                    )
                    if self.debug:
                        info_log(f"Loaded session metadata with {len(session.metadata)} keys", "Memory")
        elif self.debug:
            info_log("No session found in storage", "Memory")
        
        # Load user profile by user_id if not already loaded from current session
        if (self.user_analysis_memory_enabled 
            and self.user_id 
            and not prepared_data.get("system_prompt_injection")):
            try:
                user_sessions = await self.storage.list_agent_sessions_async(user_id=self.user_id)
                if user_sessions:
                    user_sessions_sorted = sorted(
                        user_sessions,
                        key=lambda s: s.updated_at or 0,
                        reverse=True
                    )
                    for user_session in user_sessions_sorted:
                        if user_session.user_profile:
                            if self.debug:
                                info_log(f"Found user profile from session '{user_session.session_id}' for user_id='{self.user_id}'", "Memory")
                            profile_str = self._format_profile_data(user_session.user_profile)
                            if profile_str:
                                prepared_data["system_prompt_injection"] = f"<UserProfile>\n{profile_str}\n</UserProfile>"
                                if self.debug:
                                    info_log(f"Loaded user profile with {len(user_session.user_profile)} traits from previous session", "Memory")
                            break
                    else:
                        if self.debug:
                            info_log(f"No user profile found in {len(user_sessions)} sessions for user_id='{self.user_id}'", "Memory")
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Failed to load user profile by user_id: {e}", "Memory")
        
        # Merge agent metadata with session metadata
        if agent_metadata:
            agent_meta_parts = []
            for key, value in agent_metadata.items():
                agent_meta_parts.append(f"  {key}: {value}")
            if agent_meta_parts:
                agent_meta_str = "<AgentMetadata>\n" + "\n".join(agent_meta_parts) + "\n</AgentMetadata>"
                if prepared_data["metadata_injection"]:
                    prepared_data["metadata_injection"] = agent_meta_str + "\n\n" + prepared_data["metadata_injection"]
                else:
                    prepared_data["metadata_injection"] = agent_meta_str
                if self.debug:
                    info_log(f"Added agent metadata with {len(agent_metadata)} keys", "Memory")
        
        if self.debug:
            info_log(f"Prepared memory inputs: {len(prepared_data['message_history'])} messages, "
                    f"summary={bool(prepared_data['context_injection'])}, "
                    f"profile={bool(prepared_data['system_prompt_injection'])}, "
                    f"metadata={bool(prepared_data['metadata_injection'])}", "Memory")
            
            if self.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                message_preview = []
                for msg in prepared_data['message_history'][-3:]:
                    if hasattr(msg, 'parts'):
                        msg_str = str([str(p)[:100] for p in msg.parts[:2]])[:200]
                        message_preview.append(msg_str)
                
                debug_log_level2(
                    "Memory inputs prepared",
                    "Memory",
                    debug=self.debug,
                    debug_level=self.debug_level,
                    message_count=len(prepared_data['message_history']),
                    message_preview=message_preview,
                    has_summary=bool(prepared_data['context_injection']),
                    summary_length=len(prepared_data['context_injection']) if prepared_data['context_injection'] else 0,
                    has_profile=bool(prepared_data['system_prompt_injection']),
                    profile_length=len(prepared_data['system_prompt_injection']) if prepared_data['system_prompt_injection'] else 0,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    full_session_memory=self.full_session_memory_enabled,
                    summary_memory=self.summary_memory_enabled,
                    user_analysis_memory=self.user_analysis_memory_enabled
                )
        
        return prepared_data

    async def update_memories_after_task(self, model_response, agent_run_output=None) -> None:
        """
        Updates all relevant memories after a task has been completed, saving
        the run output to the AgentSession.
        
        This method ALWAYS upserts the AgentSession to storage, regardless of
        full_session_memory setting. The full_session_memory only controls
        whether chat history is injected into the prompt.
        """
        await self._update_agent_session(model_response, agent_run_output)

    async def _update_agent_session(self, model_response, agent_run_output):
        """Helper to handle updating the AgentSession object."""
        from upsonic.run.agent.output import AgentRunOutput as RunOutputType
        
        if self.debug:
            info_log("Updating agent session...", "Memory")
        
        session = await self.storage.read_async(self.session_id, AgentSession)
        if not session:
            session = AgentSession(
                session_id=self.session_id,
                user_id=self.user_id,
                agent_id=agent_run_output.agent_id if agent_run_output else None,
                created_at=int(time.time()),
                metadata={},
                runs={},
            )
            if self.debug:
                info_log(f"Created new session: {self.session_id}", "Memory")
        
        # Create a run output if not provided
        run_to_add = agent_run_output
        if not run_to_add and model_response:
            # Try to create a minimal run output from model_response
            try:
                messages = None
                if hasattr(model_response, 'all_messages'):
                    messages = model_response.all_messages()
                elif hasattr(model_response, 'new_messages'):
                    messages = model_response.new_messages()
                
                if messages:
                    run_to_add = RunOutputType(
                        run_id=str(uuid.uuid4()),
                        session_id=self.session_id,
                        user_id=self.user_id,
                        messages=messages,
                        content=str(model_response.output) if hasattr(model_response, 'output') else None,
                    )
                    if self.debug:
                        info_log(f"Created run output from model_response with {len(messages)} messages", "Memory")
            except Exception as e:
                if self.debug:
                    info_log(f"Could not create run output from model_response: {e}", "Memory")
        
        # Always add the run output to the session if available
        if run_to_add:
            session.upsert_run(run_to_add)
            if self.debug:
                info_log(f"Added run output to session (total runs: {len(session.runs or [])})", "Memory")
            
            # Populate session_data and agent_data from run output
            session.populate_from_run_output(run_to_add)

        # Generate summary if enabled - use agent_run_output which has new_messages() method
        if self.summary_memory_enabled:
            if not self.model:
                from upsonic.utils.printing import warning_log
                warning_log("Summary memory is enabled but no model is configured. Skipping summary generation.", "MemoryStorage")
            else:
                try:
                    if self.debug:
                        info_log("Generating new session summary...", "Memory")
                    session.summary = await self._generate_new_summary(session, agent_run_output)
                    if self.debug:
                        info_log(f"Summary generated ({len(session.summary) if session.summary else 0} chars)", "Memory")
                except Exception as e:
                    from upsonic.utils.printing import warning_log
                    warning_log(f"Failed to generate session summary: {e}", "MemoryStorage")
        
        # Flatten messages if full_session_memory enabled
        if self.full_session_memory_enabled:
            session.messages = session.flatten_messages_from_runs()
            if self.debug:
                info_log(f"Flattened {len(session.messages)} messages from {len(session.runs or [])} runs", "Memory")
        
        # Update user profile if enabled - use agent_run_output which has new_messages() method
        if self.user_analysis_memory_enabled:
            if not self.model:
                from upsonic.utils.printing import warning_log
                warning_log("User analysis memory is enabled but no model is configured. Skipping user profile analysis.", "MemoryStorage")
            else:
                try:
                    updated_traits = await self._analyze_interaction_for_traits(session, agent_run_output)
                    
                    if self.debug:
                        info_log(f"Extracted traits: {updated_traits}", "Memory")
                    
                    if self.user_memory_mode == 'replace':
                        session.user_profile = updated_traits
                        if self.debug:
                            info_log(f"Replaced user profile with {len(updated_traits)} traits", "Memory")
                    elif self.user_memory_mode == 'update':
                        if not session.user_profile:
                            session.user_profile = {}
                        before_count = len(session.user_profile)
                        session.user_profile.update(updated_traits)
                        if self.debug:
                            info_log(f"Updated user profile: {before_count} -> {len(session.user_profile)} traits", "Memory")
                except Exception as e:
                    from upsonic.utils.printing import warning_log
                    warning_log(f"Failed to analyze user profile: {e}", "MemoryStorage")
        
        # Always upsert the session to storage
        session.updated_at = int(time.time())
        await self.storage.upsert_async(session)
        if self.debug:
            info_log("Agent session saved to storage", "Memory")


    def _limit_message_history(self, message_history: List) -> List:
        """
        Limits conversation history to the last N runs.
        """
        if not self.num_last_messages or self.num_last_messages <= 0:
            return message_history

        if not message_history:
            return []

        all_runs = []
        for i in range(0, len(message_history) - 1, 2):
            request = message_history[i]
            response = message_history[i+1]
            if isinstance(request, ModelRequest) and isinstance(response, ModelResponse):
                all_runs.append((request, response))

        if len(all_runs) <= self.num_last_messages:
            if self.debug:
                info_log(f"History has {len(all_runs)} runs, within limit of {self.num_last_messages}. No limiting needed.", "Memory")
            return message_history

        kept_runs = all_runs[-self.num_last_messages:]
        
        if self.debug:
            info_log(f"Limiting history from {len(all_runs)} runs to last {self.num_last_messages} runs", "Memory")
        
        if not kept_runs:
            return []

        original_system_prompt = None
        if message_history:
            for part in message_history[0].parts:
                if isinstance(part, SystemPromptPart):
                    original_system_prompt = part
                    break
        
        if not original_system_prompt:
            from upsonic.utils.printing import warning_log
            warning_log("Could not find original SystemPromptPart. History might be malformed.", "MemoryStorage")
            if self.debug:
                info_log("Warning: No system prompt found, returning limited runs without modification", "Memory")
            return [message for run in kept_runs for message in run]

        first_request_in_window = kept_runs[0][0]

        new_user_prompt = None
        for part in first_request_in_window.parts:
            if isinstance(part, UserPromptPart):
                new_user_prompt = part
                break
                
        if not new_user_prompt:
            from upsonic.utils.printing import warning_log
            warning_log("Could not find UserPromptPart in the first message of the limited window.", "MemoryStorage")
            return [message for run in kept_runs for message in run]

        modified_first_request = copy.deepcopy(first_request_in_window)
        modified_first_request.parts = [original_system_prompt, new_user_prompt]
        
        final_history = []
        final_history.append(modified_first_request)
        final_history.append(kept_runs[0][1])
        
        for run in kept_runs[1:]:
            final_history.extend(run)
            
        info_log(f"Original history had {len(all_runs)} runs. "
                f"Limited to the last {self.num_last_messages}, resulting in {len(final_history)} messages.", 
                context="Memory")

        return final_history
        

    async def _generate_new_summary(self, session: AgentSession, agent_run_output) -> str:
        from upsonic.agent.agent import Agent
        from upsonic.tasks.tasks import Task

        if not self.model:
            raise ValueError("A model must be configured on the Memory object to generate session summaries.")

        if self.debug:
            info_log("Starting summary generation...", "Memory")
        
        # Use agent_run_output which has new_messages() method
        last_turn = []
        if agent_run_output is not None and hasattr(agent_run_output, 'new_messages'):
            try:
                new_msgs = agent_run_output.new_messages()
                if new_msgs:
                    last_turn = ModelMessagesTypeAdapter.dump_python(new_msgs, mode='json')
            except Exception as e:
                if self.debug:
                    info_log(f"Could not get new_messages from agent_run_output: {e}", "Memory")
        
        if self.debug:
            info_log(f"Previous summary length: {len(session.summary) if session.summary else 0} chars", "Memory")
            info_log(f"New turn messages: {len(last_turn)} messages", "Memory")
            info_log(f"Total session runs: {len(session.runs) if session.runs else 0}", "Memory")
        
        # Get recent messages for context - from current session_id (all runs in this session)
        # Use AgentSession helper method to get all messages for this session_id
        recent_messages = await AgentSession.get_all_messages_for_session_id_async(
            storage=self.storage,
            session_id=session.session_id
        )
        
        recent_messages_str = json.dumps([str(m) for m in recent_messages], indent=2) if recent_messages else 'None'
        
        if self.debug:
            info_log(f"Recent messages for summary context: {len(recent_messages)} messages (from session_id={session.session_id}, all runs)", "Memory")
        
        # If we have no new turn and no recent messages, skip summary generation
        if not last_turn and not recent_messages:
            if self.debug:
                info_log("No messages available for summary generation, skipping", "Memory")
            return session.summary or ""
        
        summarizer = Agent(name="Summarizer", model=self.model, debug=self.debug)
        
        previous_summary_str = session.summary if session.summary else 'None (this is the first interaction)'
        
        # Build prompt based on available data
        new_turn_str = json.dumps(last_turn, indent=2) if last_turn else 'None (using chat history only)'
        
        prompt = f"""Update the conversation summary based on the new interaction.

Previous Summary: {previous_summary_str}

New Conversation Turn:
{new_turn_str}

Recent Chat History:
{recent_messages_str}

YOUR TASK: Create a concise summary that captures the key points of the entire conversation, including the new turn. Focus on important information, user preferences, and topics discussed.
"""
        task = Task(description=prompt, response_format=str)
        
        summary_response = await summarizer.do_async(task)
        summary_text = str(summary_response)
        
        if self.debug:
            info_log(f"Summary generation complete: {len(summary_text)} chars", "Memory")
        
        return summary_text



    async def _analyze_interaction_for_traits(self, session: AgentSession, agent_run_output) -> dict:
        """
        Analyzes user interaction to extract traits.
        """
        from upsonic.agent.agent import Agent
        from upsonic.tasks.tasks import Task

        if not self.model:
            raise ValueError("model must be configured for user trait analysis")

        historical_prompts_content = []
        new_prompts_content = []

        # Get historical user prompts from ALL sessions with the same user_id
        # Use AgentSession helper method to get all user prompt messages for this user_id
        current_run_id = agent_run_output.run_id if agent_run_output and hasattr(agent_run_output, 'run_id') and agent_run_output.run_id else None
        
        if self.user_id:
            try:
                historical_prompts_content = await AgentSession.get_all_user_prompt_messages_for_user_id_async(
                    storage=self.storage,
                    user_id=self.user_id,
                    exclude_run_id=current_run_id
                )
                
                if self.debug:
                    info_log(f"Retrieved {len(historical_prompts_content)} historical user prompts (user_id={self.user_id}, excluding current run_id={current_run_id})", "Memory")
            except Exception as e:
                if self.debug:
                    info_log(f"Could not get historical user prompts by user_id: {e}", "Memory")
        elif self.debug:
            info_log("No user_id available, cannot get historical user prompts", "Memory")

        # Get new user prompts from agent_run_output (current run only)
        if agent_run_output is not None and hasattr(agent_run_output, 'new_messages'):
            try:
                new_messages = agent_run_output.new_messages()
                if new_messages:
                    new_prompts_content = AgentSession._extract_user_prompts_from_messages(new_messages)
                    if self.debug:
                        info_log(f"Retrieved {len(new_prompts_content)} new user prompts from current run (agent_run_output.new_messages())", "Memory")
            except Exception as e:
                if self.debug:
                    info_log(f"Could not extract new messages from agent_run_output: {e}", "Memory")
        elif self.debug:
            info_log("No agent_run_output provided for new messages", "Memory")

        if not historical_prompts_content and not new_prompts_content:
            from upsonic.utils.printing import warning_log
            warning_log("No user prompts found in history or new messages. Cannot analyze traits.", "MemoryStorage")
            if self.debug:
                info_log("No user prompts available for trait analysis", "Memory")
            return {}

        prompt_context_parts = []
        source_log = []
        if historical_prompts_content:
            history_str = "\n".join(f"- {p}" for p in historical_prompts_content)
            prompt_context_parts.append(f"### Historical User Prompts:\n{history_str}")
            source_log.append("session history")
            if self.debug:
                info_log(f"Found {len(historical_prompts_content)} historical user prompts", "Memory")
            
        if new_prompts_content:
            new_str = "\n".join(f"- {p}" for p in new_prompts_content)
            prompt_context_parts.append(f"### Latest User Prompts:\n{new_str}")
            source_log.append("new messages")
            if self.debug:
                info_log(f"Found {len(new_prompts_content)} new user prompts", "Memory")

        conversation_context_str = "\n\n".join(prompt_context_parts)
        info_log(f"Analyzing traits using context from: {', '.join(source_log)}.", context="Memory")
        
        current_profile = session.user_profile or {}
        
        if self.debug:
            info_log(f"Current profile has {len(current_profile)} traits", "Memory")
        
        from upsonic.utils.printing import warning_log
        
        analyzer = Agent(name="User Trait Analyzer", model=self.model, debug=self.debug)

        if self.is_profile_dynamic:
            class FieldDefinition(BaseModel):
                name: str = Field(..., description="Snake_case field name")
                description: str = Field(..., description="Description of what this field represents")
            
            class ProposedSchema(BaseModel):
                fields: List[FieldDefinition] = Field(
                    ..., 
                    min_length=2,
                    description="List of 2-5 field definitions extracted from the conversation"
                )
                

            schema_generator_prompt = f"""Analyze this conversation and identify 2-5 specific traits about the user.

=== USER CONVERSATION ===
{conversation_context_str}

=== YOUR TASK ===
Create a list of field definitions where each field has:
- name: snake_case field name (e.g., preferred_name, occupation, expertise_level, primary_interest, hobbies)
- description: what that field represents

You MUST provide at least 2-3 fields based on what the user explicitly mentioned in the conversation.

Examples:
- If user says "I'm Alex interested in ML": fields like preferred_name, primary_interest, expertise_level
- If user says "I work as engineer and love coding": fields like occupation, hobbies, expertise_area
"""
            schema_task = Task(description=schema_generator_prompt, response_format=ProposedSchema)
            
            try:
                proposed_schema_response = await analyzer.do_async(schema_task)
                field_count = len(proposed_schema_response.fields) if proposed_schema_response and hasattr(proposed_schema_response, 'fields') else 0
                info_log(f"LLM generated schema with {field_count} fields", "Memory")
                if field_count > 0:
                    info_log(f"Generated field names: {[f.name for f in proposed_schema_response.fields]}", "Memory")
            except Exception as e:
                warning_log(f"Dynamic schema generation failed with error: {e}. No user traits extracted.", "Memory")
                return {}

            if not proposed_schema_response or not hasattr(proposed_schema_response, 'fields') or not proposed_schema_response.fields:
                field_count = len(proposed_schema_response.fields) if proposed_schema_response and hasattr(proposed_schema_response, 'fields') else 0
                info_log(f"Schema generation result: {field_count} fields generated", "Memory")
                warning_log(f"Dynamic schema generation returned {field_count} fields (expected at least 2). No user traits extracted.", "Memory")
                return {}

            dynamic_fields = {field_def.name: (Optional[str], Field(None, description=field_def.description)) for field_def in proposed_schema_response.fields}
            DynamicUserTraitModel = create_model('DynamicUserTraitModel', **dynamic_fields)

            trait_extractor_prompt = f"""Extract user traits from this conversation.

Current Profile Data:
{json.dumps(current_profile, indent=2)}

User's Conversation:
{conversation_context_str}

YOUR TASK: Fill in the trait fields based on what the user explicitly stated. Extract concrete, specific information from the conversation. If information is not available for a field, you may leave it as null.
"""
            trait_task = Task(description=trait_extractor_prompt, response_format=DynamicUserTraitModel)
            trait_response = await analyzer.do_async(trait_task)
            
            if trait_response and hasattr(trait_response, 'model_dump'):
                return trait_response.model_dump()
            return {}

        else:
            prompt = f"""Analyze the user's conversation and extract their traits.

Current Profile Data:
{json.dumps(current_profile, indent=2)}

User's Conversation:
{conversation_context_str}

YOUR TASK: Fill in trait fields based on what the user explicitly stated in the conversation. Extract concrete, specific information. Update existing traits if new information is provided. Leave fields as None if information is not available.
"""
            task = Task(description=prompt, response_format=self.profile_schema_model)
            
            trait_response = await analyzer.do_async(task)
            if trait_response and hasattr(trait_response, 'model_dump'):
                return trait_response.model_dump()
            return {}
    
    # ========================================================================
    # AgentSession API (delegates to storage, uses instance session_id/user_id)
    # ========================================================================
    
    
    def _run_sync(self, coro):
        """Run coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
            # If already in async context, create task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)
    
    # --- Read/Get ---
    async def get_session_async(self, session_id: Optional[str] = None) -> Optional[AgentSession]:
        """Get AgentSession by session_id (defaults to instance session_id)."""
        return await self.storage.read_async(session_id or self.session_id, AgentSession)
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[AgentSession]:
        return self._run_sync(self.get_session_async(session_id))
    
    async def get_messages_async(self, session_id: Optional[str] = None, limit: Optional[int] = None) -> List[Any]:
        """Get messages from session's runs."""
        session = await self.get_session_async(session_id)
        return session.get_messages(limit=limit) if session else []
    
    def get_messages(self, session_id: Optional[str] = None, limit: Optional[int] = None) -> List[Any]:
        return self._run_sync(self.get_messages_async(session_id, limit))
    
    # --- Write/Upsert ---
    async def save_session_async(self, session: AgentSession) -> None:
        """Save/upsert an AgentSession."""
        await self.storage.upsert_agent_session_async(session)
    
    def save_session(self, session: AgentSession) -> None:
        self._run_sync(self.save_session_async(session))
    
    async def save_run_async(self, run_output: Any, session_id: Optional[str] = None) -> None:
        """Save a run output to session (creates session if needed)."""
        sid = session_id or self.session_id
        session = await self.get_session_async(sid)
        if not session:
            session = AgentSession(
                session_id=sid, user_id=self.user_id,
                agent_id=getattr(run_output, 'agent_id', None),
                created_at=int(time.time())
            )
        session.upsert_run(run_output)
        
        # Populate session_data and agent_data from run output
        session.populate_from_run_output(run_output)
        
        await self.storage.upsert_agent_session_async(session)
    
    def save_run(self, run_output: Any, session_id: Optional[str] = None) -> None:
        self._run_sync(self.save_run_async(run_output, session_id))
    
    # --- Delete ---
    async def delete_session_async(self, session_id: Optional[str] = None) -> None:
        """Delete session by ID (defaults to instance session_id)."""
        await self.storage.delete_agent_session_async(session_id or self.session_id)
    
    def delete_session(self, session_id: Optional[str] = None) -> None:
        self._run_sync(self.delete_session_async(session_id))
    
    # --- List/Find/Clear ---
    async def list_sessions_async(
        self, agent_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[AgentSession]:
        """List sessions filtered by agent_id/user_id."""
        return await self.storage.list_agent_sessions_async(agent_id, user_id)
    
    def list_sessions(self, **kwargs) -> List[AgentSession]:
        return self._run_sync(self.list_sessions_async(**kwargs))
    
    async def find_session_async(
        self, session_id: Optional[str] = None, agent_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Optional[AgentSession]:
        """Find session by ID (priority) or agent_id/user_id filter."""
        return await self.storage.find_agent_session_async(session_id, agent_id, user_id)
    
    def find_session(self, **kwargs) -> Optional[AgentSession]:
        return self._run_sync(self.find_session_async(**kwargs))
    
    async def clear_sessions_async(
        self, agent_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> int:
        """Delete all sessions matching criteria. Returns count."""
        return await self.storage.clear_agent_sessions_async(agent_id, user_id)
    
    def clear_sessions(self, **kwargs) -> int:
        return self._run_sync(self.clear_sessions_async(**kwargs))
    
    # --- Metadata ---
    async def set_metadata_async(
        self, metadata: Dict[str, Any], session_id: Optional[str] = None, merge: bool = True
    ) -> None:
        """Set/update session metadata (merges by default)."""
        sid = session_id or self.session_id
        session = await self.get_session_async(sid)
        if not session:
            session = AgentSession(session_id=sid, user_id=self.user_id, created_at=int(time.time()))
        if merge and session.metadata:
            session.metadata.update(metadata)
        else:
            session.metadata = metadata
        await self.storage.upsert_agent_session_async(session)
    
    def set_metadata(self, metadata: Dict[str, Any], session_id: Optional[str] = None, merge: bool = True) -> None:
        self._run_sync(self.set_metadata_async(metadata, session_id, merge))
    
    async def get_metadata_async(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get session metadata."""
        session = await self.get_session_async(session_id)
        return session.metadata if session else None
    
    def get_metadata(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self._run_sync(self.get_metadata_async(session_id))
    
