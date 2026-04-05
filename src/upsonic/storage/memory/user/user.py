"""User memory implementation for Upsonic agent framework."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union

if TYPE_CHECKING:
    from pydantic import BaseModel
    from upsonic.storage.base import Storage
    from upsonic.models import Model
    from upsonic.run.agent.output import AgentRunOutput

from upsonic.storage.memory.user.base import BaseUserMemory
from upsonic.storage.memory.storage_dispatch import (
    is_async_storage_backend,
    run_awaitable_sync,
)


class UserMemory(BaseUserMemory):
    """User memory manager for storing and retrieving user profiles.
    
    This implementation:
    - Loads user profiles from storage (via sessions or user_memory table)
    - Formats profiles for system prompt injection
    - Analyzes conversations to extract user traits
    - Saves updated profiles using configured update mode
    """
    
    def __init__(
        self,
        storage: "Storage",
        user_id: str,
        enabled: bool = True,
        load_enabled: Optional[bool] = None,
        profile_schema: Optional[Type["BaseModel"]] = None,
        dynamic_profile: bool = False,
        update_mode: Literal['update', 'replace'] = 'update',
        model: Optional[Union["Model", str]] = None,
        debug: bool = False,
        debug_level: int = 1,
    ) -> None:
        super().__init__(
            storage=storage,
            user_id=user_id,
            enabled=enabled,
            load_enabled=load_enabled,
            profile_schema=profile_schema,
            dynamic_profile=dynamic_profile,
            update_mode=update_mode,
            model=model,
            debug=debug,
            debug_level=debug_level,
        )
        
        # Handle profile schema
        if self.dynamic_profile:
            if profile_schema:
                from upsonic.utils.printing import warning_log
                warning_log(
                    "`dynamic_user_profile` is True, so the provided `user_profile_schema` will be ignored.",
                    "UserMemory"
                )
            self._profile_schema_model = None
        else:
            if profile_schema:
                self._profile_schema_model = profile_schema
            else:
                from upsonic.schemas import UserTraits
                self._profile_schema_model = UserTraits
    
    def _format_profile_data(self, profile_data: Dict[str, Any]) -> Optional[str]:
        """Format user profile data into a readable string format."""
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
    
    async def aget(
        self,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get user profile formatted for prompt injection (async).

        Respects ``load_enabled`` – returns ``None`` when loading is disabled.
        """
        from upsonic.utils.printing import info_log
        
        if not self.load_enabled:
            return None

        if is_async_storage_backend(self.storage):
            try:
                user_memory = await self.storage.aget_user_memory(
                    user_id=self.user_id,
                    agent_id=agent_id,
                    team_id=team_id,
                    deserialize=True,
                )
                if user_memory and hasattr(user_memory, "user_memory") and user_memory.user_memory:
                    profile_str = self._format_profile_data(user_memory.user_memory)
                    if profile_str:
                        if self.debug:
                            info_log("Loaded user profile from storage", "UserMemory")
                        return f"<UserProfile>\n{profile_str}\n</UserProfile>"
            except Exception as e:
                if self.debug:
                    info_log(f"Could not load user memory from storage: {e}", "UserMemory")
            return None

        return self.get(agent_id=agent_id, team_id=team_id)
    
    def get(
        self,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get user profile formatted for prompt injection (sync version).

        Respects ``load_enabled`` – returns ``None`` when loading is disabled.
        """
        from upsonic.utils.printing import info_log

        if is_async_storage_backend(self.storage):
            return run_awaitable_sync(self.aget(agent_id=agent_id, team_id=team_id))
        
        if not self.load_enabled:
            return None
        
        # Get user memory from storage using correct method
        try:
            user_memory = self.storage.get_user_memory(
                user_id=self.user_id,
                agent_id=agent_id,
                team_id=team_id,
                deserialize=True
            )
            if user_memory and hasattr(user_memory, 'user_memory') and user_memory.user_memory:
                profile_str = self._format_profile_data(user_memory.user_memory)
                if profile_str:
                    if self.debug:
                        info_log("Loaded user profile from storage", "UserMemory")
                    return f"<UserProfile>\n{profile_str}\n</UserProfile>"
        except Exception as e:
            if self.debug:
                info_log(f"Could not load user memory from storage: {e}", "UserMemory")
        
        return None
    
    async def asave(
        self,
        output: Any,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> None:
        """Analyze interaction and save user profile to storage (async)."""
        from upsonic.utils.printing import info_log, warning_log
        
        if not self.enabled:
            return
        
        if not self.model:
            warning_log(
                "User analysis memory is enabled but no model is configured. Skipping.",
                "UserMemory"
            )
            return
        
        try:
            # Get current user memory from storage
            current_profile: Dict[str, Any] = {}
            if is_async_storage_backend(self.storage):
                user_memory = await self.storage.aget_user_memory(
                    user_id=self.user_id,
                    agent_id=agent_id,
                    team_id=team_id,
                    deserialize=True
                )
            else:
                user_memory = self.storage.get_user_memory(
                    user_id=self.user_id,
                    agent_id=agent_id,
                    team_id=team_id,
                    deserialize=True
                )
            
            if user_memory and hasattr(user_memory, 'user_memory') and user_memory.user_memory:
                current_profile = user_memory.user_memory
            
            # Analyze interaction for traits
            updated_traits = await self._analyze_interaction_for_traits(output, current_profile)
            
            # Aggregate trait analysis sub-agent LLM usage into output.usage
            trait_llm_usage = getattr(self, '_last_llm_usage', None)
            if trait_llm_usage is not None and output is not None:
                output_usage = output._ensure_usage()
                output_usage.incr(trait_llm_usage)
                self._last_llm_usage = None
            
            if self.debug:
                info_log(f"Extracted traits: {updated_traits}", "UserMemory")
            
            # Update profile based on mode
            if self.update_mode == 'replace':
                final_profile = updated_traits
                if self.debug:
                    info_log(f"Replaced user profile with {len(updated_traits)} traits", "UserMemory")
            elif self.update_mode == 'update':
                final_profile = current_profile.copy()
                before_count = len(final_profile)
                final_profile.update(updated_traits)
                if self.debug:
                    info_log(
                        f"Updated user profile: {before_count} -> {len(final_profile)} traits",
                        "UserMemory"
                    )
            else:
                final_profile = updated_traits
            
            # Create UserMemory instance and save to storage
            from upsonic.storage.schemas import UserMemory
            
            user_memory_instance = UserMemory(
                user_id=self.user_id,
                user_memory=final_profile,
                agent_id=agent_id,
                team_id=team_id,
            )
            
            if is_async_storage_backend(self.storage):
                await self.storage.aupsert_user_memory(user_memory_instance, deserialize=True)
            else:
                self.storage.upsert_user_memory(user_memory_instance, deserialize=True)
            
            if self.debug:
                info_log("Saved user profile to storage", "UserMemory")
            
        except Exception as e:
            warning_log(f"Failed to analyze and save user profile: {e}", "UserMemory")
    
    def save(
        self,
        output: Any,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> None:
        """Analyze interaction and save user profile (runs async pipeline in a dedicated event loop when needed)."""
        from upsonic.utils.printing import warning_log
        
        if not self.enabled:
            return
        
        if not self.model:
            warning_log(
                "User analysis memory is enabled but no model is configured. Skipping.",
                "UserMemory"
            )
            return

        run_awaitable_sync(self.asave(output, agent_id=agent_id, team_id=team_id))
    
    async def _analyze_interaction_for_traits(
        self,
        output: "AgentRunOutput",
        current_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze user interaction to extract traits."""
        from pydantic import BaseModel, Field, create_model
        from upsonic.agent.agent import Agent
        from upsonic.tasks.tasks import Task
        from upsonic.session.agent import AgentSession
        from upsonic.utils.printing import info_log, warning_log
        
        historical_prompts_content: List[str] = []
        new_prompts_content: List[str] = []
        
        # Get current run_id to exclude from historical
        current_run_id = None
        if output and hasattr(output, 'run_id') and output.run_id:
            current_run_id = output.run_id
        
        # Get historical user prompts from ALL sessions with same user_id
        if self.user_id:
            try:
                historical_prompts_content = await AgentSession.get_all_user_prompt_messages_for_user_id_async(
                    storage=self.storage,
                    user_id=self.user_id,
                    exclude_run_id=current_run_id
                )
                
                if self.debug:
                    info_log(
                        f"Retrieved {len(historical_prompts_content)} historical prompts "
                        f"(user_id={self.user_id}, excluding run_id={current_run_id})",
                        "UserMemory"
                    )
            except Exception as e:
                if self.debug:
                    info_log(f"Could not get historical user prompts: {e}", "UserMemory")
        
        # Get new user prompts from output (current run only)
        if output is not None and hasattr(output, 'new_messages'):
            try:
                new_messages = output.new_messages()
                if new_messages:
                    new_prompts_content = AgentSession._extract_user_prompts_from_messages(new_messages)
                    if self.debug:
                        info_log(
                            f"Retrieved {len(new_prompts_content)} new prompts from current run",
                            "UserMemory"
                        )
            except Exception as e:
                if self.debug:
                    info_log(f"Could not extract new messages: {e}", "UserMemory")
        
        if not historical_prompts_content and not new_prompts_content:
            warning_log(
                "No user prompts found in history or new messages. Cannot analyze traits.",
                "UserMemory"
            )
            return {}
        
        # Build context
        prompt_context_parts = []
        if historical_prompts_content:
            history_str = "\n".join(f"- {p}" for p in historical_prompts_content)
            prompt_context_parts.append(f"### Historical User Prompts:\n{history_str}")
        
        if new_prompts_content:
            new_str = "\n".join(f"- {p}" for p in new_prompts_content)
            prompt_context_parts.append(f"### Latest User Prompts:\n{new_str}")
        
        conversation_context_str = "\n\n".join(prompt_context_parts)
        
        if self.debug:
            info_log(f"Current profile has {len(current_profile)} traits", "UserMemory")
        
        analyzer = Agent(name="User Trait Analyzer", model=self.model, debug=self.debug)
        
        if self.dynamic_profile:
            # Dynamic schema generation
            class FieldDefinition(BaseModel):
                name: str = Field(..., description="Snake_case field name")
                description: str = Field(..., description="Description of what this field represents")
            
            class ProposedSchema(BaseModel):
                fields: List[FieldDefinition] = Field(
                    ...,
                    min_length=2,
                    description="List of 2-5 field definitions extracted from the conversation"
                )
            
            schema_prompt = f"""Analyze this conversation and identify 2-5 specific traits about the user.

=== USER CONVERSATION ===
{conversation_context_str}

=== YOUR TASK ===
Create a list of field definitions where each field has:
- name: snake_case field name (e.g., preferred_name, occupation, expertise_level)
- description: what that field represents

You MUST provide at least 2-3 fields based on what the user explicitly mentioned.
"""
            schema_task = Task(description=schema_prompt, response_format=ProposedSchema)
            
            try:
                schema_output = await analyzer.do_async(schema_task, return_output=True)
                # Store sub-agent usage for parent context aggregation
                if hasattr(schema_output, 'usage') and schema_output.usage:
                    if not hasattr(self, '_last_llm_usage'):
                        self._last_llm_usage = None
                    from upsonic.usage import RunUsage
                    if self._last_llm_usage is None:
                        self._last_llm_usage = RunUsage()
                    self._last_llm_usage.incr(schema_output.usage)
                proposed_schema = schema_output.output
                if not proposed_schema or not hasattr(proposed_schema, 'fields') or not proposed_schema.fields:
                    warning_log("Dynamic schema generation returned no fields", "UserMemory")
                    return {}
                
                if self.debug:
                    info_log(f"Generated schema with fields: {[f.name for f in proposed_schema.fields]}", "UserMemory")
                
                # Create dynamic model
                from typing import Optional as Opt
                dynamic_fields = {
                    field_def.name: (Opt[str], Field(None, description=field_def.description))
                    for field_def in proposed_schema.fields
                }
                DynamicUserTraitModel = create_model('DynamicUserTraitModel', **dynamic_fields)
                
                # Extract traits
                trait_prompt = f"""Extract user traits from this conversation.

Current Profile Data:
{json.dumps(current_profile, indent=2)}

User's Conversation:
{conversation_context_str}

YOUR TASK: Fill in the trait fields based on what the user explicitly stated.
"""
                trait_task = Task(description=trait_prompt, response_format=DynamicUserTraitModel)
                trait_output = await analyzer.do_async(trait_task, return_output=True)
                # Store sub-agent usage for parent context aggregation
                if hasattr(trait_output, 'usage') and trait_output.usage:
                    if not hasattr(self, '_last_llm_usage'):
                        self._last_llm_usage = None
                    from upsonic.usage import RunUsage
                    if self._last_llm_usage is None:
                        self._last_llm_usage = RunUsage()
                    self._last_llm_usage.incr(trait_output.usage)
                trait_response = trait_output.output
                
                if trait_response and hasattr(trait_response, 'model_dump'):
                    return trait_response.model_dump()
                return {}
                
            except Exception as e:
                warning_log(f"Dynamic schema generation failed: {e}", "UserMemory")
                return {}
        
        else:
            # Use provided schema
            prompt = f"""Analyze the user's conversation and extract their traits.

Current Profile Data:
{json.dumps(current_profile, indent=2)}

User's Conversation:
{conversation_context_str}

YOUR TASK: Fill in trait fields based on what the user explicitly stated. 
Extract concrete, specific information. Update existing traits if new information is provided.
Leave fields as None if information is not available.
"""
            task = Task(description=prompt, response_format=self._profile_schema_model)
            
            trait_output = await analyzer.do_async(task, return_output=True)
            # Store sub-agent usage for parent context aggregation
            if hasattr(trait_output, 'usage') and trait_output.usage:
                if not hasattr(self, '_last_llm_usage'):
                    self._last_llm_usage = None
                from upsonic.usage import RunUsage
                if self._last_llm_usage is None:
                    self._last_llm_usage = RunUsage()
                self._last_llm_usage.incr(trait_output.usage)
            trait_response = trait_output.output
            if trait_response and hasattr(trait_response, 'model_dump'):
                return trait_response.model_dump()
            return {}
