"""
CultureManager - Orchestrator for managing cultural knowledge.

This module provides the CultureManager class that handles all cultural
knowledge operations including CRUD, LLM-based extraction, and system
prompt generation.

Notice: Culture is an experimental feature and is subject to change.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from upsonic.models import Model, ModelRequest, ModelResponse
    from upsonic.storage.base import Storage

from upsonic.culture.cultural_knowledge import CulturalKnowledge
from upsonic.utils.async_utils import AsyncExecutionMixin
from upsonic.utils.printing import (
    culture_info,
    culture_debug,
    culture_warning,
    culture_error,
    culture_knowledge_added,
    culture_knowledge_updated,
    culture_knowledge_deleted,
    culture_extraction_started,
    culture_extraction_completed,
)


DEFAULT_CULTURE_MODEL = "openai/gpt-4o"


class CultureManager(AsyncExecutionMixin):
    """
    Orchestrator for managing cultural knowledge.
    
    CultureManager handles:
    - CRUD operations for CulturalKnowledge via Storage providers
    - LLM-based knowledge extraction from conversations
    - System prompt generation for agents
    - Tool generation for agentic culture updates
    
    Notice: Culture is an experimental feature and is subject to change.
    
    Attributes:
        storage: The storage provider for persisting cultural knowledge
        model: Optional LLM model for knowledge extraction
        knowledge_updated: Flag indicating if tools were called in last LLM run
        
    Example:
        ```python
        from upsonic.storage.providers import InMemoryStorage
        from upsonic.culture import CultureManager
        
        storage = InMemoryStorage()
        manager = CultureManager(storage=storage)
        
        # Add knowledge manually
        knowledge = CulturalKnowledge(
            id="k1",
            name="Communication Style",
            content="Always be concise and direct.",
            categories=["communication", "style"]
        )
        manager.add_cultural_knowledge(knowledge)
        
        # Get all knowledge
        all_knowledge = manager.get_all_knowledge()
        ```
    """

    def __init__(
        self,
        storage: "Storage",
        model: Optional[Union["Model", str]] = None,
        system_message: Optional[str] = None,
        culture_capture_instructions: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        add_knowledge: bool = True,
        update_knowledge: bool = True,
        delete_knowledge: bool = False,
        clear_knowledge: bool = False,
        debug: bool = False,
    ):
        """
        Initialize CultureManager.
        
        Args:
            storage: Storage provider for cultural knowledge persistence
            model: LLM model for knowledge extraction (string or Model instance).
                   If None, extraction features are disabled.
                   Default model is "openai/gpt-4o" when needed.
            system_message: Custom system message for knowledge extraction
            culture_capture_instructions: Instructions for what to capture as culture
            additional_instructions: Additional context-specific instructions
            add_knowledge: Enable add_cultural_knowledge tool (default True)
            update_knowledge: Enable update_cultural_knowledge tool (default True)
            delete_knowledge: Enable delete_cultural_knowledge tool (default False - risky)
            clear_knowledge: Enable clear_cultural_knowledge tool (default False - very risky)
            debug: Enable debug logging
        """
        self.storage = storage
        self._model: Optional["Model"] = None
        self._model_spec = model
        self._owns_model = False
        
        self.system_message = system_message
        self.culture_capture_instructions = culture_capture_instructions
        self.additional_instructions = additional_instructions
        
        # Tool control flags
        self.add_knowledge = add_knowledge
        self.update_knowledge = update_knowledge
        self.delete_knowledge = delete_knowledge
        self.clear_knowledge = clear_knowledge
        
        self.debug = debug
        
        # Track if knowledge was updated in last LLM run
        self.knowledge_updated: bool = False

    def _get_or_create_model(self) -> "Model":
        """
        Get or create the LLM model for culture operations.
        
        Returns:
            Model instance
        """
        if self._model is not None:
            return self._model
        
        from upsonic.models import Model, infer_model
        
        if self._model_spec is None:
            # Use default model
            self._model = infer_model(DEFAULT_CULTURE_MODEL)
            self._owns_model = True
        elif isinstance(self._model_spec, str):
            # Infer model from string
            self._model = infer_model(self._model_spec)
            self._owns_model = True
        else:
            # User provided model instance
            self._model = self._model_spec
            self._owns_model = False
        
        return self._model

    # =========================================================================
    # Core CRUD Methods (Synchronous)
    # =========================================================================

    def get_knowledge(self, knowledge_id: str) -> Optional[CulturalKnowledge]:
        """
        Get a cultural knowledge entry by ID.
        
        Args:
            knowledge_id: The unique identifier of the knowledge entry
            
        Returns:
            CulturalKnowledge instance if found, None otherwise
        """
        return self._run_async_from_sync(self.aget_knowledge(knowledge_id))

    def get_all_knowledge(self, name: Optional[str] = None) -> List[CulturalKnowledge]:
        """
        Get all cultural knowledge entries.
        
        Args:
            name: Optional filter - returns entries where name contains this string
            
        Returns:
            List of CulturalKnowledge instances
        """
        return self._run_async_from_sync(self.aget_all_knowledge(name))

    def add_cultural_knowledge(self, knowledge: CulturalKnowledge) -> Optional[str]:
        """
        Add a new cultural knowledge entry.
        
        If knowledge.id is None, a UUID will be generated.
        
        Args:
            knowledge: CulturalKnowledge instance to add
            
        Returns:
            The ID of the added knowledge entry
        """
        return self._run_async_from_sync(self.aadd_cultural_knowledge(knowledge))

    def delete_cultural_knowledge(self, knowledge_id: str) -> None:
        """
        Delete a cultural knowledge entry.
        
        Args:
            knowledge_id: The ID of the knowledge entry to delete
        """
        return self._run_async_from_sync(self.adelete_cultural_knowledge(knowledge_id))

    def clear_all_knowledge(self) -> None:
        """
        Delete ALL cultural knowledge entries.
        
        Warning: This is a destructive operation. Use with caution.
        """
        return self._run_async_from_sync(self.aclear_all_knowledge())

    # =========================================================================
    # Core CRUD Methods (Asynchronous)
    # =========================================================================

    async def aget_knowledge(self, knowledge_id: str) -> Optional[CulturalKnowledge]:
        """Async version of get_knowledge."""
        culture_debug(f"Reading knowledge: {knowledge_id}", debug=self.debug)
        result = await self.storage.read_cultural_knowledge_async(knowledge_id)
        if result:
            culture_debug(f"Found knowledge: {result.name}", debug=self.debug)
        else:
            culture_debug(f"Knowledge not found: {knowledge_id}", debug=self.debug)
        return result

    async def aget_all_knowledge(self, name: Optional[str] = None) -> List[CulturalKnowledge]:
        """Async version of get_all_knowledge."""
        filter_msg = f" (filter: {name})" if name else ""
        culture_debug(f"Listing all knowledge{filter_msg}", debug=self.debug)
        result = await self.storage.list_all_cultural_knowledge_async(name)
        culture_debug(f"Found {len(result)} knowledge entries", debug=self.debug)
        return result

    async def aadd_cultural_knowledge(self, knowledge: CulturalKnowledge) -> Optional[str]:
        """Async version of add_cultural_knowledge."""
        # Generate ID if not provided
        if knowledge.id is None:
            knowledge.id = str(uuid.uuid4())
        
        await self.storage.upsert_cultural_knowledge_async(knowledge)
        culture_knowledge_added(knowledge.name, knowledge.id, debug=self.debug)
        return knowledge.id

    async def adelete_cultural_knowledge(self, knowledge_id: str) -> None:
        """Async version of delete_cultural_knowledge."""
        await self.storage.delete_cultural_knowledge_async(knowledge_id)
        culture_knowledge_deleted(knowledge_id, debug=self.debug)

    async def aclear_all_knowledge(self) -> None:
        """Async version of clear_all_knowledge."""
        culture_warning("Clearing ALL cultural knowledge", debug=self.debug)
        await self.storage.clear_cultural_knowledge_async()
        culture_info("All cultural knowledge cleared", debug=self.debug)

    # =========================================================================
    # Tool Generation (Closure Pattern)
    # =========================================================================

    def _get_db_tools(
        self,
        enable_add_knowledge: bool = True,
        enable_update_knowledge: bool = True,
        enable_delete_knowledge: bool = False,
        enable_clear_knowledge: bool = False,
    ) -> List[Callable]:
        """
        Generate tools as closures that capture the storage instance.
        
        Each tool function has the storage in its closure scope, so the LLM
        doesn't need to pass db as a parameter.
        
        Args:
            enable_add_knowledge: Include add_cultural_knowledge tool
            enable_update_knowledge: Include update_cultural_knowledge tool
            enable_delete_knowledge: Include delete_cultural_knowledge tool (risky)
            enable_clear_knowledge: Include clear_cultural_knowledge tool (very risky)
            
        Returns:
            List of callable tool functions
        """
        tools = []
        manager = self  # Capture self in closure
        
        if enable_add_knowledge:
            def add_cultural_knowledge(
                name: str,
                summary: Optional[str] = None,
                content: Optional[str] = None,
                categories: Optional[List[str]] = None,
                notes: Optional[List[str]] = None,
            ) -> str:
                """
                Add new cultural knowledge entry.
                
                Args:
                    name: Short, specific title for the knowledge (required)
                    summary: One-line purpose or takeaway
                    content: The main principle, rule, or guideline
                    categories: List of tags (e.g., ['guardrails', 'rules', 'practices'])
                    notes: List of contextual notes, rationale, or examples
                    
                Returns:
                    Success message with knowledge ID
                """
                knowledge = CulturalKnowledge(
                    id=str(uuid.uuid4()),
                    name=name,
                    summary=summary,
                    content=content,
                    categories=categories,
                    notes=notes,
                )
                manager.storage.upsert_cultural_knowledge(knowledge)
                manager.knowledge_updated = True
                culture_knowledge_added(name, knowledge.id, debug=manager.debug)
                return f"Successfully added cultural knowledge: {knowledge.id}"
            
            tools.append(add_cultural_knowledge)
        
        if enable_update_knowledge:
            def update_cultural_knowledge(
                knowledge_id: str,
                name: str,
                summary: Optional[str] = None,
                content: Optional[str] = None,
                categories: Optional[List[str]] = None,
                notes: Optional[List[str]] = None,
            ) -> str:
                """
                Update an existing cultural knowledge entry.
                
                Args:
                    knowledge_id: The ID of the knowledge entry to update (required)
                    name: New title for the knowledge (required)
                    summary: New one-line purpose or takeaway
                    content: New main principle, rule, or guideline
                    categories: New list of tags
                    notes: New list of contextual notes
                    
                Returns:
                    Success message
                """
                # Get existing knowledge
                existing = manager.storage.read_cultural_knowledge(knowledge_id)
                if not existing:
                    culture_warning(f"Knowledge entry not found for update: {knowledge_id}", debug=manager.debug)
                    return f"Knowledge entry not found: {knowledge_id}"
                
                # Update fields
                existing.name = name
                if summary is not None:
                    existing.summary = summary
                if content is not None:
                    existing.content = content
                if categories is not None:
                    existing.categories = categories
                if notes is not None:
                    existing.notes = notes
                
                manager.storage.upsert_cultural_knowledge(existing)
                manager.knowledge_updated = True
                culture_knowledge_updated(name, knowledge_id, debug=manager.debug)
                return f"Successfully updated cultural knowledge: {knowledge_id}"
            
            tools.append(update_cultural_knowledge)
        
        if enable_delete_knowledge:
            def delete_cultural_knowledge(knowledge_id: str) -> str:
                """
                Delete a cultural knowledge entry.
                
                WARNING: This is a destructive operation.
                
                Args:
                    knowledge_id: The ID of the knowledge entry to delete
                    
                Returns:
                    Success message
                """
                manager.storage.delete_cultural_knowledge(knowledge_id)
                manager.knowledge_updated = True
                culture_knowledge_deleted(knowledge_id, debug=manager.debug)
                return f"Successfully deleted cultural knowledge: {knowledge_id}"
            
            tools.append(delete_cultural_knowledge)
        
        if enable_clear_knowledge:
            def clear_cultural_knowledge() -> str:
                """
                Clear ALL cultural knowledge entries.
                
                WARNING: This will delete ALL knowledge. Use with extreme caution.
                
                Returns:
                    Success message
                """
                culture_warning("Clearing ALL cultural knowledge via tool", debug=manager.debug)
                manager.storage.clear_cultural_knowledge()
                manager.knowledge_updated = True
                culture_info("All cultural knowledge cleared via tool", debug=manager.debug)
                return "Successfully cleared all cultural knowledge"
            
            tools.append(clear_cultural_knowledge)
        
        return tools

    def get_culture_tools(self) -> List[Callable]:
        """
        Get culture tools for agent registration (when enable_agentic_culture=True).
        
        Returns tools based on configured flags (add_knowledge, update_knowledge,
        delete_knowledge, clear_knowledge).
        
        Returns:
            List of callable tool functions
        """
        return self._get_db_tools(
            enable_add_knowledge=self.add_knowledge,
            enable_update_knowledge=self.update_knowledge,
            enable_delete_knowledge=self.delete_knowledge,
            enable_clear_knowledge=self.clear_knowledge,
        )

    # =========================================================================
    # System Prompt Generation
    # =========================================================================

    def _get_default_system_message(self) -> str:
        """Get the default system message for cultural knowledge extraction."""
        return """You are the Cultural Knowledge Manager, responsible for maintaining shared cultural knowledge for Agents and Teams.

## Your Role
Analyze conversations and inputs to extract valuable cultural knowledge that can improve agent performance across all interactions.

## Criteria for Cultural Knowledge
<knowledge_to_capture>
- Best practices and successful approaches
- Common patterns in user behavior
- Processes, design principles, rules of operation
- Guardrails, decision rationales, ethical guidelines
- Domain-specific lessons that generalize
- Communication styles that lead to better outcomes
</knowledge_to_capture>

## When to Add/Update Knowledge
- If new insights meet criteria and not already captured -> ADD
- If existing practices evolved or can be improved -> UPDATE (preserve history in notes)
- If nothing new or valuable -> respond with "No changes needed"

## How to Structure Knowledge
- `name`: short, specific title (required)
- `summary`: one-line takeaway
- `content`: reusable insight/rule (required)
- `categories`: tags like ["guardrails", "rules", "practices", "communication"]
- `notes`: context, rationale, examples

## De-duplication
- Search existing knowledge before adding
- UPDATE similar entries instead of duplicating
- Preserve lineage via notes field

## Safety
- Never include secrets, credentials, PII
- Only capture generalizable principles"""

    def _get_capture_instructions(self) -> str:
        """Get culture capture instructions."""
        if self.culture_capture_instructions:
            return self.culture_capture_instructions
        
        return """## What to Capture
Look for patterns that would benefit future interactions:
- Successful communication approaches
- User preferences and expectations
- Domain-specific best practices
- Decision-making frameworks
- Quality standards and guidelines"""

    def get_system_message(
        self,
        existing_knowledge: Optional[List[Dict[str, Any]]] = None,
        enable_add_knowledge: bool = True,
        enable_update_knowledge: bool = True,
        enable_delete_knowledge: bool = False,
        enable_clear_knowledge: bool = False,
    ) -> str:
        """
        Build the system prompt for knowledge extraction.
        
        Args:
            existing_knowledge: List of existing knowledge previews
            enable_add_knowledge: Whether add tool is available
            enable_update_knowledge: Whether update tool is available
            enable_delete_knowledge: Whether delete tool is available
            enable_clear_knowledge: Whether clear tool is available
            
        Returns:
            Complete system prompt string
        """
        parts = []
        
        # Base message
        if self.system_message:
            parts.append(self.system_message)
        else:
            parts.append(self._get_default_system_message())
        
        # Capture instructions
        parts.append(self._get_capture_instructions())
        
        # Additional instructions
        if self.additional_instructions:
            parts.append(f"\n## Additional Instructions\n{self.additional_instructions}")
        
        # Tool usage instructions
        available_tools = []
        if enable_add_knowledge:
            available_tools.append("- `add_cultural_knowledge`: Add new knowledge entries")
        if enable_update_knowledge:
            available_tools.append("- `update_cultural_knowledge`: Update existing entries")
        if enable_delete_knowledge:
            available_tools.append("- `delete_cultural_knowledge`: Delete entries (use carefully)")
        if enable_clear_knowledge:
            available_tools.append("- `clear_cultural_knowledge`: Clear ALL entries (use with extreme caution)")
        
        if available_tools:
            parts.append("\n## Available Tools\n" + "\n".join(available_tools))
        
        # Existing knowledge context
        if existing_knowledge:
            knowledge_str = "\n".join([
                f"- ID: {k.get('id')}, Name: {k.get('name')}, Categories: {k.get('categories', [])}"
                for k in existing_knowledge
            ])
            parts.append(f"\n## Existing Knowledge\n<existing_knowledge>\n{knowledge_str}\n</existing_knowledge>")
        else:
            parts.append("\n## Existing Knowledge\n<existing_knowledge>\nNo existing knowledge entries.\n</existing_knowledge>")
        
        # No-op instruction
        parts.append("\n## When No Changes Needed\nIf no valuable knowledge emerges from the input, respond exactly:\n\"No changes needed\"")
        
        return "\n\n".join(parts)

    # =========================================================================
    # LLM-Based Methods
    # =========================================================================

    def create_cultural_knowledge(
        self,
        message: Optional[str] = None,
        messages: Optional[List[Any]] = None,
    ) -> str:
        """
        Analyze messages to extract valuable cultural knowledge.
        
        SAFE FLOW: Only provides add_knowledge and update_knowledge tools.
        Does NOT provide delete or clear tools.
        
        Args:
            message: Single message to analyze
            messages: List of messages to analyze
            
        Returns:
            LLM response text
        """
        return self._run_async_from_sync(
            self.acreate_cultural_knowledge(message=message, messages=messages)
        )

    async def acreate_cultural_knowledge(
        self,
        message: Optional[str] = None,
        messages: Optional[List[Any]] = None,
    ) -> str:
        """
        Async version of create_cultural_knowledge.
        
        SAFE FLOW: Only provides add_knowledge and update_knowledge tools.
        """
        from upsonic.messages import ModelRequest, UserPromptPart, SystemPromptPart
        
        culture_extraction_started(debug=self.debug)
        
        # Reset knowledge_updated flag
        self.knowledge_updated = False
        
        # Get existing knowledge for context
        all_knowledge = await self.aget_all_knowledge()
        existing_previews = [k.preview() for k in all_knowledge]
        culture_debug(f"Found {len(all_knowledge)} existing knowledge entries for context", debug=self.debug)
        
        # Build system message
        system_message = self.get_system_message(
            existing_knowledge=existing_previews,
            enable_add_knowledge=True,
            enable_update_knowledge=True,
            enable_delete_knowledge=False,  # SAFE: No delete
            enable_clear_knowledge=False,   # SAFE: No clear
        )
        
        # Get tools (SAFE: only add + update)
        tools = self._get_db_tools(
            enable_add_knowledge=True,
            enable_update_knowledge=True,
            enable_delete_knowledge=False,
            enable_clear_knowledge=False,
        )
        
        # Build user message
        if message:
            user_content = message
        elif messages:
            user_content = "\n".join([str(m) for m in messages])
        else:
            culture_warning("No messages provided for culture extraction", debug=self.debug)
            return "No messages provided for analysis"
        
        # Get model and bind tools
        model = self._get_or_create_model()
        culture_debug(f"Using model for extraction: {model.model_name}", debug=self.debug)
        
        # Bind tools to model for the extraction task
        model.bind_tools(tools)
        
        # Build request with system prompt and user content
        request = ModelRequest(parts=[
            SystemPromptPart(content=system_message),
            UserPromptPart(content=user_content),
        ])
        
        # Run with ainvoke which handles tools properly
        try:
            response = await model.ainvoke(request)
            culture_extraction_completed(self.knowledge_updated, debug=self.debug)
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text or "No response"
            elif hasattr(response, 'parts'):
                from upsonic.messages import TextPart
                text_parts = [p.content for p in response.parts if isinstance(p, TextPart)]
                return "".join(text_parts) if text_parts else "No response"
            return "No response"
        except Exception as e:
            culture_error(f"Culture extraction failed: {e}", debug=self.debug)
            raise
        finally:
            # Clear tools from model after use
            model._tools = None

    def update_culture_task(self, task: str) -> str:
        """
        Execute explicit administrative culture management task.
        
        ADMIN FLOW: Provides ALL tools including delete and clear.
        Use for tasks like:
        - "Delete all outdated entries about old email system"
        - "Update knowledge about communication preferences"
        - "Clear all knowledge and start fresh"
        
        Args:
            task: Description of the administrative task
            
        Returns:
            LLM response text
        """
        return self._run_async_from_sync(self.aupdate_culture_task(task))

    async def aupdate_culture_task(self, task: str) -> str:
        """
        Async version of update_culture_task.
        
        ADMIN FLOW: Provides ALL tools including delete and clear.
        """
        from upsonic.messages import ModelRequest, UserPromptPart, SystemPromptPart
        
        culture_info(f"Executing admin culture task: {task[:100]}...", debug=self.debug)
        
        # Reset knowledge_updated flag
        self.knowledge_updated = False
        
        # Get existing knowledge for context
        all_knowledge = await self.aget_all_knowledge()
        existing_previews = [k.preview() for k in all_knowledge]
        culture_debug(f"Found {len(all_knowledge)} existing knowledge entries for admin task", debug=self.debug)
        
        # Build system message with ADMIN access
        system_message = self.get_system_message(
            existing_knowledge=existing_previews,
            enable_add_knowledge=True,
            enable_update_knowledge=True,
            enable_delete_knowledge=True,   # ADMIN: Enable delete
            enable_clear_knowledge=True,    # ADMIN: Enable clear
        )
        
        # Get ALL tools (ADMIN)
        tools = self._get_db_tools(
            enable_add_knowledge=True,
            enable_update_knowledge=True,
            enable_delete_knowledge=True,
            enable_clear_knowledge=True,
        )
        
        # Get model and bind tools
        model = self._get_or_create_model()
        culture_debug(f"Using model for admin task: {model.model_name}", debug=self.debug)
        
        # Bind tools to model for the admin task
        model.bind_tools(tools)
        
        # Build request
        request = ModelRequest(parts=[
            SystemPromptPart(content=system_message),
            UserPromptPart(content=f"Administrative Task: {task}"),
        ])
        
        # Run with ainvoke which handles tools properly
        try:
            response = await model.ainvoke(request)
            culture_info(f"Admin culture task completed, knowledge_updated={self.knowledge_updated}", debug=self.debug)
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text or "No response"
            elif hasattr(response, 'parts'):
                from upsonic.messages import TextPart
                text_parts = [p.content for p in response.parts if isinstance(p, TextPart)]
                return "".join(text_parts) if text_parts else "No response"
            return "No response"
        except Exception as e:
            culture_error(f"Admin culture task failed: {e}", debug=self.debug)
            raise
        finally:
            # Clear tools from model after use
            model._tools = None

    # =========================================================================
    # Context Building for Agents
    # =========================================================================

    def get_culture_context(self) -> str:
        """
        Build cultural knowledge context for agent system prompt.
        
        Returns:
            Formatted string of cultural knowledge for injection into system prompt
        """
        return self._run_async_from_sync(self.aget_culture_context())

    async def aget_culture_context(self) -> str:
        """Async version of get_culture_context."""
        culture_debug("Building culture context for system prompt", debug=self.debug)
        all_knowledge = await self.aget_all_knowledge()
        culture_debug(f"Including {len(all_knowledge)} knowledge entries in context", debug=self.debug)
        
        if not all_knowledge:
            return ""
        
        lines = ["## Cultural Knowledge"]
        lines.append("The following cultural knowledge should guide your responses:\n")
        
        for knowledge in all_knowledge:
            lines.append(f"### {knowledge.name or 'Unnamed'}")
            if knowledge.summary:
                lines.append(f"**Summary**: {knowledge.summary}")
            if knowledge.content:
                lines.append(f"**Content**: {knowledge.content}")
            if knowledge.categories:
                lines.append(f"**Categories**: {', '.join(knowledge.categories)}")
            if knowledge.notes:
                lines.append(f"**Notes**: {'; '.join(knowledge.notes)}")
            lines.append("")
        
        return "\n".join(lines)
