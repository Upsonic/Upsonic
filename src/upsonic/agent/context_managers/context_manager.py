from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional, Dict, Any, List

from upsonic.utils.printing import warning_log, error_log

# Heavy imports moved to lazy loading for faster startup
if TYPE_CHECKING:
    from upsonic.agent.agent import Agent
    from upsonic.graph.graph import State
    from upsonic.agent.context_managers.memory_manager import MemoryManager
    from upsonic.tasks.tasks import Task
    from upsonic.knowledge_base.knowledge_base import KnowledgeBase
    from upsonic.context.task import turn_task_to_string
    from upsonic.context.sources import TaskOutputSource
    from upsonic.schemas.data_models import RAGSearchResult
else:
    # Use string annotations to avoid importing heavy modules
    Agent = "Agent"
    State = "State"
    MemoryManager = "MemoryManager"
    Task = "Task"
    KnowledgeBase = "KnowledgeBase"
    turn_task_to_string = "turn_task_to_string"
    TaskOutputSource = "TaskOutputSource"
    RAGSearchResult = "RAGSearchResult"


class ContextManager:
    """A context manager for building the dynamic, task-specific context prompt."""

    def __init__(self, agent: "Agent", task: "Task", state: Optional[State] = None):
        """Initializes the ContextManager."""
        self.agent = agent
        self.task = task
        self.state = state
        self.context_prompt: str = ""

    async def _build_context_prompt(self, memory_handler: Optional["MemoryManager"]) -> str:
        """Asynchronously builds the complete contextual prompt string."""
        # Lazy import for heavy modules
        from upsonic.tasks.tasks import Task
        from upsonic.knowledge_base.knowledge_base import KnowledgeBase
        from upsonic.context.task import turn_task_to_string
        from upsonic.context.sources import TaskOutputSource
        
        final_context_parts = []

        if memory_handler:
            context_injection = memory_handler.get_context_injection()
            if context_injection:
                final_context_parts.append(context_injection)

        if self.task.context:

            knowledge_base_parts = []
            task_parts = []
            previous_task_output_parts = []
            additional_parts = []

            for item in self.task.context:
                if isinstance(item, Task):  
                    task_parts.append(f"Task ID ({item.get_task_id()}): " + turn_task_to_string(item))
                
                elif isinstance(item, KnowledgeBase): 
                    await self._process_knowledge_base_item(
                        item, 
                        knowledge_base_parts, 
                        self.task.description
                    )

                elif isinstance(item, str):
                    additional_parts.append(item)

                elif isinstance(item, TaskOutputSource) and self.state:  
                    await self._process_task_output_source(
                        item, 
                        previous_task_output_parts
                    )

            if task_parts:
                final_context_parts.append("<Tasks>\n" + "\n".join(task_parts) + "\n</Tasks>")
            if knowledge_base_parts:
                final_context_parts.append("<Knowledge Base>\n" + "\n".join(knowledge_base_parts) + "\n</Knowledge Base>")
            if previous_task_output_parts:
                final_context_parts.extend(previous_task_output_parts)
            if additional_parts:
                final_context_parts.append("<Additional Context>\n" + "\n".join(additional_parts) + "\n</Additional Context>")

        if not final_context_parts:
            return ""
        
        return "<Context>\n" + "\n\n".join(final_context_parts) + "\n</Context>"

    async def _process_knowledge_base_item(
        self, 
        knowledge_base: "KnowledgeBase", 
        knowledge_base_parts: List[str], 
        query: str
    ) -> None:
        """Process a KnowledgeBase item."""
        try:
            if knowledge_base.rag:
                await knowledge_base.setup_rag()
                
                rag_results = await knowledge_base.query_async(query, task=self.task)
                
                if rag_results:
                    formatted_results = self._format_rag_results(rag_results, knowledge_base)
                    knowledge_base_parts.append(formatted_results)
                else:
                    warning_log(f"No results found for KnowledgeBase '{knowledge_base.name}' with query: '{query}'", context="ContextManager")
            else:
                knowledge_base_parts.append(knowledge_base.markdown())
                
        except Exception as e:
            error_log(f"Error processing KnowledgeBase '{knowledge_base.name}': {str(e)}", context="ContextManager")
            try:
                knowledge_base_parts.append(knowledge_base.markdown())
            except Exception as fallback_error:
                error_log(f"Fallback also failed for KnowledgeBase '{knowledge_base.name}': {str(fallback_error)}", context="ContextManager")

    def _format_rag_results(self, rag_results: List["RAGSearchResult"], knowledge_base: "KnowledgeBase") -> str:
        """Format RAG results with enhanced context and metadata."""
        if not rag_results:
            return ""
        
        kb_info = f"Source: {knowledge_base.name}"
        if hasattr(knowledge_base, 'get_config_summary'):
            try:
                config = knowledge_base.get_config_summary()
                vector_db_info = config.get('vectordb', {})
                if isinstance(vector_db_info, dict):
                    provider = vector_db_info.get('provider', 'Unknown')
                    kb_info += f" (Vector DB: {provider})"
            except Exception:
                pass
        
        formatted_chunks = []
        for i, result in enumerate(rag_results, 1):
            cleaned_text = result.text.strip()
            metadata_str = ""
            if result.metadata:
                source = result.metadata.get('source', 'Unknown')
                page_number = result.metadata.get('page_number', 'Unknown')
                chunk_id = result.chunk_id or result.metadata.get('chunk_id', 'Unknown')

                retrieved_keys = {'source', 'page_number', 'chunk_id', 'chunk'}
                metadata_parts = [f"source: {source}"]
                if page_number is not None:
                    metadata_parts.append(f"page: {page_number}")
                if chunk_id:
                    metadata_parts.append(f"chunk_id: {chunk_id}")
                if result.score is not None:
                    metadata_parts.append(f"score: {result.score:.3f}")

                for k, v in result.metadata.items():
                    if k not in retrieved_keys:
                        metadata_parts.append(f"{k}: {v}")

                metadata_str = f" [metadata: {', '.join(metadata_parts)}]"

            formatted_chunks.append(f"[{i}]{metadata_str} {cleaned_text}")

        return f"<rag source='{kb_info}'>{' '.join(formatted_chunks)}</rag>"

    async def _process_task_output_source(
        self, 
        item: "TaskOutputSource", 
        previous_task_output_parts: List[str]
    ) -> None:
        """Process a TaskOutputSource item with error handling."""
        try:
            source_output = self.state.get_task_output(item.task_description_or_id)
            
            if source_output is not None:
                output_str = self._format_task_output(source_output)
                
                previous_task_output_parts.append(
                    f"<PreviousTaskNodeOutput id='{item.task_description_or_id}'>\n{output_str}\n</PreviousTaskNodeOutput>"
                )
            else:
                warning_log(f"No output found for task '{item.task_description_or_id}'", context="ContextManager")
                
        except Exception as e:
            error_log(f"Error processing TaskOutputSource '{item.task_description_or_id}': {str(e)}", context="ContextManager")

    def _format_task_output(self, source_output: Any) -> str:
        """Format task output with serialization."""
        try:
            if hasattr(source_output, 'model_dump_json'):
                return source_output.model_dump_json(indent=2)
            elif hasattr(source_output, 'model_dump'):
                return json.dumps(source_output.model_dump(), default=str, indent=2)
            elif hasattr(source_output, 'to_dict'):
                return json.dumps(source_output.to_dict(), default=str, indent=2)
            elif hasattr(source_output, '__dict__'):
                return json.dumps(source_output.__dict__, default=str, indent=2)
            else:
                return str(source_output)
        except Exception as e:
            error_log(f"Error formatting task output: {str(e)}", context="ContextManager")
            return str(source_output)

    def get_context_prompt(self) -> str:
        """Public getter to retrieve the constructed context prompt."""
        return self.context_prompt

    @asynccontextmanager
    async def manage_context(self, memory_handler: Optional["MemoryManager"] = None):
        """The asynchronous context manager for building the task-specific context."""
        self.context_prompt = await self._build_context_prompt(memory_handler)

        self.task.context_formatted = self.context_prompt
            
        try:
            yield self
        finally:
            pass

    async def get_knowledge_base_health_status(self) -> Dict[str, Any]:
        """Get health status of all KnowledgeBase instances in the context."""
        # Lazy import for heavy modules
        from upsonic.knowledge_base.knowledge_base import KnowledgeBase
        
        health_status = {}
        
        if self.task.context:
            for item in self.task.context:
                if isinstance(item, KnowledgeBase):
                    try:
                        health_status[item.name] = await item.health_check_async()
                    except Exception as e:
                        health_status[item.name] = {
                            "healthy": False,
                            "error": str(e)
                        }
        
        return health_status

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current context configuration."""
        summary = {
            "task": {
                "id": self.task.get_task_id() if hasattr(self.task, 'get_task_id') else "unknown",
                "description": self.task.description,
                "attachments": self.task.attachments,
                "response_format": str(self.task.response_format) if self.task.response_format else "str",
                "response_lang": self.task.response_lang,
                "not_main_task": self.task.not_main_task,
                "start_time": self.task.start_time,
                "end_time": self.task.end_time,
                "duration": self.task.duration,
                "price_id": self.task.price_id,
                "total_cost": self.task.total_cost,
                "total_input_tokens": self.task.total_input_token,
                "total_output_tokens": self.task.total_output_token,
                "tool_calls_count": len(self.task.tool_calls) if self.task.tool_calls else 0
            },
            "context": {
                "items_count": len(self.task.context) if self.task.context else 0,
                "knowledge_bases": [],
                "tasks": [],
                "task_output_sources": [],
                "additional_contexts": 0,
                "context_formatted": self.task.context_formatted is not None
            },
            "agent": {
                "id": self.agent.agent_id,
                "name": self.agent.name,
                "debug": self.agent.debug,
                "retry": self.agent.retry,
                "mode": self.agent.mode,
                "show_tool_calls": self.agent.show_tool_calls,
                "tool_call_limit": self.agent.tool_call_limit,
                "enable_thinking_tool": self.agent.enable_thinking_tool,
                "enable_reasoning_tool": self.agent.enable_reasoning_tool,
                "has_memory": self.agent.memory is not None,
                "has_knowledge": self.agent.knowledge is not None,
                "has_canvas": self.agent.canvas is not None
            },
            "state": {
                "available": self.state is not None
            }
        }
        
        if self.task.context:
            # Lazy import for heavy modules
            from upsonic.knowledge_base.knowledge_base import KnowledgeBase
            from upsonic.tasks.tasks import Task
            from upsonic.context.sources import TaskOutputSource
            
            for item in self.task.context:
                if isinstance(item, KnowledgeBase):
                    kb_info = {
                        "name": item.name,
                        "type": "rag" if item.rag else "static",
                        "is_ready": getattr(item, '_is_ready', False),
                        "knowledge_id": getattr(item, 'knowledge_id', 'unknown'),
                        "sources_count": len(item.sources) if hasattr(item, 'sources') else 0
                    }
                    
                    # Get vector database configuration from KnowledgeBase
                    if hasattr(item, 'get_config_summary'):
                        try:
                            config_summary = item.get_config_summary()
                            kb_info["vector_db"] = config_summary.get('vectordb', {})
                        except Exception:
                            # Fallback: try to get provider name from vectordb attribute
                            if hasattr(item, 'vectordb'):
                                kb_info["vector_db"] = {"provider": item.vectordb.__class__.__name__}
                            else:
                                kb_info["vector_db"] = {"provider": "Unknown"}
                    
                    if hasattr(item, 'get_collection_info_async'):
                        try:
                            kb_info["collection_info_available"] = True
                        except Exception:
                            kb_info["collection_info_available"] = False
                    
                    summary["context"]["knowledge_bases"].append(kb_info)
                    
                elif isinstance(item, Task):
                    summary["context"]["tasks"].append({
                        "id": item.get_task_id() if hasattr(item, 'get_task_id') else "unknown",
                        "description": item.description,
                        "not_main_task": item.not_main_task,
                        "has_response": item.response is not None,
                        "has_attachments": bool(item.attachments),
                        "tools_count": len(item.tools) if item.tools else 0
                    })
                    
                elif isinstance(item, TaskOutputSource):
                    summary["context"]["task_output_sources"].append({
                        "task_id": item.task_description_or_id,
                        "retrieval_mode": item.retrieval_mode,
                        "enabled": item.enabled,
                        "source_id": item.source_id
                    })
                    
                elif isinstance(item, str):
                    summary["context"]["additional_contexts"] += 1
        
        return summary