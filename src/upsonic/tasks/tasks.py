import base64
import time
from pydantic import BaseModel
from typing import Any, List, Dict, Optional, Type, Union, Callable, Literal, TYPE_CHECKING

# Heavy imports moved to lazy loading for faster startup
if TYPE_CHECKING:
    from upsonic.utils.printing import get_price_id_total_cost
    from upsonic.messages.messages import BinaryContent
    from upsonic.schemas.data_models import RAGSearchResult
    from upsonic.tools import ExternalToolCall
    from upsonic.embeddings.factory import auto_detect_best_embedding
else:
    # Use string annotations to avoid importing heavy modules
    get_price_id_total_cost = "get_price_id_total_cost"
    BinaryContent = "BinaryContent"
    RAGSearchResult = "RAGSearchResult"
    ExternalToolCall = "ExternalToolCall"
    auto_detect_best_embedding = "auto_detect_best_embedding"

# Type aliases for better type safety
CacheMethod = Literal["vector_search", "llm_call"]
CacheEntry = Dict[str, Any]

class Task(BaseModel):
    description: str
    attachments: Optional[List[str]] = None
    tools: list[Any] = None
    response_format: Union[Type[BaseModel], type[str], None] = str
    response_lang: str = "en"
    _response: Any = None
    context: Any = None
    _context_formatted: Optional[str] = None
    price_id_: Optional[str] = None
    task_id_: Optional[str] = None
    not_main_task: bool = False
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    agent: Optional[Any] = None
    response_lang: Optional[str] = None
    enable_thinking_tool: Optional[bool] = None
    enable_reasoning_tool: Optional[bool] = None
    _tool_calls: List[Dict[str, Any]] = None
    guardrail: Optional[Callable] = None
    guardrail_retries: Optional[int] = None
    is_paused: bool = False
    _tools_awaiting_external_execution: List["ExternalToolCall"] = []
    
    enable_cache: bool = False
    cache_method: Literal["vector_search", "llm_call"] = "vector_search"
    cache_threshold: float = 0.7
    cache_embedding_provider: Optional[Any] = None
    cache_duration_minutes: int = 60
    _cache_manager: Optional[Any] = None  # Will be set by Agent
    _cache_hit: bool = False
    _original_input: Optional[str] = None
    _last_cache_entry: Optional[Dict[str, Any]] = None



    def __init__(
        self, 
        description: str, 
        attachments: Optional[List[str]] = None,
        tools: list[Any] = None,
        response_format: Union[Type[BaseModel], type[str], None] = str,
        response: Any = None,
        context: Any = None,
        _context_formatted: Optional[str] = None,
        price_id_: Optional[str] = None,
        task_id_: Optional[str] = None,
        not_main_task: bool = False,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent: Optional[Any] = None,
        response_lang: Optional[str] = None,
        enable_thinking_tool: Optional[bool] = None,
        enable_reasoning_tool: Optional[bool] = None,
        guardrail: Optional[Callable] = None,
        guardrail_retries: Optional[int] = None,
        is_paused: bool = False,
        _tools_awaiting_external_execution: List["ExternalToolCall"] = None,
        enable_cache: bool = False,
        cache_method: Literal["vector_search", "llm_call"] = "vector_search",
        cache_threshold: float = 0.7,
        cache_embedding_provider: Optional[Any] = None,
        cache_duration_minutes: int = 60,
        **data
    ):
        if guardrail is not None and not callable(guardrail):
            raise TypeError("The 'guardrail' parameter must be a callable function.")
        
        if cache_method not in ("vector_search", "llm_call"):
            raise ValueError("cache_method must be either 'vector_search' or 'llm_call'")
        
        if not (0.0 <= cache_threshold <= 1.0):
            raise ValueError("cache_threshold must be between 0.0 and 1.0")
        
        if enable_cache and cache_method == "vector_search" and cache_embedding_provider is None:
            try:
                from upsonic.embeddings.factory import auto_detect_best_embedding
                cache_embedding_provider = auto_detect_best_embedding()
            except Exception:
                raise ValueError("cache_embedding_provider is required when cache_method is 'vector_search'")
        
        if description is not None:
            data["description"] = description
            
        if tools is None:
            tools = []
            
        if context is None:
            context = []

        if _tools_awaiting_external_execution is None:
            _tools_awaiting_external_execution = []
            
        data.update({
            "attachments": attachments,
            "tools": tools,
            "response_format": response_format,
            "_response": response,
            "context": context,
            "_context_formatted": _context_formatted,
            "price_id_": price_id_,
            "task_id_": task_id_,
            "not_main_task": not_main_task,
            "start_time": start_time,
            "end_time": end_time,
            "agent": agent,
            "response_lang": response_lang,
            "enable_thinking_tool": enable_thinking_tool,
            "enable_reasoning_tool": enable_reasoning_tool,
            "guardrail": guardrail,
            "guardrail_retries": guardrail_retries,
            "_tool_calls": [],
            "is_paused": is_paused,
            "_tools_awaiting_external_execution": _tools_awaiting_external_execution,
            "enable_cache": enable_cache,
            "cache_method": cache_method,
            "cache_threshold": cache_threshold,
            "cache_embedding_provider": cache_embedding_provider,
            "cache_duration_minutes": cache_duration_minutes,
            "_cache_manager": None,  # Will be set by Agent
            "_cache_hit": False,
            "_original_input": description,
            "_last_cache_entry": None
        })
        
        super().__init__(**data)
        self.validate_tools()

    @property
    def duration(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def validate_tools(self):
        """
        Validates each tool in the tools list.
        If a tool is a class and has a __control__ method, runs that method to verify it returns True.
        Raises an exception if the __control__ method returns False or raises an exception.
        """
        if not self.tools:
            return
            
        for tool in self.tools:
            # Check if the tool is a class
            if isinstance(tool, type) or hasattr(tool, '__class__'):
                # Check if the class has a __control__ method
                if hasattr(tool, '__control__') and callable(getattr(tool, '__control__')):

                        control_result = tool.__control__()

    @property
    def context_formatted(self) -> Optional[str]:
        """
        Provides read-only access to the formatted context string.
        
        This property retrieves the value of the internal `_context_formatted`
        attribute, which is expected to be populated by a context management
        process before task execution.
        """
        return self._context_formatted

    @property
    def tools_awaiting_external_execution(self) -> List["ExternalToolCall"]:
        """
        Get the list of tool calls awaiting external execution.
        When the task is paused, this list should be iterated over,
        the tools executed, and the 'result' attribute of each item set.
        """
        return self._tools_awaiting_external_execution
    
    @context_formatted.setter
    def context_formatted(self, value: Optional[str]):
        """
        Sets the internal `_context_formatted` attribute.

        This allows an external process, like a ContextManager, to set the
        final formatted context string on the task object using natural
        attribute assignment syntax.

        Args:
            value: The formatted context string to be assigned.
        """
        self._context_formatted = value
    
    async def additional_description(self, client):
        if not self.context:
            return ""
        
        # Lazy import for heavy modules
        from upsonic.knowledge_base.knowledge_base import KnowledgeBase
            
        rag_results = []
        for context in self.context:
            
            # Lazy import KnowledgeBase to avoid heavy imports
            if hasattr(context, 'rag') and context.rag == True:
                # Import KnowledgeBase only when needed
                if isinstance(context, KnowledgeBase):
                    await context.setup_rag()
                    rag_result_objects = await context.query_async(self.description)
                    # Convert RAGSearchResult objects to formatted strings
                    if rag_result_objects:
                        formatted_results = []
                        for i, result in enumerate(rag_result_objects, 1):
                            cleaned_text = result.text.strip()
                            metadata_str = ""
                            if result.metadata:
                                source = result.metadata.get('source', 'Unknown')
                                page_number = result.metadata.get('page_number')
                                chunk_id = result.chunk_id or result.metadata.get('chunk_id')
                                
                                metadata_parts = [f"source: {source}"]
                                if page_number is not None:
                                    metadata_parts.append(f"page: {page_number}")
                                if chunk_id:
                                    metadata_parts.append(f"chunk_id: {chunk_id}")
                                if result.score is not None:
                                    metadata_parts.append(f"score: {result.score:.3f}")
                                
                                metadata_str = f" [metadata: {', '.join(metadata_parts)}]"
                            
                            formatted_results.append(f"[{i}]{metadata_str} {cleaned_text}")
                        
                        rag_results.extend(formatted_results)
                
        if rag_results:
            return f"The following is the RAG data: <rag>{' '.join(rag_results)}</rag>"
        return ""


    @property
    def images_base_64(self):
        if self.images is None:
            return None
        base_64_images = []
        for image in self.images:
            with open(image, "rb") as image_file:
                base_64_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
        return base_64_images

    @property
    def price_id(self):
        if self.price_id_ is None:
            import uuid
            self.price_id_ = str(uuid.uuid4())
        return self.price_id_

    @property
    def task_id(self):
        if self.task_id_ is None:
            import uuid
            self.task_id_ = str(uuid.uuid4())
        return self.task_id_
    
    def get_task_id(self):
        return f"Task_{self.task_id[:8]}"

    @property
    def response(self):

        if self._response is None:
            return None

        if type(self._response) == str:
            return self._response



        return self._response



    def get_total_cost(self):
        if self.price_id_ is None:
            return None
        # Lazy import for heavy modules
        from upsonic.utils.printing import get_price_id_total_cost
        return get_price_id_total_cost(self.price_id)
    
    @property
    def total_cost(self) -> Optional[float]:
        """
        Get the total estimated cost of this task.
        
        Returns:
            Optional[float]: The estimated cost in USD, or None if not available
        """
        the_total_cost = self.get_total_cost()
        if the_total_cost and "estimated_cost" in the_total_cost:
            return the_total_cost["estimated_cost"]
        return None
        
    @property
    def total_input_token(self) -> Optional[int]:
        """
        Get the total number of input tokens used by this task.
        
        Returns:
            Optional[int]: The number of input tokens, or None if not available
        """
        the_total_cost = self.get_total_cost()
        if the_total_cost and "input_tokens" in the_total_cost:
            return the_total_cost["input_tokens"]
        return None
        
    @property
    def total_output_token(self) -> Optional[int]:
        """
        Get the total number of output tokens used by this task.
        
        Returns:
            Optional[int]: The number of output tokens, or None if not available
        """
        the_total_cost = self.get_total_cost()
        if the_total_cost and "output_tokens" in the_total_cost:
            return the_total_cost["output_tokens"]
        return None

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """
        Get all tool calls made during this task's execution.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing information about tool calls,
            including tool name, parameters, and result.
        """
        if self._tool_calls is None:
            self._tool_calls = []
        return self._tool_calls
        
    def add_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """
        Add a tool call to the task's history.
        
        Args:
            tool_call (Dict[str, Any]): Dictionary containing information about the tool call.
                Should include 'tool_name', 'params', and 'tool_result' keys.
        """
        if self._tool_calls is None:
            self._tool_calls = []
        self._tool_calls.append(tool_call)



    def canvas_agent_description(self):
        return "You are a canvas agent. You have tools. You can edit the canvas and get the current text of the canvas."

    def add_canvas(self, canvas):
        # Check if canvas tools have already been added to prevent duplicates
        canvas_functions = canvas.functions()
        canvas_description = self.canvas_agent_description()
        
        # Check if canvas tools are already present
        canvas_already_added = False
        if canvas_functions:
            # Check if any of the canvas functions are already in tools
            for canvas_func in canvas_functions:
                if canvas_func in self.tools:
                    canvas_already_added = True
                    break
        
        # Only add canvas tools if they haven't been added before
        if not canvas_already_added:
            self.tools += canvas_functions
            
        # Check if canvas description is already in the task description
        if canvas_description not in self.description:
            self.description += canvas_description



    def task_start(self, agent):
        self.start_time = time.time()
        if agent.canvas:
            self.add_canvas(agent.canvas)


    def task_end(self):
        self.end_time = time.time()

    def task_response(self, model_response):
        self._response = model_response.output



    def build_agent_input(self):
        """
        Builds the input for the agent, using and then clearing the formatted context.
        """
        # Lazy import for heavy modules
        from upsonic.messages.messages import BinaryContent
        
        final_description = self.description
        if self.context_formatted and isinstance(self.context_formatted, str):
            final_description += "\n" + self.context_formatted

        self.context_formatted = None

        if not self.attachments:
            return final_description

        input_list = [final_description]
        
        for attachment_path in self.attachments:
            try:
                with open(attachment_path, "rb") as attachment_file:
                    attachment_data  = attachment_file.read()
                
                # Using mimetypes is more robust than just checking extensions
                import mimetypes
                media_type, _ = mimetypes.guess_type(attachment_path)
                if media_type is None:
                    media_type = "application/octet-stream" # Fallback
                    
                input_list.append(BinaryContent(data=attachment_data, media_type=media_type))
                
            except Exception as e:
                print(f"Warning: Could not load image {attachment_path}: {e}")

        return input_list

    
    def set_cache_manager(self, cache_manager: Any):
        """Set the cache manager for this task."""
        self._cache_manager = cache_manager
    
    async def get_cached_response(self, input_text: str, llm_provider: Optional[Any] = None) -> Optional[Any]:
        """
        Get cached response for the given input text.
        
        Args:
            input_text: The input text to search for in cache
            llm_provider: LLM provider for semantic comparison (for llm_call method)
            
        Returns:
            Cached response if found, None otherwise
        """
        if not self.enable_cache or not self._cache_manager:
            return None
        
        cached_response = await self._cache_manager.get_cached_response(
            input_text=input_text,
            cache_method=self.cache_method,
            cache_threshold=self.cache_threshold,
            duration_minutes=self.cache_duration_minutes,
            embedding_provider=self.cache_embedding_provider,
            llm_provider=llm_provider
        )
        
        if cached_response is not None:
            self._cache_hit = True
            self._last_cache_entry = {"output": cached_response}
        
        return cached_response
    
    async def store_cache_entry(self, input_text: str, output: Any):
        """
        Store a new cache entry.
        
        Args:
            input_text: The input text
            output: The corresponding output
        """
        if not self.enable_cache or not self._cache_manager:
            return
        
        await self._cache_manager.store_cache_entry(
            input_text=input_text,
            output=output,
            cache_method=self.cache_method,
            embedding_provider=self.cache_embedding_provider
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._cache_manager:
            return {
                "total_entries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "hit_rate": 0.0,
                "cache_method": self.cache_method,
                "cache_threshold": self.cache_threshold,
                "cache_duration_minutes": self.cache_duration_minutes,
                "session_id": None
            }
        
        stats = self._cache_manager.get_cache_stats()
        stats.update({
            "cache_method": self.cache_method,
            "cache_threshold": self.cache_threshold,
            "cache_duration_minutes": self.cache_duration_minutes,
            "cache_hit": self._cache_hit
        })
        
        return stats
    
    def clear_cache(self):
        """Clear all cache entries."""
        if self._cache_manager:
            self._cache_manager.clear_cache()
        self._cache_hit = False