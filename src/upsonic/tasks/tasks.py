import base64
import time
from pydantic import BaseModel
from typing import Any, List, Dict, Optional, Type, Union, Callable, Literal, TYPE_CHECKING
from upsonic.exceptions import FileNotFoundError

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


CacheMethod = Literal["vector_search", "llm_call"]
CacheEntry = Dict[str, Any]

class Task(BaseModel):
    description: str
    attachments: Optional[List[str]] = None
    tools: list[Any] = None
    response_format: Union[Type[BaseModel], type[str], None] = str
    response_lang: Optional[str] = "en"
    _response: Any = None
    context: Any = None
    _context_formatted: Optional[str] = None
    price_id_: Optional[str] = None
    task_id_: Optional[str] = None
    not_main_task: bool = False
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    agent: Optional[Any] = None
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
    
    # Durable execution support
    durable_execution: Optional[Any] = None  # DurableExecution instance
    durable_checkpoint_enabled: bool = False

    # DeepAgent planning support
    _task_todos: Optional[List[Any]] = None  # List of Todo objects for task planning
    
    # Task-specific tool tracking (similar to Agent's registered_agent_tools)
    registered_task_tools: Dict[str, Any] = {}  # Maps tool names to wrapped Tool objects
    task_builtin_tools: List[Any] = []
        
    # Vector search parameters (override config defaults when provided)
    vector_search_top_k: Optional[int] = None
    vector_search_alpha: Optional[float] = None
    vector_search_fusion_method: Optional[Literal['rrf', 'weighted']] = None
    vector_search_similarity_threshold: Optional[float] = None
    vector_search_filter: Optional[Dict[str, Any]] = None

    @staticmethod
    def _is_file_path(item: Any) -> bool:
        """
        Check if an item is a valid file path.
        
        Args:
            item: Any object to check
            
        Returns:
            bool: True if the item is a string representing an existing file path
            
        Raises:
            FileNotFoundError: If the file path exists but cannot be accessed, or if it looks like a file path but doesn't exist
        """
        if not isinstance(item, str):
            return False
        
        import os
        
        # Check if it's a valid file path and the file exists
        try:
            if os.path.isfile(item):
                # Additional check to ensure file is readable
                if not os.access(item, os.R_OK):
                    raise FileNotFoundError(item, "File exists but is not readable")
                return True
            elif os.path.isdir(item):
                # It's a directory, not a file
                return False
            else:
                # Check if it looks like a file path but doesn't exist
                if (item.endswith(('.txt', '.pdf', '.docx', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv')) or 
                    ('/' in item or '\\' in item) and '.' in item):
                    raise FileNotFoundError(item, "File does not exist")
                return False
        except (TypeError, ValueError, OSError) as e:
            # If it's a string that looks like a file path but can't be accessed, raise error
            if isinstance(item, str) and (item.endswith(('.txt', '.pdf', '.docx', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv')) or '/' in item or '\\' in item):
                raise FileNotFoundError(item, f"Cannot access file: {str(e)}")
            return False
    
    @staticmethod
    def _is_folder_path(item: Any) -> bool:
        """
        Check if an item is a valid folder/directory path.
        
        Args:
            item: Any object to check
            
        Returns:
            bool: True if the item is a string representing an existing directory
            
        Raises:
            FileNotFoundError: If the folder path exists but cannot be accessed, or if it looks like a directory path but doesn't exist
        """
        if not isinstance(item, str):
            return False
        
        import os
        
        # Check if it's a valid directory path and the directory exists
        try:
            if os.path.isdir(item):
                # Additional check to ensure directory is readable
                if not os.access(item, os.R_OK):
                    raise FileNotFoundError(item, "Directory exists but is not readable")
                return True
            else:
                # Check if it looks like a directory path but doesn't exist
                # A path looks like a directory if it ends with / or \, or if it contains path separators
                if (item.endswith('/') or item.endswith('\\') or 
                    (('/' in item or '\\' in item) and not os.path.isfile(item))):
                    raise FileNotFoundError(item, "Directory does not exist")
                return False
        except (TypeError, ValueError, OSError) as e:
            # If it's a string that looks like a directory path but can't be accessed, raise error
            if isinstance(item, str) and (item.endswith('/') or item.endswith('\\') or '/' in item or '\\' in item):
                raise FileNotFoundError(item, f"Cannot access directory: {str(e)}")
            return False
    
    @staticmethod
    def _get_files_from_folder(folder_path: str) -> List[str]:
        """
        Recursively get all file paths from a folder.
        
        Args:
            folder_path: Path to the folder
            
        Returns:
            List[str]: List of all file paths in the folder and subfolders
            
        Raises:
            FileNotFoundError: If the folder cannot be accessed
        """
        import os
        
        files = []
        try:
            for root, dirs, filenames in os.walk(folder_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
        except (OSError, PermissionError) as e:
            # If we can't access the folder, raise a proper error
            raise FileNotFoundError(folder_path, f"Cannot access folder: {str(e)}")
        
        return files
    
    @staticmethod
    def _extract_files_from_context(context: Any) -> tuple[Any, List[str]]:
        """
        Extract file paths from context and return cleaned context and file list.
        Also handles folders by extracting all files from them recursively.
        
        Args:
            context: The context parameter (can be a list, dict, or any other type)
            
        Returns:
            tuple: (cleaned_context, extracted_files)
                - cleaned_context: Context with file/folder paths removed
                - extracted_files: List of file paths found (including files from folders)
                
        Raises:
            FileNotFoundError: If any file or folder in context cannot be accessed
        """
        extracted_files = []
        
        # If context is None or empty, return as is
        if not context:
            return context, extracted_files
        
        # Handle list context
        if isinstance(context, list):
            cleaned_context = []
            for item in context:
                try:
                    if Task._is_file_path(item):
                        extracted_files.append(item)
                    elif Task._is_folder_path(item):
                        # Extract all files from the folder
                        folder_files = Task._get_files_from_folder(item)
                        extracted_files.extend(folder_files)
                    else:
                        cleaned_context.append(item)
                except FileNotFoundError:
                    # Re-raise the exception with context
                    raise
            return cleaned_context, extracted_files
        
        # Handle dict context - check values
        elif isinstance(context, dict):
            cleaned_context = {}
            for key, value in context.items():
                try:
                    if Task._is_file_path(value):
                        extracted_files.append(value)
                    elif Task._is_folder_path(value):
                        # Extract all files from the folder
                        folder_files = Task._get_files_from_folder(value)
                        extracted_files.extend(folder_files)
                    elif isinstance(value, list):
                        # Recursively process lists in dict values
                        cleaned_list = []
                        for item in value:
                            try:
                                if Task._is_file_path(item):
                                    extracted_files.append(item)
                                elif Task._is_folder_path(item):
                                    # Extract all files from the folder
                                    folder_files = Task._get_files_from_folder(item)
                                    extracted_files.extend(folder_files)
                                else:
                                    cleaned_list.append(item)
                            except FileNotFoundError:
                                # Re-raise the exception with context
                                raise
                        cleaned_context[key] = cleaned_list
                    else:
                        cleaned_context[key] = value
                except FileNotFoundError:
                    # Re-raise the exception with context
                    raise
            return cleaned_context, extracted_files
        
        # Handle single string that might be a file path or folder
        try:
            if Task._is_file_path(context):
                extracted_files.append(context)
                return [], extracted_files
            elif Task._is_folder_path(context):
                # Extract all files from the folder
                folder_files = Task._get_files_from_folder(context)
                extracted_files.extend(folder_files)
                return [], extracted_files
        except FileNotFoundError:
            raise
        
        else:
            return context, extracted_files

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
        durable_execution: Optional[Any] = None,
        _task_todos: Optional[List[Any]] = None,
        vector_search_top_k: Optional[int] = None,
        vector_search_alpha: Optional[float] = None,
        vector_search_fusion_method: Optional[Literal['rrf', 'weighted']] = None,
        vector_search_similarity_threshold: Optional[float] = None,
        vector_search_filter: Optional[Dict[str, Any]] = None,
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
            
        if tools is None:
            tools = []
            
        if context is None:
            context = []

        if _tools_awaiting_external_execution is None:
            _tools_awaiting_external_execution = []
        
        try:
            context, extracted_files = self._extract_files_from_context(context)
        except FileNotFoundError as e:
            raise FileNotFoundError(e.file_path, f"File specified in context cannot be accessed: {e.reason}")
        
        if attachments is None:
            attachments = []
        
        if extracted_files:
            attachments.extend(extracted_files)
            
        super().__init__(**{
            "description": description,
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
            "_last_cache_entry": None,
            "durable_execution": durable_execution,
            "durable_checkpoint_enabled": durable_execution is not None,
            "_task_todos": _task_todos or [],
            "registered_task_tools": {},  # Initialize empty tool registry
            "vector_search_top_k": vector_search_top_k,
            "vector_search_alpha": vector_search_alpha,
            "vector_search_fusion_method": vector_search_fusion_method,
            "vector_search_similarity_threshold": vector_search_similarity_threshold,
            "vector_search_filter": vector_search_filter,
        })
        
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

    def add_tools(self, tools: Union[Any, List[Any]]) -> None:
        """
        Add tools to the task's tool list.
        
        This method simply adds tools to self.tools without processing them.
        Tools are processed at runtime when the agent executes the task.
        
        Note: If plan_and_execute is added explicitly, it will be treated as a
        regular tool (not auto-managed by enable_thinking_tool).
        
        Args:
            tools: A single tool or list of tools to add
        """
        if not isinstance(tools, list):
            tools = [tools]
        
        # Initialize self.tools if it's None
        if self.tools is None:
            self.tools = []
        
        # Add tools to self.tools
        for tool in tools:
            if tool not in self.tools:
                self.tools.append(tool)
    
    def remove_tools(self, tools: Union[str, List[str], Any, List[Any]], agent: Any) -> None:
        """
        Remove tools from the task.
        
        This method requires an agent instance because task tools are registered
        at runtime (not in __init__), so we need access to the agent's ToolManager
        to properly remove tools from all relevant data structures.
        
        Supports removing:
        - Tool names (strings)
        - Function objects
        - Agent objects
        - MCP handlers (and all their tools)
        - Class instances (ToolKit or regular classes, and all their tools)
        - Builtin tools (AbstractBuiltinTool instances)
        
        Args:
            tools: Single tool or list of tools to remove (any type)
            agent: Agent instance for accessing ToolManager
        """
        if not isinstance(tools, list):
            tools = [tools]
        
        # Separate builtin tools from regular tools
        from upsonic.tools.builtin_tools import AbstractBuiltinTool
        builtin_tools_to_remove = []
        regular_tools_to_remove = []
        
        for tool in tools:
            if tool is not None and isinstance(tool, AbstractBuiltinTool):
                builtin_tools_to_remove.append(tool)
            else:
                regular_tools_to_remove.append(tool)
        
        # Handle regular tools through ToolManager
        removed_tool_names = []
        removed_objects = []
        
        if regular_tools_to_remove:
            # Call ToolManager to handle all removal logic for regular tools
            # Pass self.registered_task_tools instead of agent.registered_agent_tools
            removed_tool_names, removed_objects = agent.tool_manager.remove_tools(
                tools=regular_tools_to_remove,
                registered_tools=self.registered_task_tools
            )
            
            # Update self.registered_task_tools - remove the tool names
            for tool_name in removed_tool_names:
                if tool_name in self.registered_task_tools:
                    del self.registered_task_tools[tool_name]
        
        # Handle builtin tools separately - they don't use ToolManager/ToolProcessor
        if builtin_tools_to_remove and hasattr(self, 'task_builtin_tools'):
            # Remove from task_builtin_tools by unique_id
            builtin_ids_to_remove = {tool.unique_id for tool in builtin_tools_to_remove}
            self.task_builtin_tools = [
                tool for tool in self.task_builtin_tools 
                if tool.unique_id not in builtin_ids_to_remove
            ]
            # Add to removed_objects for self.tools cleanup
            removed_objects.extend(builtin_tools_to_remove)
        
        # Update self.tools - remove all removed objects (regular + builtin)
        if self.tools and removed_objects:
            self.tools = [t for t in self.tools if t not in removed_objects]

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
    
    @property
    def durable_execution_id(self) -> Optional[str]:
        """
        Get the durable execution ID for this task.
        
        Returns:
            The execution ID if durable execution is enabled, None otherwise
        """
        if self.durable_execution:
            return self.durable_execution.execution_id
        return None
    
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
                    rag_result_objects = await context.query_async(self.description, task=self)
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
    def attachments_base64(self):
        """
        Convert all attachment files to base64 encoded strings.
        
        Base64 encoding works with any file type (images, PDFs, documents, etc.)
        and is commonly used for embedding binary data in text-based formats.
        
        Returns:
            List[str]: List of base64 encoded strings, one for each attachment
            None: If no attachments are present
        """
        if self.attachments is None:
            return None
        base64_attachments = []
        for attachment_path in self.attachments:
            try:
                with open(attachment_path, "rb") as attachment_file:
                    file_data = attachment_file.read()
                    base64_encoded = base64.b64encode(file_data).decode('utf-8')
                    base64_attachments.append(base64_encoded)
            except Exception as e:
                # Log the error but continue with other attachments
                from upsonic.utils.printing import warning_log
                warning_log(f"Could not encode attachment {attachment_path} to base64: {e}", "TaskProcessor")
        return base64_attachments


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
    def cache_hit(self) -> bool:
        """
        Check if the last response was retrieved from cache.
        
        Returns:
            bool: True if the response came from cache, False otherwise
        """
        return self._cache_hit

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
                from upsonic.utils.printing import warning_log
                warning_log(f"Could not load attachment {attachment_path}: {e}", "TaskProcessor")

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