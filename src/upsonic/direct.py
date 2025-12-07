from __future__ import annotations

import asyncio
from typing import Any, Optional, Union

from upsonic.models.settings import ModelSettings
from upsonic.tasks.tasks import Task
from upsonic.output import OutputObjectDefinition
from upsonic.profiles import ModelProfileSpec
from upsonic.providers import Provider


class Direct:
    """Simplified, high-speed interface for LLM interactions.
    
    This class provides a streamlined way to interact with LLMs without
    the complexity of memory, knowledge base, or tool calls. It focuses
    on maximum speed and direct data retrieval.
    
    Example:
        ```python
        from upsonic import Direct, Task
        from pydantic import BaseModel
        
        my_direct = Direct(model="openai/gpt-4o")
        
        class MyResponse(BaseModel):
            tax_number: str
        
        my_task = Task(
            "Read the paper and return me the tax number", 
            context=["my.pdf", "my.png"], 
            response_format=MyResponse
        )
        
        result = my_direct.do(my_task)
        print(result)
        ```
    """
    
    def __init__(
        self,
        model: Union[str, Any, None] = None,
        *,
        settings: Optional[ModelSettings] = None,
        profile: Optional[ModelProfileSpec] = None,
        provider: Optional[Union[str, Provider]] = None
    ):
        """Initialize the Direct instance.
        
        Args:
            model: Model name (e.g., "openai/gpt-4o"), Model instance, or None
            settings: Optional model settings
            profile: Optional model profile
            provider: Optional provider name or Provider instance
        """
        self._model = None
        self._settings = settings
        self._profile = profile
        self._provider = provider
        
        if model is not None:
            self._set_model(model)
    
    def _set_model(self, model: Union[str, Any]) -> None:
        """Set the model for this Direct instance."""
        if isinstance(model, str):
            from upsonic.models import infer_model
            self._model = infer_model(model)
        elif hasattr(model, 'request'):  # Check if it's a Model-like object
            self._model = model
        else:
            raise ValueError(f"Invalid model type: {type(model)}")
    
    def with_model(self, model: Union[str, Any]) -> "Direct":
        """Create a new Direct instance with the specified model.
        
        Args:
            model: Model name or Model instance
            
        Returns:
            New Direct instance with the specified model
        """
        new_direct = Direct(
            settings=self._settings,
            profile=self._profile,
            provider=self._provider
        )
        new_direct._set_model(model)
        return new_direct
    
    def with_settings(self, settings: ModelSettings) -> "Direct":
        """Create a new Direct instance with the specified settings.
        
        Args:
            settings: Model settings
            
        Returns:
            New Direct instance with the specified settings
        """
        new_direct = Direct(
            model=self._model,
            settings=settings,
            profile=self._profile,
            provider=self._provider
        )
        return new_direct
    
    def with_profile(self, profile: ModelProfileSpec) -> "Direct":
        """Create a new Direct instance with the specified profile.
        
        Args:
            profile: Model profile
            
        Returns:
            New Direct instance with the specified profile
        """
        new_direct = Direct(
            model=self._model,
            settings=self._settings,
            profile=profile,
            provider=self._provider
        )
        return new_direct
    
    def with_provider(self, provider: Union[str, Provider]) -> "Direct":
        """Create a new Direct instance with the specified provider.
        
        Args:
            provider: Provider name or Provider instance
            
        Returns:
            New Direct instance with the specified provider
        """
        new_direct = Direct(
            model=self._model,
            settings=self._settings,
            profile=self._profile,
            provider=provider
        )
        return new_direct
    
    def _prepare_model(self) -> Any:
        """Prepare the model for use, creating one if necessary."""
        if self._model is None:
            # Use default model if none specified
            from upsonic.models import infer_model
            self._model = infer_model("openai/gpt-4o")
        
        # Apply settings and profile if provided
        if self._settings is not None:
            self._model._settings = self._settings
        
        if self._profile is not None:
            self._model._profile = self._profile
        
        return self._model
    
    def _build_messages_from_task(self, task: Task) -> list:
        """Build messages from a Task object."""
        from upsonic.messages import ModelRequest, UserPromptPart, BinaryContent
        import mimetypes
        
        # Start with the task description
        user_part = UserPromptPart(content=task.description)
        parts = [user_part]
        
        # Add attachments if present
        if task.attachments:
            for attachment_path in task.attachments:
                try:
                    with open(attachment_path, "rb") as attachment_file:
                        attachment_data = attachment_file.read()
                    
                    # Determine media type
                    media_type, _ = mimetypes.guess_type(attachment_path)
                    if media_type is None:
                        media_type = "application/octet-stream"
                    
                    parts.append(BinaryContent(data=attachment_data, media_type=media_type))
                except Exception as e:
                    print(f"Warning: Could not load attachment {attachment_path}: {e}")
        
        return [ModelRequest(parts=parts)]
    
    def _build_request_parameters(self, task: Task):
        """Build model request parameters from task."""
        from upsonic.models import ModelRequestParameters
        from pydantic import BaseModel
        
        # Handle response format
        output_mode = 'text'
        output_object = None
        allow_text_output = True
        
        if task.response_format and task.response_format != str and task.response_format is not str:
            if isinstance(task.response_format, type) and issubclass(task.response_format, BaseModel):
                output_mode = 'native'
                allow_text_output = False
                
                schema = task.response_format.model_json_schema()
                output_object = OutputObjectDefinition(
                    json_schema=schema,
                    name=task.response_format.__name__,
                    description=task.response_format.__doc__,
                    strict=True
                )
        
        return ModelRequestParameters(
            function_tools=[],
            builtin_tools=[],
            output_mode=output_mode,
            output_object=output_object,
            allow_text_output=allow_text_output
        )
    
    def _extract_output(self, response, task: Task) -> Any:
        """Extract output from model response."""
        from upsonic.messages import TextPart, FilePart, BinaryImage
        
        # Check for image outputs first
        image_parts = [
            part.content for part in response.parts 
            if isinstance(part, FilePart) and isinstance(part.content, BinaryImage)
        ]
        
        if image_parts:
            # If there are multiple images, return a list; if single, return the image data
            if len(image_parts) == 1:
                return image_parts[0].data
            else:
                return [img.data for img in image_parts]
        
        # Extract text parts
        text_parts = [
            part.content for part in response.parts 
            if isinstance(part, TextPart)
        ]
        
        if task.response_format == str or task.response_format is str:
            return "".join(text_parts)
        
        text_content = "".join(text_parts)
        
        if task.response_format and text_content:
            try:
                import json
                parsed = json.loads(text_content)
                if hasattr(task.response_format, 'model_validate'):
                    return task.response_format.model_validate(parsed)
                return parsed
            except Exception:
                # If parsing fails, return as text
                pass
        
        # Default: return as string
        return text_content
    
    def do(self, task: Task, show_output: bool = True) -> Any:
        """Execute a task synchronously.
        
        Args:
            task: Task object containing description, context, and response format
            show_output: Whether to show visual output (default: True)
            
        Returns:
            The model's response (extracted output)
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.do_async(task, show_output=show_output))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.do_async(task, show_output=show_output))
    
    async def do_async(self, task: Task, show_output: bool = True) -> Any:
        """Execute a task asynchronously.
        
        Args:
            task: Task object containing description, context, and response format
            show_output: Whether to show visual output (default: True)
            
        Returns:
            The model's response (extracted output)
        """
        import time
        
        model = self._prepare_model()
        
        # Get response format name
        response_format_name = "str"
        if task.response_format and task.response_format != str:
            if hasattr(task.response_format, '__name__'):
                response_format_name = task.response_format.__name__
            else:
                response_format_name = str(task.response_format)
        
        # Show start message
        if show_output:
            from upsonic.utils.printing import direct_started
            direct_started(
                model_name=model.model_name,
                task_description=task.description,
                response_format=response_format_name
            )
        
        start_time = time.time()
        
        try:
            # Build messages from task
            messages = self._build_messages_from_task(task)
            
            # Build request parameters
            model_params = self._build_request_parameters(task)
            model_params = model.customize_request_parameters(model_params)
            
            # Make the request
            response = await model.request(
                messages=messages,
                model_settings=model.settings,
                model_request_parameters=model_params
            )
            
            end_time = time.time()
            
            # Extract output
            result = self._extract_output(response, task)
            
            # Show completion message with metrics
            if show_output:
                from upsonic.utils.printing import direct_completed
                
                # Get usage information
                usage = {
                    'input_tokens': response.usage.input_tokens if hasattr(response, 'usage') and response.usage else 0,
                    'output_tokens': response.usage.output_tokens if hasattr(response, 'usage') and response.usage else 0
                }
                
                direct_completed(
                    result=result,
                    model=model,
                    response_format=response_format_name,
                    start_time=start_time,
                    end_time=end_time,
                    usage=usage,
                    debug=False,
                    task_description=task.description
                )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            
            if show_output:
                from upsonic.utils.printing import direct_error
                direct_error(
                    error_message=str(e),
                    model_name=model.model_name if model else None,
                    task_description=task.description,
                    execution_time=end_time - start_time
                )
            raise
    
    def print_do(self, task: Task) -> Any:
        """Execute a task synchronously and print the result with visual output.
        
        Args:
            task: Task object containing description, context, and response format
            
        Returns:
            The model's response (extracted output)
        """
        # show_output is True by default in do() method
        return self.do(task, show_output=True)
    
    async def print_do_async(self, task: Task) -> Any:
        """Execute a task asynchronously and print the result with visual output.
        
        Args:
            task: Task object containing description, context, and response format
            
        Returns:
            The model's response (extracted output)
        """
        # show_output is True by default in do_async() method
        return await self.do_async(task, show_output=True)
    
    @property
    def model(self) -> Optional[Any]:
        """Get the current model."""
        return self._model
    
    @property
    def settings(self) -> Optional[ModelSettings]:
        """Get the current settings."""
        return self._settings
    
    @property
    def profile(self) -> Optional[ModelProfileSpec]:
        """Get the current profile."""
        return self._profile
    
    @property
    def provider(self) -> Optional[Union[str, Provider]]:
        """Get the current provider."""
        return self._provider
