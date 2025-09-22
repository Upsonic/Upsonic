from __future__ import annotations

import base64
import asyncio
import mimetypes
import io
import json
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
)


# Import the official OpenAI library and its error types
import httpx
from openai import APIStatusError, AsyncOpenAI, NOT_GIVEN
from openai.types import chat, responses
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai.types.chat.chat_completion_content_part_param import File, FileFile
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from openai.types.responses import (
    Response,
    ResponseCodeInterpreterToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseFunctionWebSearch,
    ResponseInputFileParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputMessageItem,
    ResponseInputTextParam,
    ResponseFunctionToolCallItem,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from pydantic import Field

from upsonic.models.settings import BaseModelSettings
from upsonic.messages.streaming import (
    FinalResultEvent,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)

# Import all necessary message models from your framework
from upsonic.messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
    TokenUsage,
)
from upsonic.messages.base import ModelResponsePart
from upsonic.models.base import ModelProvider, ModelSettings
from upsonic.messages.types import FinishReason

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
    )
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChatCompletionChunk
    from openai.types.responses import ResponseStreamEvent
    from upsonic.messages import ModelRequest
    from upsonic.messages.streaming import ModelResponseStreamEvent

# ==============================================================================
# 1. OpenAI-Specific Settings Class (from Step 2)
# ==============================================================================

class OpenAIModelSettings(BaseModelSettings):
    """
    A comprehensive model for all OpenAI-specific settings.

    This class inherits from BaseModelSettings and adds parameters that are unique
    to the OpenAI API, covering both the Chat Completions and the Responses APIs.
    """
    seed: Optional[int] = Field(
        default=None,
        description="For reproducible outputs, the model will make a best effort to sample deterministically.",
    )
    logprobs: Optional[bool] = Field(
        default=None, description="Whether to return log probabilities of the output tokens."
    )
    top_logprobs: Optional[int] = Field(
        default=None,
        description="An integer between 0 and 5 specifying the number of most likely tokens to return at each token position.",
        ge=0,
        le=5,
    )
    user: Optional[str] = Field(
        default=None,
        description="A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.",
    )
    response_format: Optional[Dict[str, str]] = Field(
        default=None,
        description="An object specifying the format that the model must output, e.g., {'type': 'json_object'}.",
    )
    tool_choice: Optional[str | Dict] = Field(
        default=None,
        description="Controls which function is called by the model. e.g., 'none', 'auto', or a specific function.",
    )
    parallel_tool_calls: Optional[bool] = Field(
        default=None,
        description="Whether to enable parallel function calling.",
    )
    service_tier: Optional[Literal['auto', 'default']] = Field(
        default=None,
        description="Specifies the latency tier to use for processing the request."
    )
    
    openai_reasoning_effort: Optional[Literal["low", "medium", "high", "auto"]] = Field(
        default=None,
        description="[Responses API only] Constrains the effort on reasoning. Lower effort can result in faster responses."
    )
    openai_truncation: Optional[Literal["disabled", "auto"]] = Field(
        default=None,
        description="[Responses API only] Truncation strategy if the context exceeds the window size."
    )
    openai_previous_response_id: Optional[str] = Field(
        default=None,
        description="[Responses API only] The ID of a previous response to continue a conversation statefully."
    )
    openai_text_verbosity: Optional[Literal['low', 'medium', 'high']] = Field(
        default=None,
        description="[Responses API only] Constrains the verbosity of the model's text response."
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="A list of tool definitions the model may call. Each tool should be a JSON Schema object."
    )


# ==============================================================================
# 2. OpenAI Provider Class
# ==============================================================================

class OpenAIProvider(ModelProvider):
    """
    A comprehensive provider for interacting with OpenAI's APIs.

    This class implements the ModelProvider interface to act as a bridge between
    the agent framework and OpenAI. It supports both the Chat Completions API
    and the newer Responses API, handles streaming, batching, and provides
    dedicated methods for multi-modal generation (images, audio).
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_mode: Literal["chat", "responses"] = "chat",
        default_settings: OpenAIModelSettings = OpenAIModelSettings(),
        **kwargs,
    ):
        """
        Initializes the OpenAIProvider.

        Args:
            model_name: The name of the model to use.
            api_key: Your OpenAI API key.
            base_url: Can be used to point to a custom endpoint (e.g., for Azure).
            api_mode: The OpenAI API to target. 'chat' for Chat Completions,
                      'responses' for the newer Responses API.
            default_settings: Default settings to apply to all requests.
            **kwargs: Additional options to pass to the AsyncOpenAI client.
        """
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)
        self.api_mode = api_mode
        self.default_settings = default_settings
        # Don't create a persistent http client to avoid event loop issues
        self._http_client = None

    # --- Core Method Implementations (Stubs for Step 4) ---

    async def run_async(
        self, request: ModelRequest, settings: Optional[ModelSettings] = None
    ) -> ModelResponse:
        """
        Asynchronously executes a single, non-streaming request.
        """
        if settings is None:
            final_settings = self.default_settings
        elif isinstance(settings, dict):
            merged_settings = self.default_settings.model_copy(update=settings)
            final_settings = OpenAIModelSettings.model_validate(merged_settings)
        else:
            # settings is already a ModelSettings object
            final_settings = settings

        try:
            if self.api_mode == "chat":
                params = await self._prepare_chat_request(request, final_settings)
                openai_response = await self.client.chat.completions.create(
                    model=self.model_name, **params
                )
                return self._parse_chat_response(openai_response)
            elif self.api_mode == "responses":
                params = await self._prepare_responses_request(request, final_settings)
                openai_response = await self.client.responses.create(
                    model=self.model_name, **params
                )
                return self._parse_responses_api_response(openai_response)
            else:
                raise ValueError(f"Unknown api_mode: {self.api_mode}")
        except APIStatusError as e:
            # Re-raise as a framework-specific error in a real implementation
            print(f"OpenAI API Error: {e.status_code} - {e.response}")
            raise e

    async def run_stream_async(
        self, request: ModelRequest, settings: Optional[ModelSettings] = None
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        """
        Asynchronously executes a streaming request.
        """
        if settings is None:
            final_settings = self.default_settings
        elif isinstance(settings, dict):
            merged_settings = self.default_settings.model_copy(update=settings)
            final_settings = OpenAIModelSettings.model_validate(merged_settings)
        else:
            # settings is already a ModelSettings object
            final_settings = settings

        try:
            if self.api_mode == "chat":
                params = await self._prepare_chat_request(request, final_settings)
                params["stream"] = True
                stream = await self.client.chat.completions.create(
                    model=self.model_name, **params
                )
                async for event in self._process_chat_stream(stream):
                    yield event
            elif self.api_mode == "responses":
                params = await self._prepare_responses_request(request, final_settings)
                params["stream"] = True
                stream = await self.client.responses.create(
                    model=self.model_name, **params
                )
                async for event in self._process_responses_stream(stream):
                    yield event
            else:
                raise ValueError(f"Unknown api_mode: {self.api_mode}")
        except APIStatusError as e:
            print(f"OpenAI API Error: {e.status_code} - {e.response}")
            raise e

    async def run_batch_async(
        self, requests: List[ModelRequest], settings: Optional[ModelSettings] = None
    ) -> List[ModelResponse]:
        """
        Asynchronously executes a batch of requests using the OpenAI Batch API.

        This method follows the required four-step process:
        1. Creates and uploads a JSONL file with all requests.
        2. Creates a batch processing job.
        3. Polls the job status until completion.
        4. Downloads, parses, and returns the results.
        
        Note: Polling is used for simplicity. For long-running production jobs,
        a non-blocking webhook or separate monitoring task is recommended.
        """
        if self.api_mode != "chat":
            raise NotImplementedError("Batch processing is only supported for 'chat' api_mode.")

        # --- 1. Prepare and Upload Input File ---
        jsonl_data = await self._prepare_batch_jsonl(requests, settings)
        
        batch_input_file = await self.client.files.create(
            file=io.BytesIO(jsonl_data.encode("utf-8")), purpose="batch"
        )

        # --- 2. Create Batch Job ---
        batch_job = await self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        # --- 3. Poll for Results ---
        while batch_job.status not in ["completed", "failed", "cancelled"]:
            await asyncio.sleep(10)  # Poll every 10 seconds
            batch_job = await self.client.batches.retrieve(batch_job.id)

        if batch_job.status != "completed":
            raise RuntimeError(f"Batch job failed or was cancelled. Status: {batch_job.status}")

        # --- 4. Retrieve and Parse Output ---
        output_file_id = batch_job.output_file_id
        if not output_file_id:
            raise RuntimeError("Batch job completed but returned no output file.")
            
        result_content = await self.client.files.content(output_file_id)
        result_text = result_content.read().decode("utf-8")
        result_lines = result_text.strip().split("\n")

        # Map results back to original requests using custom_id
        responses_map: Dict[str, ModelResponse] = {}
        for line in result_lines:
            result = json.loads(line)
            custom_id = result.get("custom_id")
            response_body = result.get("response", {}).get("body")
            
            if custom_id and response_body:
                # Re-create the ChatCompletion object to use our existing parser
                chat_completion = ChatCompletion.model_validate(response_body)
                responses_map[custom_id] = self._parse_chat_response(chat_completion)
        
        # Return responses in the same order as the original requests
        ordered_responses = [responses_map[f"request-{i}"] for i in range(len(requests))]
        return ordered_responses

    async def generate_image_async(
        self, prompt: str, **kwargs: Any
    ) -> Any:
        """
        Generates an image using a DALL-E model.

        Args:
            prompt: The text prompt for the image.
            **kwargs: Additional parameters for the Images API, such as
                      'model' (e.g., "dall-e-3"), 'n', 'size', 'quality', 'style'.

        Returns:
            The full ImagesResponse object from the OpenAI client, containing
            URLs and/or base64-encoded image data.
        """
        # Remove model from kwargs to avoid duplicate parameter
        kwargs_copy = kwargs.copy()
        model = kwargs_copy.pop("model", "dall-e-3")
        
        return await self.client.images.generate(
            prompt=prompt, model=model, **kwargs_copy
        )

    async def generate_speech_async(self, text: str, **kwargs: Any) -> bytes:
        """
        Generates audio from text using a Text-to-Speech (TTS) model.

        Args:
            text: The text to convert to speech.
            **kwargs: Additional parameters for the Audio API, such as
                      'model' (e.g., "tts-1"), 'voice' (e.g., "alloy").

        Returns:
            The raw audio data as bytes (e.g., in MP3 format).
        """
        # Remove voice from kwargs to avoid duplicate parameter
        kwargs_copy = kwargs.copy()
        voice = kwargs_copy.pop("voice", "alloy")
        model = kwargs_copy.pop("model", "tts-1")
        
        response = await self.client.audio.speech.create(
            input=text,
            model=model,
            voice=voice,
            **kwargs_copy,
        )
        return await response.aread()

    async def transcribe_audio_async(
        self, audio_data: bytes, filename: str, **kwargs: Any
    ) -> str:
        """
        Transcribes audio to text using the Whisper model.

        Args:
            audio_data: The raw audio data as bytes.
            filename: The name of the file including its extension (e.g., "speech.mp3").
                      The extension is crucial for the API to process the format correctly.
            **kwargs: Additional parameters for the Transcriptions API, such as
                      'model' (e.g., "whisper-1"), 'language', 'prompt'.

        Returns:
            The transcribed text as a string.
        """
        file_tuple = (filename, audio_data)
        # Remove model from kwargs to avoid duplicate parameter
        kwargs_copy = kwargs.copy()
        model = kwargs_copy.pop("model", "whisper-1")
        
        transcription = await self.client.audio.transcriptions.create(
            model=model, file=file_tuple, **kwargs_copy
        )
        return transcription.text

    # --- Private Helper for Batching ---

    async def _prepare_batch_jsonl(
        self, requests: List[ModelRequest], settings: Optional[ModelSettings] = None
    ) -> str:
        """
        Converts a list of ModelRequests into the JSONL format required by the Batch API.
        """
        jsonl_lines = []
        for i, request in enumerate(requests):
            # We use the existing chat request preparer to build the body
            final_settings = self.default_settings.model_copy(update=settings or {})
            body_params = await self._prepare_chat_request(request, final_settings)
            
            # The Batch API requires the model to be inside the body
            body_params["model"] = self.model_name
            
            line = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body_params,
            }
            jsonl_lines.append(json.dumps(line))
        
        return "\n".join(jsonl_lines)

    # ==============================================================================
    # 3. Core Logic Implementation (THIS IS STEP 3)
    # ==============================================================================

    async def _prepare_chat_request(
        self, request: "ModelRequest", settings: OpenAIModelSettings
    ) -> Dict[str, Any]:
        """
        Translates a framework ModelRequest to the dictionary of parameters
        required by the OpenAI Chat Completions API.
        """
        # --- 1. Map Messages ---
        openai_messages: List[chat.ChatCompletionMessageParam] = []
        for message in request.parts:
            if isinstance(message, SystemPromptPart):
                openai_messages.append(
                    chat.ChatCompletionSystemMessageParam(
                        role="system", content=message.content
                    )
                )
            elif isinstance(message, ToolReturnPart):
                openai_messages.append(
                    chat.ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=message.tool_call_id,
                        content=str(message.content),
                    )
                )
            elif isinstance(message, RetryPromptPart):
                if message.tool_name:
                    openai_messages.append(
                        chat.ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=message.tool_call_id,
                            content=str(message.content),
                        )
                    )
                else:
                    openai_messages.append(
                        chat.ChatCompletionUserMessageParam(
                            role="user", content=str(message.content)
                        )
                    )
            elif isinstance(message, UserPromptPart):
                openai_messages.append(
                    await self._map_chat_user_prompt(message)
                )
            # BuiltinToolReturnPart is handled by the framework and not sent back
            elif isinstance(message, BuiltinToolReturnPart):
                pass

            elif isinstance(message, ModelResponse):
                assistant_parts = {"content": "", "tool_calls": []}
                for part in message.parts:
                    if isinstance(part, TextPart):
                        assistant_parts["content"] += part.content
                    elif isinstance(part, ToolCallPart):
                        assistant_parts["tool_calls"].append(
                            ChatCompletionMessageToolCallParam(
                                id=part.tool_call_id,
                                type="function",
                                function={
                                    "name": part.tool_name,
                                    "arguments": json.dumps(part.args)
                                }
                            )
                        )
                # Add the reconstructed assistant message to the history
                if assistant_parts["content"] or assistant_parts["tool_calls"]:
                     openai_messages.append(
                         chat.ChatCompletionAssistantMessageParam(
                             role="assistant",
                             content=assistant_parts["content"] or None,
                             tool_calls=assistant_parts["tool_calls"] or None
                         )
                     )
            else:
                raise TypeError(f"Unhandled request part type: {type(part)}")

        # --- 2. Map Tools (if any) ---
        # This assumes you have a ToolDefinition model in your framework
        # For this example, we'll assume tools are passed via settings
        tools = settings.model_dump().get("tools", NOT_GIVEN)
        
        # --- 3. Combine Settings ---
        # This merges base settings with OpenAI-specific ones
        request_params = settings.model_dump(exclude_none=True)

        params: Dict[str, Any] = {
            "messages": openai_messages,
        }
        
        # Only add parameters that are not None
        if tools is not NOT_GIVEN:
            params["tools"] = tools
        if request_params.get("temperature") is not None:
            params["temperature"] = request_params["temperature"]
        if request_params.get("max_tokens") is not None:
            params["max_tokens"] = request_params["max_tokens"]
        if request_params.get("top_p") is not None:
            params["top_p"] = request_params["top_p"]
        if request_params.get("presence_penalty") is not None:
            params["presence_penalty"] = request_params["presence_penalty"]
        if request_params.get("frequency_penalty") is not None:
            params["frequency_penalty"] = request_params["frequency_penalty"]
        if request_params.get("stop_sequences") is not None:
            params["stop"] = request_params["stop_sequences"]
        if request_params.get("timeout") is not None:
            params["timeout"] = request_params["timeout"]
        if request_params.get("seed") is not None:
            params["seed"] = request_params["seed"]
        if request_params.get("logprobs") is not None:
            params["logprobs"] = request_params["logprobs"]
        if request_params.get("top_logprobs") is not None:
            params["top_logprobs"] = request_params["top_logprobs"]
        if request_params.get("user") is not None:
            params["user"] = request_params["user"]
        if request_params.get("response_format") is not None:
            params["response_format"] = request_params["response_format"]
        if request_params.get("tool_choice") is not None:
            params["tool_choice"] = request_params["tool_choice"]
        if request_params.get("parallel_tool_calls") is not None:
            params["parallel_tool_calls"] = request_params["parallel_tool_calls"]
        if request_params.get("service_tier") is not None:
            params["service_tier"] = request_params["service_tier"]
        if request_params.get("extra_body") is not None:
            params["extra_body"] = request_params["extra_body"]
        return params

    def _parse_chat_response(
        self, openai_response: ChatCompletion
    ) -> "ModelResponse":
        """
        Translates a ChatCompletion object from the OpenAI library
        into your framework's ModelResponse.
        """
        choice = openai_response.choices[0]
        message = choice.message

        finish_reason = self._map_chat_finish_reason(choice.finish_reason)
        usage = TokenUsage(
            input_tokens=openai_response.usage.prompt_tokens if openai_response.usage else 0,
            output_tokens=openai_response.usage.completion_tokens if openai_response.usage else 0,
        )

        parts: List[ModelResponsePart] = []
        if message.content:
            parts.append(TextPart(content=message.content))

        if message.tool_calls:
            for tool_call in message.tool_calls:
                parts.append(
                    self._parse_tool_call(tool_call)
                )
        
        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=openai_response.model,
            provider_name="openai",
            provider_response_id=openai_response.id,
            finish_reason=finish_reason,
            provider_details={"system_fingerprint": openai_response.system_fingerprint},
        )

    async def _prepare_responses_request(
        self, request: "ModelRequest", settings: OpenAIModelSettings
    ) -> Dict[str, Any]:
        """
        Translates a framework ModelRequest, including history, to the dictionary
        of parameters required by the OpenAI Responses API.
        """
        instructions = request.instructions
        input_items: List[ResponseInputItemParam] = []

        for message in request.parts:
            # --- Handle User Input & Tool Returns ---
            if isinstance(message, (SystemPromptPart, ToolReturnPart, UserPromptPart)):
                 if isinstance(message, SystemPromptPart):
                    input_items.append(ResponseInputMessageItem(id=f"msg_{uuid.uuid4().hex}", role="system", content=[ResponseInputTextParam(text=message.content, type="input_text")]))
                 elif isinstance(message, ToolReturnPart):
                    input_items.append(ResponseFunctionToolCallOutputItem(id=f"output_{uuid.uuid4().hex}", type="function_call_output", call_id=message.tool_call_id, output=str(message.content)))
                 elif isinstance(message, UserPromptPart):
                    input_items.append(await self._map_responses_user_prompt(message))

            # --- HANDLE CHAT HISTORY (ModelResponse objects) ---
            elif isinstance(message, ModelResponse):
                # Re-hydrate the previous ModelResponse into the format the Responses API expects
                for part in message.parts:
                    if isinstance(part, TextPart):
                        input_items.append(ResponseOutputMessage(id=f"msg_{uuid.uuid4().hex}", role="assistant", content=[{"type": "output_text", "text": part.content}]))
                    elif isinstance(part, ThinkingPart):
                        input_items.append(ResponseReasoningItem(id=f"rsn_{uuid.uuid4().hex}", type="reasoning", summary=[{"type": "summary_text", "text": part.content}]))
                    elif isinstance(part, ToolCallPart):
                        input_items.append(ResponseFunctionToolCallItem(id=f"call_{uuid.uuid4().hex}", type="function_call", name=part.tool_name, arguments=json.dumps(part.args), call_id=part.tool_call_id))
                    elif isinstance(part, BuiltinToolCallPart):
                        # Re-hydrating built-in tool calls for history
                        if part.tool_name == "code_interpreter":
                             input_items.append(ResponseCodeInterpreterToolCall(id=part.tool_call_id, type="code_interpreter_call", **part.args))
                        elif part.tool_name == "web_search":
                             input_items.append(ResponseFunctionWebSearch(id=part.tool_call_id, type="web_search_call", action=part.args))

        # --- 3. Combine Settings ---
        request_params = settings.model_dump(exclude_none=True)

        params: Dict[str, Any] = {
            "input": input_items,
            "instructions": instructions or NOT_GIVEN,
            "temperature": request_params.get("temperature", NOT_GIVEN),
            "max_output_tokens": request_params.get("max_tokens", NOT_GIVEN),
            "top_p": request_params.get("top_p", NOT_GIVEN),
            "timeout": request_params.get("timeout", NOT_GIVEN),
            "user": request_params.get("user", NOT_GIVEN),
        }
        
        # Only add parameters that are supported by the Responses API
        if request_params.get("openai_truncation") is not None:
            params["truncation"] = request_params["openai_truncation"]
        if request_params.get("openai_previous_response_id") is not None:
            params["previous_response_id"] = request_params["openai_previous_response_id"]
        if request_params.get("openai_text_verbosity") is not None:
            params["text"] = {"verbosity": request_params["openai_text_verbosity"]}
        return params

    def _parse_responses_api_response(
        self, openai_response: Response
    ) -> "ModelResponse":
        """
        Translates a Response object from the OpenAI library
        into your framework's ModelResponse.
        """
        finish_reason = self._map_responses_finish_reason(
            str(openai_response.status), openai_response.incomplete_details
        )
        usage = TokenUsage(
            input_tokens=openai_response.usage.input_tokens if openai_response.usage else 0,
            output_tokens=openai_response.usage.output_tokens if openai_response.usage else 0,
        )
        parts: List[ModelResponsePart] = []
        for item in openai_response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        parts.append(TextPart(content=content.text))
            elif item.type == "reasoning":
                if item.summary:
                    for summary in item.summary:
                        parts.append(ThinkingPart(content=summary.text))
            elif item.type == "function_call":
                parts.append(
                    self._parse_tool_call(item)
                )
            elif item.type == "code_interpreter_call":
                call_part, return_part = self._map_code_interpreter_call(item)
                parts.extend([call_part, return_part])
            elif item.type == "web_search_call":
                call_part, return_part = self._map_web_search_call(item)
                parts.extend([call_part, return_part])

        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=openai_response.model,
            provider_name="openai",
            provider_response_id=openai_response.id,
            finish_reason=finish_reason,
            provider_details={"status": openai_response.status},
        )
    
    # --- Private Helper & Mapping Methods (Stubs for Step 4) ---

    async def _process_chat_stream(
        self, stream: AsyncIterator[ChatCompletionChunk]
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        """Processes a ChatCompletionChunk stream and yields your stream events."""
        manager = _PartsManager()

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                yield manager.handle_text_delta(delta.content)

            if delta.tool_calls:
                for tool_delta in delta.tool_calls:
                    yield manager.handle_tool_call_delta(tool_delta)
        
        yield FinalResultEvent()

    async def _process_responses_stream(
        self, stream: AsyncIterator[ResponseStreamEvent]
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        """Processes a ResponseStreamEvent stream and yields your stream events."""
        manager = _PartsManager()

        async for event in stream:
            if event.type == "text.delta":
                yield manager.handle_text_delta(event.delta, part_id=event.item_id)
            elif event.type == "reasoning.summary.text.delta":
                yield manager.handle_thinking_delta(event.delta, part_id=event.item_id)
            elif event.type == "output.item.added" and event.item.type == "function_call":
                # A new tool call has started
                yield manager.start_tool_call(
                    part_id=event.item.id,
                    tool_call_id=event.item.call_id,
                    tool_name=event.item.name
                )
            elif event.type == "function_call.arguments.delta":
                yield manager.handle_tool_args_delta(
                    part_id=event.item_id, args_delta=event.delta
                )
        
        yield FinalResultEvent()

    # ==============================================================================
    # 4. Internal Helper Utilities
    # ==============================================================================

    async def _download_content_to_data_uri(self, item: ImageUrl | AudioUrl | DocumentUrl) -> str:
        """Downloads URL content and converts it to a base64 data URI."""
        try:
            # Use a new client instance to avoid context manager issues
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(item.url)
                response.raise_for_status()  # Raise an exception for bad status codes
                data = response.content
                
                # Guess MIME type if not obvious
                media_type = mimetypes.guess_type(item.url)[0]
                if not media_type:
                    if isinstance(item, ImageUrl):
                        media_type = "image/jpeg"  # Default
                    elif isinstance(item, AudioUrl):
                        media_type = "audio/mpeg" # Default
                    else:
                        media_type = "application/octet-stream"

                base64_data = base64.b64encode(data).decode("utf-8")
                return f"data:{media_type};base64,{base64_data}"
        except Exception as e:
            # Handle download/encoding errors gracefully
            print(f"Error downloading {item.url}: {e}")
            return "" # Return empty string or raise an error

    async def _map_chat_user_prompt(
        self, part: UserPromptPart
    ) -> chat.ChatCompletionUserMessageParam:
        """Maps your UserPromptPart to the Chat API's message param."""
        if isinstance(part.content, str):
            return chat.ChatCompletionUserMessageParam(
                role="user", content=part.content
            )

        content_parts: List[ChatCompletionContentPartParam] = []
        for item in part.content:
            if isinstance(item, str):
                content_parts.append(
                    ChatCompletionContentPartTextParam(text=item, type="text")
                )
            elif isinstance(item, (ImageUrl, BinaryContent)) and "image" in item.media_type:
                url = await self._get_data_url(item)
                content_parts.append(
                    ChatCompletionContentPartImageParam(
                        image_url=ImageURL(url=url), type="image_url"
                    )
                )
            elif isinstance(item, (AudioUrl, BinaryContent)) and "audio" in item.media_type:
                base64_data = await self._get_base64_from_item(item)
                # Infer format from media_type (e.g., 'audio/mp3' -> 'mp3')
                audio_format = item.media_type.split("/")[-1]
                # Ensure format is one of the supported types
                if audio_format not in ['wav', 'mp3']:
                    audio_format = 'mp3'  # Default to mp3
                content_parts.append(
                    ChatCompletionContentPartInputAudioParam(
                        input_audio=InputAudio(data=base64_data, format=audio_format),
                        type="input_audio",
                    )
                )
            elif isinstance(item, (DocumentUrl, BinaryContent)):
                # Handle documents by creating a data URI and passing as a File
                file_data_uri = await self._get_data_url(item)
                filename = getattr(item, 'url', 'file.bin').split("/")[-1]
                content_parts.append(
                    File(
                        file=FileFile(file_data=file_data_uri, filename=filename),
                        type="file",
                    )
                )
            elif isinstance(item, VideoUrl):
                # The Chat Completions API does not support video file inputs.
                raise NotImplementedError("VideoUrl is not supported for OpenAI Chat Completions.")
        
        return chat.ChatCompletionUserMessageParam(role="user", content=content_parts)

    async def _map_responses_user_prompt(self, part: UserPromptPart) -> ResponseInputMessageItem:
        """Maps your UserPromptPart to the Responses API's message param."""
        content = [part.content] if isinstance(part.content, str) else part.content
        
        content_parts: List[Union[ResponseInputTextParam, ResponseInputImageParam, ResponseInputFileParam]] = []
        for item in content:
            if isinstance(item, str):
                content_parts.append(ResponseInputTextParam(text=item, type="input_text"))
            
            elif isinstance(item, (ImageUrl, BinaryContent)) and "image" in item.media_type:
                url = await self._get_data_url(item)
                content_parts.append(ResponseInputImageParam(image_url=url, type="input_image", detail="auto"))
            
            elif isinstance(item, (DocumentUrl, AudioUrl, BinaryContent)):
                # For Responses API, documents, audio, and other binary types are all handled as files.
                data_uri = await self._get_data_url(item)
                filename = getattr(item, 'url', 'file.bin').split("/")[-1]
                content_parts.append(
                    ResponseInputFileParam(
                        file_data=data_uri,
                        filename=filename,
                        type="input_file"
                    )
                )
            elif isinstance(item, VideoUrl):
                 raise NotImplementedError("VideoUrl is not supported for OpenAI Responses API.")

        return ResponseInputMessageItem(
            id=f"msg_{uuid.uuid4().hex}", # Dynamic UUID
            role="user", 
            content=content_parts
        )

    async def _get_data_url(self, item: ImageUrl | AudioUrl | DocumentUrl | BinaryContent) -> str:
        if isinstance(item, BinaryContent):
            base64_data = base64.b64encode(item.data).decode("utf-8")
            return f"data:{item.media_type};base64,{base64_data}"
        if not item.url.startswith("data:"):
            return await self._download_content_to_data_uri(item)
        return item.url

    def _parse_tool_call(self, tool_call: Union[ChatCompletionMessageToolCall, ResponseFunctionToolCall]) -> ToolCallPart:
        if isinstance(tool_call, ChatCompletionMessageToolCall):
            name = tool_call.function.name
            args_str = tool_call.function.arguments
            tool_id = tool_call.id
        else: # ResponseFunctionToolCall
            name = tool_call.name
            args_str = tool_call.arguments
            tool_id = tool_call.call_id

        try:
            args = json.loads(args_str)
        except Exception:
            args = {"__raw_args__": args_str}
        
        return ToolCallPart(tool_name=name, args=args, tool_call_id=tool_id)

    async def _get_base64_from_item(self, item: Union[ImageUrl, AudioUrl, DocumentUrl, BinaryContent]) -> str:
        """Helper to get base64 data from either a URL or binary content."""
        if isinstance(item, BinaryContent):
            return base64.b64encode(item.data).decode("utf-8")
        
        # If it's already a data URI, extract the base64 part
        if item.url.startswith("data:"):
            if "," in item.url:
                return item.url.split(",")[1]
            else:
                return ""
        
        data_uri = await self._download_content_to_data_uri(item)
        if "," in data_uri:
            return data_uri.split(",")[1]
        else:
            return ""
    
    def _map_chat_finish_reason(self, reason: str | None) -> FinishReason | None:
        return FinishReason.from_provider_reason(reason, "openai") if reason else None

    def _map_responses_finish_reason(self, status: str, details: Any) -> FinishReason | None:
        if status == "completed": return FinishReason.STOP
        if details: return FinishReason.from_provider_reason(details.reason, "openai")
        return FinishReason.ERROR

    def _map_code_interpreter_call(
        self, item: ResponseCodeInterpreterToolCall
    ) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
        """Maps a Code Interpreter call to your framework's built-in parts."""
        call_part = BuiltinToolCallPart(
            tool_name="code_interpreter",
            args={"code": item.code, "container_id": item.container_id},
            tool_call_id=item.id,
            provider_name="openai",
        )
        return_part = BuiltinToolReturnPart(
            tool_name="code_interpreter",
            content={
                "status": item.status,
                "outputs": [o.model_dump() for o in item.outputs] if item.outputs else [],
            },
            tool_call_id=item.id,
            provider_name="openai",
        )
        return call_part, return_part

    def _map_web_search_call(
        self, item: ResponseFunctionWebSearch
    ) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
        """Maps a Web Search call to your framework's built-in parts."""
        args = item.action.model_dump() if item.action else {}
        sources = args.pop("sources", [])
        
        call_part = BuiltinToolCallPart(
            tool_name="web_search",
            args=args,
            tool_call_id=item.id,
            provider_name="openai",
        )
        return_part = BuiltinToolReturnPart(
            tool_name="web_search",
            content={"status": item.status, "sources": sources},
            tool_call_id=item.id,
            provider_name="openai",
        )
        return call_part, return_part

class _PartsManager:
    """A helper class to manage state during a streaming response."""
    def __init__(self):
        self._parts_by_id: Dict[str, ModelResponsePart] = {}
        self._parts_by_index: Dict[int, ModelResponsePart] = {}
        self._next_index = 0

    def _get_or_create_part(self, part_id: str, part_type: type) -> tuple[ModelResponsePart, bool]:
        """Gets a part or creates it if it's new."""
        is_new = part_id not in self._parts_by_id
        if is_new:
            if part_type == TextPart:
                part: ModelResponsePart = TextPart(content="")
            elif part_type == ThinkingPart:
                part = ThinkingPart(content="")
            elif part_type == ToolCallPart:
                part = ToolCallPart(tool_name="", args={}, tool_call_id="")
            else:
                raise TypeError(f"Unsupported part type for streaming: {part_type}")
            
            self._parts_by_id[part_id] = part
            self._parts_by_index[self._next_index] = part
            self._next_index += 1
        return self._parts_by_id[part_id], is_new

    def handle_text_delta(self, delta: str, part_id: str = "text_0") -> ModelResponseStreamEvent:
        part, is_new = self._get_or_create_part(part_id, TextPart)
        cast(TextPart, part).content += delta
        index = next(k for k, v in self._parts_by_index.items() if v is part)
        if is_new:
            return PartStartEvent(index=index, part=part.model_copy())
        return PartDeltaEvent(index=index, delta=TextPartDelta(content_delta=delta))

    def handle_thinking_delta(self, delta: str, part_id: str) -> ModelResponseStreamEvent:
        part, is_new = self._get_or_create_part(part_id, ThinkingPart)
        cast(ThinkingPart, part).content += delta
        index = next(k for k, v in self._parts_by_index.items() if v is part)
        if is_new:
            return PartStartEvent(index=index, part=part.model_copy())
        return PartDeltaEvent(index=index, delta=ThinkingPartDelta(content_delta=delta))

    def handle_tool_call_delta(self, delta: ChoiceDeltaToolCall) -> ModelResponseStreamEvent:
        part_id = f"tool_{delta.index}"
        part, is_new = self._get_or_create_part(part_id, ToolCallPart)
        part = cast(ToolCallPart, part)
        
        delta_event = ToolCallPartDelta()
        if delta.id:
            part.tool_call_id = delta.id
        if delta.function and delta.function.name:
            part.tool_name += delta.function.name
            delta_event.tool_name_delta = delta.function.name
        if delta.function and delta.function.arguments:
            # This assumes args are streamed as a raw string
            raw_args = part.args.get("__raw_args__", "") + delta.function.arguments
            part.args["__raw_args__"] = raw_args
            delta_event.args_delta = delta.function.arguments

        index = next(k for k, v in self._parts_by_index.items() if v is part)
        if is_new:
            return PartStartEvent(index=index, part=part.model_copy())
        return PartDeltaEvent(index=index, delta=delta_event)

    def start_tool_call(self, part_id: str, tool_call_id: str, tool_name: str) -> PartStartEvent:
        part, _ = self._get_or_create_part(part_id, ToolCallPart)
        part = cast(ToolCallPart, part)
        part.tool_call_id = tool_call_id
        part.tool_name = tool_name
        index = next(k for k, v in self._parts_by_index.items() if v is part)
        return PartStartEvent(index=index, part=part.model_copy())

    def handle_tool_args_delta(self, part_id: str, args_delta: str) -> PartDeltaEvent:
        part = cast(ToolCallPart, self._parts_by_id[part_id])
        raw_args = part.args.get("__raw_args__", "") + args_delta
        part.args["__raw_args__"] = raw_args
        index = next(k for k, v in self._parts_by_index.items() if v is part)
        return PartDeltaEvent(index=index, delta=ToolCallPartDelta(args_delta=args_delta))
