---
name: messages-data-model
description: Use when working with Upsonic's wire-level message and part data model that flows between Agent and Model adapters. Use when a user asks to construct, inspect, serialize, or stream ModelRequest/ModelResponse objects, build multi-modal user prompts, handle tool-call lifecycles, persist chat history, or emit OpenTelemetry GenAI parts. Trigger when the user mentions ModelMessage, ModelRequest, ModelResponse, ModelMessagesTypeAdapter, SystemPromptPart, UserPromptPart, ToolReturnPart, RetryPromptPart, TextPart, ThinkingPart, FilePart, ToolCallPart, BuiltinToolCallPart, BuiltinToolReturnPart, BinaryContent, BinaryImage, ImageUrl, AudioUrl, VideoUrl, DocumentUrl, FileUrl, CachePoint, ToolReturn, MultiModalContent, UserContent, FinishReason, TextPartDelta, ThinkingPartDelta, ToolCallPartDelta, PartStartEvent, PartDeltaEvent, PartEndEvent, FinalResultEvent, FunctionToolCallEvent, FunctionToolResultEvent, ModelResponseStreamEvent, _otel_messages, otel_message_parts, part_kind, tool_call_id, streaming deltas, or chat history serialization.
---

# `src/upsonic/messages/` — Message and Part Data Model

## 1. What this folder is

This folder defines the **wire-level data model** that Upsonic uses to talk to LLM providers. Every interaction between an `Agent` (the application side) and a `Model` (the provider adapter) is normalized into a list of `ModelMessage` values, each of which is either a `ModelRequest` (Upsonic → model) or a `ModelResponse` (model → Upsonic).

These message objects are not provider-specific. They are the canonical, provider-agnostic representation of a chat history. Each message holds an ordered sequence of *parts*: small dataclasses that represent a single semantic unit (a system prompt, a user prompt, a chunk of model text, a chain-of-thought, a tool call, a tool result, an error retry, a generated file, and so on). The same data model is used for:

- The Agent <-> Model request/response cycle.
- Persisted chat history (the `ModelMessagesTypeAdapter` Pydantic adapter handles JSON round-tripping).
- Streaming: `ModelResponseStreamEvent`s (`PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FinalResultEvent`) progressively build a `ModelResponse` while preserving the same part vocabulary.
- Multi-modal payloads (URLs and inline binary content for images, audio, video, documents).
- OpenTelemetry instrumentation: every part knows how to emit OTel `LogRecord`s and `_otel_messages.MessagePart` payloads (a separate, GenAI-spec-aligned representation defined in `_otel_messages.py`).

In short: this folder *is* the schema for everything that flows through the agent loop.

Source files:

| Path | Lines | Role |
| --- | ---: | --- |
| `__init__.py` | 4 | Re-exports everything from `messages.py` via `from .messages import *`. |
| `messages.py` | 2114 | Core data model: parts, deltas, events, request/response messages, multi-modal types, type adapters, OTel hooks. |
| `_otel_messages.py` | 102 | Internal `TypedDict` mirror of the OpenTelemetry GenAI semconv message-parts vocabulary (used only when emitting telemetry). |

## 2. Folder layout

```
src/upsonic/messages/
├── __init__.py            # `from .messages import *`
├── _otel_messages.py      # OTel GenAI-spec TypedDicts (text/tool_call/tool_call_response/uri/blob/...)
└── messages.py            # All public message/part/delta/event classes + Pydantic TypeAdapters
```

The single Python module `messages.py` is the canonical source. `__init__.py` simply re-exports it, so `from upsonic.messages import ModelRequest, TextPart, ...` works.

## 3. Top-level files

### 3.1 `__init__.py`

```python
from .messages import *

__all__ = [
    'messages',
]
```

A pure re-export. It does not introduce new symbols. Anything described below in `messages.py` is reachable as `upsonic.messages.<Symbol>`.

### 3.2 `_otel_messages.py`

Internal-only module (leading underscore). It defines `TypedDict`s that match the OpenTelemetry GenAI semantic conventions for chat message parts. These are *not* the runtime classes used inside the agent loop — those live in `messages.py`. These TypedDicts are the *output* shape produced by `*.otel_message_parts(settings)` calls when Upsonic emits spans/logs.

| Class | `type` literal | Required fields | Optional fields | Purpose |
| --- | --- | --- | --- | --- |
| `TextPart` | `'text'` | — | `content: str` | Plain assistant/user/system text. |
| `ToolCallPart` | `'tool_call'` | `id: str`, `name: str` | `arguments: JsonValue`, `builtin: bool` | Function tool call invocation. |
| `ToolCallResponsePart` | `'tool_call_response'` | `id: str`, `name: str` | `result: JsonValue`, `builtin: bool` | Function tool call result. |
| `MediaUrlPart` | `'image-url' \| 'audio-url' \| 'video-url' \| 'document-url'` | — | `url: str` | Pre-v4 OTel multimodal URL. |
| `UriPart` | `'uri'` | — | `modality: 'image'\|'audio'\|'video'`, `uri: str`, `mime_type: str` | v4+ OTel URI part. |
| `BinaryDataPart` | `'binary'` | `media_type: str` | `content: str` (base64) | Pre-v4 OTel inline binary. |
| `BlobPart` | `'blob'` | — | `modality`, `mime_type`, `content` (base64) | v4+ OTel inline binary. |
| `ThinkingPart` | `'thinking'` | — | `content: str` | Chain-of-thought / reasoning. |

There are also helper aliases:

```python
MessagePart: TypeAlias = (
    'TextPart | ToolCallPart | ToolCallResponsePart | MediaUrlPart | UriPart '
    '| BinaryDataPart | BlobPart | ThinkingPart'
)

Role = Literal['system', 'user', 'assistant']

class ChatMessage(TypedDict):
    role: Role
    parts: list[MessagePart]

InputMessages: TypeAlias = list[ChatMessage]

class OutputMessage(ChatMessage):
    finish_reason: NotRequired[str]

OutputMessages: TypeAlias = list[OutputMessage]
```

These types are consumed by `upsonic.models.instrumented.InstrumentedModel` (referenced indirectly through `if TYPE_CHECKING` and inline `from upsonic.models.instrumented import InstrumentedModel`) when serializing a request/response to OTel.

### 3.3 `messages.py`

This module is structured into four layers:

1. Multi-modal building blocks (media types, `FileUrl` hierarchy, `BinaryContent`, `BinaryImage`, `CachePoint`, `ToolReturn`).
2. Request-side parts (`SystemPromptPart`, `UserPromptPart`, `ToolReturnPart`, `BuiltinToolReturnPart`, `RetryPromptPart`) and the `ModelRequest` message.
3. Response-side parts (`TextPart`, `ThinkingPart`, `FilePart`, `ToolCallPart`, `BuiltinToolCallPart`, `BuiltinToolReturnPart`) and the `ModelResponse` message.
4. Streaming primitives (`*PartDelta`, `PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FinalResultEvent`, `FunctionToolCallEvent`, `FunctionToolResultEvent`, deprecated `BuiltinToolCallEvent`/`BuiltinToolResultEvent`).

#### 3.3.1 Module-level constants and aliases

| Symbol | Type | Purpose |
| --- | --- | --- |
| `_mime_types` | `mimetypes.MimeTypes` | Custom MIME registry pre-populated with extra rich-media + doc + YAML/TOML/XML overrides. |
| `AudioMediaType` | `Literal[...]` | Whitelist of supported audio MIME types. |
| `ImageMediaType` | `Literal[...]` | Whitelist of supported image MIME types. |
| `DocumentMediaType` | `Literal[...]` | Whitelist of supported document MIME types. |
| `VideoMediaType` | `Literal[...]` | Whitelist of supported video MIME types. |
| `AudioFormat`, `ImageFormat`, `DocumentFormat`, `VideoFormat` | `Literal[...]` | Short-form format strings (`'mp4'`, `'pdf'`, ...). |
| `FinishReason` | `Literal['stop','length','content_filter','tool_call','error']` | Normalized OTel finish reason. |
| `ForceDownloadMode` | `bool \| Literal['allow-local']` | Controls whether a `FileUrl` is sent as URL or downloaded locally with SSRF protection. |
| `ProviderDetailsDelta` | `dict[str, Any] \| Callable[[dict\|None], dict] \| None` | Used by deltas to merge or transform `provider_details`. |
| `MULTI_MODAL_CONTENT_TYPES` | `tuple` | `(ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent)` for `isinstance` checks. |
| `MultiModalContent` | `Annotated[... pydantic.Discriminator('kind')]` | Tagged union of the five multi-modal types. |
| `UserContent` | `str \| MultiModalContent \| CachePoint` | Anything legal inside `UserPromptPart.content`. |
| `_document_format_lookup`, `_audio_format_lookup`, `_image_format_lookup`, `_video_format_lookup` | `dict[str, *Format]` | MIME → short format tables used by the `format` property. |
| `_kind_to_modality_lookup` | `dict[str, Literal['image','audio','video']]` | URL part kind → OTel modality. |
| `tool_return_ta` | `pydantic.TypeAdapter[Any]` | Used to JSON-serialize tool return content (with base64 byte handling). |
| `error_details_ta` | `pydantic.TypeAdapter[list[ErrorDetails]]` | Pydantic adapter for retry-prompt validation errors. |
| `ModelRequestPart` | `Annotated[... Discriminator('part_kind')]` | Discriminated union of the four request parts. |
| `ModelResponsePart` | `Annotated[... Discriminator('part_kind')]` | Discriminated union of the six response parts. |
| `ModelMessage` | `Annotated[ModelRequest \| ModelResponse, Discriminator('kind')]` | Top-level message union. |
| `ModelMessagesTypeAdapter` | `pydantic.TypeAdapter[list[ModelMessage]]` | (De)serialize chat histories. |
| `ModelResponsePartTypeAdapter` | `pydantic.TypeAdapter[list[ModelResponsePart]]` | (De)serialize partial response parts. |
| `BinaryContentTypeAdapter` | `pydantic.TypeAdapter[list[BinaryContent]]` | (De)serialize lists of binary content. |
| `ModelResponsePartDelta` | `Annotated[TextPartDelta\|ThinkingPartDelta\|ToolCallPartDelta, Discriminator('part_delta_kind')]` | Streaming delta union. |
| `ModelResponseStreamEvent` | `Annotated[PartStartEvent\|PartDeltaEvent\|PartEndEvent\|FinalResultEvent, Discriminator('event_kind')]` | Streaming event union. |

#### 3.3.2 Multi-modal building blocks

##### `FileUrl` (abstract base)

`@pydantic_dataclass` ABC. Common base for `VideoUrl`, `AudioUrl`, `ImageUrl`, `DocumentUrl`.

| Field | Type | Description |
| --- | --- | --- |
| `url` | `str` | The URL of the file (positional, required). |
| `force_download` | `ForceDownloadMode` | `False` (send as URL when supported), `True` (always download, with SSRF protection blocking private + cloud-metadata IPs), `'allow-local'` (download but allow private IPs). |
| `vendor_metadata` | `dict[str, Any] \| None` | Provider-specific overrides (e.g. Google `video_metadata`, OpenAI/XAI image `detail`). |
| `_media_type` | `str \| None` (alias `media_type`) | Optional explicit MIME override. |
| `_identifier` | `str \| None` (alias `identifier`) | Stable handle the LLM can refer to in tool args. |
| `media_type` (computed) | `str` | Returns explicit `_media_type` or calls `_infer_media_type()` (subclass-defined). |
| `identifier` (computed) | `str` | Returns explicit `_identifier` or `sha1(url)[:6]`. |
| `format` (abstract property) | `str` | Subclass returns `'mp4'`, `'png'`, etc. |

`_multi_modal_content_identifier(identifier: str | bytes) -> str` is the helper that produces stable 6-char SHA1 identifiers.

##### Concrete `FileUrl` subclasses

| Class | `kind` | `_infer_media_type` | `format` |
| --- | --- | --- | --- |
| `VideoUrl` | `'video-url'` | YouTube → `'video/mp4'`; otherwise `mimetypes.guess_type`. Has `is_youtube` property checking hostname. | `_video_format_lookup[media_type]` |
| `AudioUrl` | `'audio-url'` | `mimetypes.guess_type(url)`. | `_audio_format_lookup[media_type]` |
| `ImageUrl` | `'image-url'` | `mimetypes.guess_type(url)`. | `_image_format_lookup[media_type]` |
| `DocumentUrl` | `'document-url'` | `mimetypes.guess_type(url)`. | `_document_format_lookup[media_type]` |

All four raise `ValueError` if the MIME type cannot be inferred and no explicit `media_type` was provided.

##### `BinaryContent`

```python
@pydantic_dataclass(config=pydantic.ConfigDict(ser_json_bytes='base64', val_json_bytes='base64'))
class BinaryContent:
    data: bytes
    media_type: AudioMediaType | ImageMediaType | DocumentMediaType | str
    vendor_metadata: dict[str, Any] | None = None
    _identifier: str | None  # alias `identifier`, defaults to sha1(data)[:6]
    kind: Literal['binary'] = 'binary'
```

| Method | Returns | Behavior |
| --- | --- | --- |
| `narrow_type(bc)` (static) | `BinaryContent \| BinaryImage` | If MIME starts with `image/`, returns a `BinaryImage`; else returns the input. |
| `from_data_uri(data_uri)` (classmethod) | `BinaryContent` | Parses `data:<mime>;base64,<...>`. |
| `from_path(path)` (classmethod) | `BinaryContent` | Reads bytes, infers MIME, defaults to `application/octet-stream`. Raises on missing/unreadable files. |
| `identifier` (computed) | `str` | `_identifier` or `sha1(data)[:6]`. |
| `data_uri` | `str` | `f'data:{media_type};base64,{base64}'`. |
| `base64` | `str` | UTF-8-decoded base64. |
| `is_audio` / `is_image` / `is_video` / `is_document` | `bool` | MIME prefix or `_document_format_lookup` membership. |
| `format` | `str` | Looks up the right format dict based on `is_*`. |

##### `BinaryImage(BinaryContent)`

Same `kind='binary'` discriminator (so it round-trips as a `BinaryContent`), but enforces `is_image` in `__post_init__`. Used by `FilePart.content`'s `pydantic.AfterValidator(BinaryImage.narrow_type)` to auto-narrow.

##### `CachePoint`

Plain `@dataclass`. Marker placed inside `UserPromptPart.content` to delimit a prompt-cache window. Filtered out by models that do not support caching. Fields:

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `kind` | `Literal['cache-point']` | `'cache-point'` | Discriminator. |
| `ttl` | `Literal['5m', '1h']` | `'5m'` | Cache TTL (Anthropic). |

##### `ToolReturn`

A *user-side* helper for tool authors. Lets a tool return both a value and rich follow-up content for the model.

| Field | Type | Description |
| --- | --- | --- |
| `return_value` | `ToolReturnContent` | The actual returned value. |
| `content` | `str \| Sequence[UserContent] \| None` | Follow-up content sent as a `UserPromptPart`. |
| `metadata` | `Any` | Application-only data. Not sent to the LLM. |
| `kind` | `Literal['tool-return']` | Discriminator. |

`ToolReturnContent` is recursively defined as `MultiModalContent | Sequence[ToolReturnContent] | Mapping[str, ToolReturnContent] | Any` (a `TypeAliasType` at runtime, simpler `MultiModalContent | Sequence[Any] | Mapping[str, Any] | Any` under `TYPE_CHECKING` to keep static analyzers happy).

#### 3.3.3 Request-side parts

All request parts share `part_kind: Literal[...]` for discrimination and an `otel_event(settings)` + `otel_message_parts(settings)` pair for telemetry.

##### `SystemPromptPart`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `content` | `str` | — | The text of the system prompt. |
| `timestamp` | `datetime` | `_now_utc()` | When the prompt was created. |
| `dynamic_ref` | `str \| None` | `None` | Reference to the dynamic system prompt function that produced it. |
| `part_kind` | `Literal['system-prompt']` | `'system-prompt'` | Discriminator. |

OTel: `event.name='gen_ai.system.message'`, body `{'role':'system','content':...}` (content omitted when `include_content=False`).

##### `UserPromptPart`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `content` | `str \| Sequence[UserContent]` | — | Text and/or multi-modal items and/or `CachePoint` markers. |
| `timestamp` | `datetime` | `_now_utc()` | When the prompt was created. |
| `part_kind` | `Literal['user-prompt']` | `'user-prompt'` | Discriminator. |

`otel_message_parts` walks the content sequence and emits:

- `TextPart` for `str` items;
- `UriPart` (v4+) or `MediaUrlPart` (legacy) for `FileUrl` subclasses, including `modality` from `_kind_to_modality_lookup`;
- `BlobPart` (v4+) or `BinaryDataPart` (legacy) for `BinaryContent`;
- `CachePoint`s are skipped (markers, not content).

##### `BaseToolReturnPart` / `ToolReturnPart` / `BuiltinToolReturnPart`

`BaseToolReturnPart` is the shared base. It carries the result of executing a tool back to the model.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `tool_name` | `str` | — | Tool that was called. |
| `content` | `ToolReturnContent` | — | The return value. |
| `tool_call_id` | `str` | `_generate_tool_call_id()` | Matches the originating `ToolCallPart`. |
| `metadata` | `Any` | `None` | App-only sidecar. |
| `timestamp` | `datetime` | `_now_utc()` | When the tool returned. |

Methods:

- `model_response_str() -> str` — string view of `content` (raw if `str`, else JSON-dumped via `tool_return_ta`).
- `model_response_object() -> dict[str, Any]` — JSON-mode dict (wraps non-dicts as `{'return_value': ...}` for Gemini compatibility).
- `otel_event(settings)` — `event.name='gen_ai.tool.message'`, body has role/id/name (+ content if allowed).
- `otel_message_parts(settings)` — emits a single `_otel_messages.ToolCallResponsePart` (with `result` if content is included).
- `has_content() -> bool` — `content is not None`.

`ToolReturnPart` adds `part_kind: Literal['tool-return']`. `BuiltinToolReturnPart` adds `provider_name`, `provider_details`, and `part_kind: Literal['builtin-tool-return']`. (Note: `BuiltinToolReturnPart` is *both* a request-side part subclass of `BaseToolReturnPart` *and* an entry in `ModelResponsePart`'s union, because builtin tool calls + their returns are emitted by the model itself in the same response.)

##### `RetryPromptPart`

A request-side message that asks the model to fix something and retry. Used for tool-arg validation errors, missing tools, plain-text-when-structured, output validators raising `ModelRetry`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `content` | `list[pydantic_core.ErrorDetails] \| str` | — | Either freeform feedback or pydantic error rows. |
| `tool_name` | `str \| None` | `None` | Tool that produced the failure (if any). |
| `tool_call_id` | `str` | `_generate_tool_call_id()` | Matching call id. |
| `timestamp` | `datetime` | `_now_utc()` | When retry was requested. |
| `part_kind` | `Literal['retry-prompt']` | `'retry-prompt'` | Discriminator. |

`model_response()` formats either a "Validation feedback:\n..." block or a JSON-dumped pydantic error block with `Fix the errors and try again.` appended. `otel_event` emits a `gen_ai.user.message` if no `tool_name`, otherwise a `gen_ai.tool.message`.

##### `ModelRequest`

```python
ModelRequestPart = Annotated[
    SystemPromptPart | UserPromptPart | ToolReturnPart | RetryPromptPart,
    pydantic.Discriminator('part_kind'),
]

@dataclass(repr=False)
class ModelRequest:
    parts: Sequence[ModelRequestPart]
    timestamp: datetime | None = None
    instructions: str | None = None
    kind: Literal['request'] = 'request'
    run_id: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def user_text_prompt(cls, user_prompt: str, *, instructions: str | None = None) -> ModelRequest:
        return cls(parts=[UserPromptPart(user_prompt)], instructions=instructions)
```

`run_id` lets persisted history know which agent run a message belongs to. `instructions` carries the resolved system instruction string (separately from any `SystemPromptPart`). `timestamp` defaults to `None` for backward-compat with messages persisted before the field existed.

#### 3.3.4 Response-side parts

##### `TextPart`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `content` | `str` | — | The plain text. |
| `id` | `str \| None` | `None` | Provider-specific text part id (requires `provider_name`). |
| `provider_name` | `str \| None` | `None` | Required when `id` or `provider_details` is set. |
| `provider_details` | `dict[str, Any] \| None` | `None` | Round-trippable provider sidecar. |
| `part_kind` | `Literal['text']` | `'text'` | Discriminator. |

`has_content()` checks `bool(self.content)`.

##### `ThinkingPart`

Same structure as `TextPart` but discriminator `'thinking'` and an extra `signature: str | None`. `signature` round-trips under different names per provider:

| Provider | Field name |
| --- | --- |
| Anthropic | `signature` |
| Bedrock | `signature` |
| Google | `thought_signature` |
| OpenAI | `encrypted_content` |

Signatures are only forwarded back to the originating provider.

##### `FilePart`

A model-generated file (typically an image). `content` is a `BinaryContent`, but with a `pydantic.AfterValidator(BinaryImage.narrow_type)` so deserialization automatically upgrades image MIME types into a `BinaryImage`. Same `id`/`provider_name`/`provider_details` triple. `part_kind = 'file'`. `has_content()` checks `bool(self.content.data)`.

##### `BaseToolCallPart` / `ToolCallPart` / `BuiltinToolCallPart`

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `tool_name` | `str` | — | Function tool name. |
| `args` | `str \| dict[str, Any] \| None` | `None` | Either a JSON string or a parsed dict, depending on what the provider gave us. |
| `tool_call_id` | `str` | `_generate_tool_call_id()` | Match key for the eventual `ToolReturnPart`. |
| `id` | `str \| None` | `None` | Distinct from `tool_call_id` (used by OpenAI Responses). |
| `provider_name` | `str \| None` | `None` | Required when `id` or `provider_details` is set. |
| `provider_details` | `dict[str, Any] \| None` | `None` | Provider sidecar. |

Methods:

- `args_as_dict() -> dict[str, Any]` — parses JSON if needed, asserts dict.
- `args_as_json_str() -> str` — stringifies dict if needed, returns `'{}'` for empty.
- `has_content() -> bool` — `any(self.args.values())` for dicts, `bool(self.args)` otherwise.

`ToolCallPart` adds `part_kind = 'tool-call'`, `BuiltinToolCallPart` adds `part_kind = 'builtin-tool-call'`.

##### `ModelResponsePart` and `ModelResponse`

```python
ModelResponsePart = Annotated[
    TextPart | ToolCallPart | BuiltinToolCallPart | BuiltinToolReturnPart | ThinkingPart | FilePart,
    pydantic.Discriminator('part_kind'),
]
```

`ModelResponse` fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `parts` | `Sequence[ModelResponsePart]` | — | Assistant turn parts. |
| `usage` | `RequestUsage` | `RequestUsage()` | Token + request counts. |
| `model_name` | `str \| None` | `None` | The model id that produced this response. |
| `timestamp` | `datetime` | `_now_utc()` | High-precision local time. Provider clocks live in `provider_details['timestamp']`. |
| `kind` | `Literal['response']` | `'response'` | Discriminator. |
| `provider_name` | `str \| None` | `None` | LLM provider. |
| `provider_url` | `str \| None` | `None` | Provider base URL. |
| `provider_details` | `dict[str, Any] \| None` | `None` | Provider sidecar (also accepts deprecated `vendor_details` on input). |
| `provider_response_id` | `str \| None` | `None` | Provider request id (also accepts deprecated `vendor_id`). |
| `finish_reason` | `FinishReason \| None` | `None` | Normalized OTel value. |
| `run_id` | `str \| None` | `None` | Agent run id. |
| `metadata` | `dict[str, Any] \| None` | `None` | App-only sidecar. |

Computed properties:

| Property | Type | Behavior |
| --- | --- | --- |
| `text` | `str \| None` | Joins `TextPart`s. Adjacent text parts are concatenated; non-adjacent ones (separated by tool calls etc.) are joined with `\n\n`. Returns `None` if no `TextPart` exists. |
| `thinking` | `str \| None` | All `ThinkingPart` contents joined by `\n\n`. |
| `files` | `list[BinaryContent]` | `[part.content for part in parts if isinstance(part, FilePart)]`. |
| `images` | `list[BinaryImage]` | Subset of `files` that are `BinaryImage`. |
| `tool_calls` | `list[ToolCallPart]` | Function (non-builtin) tool calls. |
| `builtin_tool_calls` | `list[tuple[BuiltinToolCallPart, BuiltinToolReturnPart]]` | Pairs each builtin call with its matching return by `tool_call_id`. |

Pricing helpers:

```python
def cost(self) -> genai_types.PriceCalculation:
    assert self.model_name, 'Model name is required to calculate price'
    if self.provider_url:
        try:
            return calc_price(
                self.usage, self.model_name,
                provider_api_url=self.provider_url,
                genai_request_timestamp=self.timestamp,
            )
        except LookupError:
            pass
    return calc_price(
        self.usage, self.model_name,
        provider_id=self.provider_name,
        genai_request_timestamp=self.timestamp,
    )
```

Uses `genai_prices.calc_price` to compute a `PriceCalculation`. `price()` is a deprecated alias.

OTel:

- `otel_events(settings)` returns a list of `gen_ai.assistant.message` `LogRecord`s, splitting per assistant turn. Tool calls go into `body['tool_calls']`; text/thinking go into `body['content']`; files become `{'kind': 'binary', 'media_type': ..., 'binary_content': base64?}`. If the entire content is one text item, the body collapses to a plain string.
- `otel_message_parts(settings)` returns `list[_otel_messages.MessagePart]`, mapping each `ModelResponsePart` to its OTel-spec analogue (`TextPart`, `ThinkingPart`, `BlobPart`/`BinaryDataPart`, `ToolCallPart` with optional `builtin=True`, `ToolCallResponsePart` for `BuiltinToolReturnPart`).

Three deprecated properties remain: `vendor_details` (use `provider_details`), `vendor_id` and `provider_request_id` (both use `provider_response_id`).

#### 3.3.5 Streaming primitives

##### `*PartDelta`

| Class | Discriminator | What it can update | `apply(part)` accepts |
| --- | --- | --- | --- |
| `TextPartDelta` | `part_delta_kind='text'` | `content_delta`, `provider_name`, `provider_details` | Existing `TextPart` only. |
| `ThinkingPartDelta` | `part_delta_kind='thinking'` | `content_delta`, `signature_delta`, `provider_name`, `provider_details` (dict or callable) | `ThinkingPart` (returns merged `ThinkingPart`) or another `ThinkingPartDelta` (returns merged delta with chained provider_details if callables). |
| `ToolCallPartDelta` | `part_delta_kind='tool_call'` | `tool_name_delta`, `args_delta` (str or dict), `tool_call_id`, `provider_name`, `provider_details` | `ToolCallPart`/`BuiltinToolCallPart` (returns updated part) or `ToolCallPartDelta` (returns merged delta, possibly upgraded to a full `ToolCallPart` once `tool_name_delta` is present). Also has `as_part()` which builds a fresh `ToolCallPart` if `tool_name_delta` is set. |

Important invariants enforced by `ToolCallPartDelta.apply`:

- You cannot mix string and dict args — applying a JSON delta to dict args (or vice versa) raises `UnexpectedModelBehavior`.
- `tool_call_id` cannot be reassigned to a non-matching value silently; it can only fill in `None`.
- `provider_details` for `ThinkingPartDelta` may be a callable, in which case applying chains the callables (or wraps a dict appropriately).

```python
ModelResponsePartDelta = Annotated[
    TextPartDelta | ThinkingPartDelta | ToolCallPartDelta,
    pydantic.Discriminator('part_delta_kind'),
]
```

##### Stream events

| Class | Discriminator | Fields | Meaning |
| --- | --- | --- | --- |
| `PartStartEvent` | `event_kind='part_start'` | `index: int`, `part: ModelResponsePart`, `previous_part_kind` | A new part just appeared at `index`. If a prior `PartStartEvent` already used this index, it is replaced. |
| `PartDeltaEvent` | `event_kind='part_delta'` | `index: int`, `delta: ModelResponsePartDelta` | Apply `delta` to the part at `index`. |
| `PartEndEvent` | `event_kind='part_end'` | `index: int`, `part: ModelResponsePart`, `next_part_kind` | The part at `index` is now finalized. |
| `FinalResultEvent` | `event_kind='final_result'` | `tool_name: str \| None`, `tool_call_id: str \| None` | The current model response satisfies the agent's output schema; downstream consumers can stop streaming further parts. |

Combined union:

```python
ModelResponseStreamEvent = Annotated[
    PartStartEvent | PartDeltaEvent | PartEndEvent | FinalResultEvent,
    pydantic.Discriminator('event_kind'),
]
```

##### Tool-call lifecycle events

These are emitted by the agent loop *outside* `ModelResponseStreamEvent`. They surface tool execution to the user-facing streaming API.

| Class | Discriminator | Fields | Meaning |
| --- | --- | --- | --- |
| `FunctionToolCallEvent` | `event_kind='function_tool_call'` | `part: ToolCallPart`, `args_valid: bool \| None` | A function tool is about to run. `args_valid` is `True` if both schema + custom validators passed, `False` if validation failed, `None` if not validated. |
| `FunctionToolResultEvent` | `event_kind='function_tool_result'` | `result: ToolReturnPart \| RetryPromptPart`, `content: str \| Sequence[UserContent] \| None` | The tool finished — `result` is the corresponding part that will be added to the request, `content` is any extra `UserPromptPart` content the tool emitted via `ToolReturn`. |
| `BuiltinToolCallEvent` (deprecated) | `event_kind='builtin_tool_call'` | `part: BuiltinToolCallPart` | Use `PartStartEvent` with `BuiltinToolCallPart` instead. |
| `BuiltinToolResultEvent` (deprecated) | `event_kind='builtin_tool_result'` | `result: BuiltinToolReturnPart` | Use `PartStartEvent`/`PartDeltaEvent` with `BuiltinToolReturnPart` instead. |

Both lifecycle events expose a `tool_call_id` property for matching call ↔ result.

## 4. Subfolders walked through

There are no subfolders. `messages/` is intentionally a flat module: the data model is small enough to live in one file, and keeping every part class colocated makes the discriminated unions easy to maintain. The companion `_otel_messages.py` is leading-underscore to signal that it is internal.

## 5. Cross-file relationships

### 5.1 Inside the folder

- `__init__.py` imports `messages.py` via `*`. There is no other internal import.
- `messages.py` imports `_otel_messages` to translate runtime parts into telemetry shapes:

```python
from upsonic.messages import _otel_messages
```

Each request/response part has an `otel_message_parts(settings: InstrumentationSettings) -> list[_otel_messages.MessagePart]` method that maps:
- `SystemPromptPart` → `_otel_messages.TextPart`
- `UserPromptPart` → mix of `TextPart` / `UriPart` (v4+) or `MediaUrlPart` (legacy) / `BlobPart` (v4+) or `BinaryDataPart` (legacy); `CachePoint` skipped.
- `ToolReturnPart` / `BuiltinToolReturnPart` → `ToolCallResponsePart` (with `builtin=True` for the latter).
- `RetryPromptPart` → either `TextPart` (no `tool_name`) or `ToolCallResponsePart` (with `tool_name`).
- `TextPart` → `TextPart`.
- `ThinkingPart` → `ThinkingPart`.
- `FilePart` → `BlobPart`/`BinaryDataPart` via `_convert_binary_to_otel_part`.
- `ToolCallPart` / `BuiltinToolCallPart` → `ToolCallPart` (with `builtin=True` for the latter).

### 5.2 Outside the folder

| Imported from | Used as |
| --- | --- |
| `upsonic._utils.generate_tool_call_id` | Default `tool_call_id` for `BaseToolReturnPart`, `RetryPromptPart`, `BaseToolCallPart`, and `ToolCallPartDelta.as_part`. |
| `upsonic._utils.now_utc` | Default `timestamp` on every part. |
| `upsonic._utils.dataclasses_no_defaults_repr` | The `__repr__` of every dataclass — only non-default fields are shown. |
| `upsonic.utils.package.exception.UnexpectedModelBehavior` | Raised by `ToolCallPartDelta.apply` when args delta types collide. |
| `upsonic.usage.RequestUsage` | The `usage` field on `ModelResponse`. |
| `upsonic.models.instrumented.InstrumentationSettings` | Threaded through all `otel_*` methods (TYPE_CHECKING-only import). |
| `upsonic.models.instrumented.InstrumentedModel.serialize_any` | Used to serialize tool-call `args` and tool-return `content` for OTel. |
| `genai_prices.calc_price` / `genai_prices.types` | Powers `ModelResponse.cost()`. |
| `pydantic` / `pydantic_core` | Validation, discriminated unions, `from_json`/`to_json`, base64 byte handling. |
| `opentelemetry._logs.LogRecord` / `opentelemetry.util.types.AnyValue` | The `otel_event(settings)` return type. |

### 5.3 Serialization conventions

- Every `BinaryContent` and any TypeAdapter that serializes it uses `pydantic.ConfigDict(ser_json_bytes='base64', val_json_bytes='base64')` — bytes round-trip as base64 strings in JSON.
- `defer_build=True` is set on the public TypeAdapters (`ModelMessagesTypeAdapter`, `ModelResponsePartTypeAdapter`, `BinaryContentTypeAdapter`, `tool_return_ta`, `error_details_ta`) to avoid building schemas at import time.
- `ModelResponse` accepts the deprecated input field names `vendor_details` and `vendor_id` via `pydantic.AliasChoices` so older persisted data still loads.

## 6. Public API

Anything that is reachable as `upsonic.messages.<Symbol>` or imported from `upsonic.messages` is part of the public API:

- **Multi-modal**: `FileUrl` (abstract), `VideoUrl`, `AudioUrl`, `ImageUrl`, `DocumentUrl`, `BinaryContent`, `BinaryImage`, `CachePoint`, `MultiModalContent`, `MULTI_MODAL_CONTENT_TYPES`, `UserContent`.
- **Helpers**: `ToolReturn`, `ToolReturnContent`, `FinishReason`, `ForceDownloadMode`, `ProviderDetailsDelta`, `AudioMediaType`, `ImageMediaType`, `DocumentMediaType`, `VideoMediaType`, `AudioFormat`, `ImageFormat`, `DocumentFormat`, `VideoFormat`.
- **Request parts**: `SystemPromptPart`, `UserPromptPart`, `ToolReturnPart`, `BuiltinToolReturnPart`, `RetryPromptPart`, `BaseToolReturnPart`, `ModelRequestPart`.
- **Response parts**: `TextPart`, `ThinkingPart`, `FilePart`, `ToolCallPart`, `BuiltinToolCallPart`, `BaseToolCallPart`, `ModelResponsePart`.
- **Messages**: `ModelRequest`, `ModelResponse`, `ModelMessage`.
- **Type adapters**: `ModelMessagesTypeAdapter`, `ModelResponsePartTypeAdapter`, `BinaryContentTypeAdapter`.
- **Streaming deltas**: `TextPartDelta`, `ThinkingPartDelta`, `ToolCallPartDelta`, `ModelResponsePartDelta`.
- **Streaming events**: `PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FinalResultEvent`, `ModelResponseStreamEvent`.
- **Tool lifecycle events**: `FunctionToolCallEvent`, `FunctionToolResultEvent`, `BuiltinToolCallEvent` (deprecated), `BuiltinToolResultEvent` (deprecated).

## 7. Integration with the rest of Upsonic

### 7.1 Agent loop

`upsonic/agent/agent.py` (the `Direct`/`Agent` class) maintains a `list[ModelMessage]` for an agent run. On each turn it:

1. Builds a `ModelRequest` whose `parts` start with optional `SystemPromptPart`s, then user input as a `UserPromptPart`, then any pending `ToolReturnPart`/`BuiltinToolReturnPart`/`RetryPromptPart` from the prior turn.
2. Sends it to the `Model`. The model returns a `ModelResponse` (sync) or yields `ModelResponseStreamEvent`s that are reduced into a `ModelResponse` (streaming).
3. Inspects `response.parts`: every `ToolCallPart` becomes a `FunctionToolCallEvent`, the tool runs, and the result becomes a `ToolReturnPart` or `RetryPromptPart` plus a `FunctionToolResultEvent`.
4. Loops, appending the new `ModelResponse` and the next `ModelRequest` to history, until the response satisfies the output schema (signaled by `FinalResultEvent`) or hits a final text answer.

### 7.2 Chat history persistence

Chat history (memory, sessions, conversation logs, `Team` coordination) is just `list[ModelMessage]`. The Pydantic `ModelMessagesTypeAdapter` JSON-(de)serializes that list, with full discriminator support so a stored `kind='request'`/`kind='response'` plus per-part `part_kind` round-trips losslessly. `BinaryContent` data is base64-encoded automatically.

### 7.3 Tool calls

The lifecycle:

```
[ModelResponse]
   parts = [..., ToolCallPart(tool_name='X', args=..., tool_call_id='abc')]

   -> FunctionToolCallEvent(part=ToolCallPart, args_valid=...)
   -> tool runs (may return a `ToolReturn` for rich output)
   -> FunctionToolResultEvent(result=ToolReturnPart(...), content=...)

[Next ModelRequest]
   parts = [ToolReturnPart(tool_name='X', tool_call_id='abc', content=...),
            UserPromptPart(content=ToolReturn.content),  # only if non-None
            ...]
```

`tool_call_id` is the join key. `BuiltinToolCallPart`/`BuiltinToolReturnPart` follow the same pattern but live *inside* a single `ModelResponse` because the provider executes them server-side.

### 7.4 Streaming

Streaming model adapters (e.g. `OpenAIModel.request_stream`) yield `ModelResponseStreamEvent`s. Reduction is straightforward:

```python
parts: list[ModelResponsePart] = []
async for event in stream:
    if isinstance(event, PartStartEvent):
        if event.index < len(parts):
            parts[event.index] = event.part   # replace
        else:
            parts.append(event.part)
    elif isinstance(event, PartDeltaEvent):
        parts[event.index] = event.delta.apply(parts[event.index])
    elif isinstance(event, PartEndEvent):
        parts[event.index] = event.part   # finalize
    elif isinstance(event, FinalResultEvent):
        # Output schema satisfied; consumer can stop
        break

response = ModelResponse(parts=parts, ...)
```

Provider adapters typically also emit `FunctionToolCallEvent`/`FunctionToolResultEvent` to user code through the agent's stream wrapper.

### 7.5 Telemetry

`upsonic.models.instrumented.InstrumentedModel` reads the `InstrumentationSettings` (version, `include_content`, `include_binary_content`) and:

- For each request/response, calls `part.otel_message_parts(settings)` and assembles `_otel_messages.InputMessages`/`OutputMessages` arrays for spans.
- For events log (the older OTel pattern), calls `part.otel_event(settings)` to produce `LogRecord`s.

The settings fan out into:

- `version >= 4` switches `MediaUrlPart`→`UriPart`, `BinaryDataPart`→`BlobPart`, and adds `modality`.
- `include_content=False` strips text/URL/binary payloads (only structural metadata is emitted).
- `include_binary_content=False` strips `content`/`binary_content` from blob/binary parts even when the rest of the payload is included.

### 7.6 Pricing and usage

`ModelResponse.usage: RequestUsage` is populated by each provider adapter. `ModelResponse.cost()` calls `genai_prices.calc_price` (preferring `provider_url`, falling back to `provider_name`) to produce a `PriceCalculation`. The `Agent.cost` aggregator sums `cost()` across every response in a run (see `src/upsonic/agent/`).

## 8. End-to-end flow of a message round-trip

The following diagram shows what happens when a user calls `agent.do("Read this PDF and summarize")` with a PDF attachment, and the model decides to call a tool before producing a final answer.

```
USER
 └── agent.do("Read this PDF...", attachments=[DocumentUrl(url=...)])
        │
        ▼
ModelRequest(kind='request', parts=[
    SystemPromptPart(content='You are a helpful agent...'),
    UserPromptPart(content=[
        'Read this PDF...',
        DocumentUrl(url='https://...', _media_type='application/pdf'),
    ]),
], instructions='You are a helpful agent...', run_id='run-1')
        │
        ▼  (provider adapter serializes to OpenAI/Anthropic/etc. request)
LLM PROVIDER
        │
        ▼  (streaming response — events shown in arrival order)
PartStartEvent(index=0, part=ThinkingPart(content=''))
PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Let me look at the PDF'))
PartEndEvent(index=0, part=ThinkingPart(content='Let me look at the PDF'))
PartStartEvent(index=1, part=ToolCallPart(tool_name='extract_pdf', args=None, tool_call_id='call_1'))
PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta='{"url":"https://..."}'))
PartEndEvent(index=1, part=ToolCallPart(tool_name='extract_pdf', args='{"url":"https://..."}', tool_call_id='call_1'))
        │
        ▼  (events reduced)
ModelResponse(kind='response', parts=[
    ThinkingPart(content='Let me look at the PDF'),
    ToolCallPart(tool_name='extract_pdf', args='{"url":"https://..."}', tool_call_id='call_1'),
], usage=..., model_name='gpt-...', finish_reason='tool_call', run_id='run-1')
        │
        ▼  (agent loop)
FunctionToolCallEvent(part=<the ToolCallPart>, args_valid=True)
        │   tool runs:
        │     def extract_pdf(url): return ToolReturn(return_value={'text': '...'},
        │                                             content='Extraction succeeded.')
FunctionToolResultEvent(result=ToolReturnPart(tool_name='extract_pdf',
                                              content={'text': '...'},
                                              tool_call_id='call_1'),
                         content='Extraction succeeded.')
        │
        ▼
ModelRequest(kind='request', parts=[
    ToolReturnPart(tool_name='extract_pdf', content={'text': '...'},
                   tool_call_id='call_1'),
    UserPromptPart(content='Extraction succeeded.'),
], run_id='run-1')
        │
        ▼  (second turn, streaming)
PartStartEvent(index=0, part=TextPart(content=''))
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='The PDF is about '))
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='quarterly earnings.'))
FinalResultEvent(tool_name=None, tool_call_id=None)
PartEndEvent(index=0, part=TextPart(content='The PDF is about quarterly earnings.'))
        │
        ▼
ModelResponse(parts=[TextPart(content='The PDF is about quarterly earnings.')],
              finish_reason='stop', ...)
        │
        ▼  (agent extracts response.text → user)
'The PDF is about quarterly earnings.'
```

Throughout the round-trip, **every byte of state** lives in the dataclasses defined in this folder:

- The chat history accumulates as `[ModelRequest, ModelResponse, ModelRequest, ModelResponse]`.
- Tool execution materializes as `ToolCallPart` → `ToolReturnPart` (joined by `tool_call_id`).
- Reasoning shows up as `ThinkingPart` (with optional provider-specific `signature`).
- Multi-modal inputs ride inside `UserPromptPart.content` as `DocumentUrl`/`ImageUrl`/`AudioUrl`/`VideoUrl`/`BinaryContent` items, with optional `CachePoint` markers.
- Errors that the model should recover from become `RetryPromptPart`s attached to the next `ModelRequest`.
- Telemetry projects each part through `otel_event` / `otel_message_parts` into the OpenTelemetry GenAI vocabulary defined by `_otel_messages.py`.
- Persistence flows through `ModelMessagesTypeAdapter`, with bytes safely base64-encoded and discriminators (`kind`, `part_kind`, `part_delta_kind`, `event_kind`) keeping every variant unambiguous.

This single, sharply typed schema is what lets Upsonic swap providers, replay sessions, fan out tool calls, stream live UI events, and emit GenAI-spec telemetry — all without any layer of the framework needing to know how the wire bytes were originally shaped.
