"""Logic related to making requests to an LLM.

The aim here is to make a common interface for different LLMs, so that the rest of the code can be agnostic to the
specific LLM being used.

Also includes model selection capabilities for choosing the best model for a given task.
"""

from __future__ import annotations as _annotations

import os
import base64
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

import httpx
from typing_extensions import TypeAliasType, TypedDict

from upsonic import _utils
from upsonic._json_schema import JsonSchemaTransformer
from upsonic.output import OutputObjectDefinition, DEFAULT_OUTPUT_TOOL_NAME
from upsonic._output import PromptedOutputSchema
from upsonic._parts_manager import ModelResponsePartsManager
from upsonic.tools.builtin_tools import AbstractBuiltinTool
from upsonic.utils.package.exception import UserError
from upsonic.messages import (
    BaseToolCallPart,
    BinaryImage,
    FilePart,
    FileUrl,
    FinalResultEvent,
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    VideoUrl,
)
from upsonic.output import OutputMode
from upsonic.profiles import DEFAULT_PROFILE, ModelProfile, ModelProfileSpec
from upsonic.providers import Provider, infer_provider
from upsonic.models.settings import ModelSettings, merge_model_settings
from upsonic.tools import ToolDefinition
from upsonic.usage import RequestUsage
from upsonic.uel import Runnable

if TYPE_CHECKING:
    from upsonic.messages import ToolReturnPart

from dotenv import load_dotenv
load_dotenv()


KnownModelName = TypeAliasType(
    'KnownModelName',
    Literal[
        'anthropic/claude-3-5-haiku-20241022',
        'anthropic/claude-3-5-haiku-latest',
        'anthropic/claude-3-7-sonnet-20250219',
        'anthropic/claude-3-7-sonnet-latest',
        'anthropic/claude-3-haiku-20240307',
        'anthropic/claude-3-opus-20240229',
        'anthropic/claude-3-opus-latest',
        'anthropic/claude-4-opus-20250514',
        'anthropic/claude-4-sonnet-20250514',
        'anthropic/claude-haiku-4-5-20251001',
        'anthropic/claude-haiku-4-5',
        'anthropic/claude-opus-4-0',
        'anthropic/claude-opus-4-1-20250805',
        'anthropic/claude-opus-4-20250514',
        'anthropic/claude-opus-4-5-20251101',
        'anthropic/claude-opus-4-5',
        'anthropic/claude-sonnet-4-0',
        'anthropic/claude-sonnet-4-20250514',
        'anthropic/claude-sonnet-4-5-20250929',
        'anthropic/claude-sonnet-4-5',
        'bedrock/amazon.titan-text-express-v1',
        'bedrock/amazon.titan-text-lite-v1',
        'bedrock/amazon.titan-tg1-large',
        'bedrock/anthropic.claude-3-5-haiku-20241022-v1:0',
        'bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0',
        'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0',
        'bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0',
        'bedrock/anthropic.claude-3-haiku-20240307-v1:0',
        'bedrock/anthropic.claude-3-opus-20240229-v1:0',
        'bedrock/anthropic.claude-3-sonnet-20240229-v1:0',
        'bedrock/anthropic.claude-haiku-4-5-20251001-v1:0',
        'bedrock/anthropic.claude-instant-v1',
        'bedrock/anthropic.claude-opus-4-20250514-v1:0',
        'bedrock/anthropic.claude-sonnet-4-20250514-v1:0',
        'bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0',
        'bedrock/anthropic.claude-v2:1',
        'bedrock/anthropic.claude-v2',
        'bedrock/cohere.command-light-text-v14',
        'bedrock/cohere.command-r-plus-v1:0',
        'bedrock/cohere.command-r-v1:0',
        'bedrock/cohere.command-text-v14',
        'bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0',
        'bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0',
        'bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'bedrock/global.anthropic.claude-opus-4-5-20251101-v1:0',
        'bedrock/meta.llama3-1-405b-instruct-v1:0',
        'bedrock/meta.llama3-1-70b-instruct-v1:0',
        'bedrock/meta.llama3-1-8b-instruct-v1:0',
        'bedrock/meta.llama3-70b-instruct-v1:0',
        'bedrock/meta.llama3-8b-instruct-v1:0',
        'bedrock/mistral.mistral-7b-instruct-v0:2',
        'bedrock/mistral.mistral-large-2402-v1:0',
        'bedrock/mistral.mistral-large-2407-v1:0',
        'bedrock/mistral.mixtral-8x7b-instruct-v0:1',
        'bedrock/us.amazon.nova-lite-v1:0',
        'bedrock/us.amazon.nova-micro-v1:0',
        'bedrock/us.amazon.nova-pro-v1:0',
        'bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0',
        'bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0',
        'bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'bedrock/us.anthropic.claude-3-haiku-20240307-v1:0',
        'bedrock/us.anthropic.claude-3-opus-20240229-v1:0',
        'bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0',
        'bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0',
        'bedrock/us.anthropic.claude-opus-4-20250514-v1:0',
        'bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0',
        'bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'bedrock/us.meta.llama3-1-70b-instruct-v1:0',
        'bedrock/us.meta.llama3-1-8b-instruct-v1:0',
        'bedrock/us.meta.llama3-2-11b-instruct-v1:0',
        'bedrock/us.meta.llama3-2-1b-instruct-v1:0',
        'bedrock/us.meta.llama3-2-3b-instruct-v1:0',
        'bedrock/us.meta.llama3-2-90b-instruct-v1:0',
        'bedrock/us.meta.llama3-3-70b-instruct-v1:0',
        'cerebras/gpt-oss-120b',
        'cerebras/llama-3.3-70b',
        'cerebras/llama3.1-8b',
        'cerebras/qwen-3-235b-a22b-instruct-2507',
        'cerebras/qwen-3-32b',
        'cerebras/zai-glm-4.6',
        'cohere/c4ai-aya-expanse-32b',
        'cohere/c4ai-aya-expanse-8b',
        'cohere/command-nightly',
        'cohere/command-r-08-2024',
        'cohere/command-r-plus-08-2024',
        'cohere/command-r7b-12-2024',
        'deepseek/deepseek-chat',
        'deepseek/deepseek-reasoner',
        'gateway/anthropic/claude-3-5-haiku-20241022',
        'gateway/anthropic/claude-3-5-haiku-latest',
        'gateway/anthropic/claude-3-7-sonnet-20250219',
        'gateway/anthropic/claude-3-7-sonnet-latest',
        'gateway/anthropic/claude-3-haiku-20240307',
        'gateway/anthropic/claude-3-opus-20240229',
        'gateway/anthropic/claude-3-opus-latest',
        'gateway/anthropic/claude-4-opus-20250514',
        'gateway/anthropic/claude-4-sonnet-20250514',
        'gateway/anthropic/claude-haiku-4-5-20251001',
        'gateway/anthropic/claude-haiku-4-5',
        'gateway/anthropic/claude-opus-4-0',
        'gateway/anthropic/claude-opus-4-1-20250805',
        'gateway/anthropic/claude-opus-4-20250514',
        'gateway/anthropic/claude-opus-4-5-20251101',
        'gateway/anthropic/claude-opus-4-5',
        'gateway/anthropic/claude-sonnet-4-0',
        'gateway/anthropic/claude-sonnet-4-20250514',
        'gateway/anthropic/claude-sonnet-4-5-20250929',
        'gateway/anthropic/claude-sonnet-4-5',
        'gateway/bedrock/amazon.titan-text-express-v1',
        'gateway/bedrock/amazon.titan-text-lite-v1',
        'gateway/bedrock/amazon.titan-tg1-large',
        'gateway/bedrock/anthropic.claude-3-5-haiku-20241022-v1:0',
        'gateway/bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0',
        'gateway/bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0',
        'gateway/bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0',
        'gateway/bedrock/anthropic.claude-3-haiku-20240307-v1:0',
        'gateway/bedrock/anthropic.claude-3-opus-20240229-v1:0',
        'gateway/bedrock/anthropic.claude-3-sonnet-20240229-v1:0',
        'gateway/bedrock/anthropic.claude-haiku-4-5-20251001-v1:0',
        'gateway/bedrock/anthropic.claude-instant-v1',
        'gateway/bedrock/anthropic.claude-opus-4-20250514-v1:0',
        'gateway/bedrock/anthropic.claude-sonnet-4-20250514-v1:0',
        'gateway/bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0',
        'gateway/bedrock/anthropic.claude-v2:1',
        'gateway/bedrock/anthropic.claude-v2',
        'gateway/bedrock/cohere.command-light-text-v14',
        'gateway/bedrock/cohere.command-r-plus-v1:0',
        'gateway/bedrock/cohere.command-r-v1:0',
        'gateway/bedrock/cohere.command-text-v14',
        'gateway/bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0',
        'gateway/bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0',
        'gateway/bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'gateway/bedrock/global.anthropic.claude-opus-4-5-20251101-v1:0',
        'gateway/bedrock/meta.llama3-1-405b-instruct-v1:0',
        'gateway/bedrock/meta.llama3-1-70b-instruct-v1:0',
        'gateway/bedrock/meta.llama3-1-8b-instruct-v1:0',
        'gateway/bedrock/meta.llama3-70b-instruct-v1:0',
        'gateway/bedrock/meta.llama3-8b-instruct-v1:0',
        'gateway/bedrock/mistral.mistral-7b-instruct-v0:2',
        'gateway/bedrock/mistral.mistral-large-2402-v1:0',
        'gateway/bedrock/mistral.mistral-large-2407-v1:0',
        'gateway/bedrock/mistral.mixtral-8x7b-instruct-v0:1',
        'gateway/bedrock/us.amazon.nova-lite-v1:0',
        'gateway/bedrock/us.amazon.nova-micro-v1:0',
        'gateway/bedrock/us.amazon.nova-pro-v1:0',
        'gateway/bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0',
        'gateway/bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0',
        'gateway/bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'gateway/bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0',
        'gateway/bedrock/us.anthropic.claude-3-haiku-20240307-v1:0',
        'gateway/bedrock/us.anthropic.claude-3-opus-20240229-v1:0',
        'gateway/bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0',
        'gateway/bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0',
        'gateway/bedrock/us.anthropic.claude-opus-4-20250514-v1:0',
        'gateway/bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0',
        'gateway/bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'gateway/bedrock/us.meta.llama3-1-70b-instruct-v1:0',
        'gateway/bedrock/us.meta.llama3-1-8b-instruct-v1:0',
        'gateway/bedrock/us.meta.llama3-2-11b-instruct-v1:0',
        'gateway/bedrock/us.meta.llama3-2-1b-instruct-v1:0',
        'gateway/bedrock/us.meta.llama3-2-3b-instruct-v1:0',
        'gateway/bedrock/us.meta.llama3-2-90b-instruct-v1:0',
        'gateway/bedrock/us.meta.llama3-3-70b-instruct-v1:0',
        'gateway/google-vertex/gemini-2.0-flash-lite',
        'gateway/google-vertex/gemini-2.0-flash',
        'gateway/google-vertex/gemini-2.5-flash-image',
        'gateway/google-vertex/gemini-2.5-flash-lite-preview-09-2025',
        'gateway/google-vertex/gemini-2.5-flash-lite',
        'gateway/google-vertex/gemini-2.5-flash-preview-09-2025',
        'gateway/google-vertex/gemini-2.5-flash',
        'gateway/google-vertex/gemini-2.5-pro',
        'gateway/google-vertex/gemini-3-pro-image-preview',
        'gateway/google-vertex/gemini-3-pro-preview',
        'gateway/google-vertex/gemini-flash-latest',
        'gateway/google-vertex/gemini-flash-lite-latest',
        'gateway/groq/deepseek-r1-distill-llama-70b',
        'gateway/groq/deepseek-r1-distill-qwen-32b',
        'gateway/groq/distil-whisper-large-v3-en',
        'gateway/groq/gemma2-9b-it',
        'gateway/groq/llama-3.1-8b-instant',
        'gateway/groq/llama-3.2-11b-vision-preview',
        'gateway/groq/llama-3.2-1b-preview',
        'gateway/groq/llama-3.2-3b-preview',
        'gateway/groq/llama-3.2-90b-vision-preview',
        'gateway/groq/llama-3.3-70b-specdec',
        'gateway/groq/llama-3.3-70b-versatile',
        'gateway/groq/llama-guard-3-8b',
        'gateway/groq/llama3-70b-8192',
        'gateway/groq/llama3-8b-8192',
        'gateway/groq/mistral-saba-24b',
        'gateway/groq/moonshotai/kimi-k2-instruct',
        'gateway/groq/playai-tts-arabic',
        'gateway/groq/playai-tts',
        'gateway/groq/qwen-2.5-32b',
        'gateway/groq/qwen-2.5-coder-32b',
        'gateway/groq/qwen-qwq-32b',
        'gateway/groq/whisper-large-v3-turbo',
        'gateway/groq/whisper-large-v3',
        'gateway/openai/chatgpt-4o-latest',
        'gateway/openai/codex-mini-latest',
        'gateway/openai/computer-use-preview-2025-03-11',
        'gateway/openai/computer-use-preview',
        'gateway/openai/gpt-3.5-turbo-0125',
        'gateway/openai/gpt-3.5-turbo-0301',
        'gateway/openai/gpt-3.5-turbo-0613',
        'gateway/openai/gpt-3.5-turbo-1106',
        'gateway/openai/gpt-3.5-turbo-16k-0613',
        'gateway/openai/gpt-3.5-turbo-16k',
        'gateway/openai/gpt-3.5-turbo',
        'gateway/openai/gpt-4-0125-preview',
        'gateway/openai/gpt-4-0314',
        'gateway/openai/gpt-4-0613',
        'gateway/openai/gpt-4-1106-preview',
        'gateway/openai/gpt-4-32k-0314',
        'gateway/openai/gpt-4-32k-0613',
        'gateway/openai/gpt-4-32k',
        'gateway/openai/gpt-4-turbo-2024-04-09',
        'gateway/openai/gpt-4-turbo-preview',
        'gateway/openai/gpt-4-turbo',
        'gateway/openai/gpt-4-vision-preview',
        'gateway/openai/gpt-4.1-2025-04-14',
        'gateway/openai/gpt-4.1-mini-2025-04-14',
        'gateway/openai/gpt-4.1-mini',
        'gateway/openai/gpt-4.1-nano-2025-04-14',
        'gateway/openai/gpt-4.1-nano',
        'gateway/openai/gpt-4.1',
        'gateway/openai/gpt-4',
        'gateway/openai/gpt-4o-2024-05-13',
        'gateway/openai/gpt-4o-2024-08-06',
        'gateway/openai/gpt-4o-2024-11-20',
        'gateway/openai/gpt-4o-audio-preview-2024-10-01',
        'gateway/openai/gpt-4o-audio-preview-2024-12-17',
        'gateway/openai/gpt-4o-audio-preview-2025-06-03',
        'gateway/openai/gpt-4o-audio-preview',
        'gateway/openai/gpt-4o-mini-2024-07-18',
        'gateway/openai/gpt-4o-mini-audio-preview-2024-12-17',
        'gateway/openai/gpt-4o-mini-audio-preview',
        'gateway/openai/gpt-4o-mini-search-preview-2025-03-11',
        'gateway/openai/gpt-4o-mini-search-preview',
        'gateway/openai/gpt-4o-mini',
        'gateway/openai/gpt-4o-search-preview-2025-03-11',
        'gateway/openai/gpt-4o-search-preview',
        'gateway/openai/gpt-4o',
        'gateway/openai/gpt-5-2025-08-07',
        'gateway/openai/gpt-5-chat-latest',
        'gateway/openai/gpt-5-codex',
        'gateway/openai/gpt-5-mini-2025-08-07',
        'gateway/openai/gpt-5-mini',
        'gateway/openai/gpt-5-nano-2025-08-07',
        'gateway/openai/gpt-5-nano',
        'gateway/openai/gpt-5-pro-2025-10-06',
        'gateway/openai/gpt-5-pro',
        'gateway/openai/gpt-5.1-2025-11-13',
        'gateway/openai/gpt-5.1-chat-latest',
        'gateway/openai/gpt-5.1-codex-max',
        'gateway/openai/gpt-5.1-codex',
        'gateway/openai/gpt-5.1-mini',
        'gateway/openai/gpt-5.1',
        'gateway/openai/gpt-5.2-2025-12-11',
        'gateway/openai/gpt-5.2-chat-latest',
        'gateway/openai/gpt-5.2-pro-2025-12-11',
        'gateway/openai/gpt-5.2-pro',
        'gateway/openai/gpt-5.2',
        'gateway/openai/gpt-5',
        'gateway/openai/o1-2024-12-17',
        'gateway/openai/o1-mini-2024-09-12',
        'gateway/openai/o1-mini',
        'gateway/openai/o1-preview-2024-09-12',
        'gateway/openai/o1-preview',
        'gateway/openai/o1-pro-2025-03-19',
        'gateway/openai/o1-pro',
        'gateway/openai/o1',
        'gateway/openai/o3-2025-04-16',
        'gateway/openai/o3-deep-research-2025-06-26',
        'gateway/openai/o3-deep-research',
        'gateway/openai/o3-mini-2025-01-31',
        'gateway/openai/o3-mini',
        'gateway/openai/o3-pro-2025-06-10',
        'gateway/openai/o3-pro',
        'gateway/openai/o3',
        'gateway/openai/o4-mini-2025-04-16',
        'gateway/openai/o4-mini-deep-research-2025-06-26',
        'gateway/openai/o4-mini-deep-research',
        'gateway/openai/o4-mini',
        'google-gla/gemini-2.0-flash-lite',
        'google-gla/gemini-2.0-flash',
        'google-gla/gemini-2.5-flash-image',
        'google-gla/gemini-2.5-flash-lite-preview-09-2025',
        'google-gla/gemini-2.5-flash-lite',
        'google-gla/gemini-2.5-flash-preview-09-2025',
        'google-gla/gemini-2.5-flash',
        'google-gla/gemini-2.5-pro',
        'google-gla/gemini-3-pro-image-preview',
        'google-gla/gemini-3-pro-preview',
        'google-gla/gemini-flash-latest',
        'google-gla/gemini-flash-lite-latest',
        'google-vertex/gemini-2.0-flash-lite',
        'google-vertex/gemini-2.0-flash',
        'google-vertex/gemini-2.5-flash-image',
        'google-vertex/gemini-2.5-flash-lite-preview-09-2025',
        'google-vertex/gemini-2.5-flash-lite',
        'google-vertex/gemini-2.5-flash-preview-09-2025',
        'google-vertex/gemini-2.5-flash',
        'google-vertex/gemini-2.5-pro',
        'google-vertex/gemini-3-pro-image-preview',
        'google-vertex/gemini-3-pro-preview',
        'google-vertex/gemini-flash-latest',
        'google-vertex/gemini-flash-lite-latest',
        'grok/grok-2-image-1212',
        'grok/grok-2-vision-1212',
        'grok/grok-3-fast',
        'grok/grok-3-mini-fast',
        'grok/grok-3-mini',
        'grok/grok-3',
        'grok/grok-4-0709',
        'grok/grok-4-1-fast-non-reasoning',
        'grok/grok-4-1-fast-reasoning',
        'grok/grok-4-1-fast',
        'grok/grok-4-fast-non-reasoning',
        'grok/grok-4-fast-reasoning',
        'grok/grok-4-fast',
        'grok/grok-4',
        'grok/grok-code-fast-1',
        'groq/deepseek-r1-distill-llama-70b',
        'groq/deepseek-r1-distill-qwen-32b',
        'groq/distil-whisper-large-v3-en',
        'groq/gemma2-9b-it',
        'groq/llama-3.1-8b-instant',
        'groq/llama-3.2-11b-vision-preview',
        'groq/llama-3.2-1b-preview',
        'groq/llama-3.2-3b-preview',
        'groq/llama-3.2-90b-vision-preview',
        'groq/llama-3.3-70b-specdec',
        'groq/llama-3.3-70b-versatile',
        'groq/llama-guard-3-8b',
        'groq/llama3-70b-8192',
        'groq/llama3-8b-8192',
        'groq/mistral-saba-24b',
        'groq/moonshotai/kimi-k2-instruct',
        'groq/playai-tts-arabic',
        'groq/playai-tts',
        'groq/qwen-2.5-32b',
        'groq/qwen-2.5-coder-32b',
        'groq/qwen-qwq-32b',
        'groq/whisper-large-v3-turbo',
        'groq/whisper-large-v3',
        'heroku/amazon-rerank-1-0',
        'heroku/claude-3-5-haiku',
        'heroku/claude-3-5-sonnet-latest',
        'heroku/claude-3-7-sonnet',
        'heroku/claude-3-haiku',
        'heroku/claude-4-5-haiku',
        'heroku/claude-4-5-sonnet',
        'heroku/claude-4-sonnet',
        'heroku/cohere-rerank-3-5',
        'heroku/gpt-oss-120b',
        'heroku/nova-lite',
        'heroku/nova-pro',
        'huggingface/deepseek-ai/DeepSeek-R1',
        'huggingface/meta-llama/Llama-3.3-70B-Instruct',
        'huggingface/meta-llama/Llama-4-Maverick-17B-128E-Instruct',
        'huggingface/meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'huggingface/Qwen/Qwen2.5-72B-Instruct',
        'huggingface/Qwen/Qwen3-235B-A22B',
        'huggingface/Qwen/Qwen3-32B',
        'huggingface/Qwen/QwQ-32B',
        'mistral/codestral-latest',
        'mistral/mistral-large-latest',
        'mistral/mistral-moderation-latest',
        'mistral/mistral-small-latest',
        'moonshotai/kimi-k2-0711-preview',
        'moonshotai/kimi-latest',
        'moonshotai/kimi-thinking-preview',
        'moonshotai/moonshot-v1-128k-vision-preview',
        'moonshotai/moonshot-v1-128k',
        'moonshotai/moonshot-v1-32k-vision-preview',
        'moonshotai/moonshot-v1-32k',
        'moonshotai/moonshot-v1-8k-vision-preview',
        'moonshotai/moonshot-v1-8k',
        'openai/chatgpt-4o-latest',
        'openai/codex-mini-latest',
        'openai/computer-use-preview-2025-03-11',
        'openai/computer-use-preview',
        'openai/gpt-3.5-turbo-0125',
        'openai/gpt-3.5-turbo-0301',
        'openai/gpt-3.5-turbo-0613',
        'openai/gpt-3.5-turbo-1106',
        'openai/gpt-3.5-turbo-16k-0613',
        'openai/gpt-3.5-turbo-16k',
        'openai/gpt-3.5-turbo',
        'openai/gpt-4-0125-preview',
        'openai/gpt-4-0314',
        'openai/gpt-4-0613',
        'openai/gpt-4-1106-preview',
        'openai/gpt-4-32k-0314',
        'openai/gpt-4-32k-0613',
        'openai/gpt-4-32k',
        'openai/gpt-4-turbo-2024-04-09',
        'openai/gpt-4-turbo-preview',
        'openai/gpt-4-turbo',
        'openai/gpt-4-vision-preview',
        'openai/gpt-4.1-2025-04-14',
        'openai/gpt-4.1-mini-2025-04-14',
        'openai/gpt-4.1-mini',
        'openai/gpt-4.1-nano-2025-04-14',
        'openai/gpt-4.1-nano',
        'openai/gpt-4.1',
        'openai/gpt-4',
        'openai/gpt-4o-2024-05-13',
        'openai/gpt-4o-2024-08-06',
        'openai/gpt-4o-2024-11-20',
        'openai/gpt-4o-audio-preview-2024-10-01',
        'openai/gpt-4o-audio-preview-2024-12-17',
        'openai/gpt-4o-audio-preview-2025-06-03',
        'openai/gpt-4o-audio-preview',
        'openai/gpt-4o-mini-2024-07-18',
        'openai/gpt-4o-mini-audio-preview-2024-12-17',
        'openai/gpt-4o-mini-audio-preview',
        'openai/gpt-4o-mini-search-preview-2025-03-11',
        'openai/gpt-4o-mini-search-preview',
        'openai/gpt-4o-mini',
        'openai/gpt-4o-search-preview-2025-03-11',
        'openai/gpt-4o-search-preview',
        'openai/gpt-4o',
        'openai/gpt-5-2025-08-07',
        'openai/gpt-5-chat-latest',
        'openai/gpt-5-codex',
        'openai/gpt-5-mini-2025-08-07',
        'openai/gpt-5-mini',
        'openai/gpt-5-nano-2025-08-07',
        'openai/gpt-5-nano',
        'openai/gpt-5-pro-2025-10-06',
        'openai/gpt-5-pro',
        'openai/gpt-5.1-2025-11-13',
        'openai/gpt-5.1-chat-latest',
        'openai/gpt-5.1-codex-max',
        'openai/gpt-5.1-codex',
        'openai/gpt-5.1-mini',
        'openai/gpt-5.1',
        'openai/gpt-5.2-2025-12-11',
        'openai/gpt-5.2-chat-latest',
        'openai/gpt-5.2-pro-2025-12-11',
        'openai/gpt-5.2-pro',
        'openai/gpt-5.2',
        'openai/gpt-5',
        'openai/o1-2024-12-17',
        'openai/o1-mini-2024-09-12',
        'openai/o1-mini',
        'openai/o1-preview-2024-09-12',
        'openai/o1-preview',
        'openai/o1-pro-2025-03-19',
        'openai/o1-pro',
        'openai/o1',
        'openai/o3-2025-04-16',
        'openai/o3-deep-research-2025-06-26',
        'openai/o3-deep-research',
        'openai/o3-mini-2025-01-31',
        'openai/o3-mini',
        'openai/o3-pro-2025-06-10',
        'openai/o3-pro',
        'openai/o3',
        'openai/o4-mini-2025-04-16',
        'openai/o4-mini-deep-research-2025-06-26',
        'openai/o4-mini-deep-research',
        'openai/o4-mini',
        'test',
    ],
)
"""Known model names that can be used with the `model` parameter of [`Agent`][upsonic.Agent].

`KnownModelName` is provided as a concise way to specify a model.
"""


@dataclass(repr=False, kw_only=True)
class ModelRequestParameters:
    """Configuration for an agent's request to a model, specifically related to tools and output handling."""

    function_tools: list[ToolDefinition] = field(default_factory=list)
    builtin_tools: list[AbstractBuiltinTool] = field(default_factory=list)

    output_mode: OutputMode = 'text'
    output_object: OutputObjectDefinition | None = None
    output_tools: list[ToolDefinition] = field(default_factory=list)
    prompted_output_template: str | None = None
    allow_text_output: bool = True
    allow_image_output: bool = False

    @cached_property
    def tool_defs(self) -> dict[str, ToolDefinition]:
        return {tool_def.name: tool_def for tool_def in [*self.function_tools, *self.output_tools]}

    @cached_property
    def prompted_output_instructions(self) -> str | None:
        if self.output_mode == 'prompted' and self.prompted_output_template and self.output_object:
            return PromptedOutputSchema.build_instructions(self.prompted_output_template, self.output_object)
        return None

    __repr__ = _utils.dataclasses_no_defaults_repr


class Model(Runnable[Any, Any]):
    """Abstract class for a model.
    
    Model inherits from Runnable, making it compatible with UEL.
    This allows models to be used in chains with the pipe operator (|) for powerful composition:
    
    Example:
        ```python
        from upsonic.uel import ChatPromptTemplate
        from upsonic import infer_model
        
        prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
        model = infer_model("openai/gpt-4o")
        
        # Chain components together with the pipe operator
        chain = prompt | model
        result = await chain.invoke({"topic": "AI"})
        ```
    
    The Model class can also be configured with memory and tools:
        ```python
        # Add memory for conversation history
        model_with_memory = model.add_memory(history=True)
        
        # Bind tools for function calling
        model_with_tools = model.bind_tools([my_tool])
        
        # Configure structured output
        model_with_structure = model.with_structured_output(MyPydanticModel)
        
        # Chain all features together
        configured = model.add_memory(history=True).bind_tools([tool]).with_structured_output(Schema)
        
        # Use in chains
        chain = prompt | configured
        ```
    """

    _profile: ModelProfileSpec | None = None
    _settings: ModelSettings | None = None
    
    # UEL integration attributes
    _memory: Any = None  # Memory instance for chat history
    _tools: list[Any] | None = None  # List of tools to bind
    _tool_manager: Any = None  # ToolManager instance
    _tool_metrics: Any = None  # ToolMetrics for tool execution tracking
    _response_format: Any = None  # Pydantic model for structured output
    _tool_call_limit: int = 5  # Maximum tool calls per invocation
    _tool_call_count: int = 0  # Current tool call count

    def __init__(
        self,
        *,
        settings: ModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ) -> None:
        """Initialize the model with optional settings and profile.

        Args:
            settings: Model-specific settings that will be used as defaults for this model.
            profile: The model profile to use.
        """
        self._settings = settings
        self._profile = profile

    @property
    def settings(self) -> ModelSettings | None:
        """Get the model settings."""
        return self._settings

    @abstractmethod
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the model."""
        raise NotImplementedError()

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        """Make a request to the model for counting tokens."""
        # This method is not required, but you need to implement it if you want to support `UsageLimits.count_tokens_before_request`.
        raise NotImplementedError(f'Token counting ahead of the request is not supported by {self.__class__.__name__}')

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a request to the model and return a streaming response."""
        # This method is not required, but you need to implement it if you want to support streamed responses
        raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
        # yield is required to make this a generator for type checking
        # noinspection PyUnreachableCode
        yield  # pragma: no cover

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        """Customize the request parameters for the model.

        This method can be overridden by subclasses to modify the request parameters before sending them to the model.
        In particular, this method can be used to make modifications to the generated tool JSON schemas if necessary
        for vendor/model-specific reasons.
        """
        if transformer := self.profile.json_schema_transformer:
            model_request_parameters = replace(
                model_request_parameters,
                function_tools=[_customize_tool_def(transformer, t) for t in model_request_parameters.function_tools],
                output_tools=[_customize_tool_def(transformer, t) for t in model_request_parameters.output_tools],
            )
            if output_object := model_request_parameters.output_object:
                model_request_parameters = replace(
                    model_request_parameters,
                    output_object=_customize_output_object(transformer, output_object),
                )

        return model_request_parameters

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        """Prepare request inputs before they are passed to the provider.

        This merges the given `model_settings` with the model's own `settings` attribute and ensures
        `customize_request_parameters` is applied to the resolved
        [`ModelRequestParameters`][upsonic.models.ModelRequestParameters]. Subclasses can override this method if
        they need to customize the preparation flow further, but most implementations should simply call
        `self.prepare_request(...)` at the start of their `request` (and related) methods.
        """
        model_settings = merge_model_settings(self.settings, model_settings)

        params = self.customize_request_parameters(model_request_parameters)

        if builtin_tools := params.builtin_tools:
            # Deduplicate builtin tools
            params = replace(
                params,
                builtin_tools=list({tool.unique_id: tool for tool in builtin_tools}.values()),
            )

        if params.output_mode == 'auto':
            output_mode = self.profile.default_structured_output_mode
            params = replace(
                params,
                output_mode=output_mode,
                allow_text_output=output_mode in ('native', 'prompted'),
            )

        # Reset irrelevant fields
        if params.output_tools and params.output_mode != 'tool':
            params = replace(params, output_tools=[])
        if params.output_object and params.output_mode not in ('native', 'prompted'):
            params = replace(params, output_object=None)
        if params.prompted_output_template and params.output_mode != 'prompted':
            params = replace(params, prompted_output_template=None)  # pragma: no cover

        # Set default prompted output template
        if params.output_mode == 'prompted' and not params.prompted_output_template:
            params = replace(params, prompted_output_template=self.profile.prompted_output_template)

        # Check if output mode is supported
        if params.output_mode == 'native' and not self.profile.supports_json_schema_output:
            raise UserError('Native structured output is not supported by this model.')
        if params.output_mode == 'tool' and not self.profile.supports_tools:
            raise UserError('Tool output is not supported by this model.')
        if params.allow_image_output and not self.profile.supports_image_output:
            raise UserError('Image output is not supported by this model.')

        return model_settings, params

    
    
    def add_memory(
        self, 
        history: bool = False, 
        memory: Any = None
    ) -> "Model":
        """Add memory/chat history to the model for UEL chains.
        
        This enables the model to maintain conversation context across multiple invocations.
        
        Args:
            history: If True, creates an in-memory storage for chat history
            memory: Optional Memory instance to use for chat history management
            
        Returns:
            Self for method chaining
        """
        if memory is not None:
            self._memory = memory
        elif history:
            from upsonic.storage.memory import Memory
            from upsonic.storage.providers.in_memory import InMemoryStorage
            import uuid

            session_id = str(uuid.uuid4())
            storage = InMemoryStorage()
            self._memory = Memory(storage=storage, session_id=session_id, full_session_memory=True)
        
        return self
    
    def bind_tools(self, tools: list[Any], *, tool_call_limit: int | None = None) -> "Model":
        """Bind tools to the model for UEL chains.
        
        This enables the model to call tools during execution.
        
        Args:
            tools: List of tools (can be functions, Tool instances, or ToolConfig instances)
            tool_call_limit: Optional maximum number of tool calls per invocation (default: 5)
            
        Returns:
            Self for method chaining
        """
        self._tools = tools if tools else []
        if tool_call_limit is not None:
            self._tool_call_limit = tool_call_limit
        return self
    
    def with_structured_output(self, schema: Any) -> "Model":
        """Configure the model to return structured output in a Pydantic model format.
        
        This enables the model to return responses that conform to a specific schema,
        ensuring type-safe and validated outputs.
        
        Args:
            schema: A Pydantic model class that defines the output structure
            
        Returns:
            Self for method chaining
        """
        from pydantic import BaseModel
        
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise ValueError(
                f"schema must be a Pydantic BaseModel class, got {type(schema)}"
            )
        
        self._response_format = schema
        return self
    
    def invoke(
        self,
        input: str | "ModelRequest" | list["ModelMessage"],
        config: dict[str, Any] | None = None
    ) -> ModelResponse:
        """Execute the model synchronously for UEL compatibility.
        
        This is the synchronous version of invoke for UEL chains.
        It runs the async version in an event loop.
        """
        import asyncio
        
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.ainvoke(input, config))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input, config))
    
    async def ainvoke(
        self,
        input: str | "ModelRequest" | list["ModelMessage"],
        config: dict[str, Any] | None = None
    ) -> ModelResponse:
        """Execute the model with the configured memory and tools.
        
        This is the main entry point for UEL execution. It handles:
        - Converting input to proper message format
        - Loading chat history from memory (if configured)
        - Setting up and executing tools (if configured)
        - Handling tool calls in model responses
        - Saving conversation to memory (if configured)
        
        Args:
            input: Can be a string, ModelRequest, or list of ModelMessages
            config: Optional configuration dictionary
            
        Returns:
            The model's response (extracted output)
        """
        
        self._tool_call_count = 0
        
        messages = self._prepare_messages_for_invoke(input)
        
        if self._memory:
            messages = await self._build_messages_with_memory(messages)
        
        if self._tools:
            self._setup_tools_for_invoke()
        
        task = self._build_task_for_invoke(messages)

        task.price_id_ = None
        _ = task.price_id
        task._tool_calls = []
        
        model_params = self._build_parameters_for_invoke()
        model_params = self.customize_request_parameters(model_params)
        
        response = await self.request(
            messages=messages,
            model_settings=self.settings,
            model_request_parameters=model_params
        )
        
        if self._tools:
            response = await self._handle_response_with_tools(response, messages, model_params)
        
        output = self._extract_output_for_invoke(response)
        
        try:
            from upsonic.agent.context_managers import CallManager
            from upsonic.agent.run_result import RunResult
            
            run_result = RunResult(output=output)
            call_manager = CallManager(
                self,  # model
                task,
                debug=False,
                show_tool_calls=True
            )
            
            async with call_manager.manage_call() as call_handler:
                call_handler.process_response(run_result)
        except Exception:
            pass
        
        try:
            from upsonic.agent.context_managers import TaskManager
            from upsonic.agent.run_result import RunResult
            
            run_result_tm = RunResult(output=output)
            task_manager = TaskManager(task, None)
            async with task_manager.manage_task() as task_handler:
                task_handler.process_response(run_result_tm)
        except Exception:
            pass
        
        if self._memory:
            await self._save_to_memory(messages, response)
        
        return response
    
    def _prepare_messages_for_invoke(
        self, 
        input: str | "ModelRequest" | list["ModelMessage"]
    ) -> list["ModelMessage"]:
        """Convert input to list[ModelMessage] format.
        
        Args:
            input: String, ModelRequest, or list of ModelMessages
            
        Returns:
            List of ModelMessages ready for model request
        """
        from upsonic.messages import ModelRequest, UserPromptPart, ModelMessage
        
        if isinstance(input, str):
            user_part = UserPromptPart(content=input)
            request = ModelRequest(parts=[user_part])
            return [request]
        elif isinstance(input, ModelRequest):
            return [input]
        elif isinstance(input, list):
            return input
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    async def _build_messages_with_memory(
        self, 
        messages: list["ModelMessage"]
    ) -> list["ModelMessage"]:
        """Build messages including chat history from memory.
        
        Ensures that only the very first SystemPromptPart is kept in the
        final message list, removing any duplicate system prompts that might
        appear in later messages by directly modifying the parts attribute.
        
        Args:
            messages: Current messages to send
            
        Returns:
            Messages with history prepended and system prompts deduplicated
        """
        from upsonic.agent.context_managers import MemoryManager
        from upsonic.messages import ModelRequest, SystemPromptPart
        
        memory_manager = MemoryManager(self._memory)
        async with memory_manager.manage_memory() as memory_handler:
            message_history = memory_handler.get_message_history()
            
            combined_messages = message_history + list(messages)
            
            if not combined_messages:
                return combined_messages
            
            has_system_prompt_in_first = False
            if combined_messages and isinstance(combined_messages[0], ModelRequest):
                has_system_prompt_in_first = any(
                    isinstance(part, SystemPromptPart) 
                    for part in combined_messages[0].parts
                )
            
            if not has_system_prompt_in_first:
                return combined_messages
            

            for i, msg in enumerate(combined_messages[1:], start=1):
                if isinstance(msg, ModelRequest):
                    has_system_prompts = any(
                        isinstance(part, SystemPromptPart) 
                        for part in msg.parts
                    )
                    
                    if has_system_prompts:
                        filtered_parts = [
                            part for part in msg.parts 
                            if not isinstance(part, SystemPromptPart)
                        ]
                        
                        from dataclasses import replace
                        combined_messages[i] = replace(msg, parts=filtered_parts)
            
            return combined_messages
    
    def _setup_tools_for_invoke(self) -> None:
        """Setup ToolManager and register tools for invocation."""
        from upsonic.tools import ToolManager, ToolMetrics
        
        if not self._tool_manager:
            self._tool_manager = ToolManager()
        
        if not self._tool_metrics:
            self._tool_metrics = ToolMetrics(
                tool_call_count=self._tool_call_count,
                tool_call_limit=self._tool_call_limit
            )
        
        self._tool_manager.register_tools(
            tools=self._tools,
            task=None,
            agent_instance=None
        )
    
    def _build_parameters_for_invoke(self) -> ModelRequestParameters:
        """Build model request parameters including tools and structured output.
        
        Returns:
            ModelRequestParameters with tool definitions and output configuration
        """
        from pydantic import BaseModel
        from upsonic.output import OutputObjectDefinition
        
        tool_definitions = []
        builtin_tools = []
        
        if self._tools and self._tool_manager:
            tool_definitions = self._tool_manager.get_tool_definitions()
            builtin_tools = self._tool_manager.processor.extract_builtin_tools(self._tools)
        
        output_mode = 'text'
        output_object = None
        output_tools = []
        allow_text_output = True
        
        if self._response_format and self._response_format != str and self._response_format is not str:
            if isinstance(self._response_format, type) and issubclass(self._response_format, BaseModel):
                output_mode = 'auto'
                allow_text_output = False
                
                schema = self._response_format.model_json_schema()
                output_object = OutputObjectDefinition(
                    json_schema=schema,
                    name=self._response_format.__name__,
                    description=self._response_format.__doc__,
                    strict=True
                )
                
                # Create output tool for tool-based structured output
                from upsonic.tools import ToolDefinition
                output_tools = [ToolDefinition(
                    name=DEFAULT_OUTPUT_TOOL_NAME,
                    parameters_json_schema=schema,
                    description=self._response_format.__doc__ or f"Return the final result as a {self._response_format.__name__}",
                    kind='output',
                    strict=True
                )]
        
        return ModelRequestParameters(
            function_tools=tool_definitions,
            builtin_tools=builtin_tools,
            output_mode=output_mode,
            output_object=output_object,
            output_tools=output_tools,
            allow_text_output=allow_text_output
        )

    def _build_task_for_invoke(self, messages: list["ModelMessage"]) -> Any:
        """Create a Task object based on the latest user input and current configuration.
        
        Extracts the user's prompt from the last ModelRequest and builds a Task with
        description, tools, and response_format set appropriately.
        """
        try:
            from upsonic.tasks.tasks import Task
            from upsonic.messages import ModelRequest, UserPromptPart
            
            last_request = None
            for msg in reversed(messages):
                if isinstance(msg, ModelRequest):
                    last_request = msg
                    break
            
            description = ""
            if last_request is not None:
                for part in last_request.parts:
                    if isinstance(part, UserPromptPart):
                        description = str(part.content)
                        break
            
            task = Task(
                description=description,
                tools=list(self._tools or []),
                response_format=self._response_format if self._response_format is not None else str,
            )
            return task
        except Exception:
            # In case Task is unavailable or something goes wrong, return a minimal shim
            class _ShimTask:
                def __init__(self, description: str):
                    self.description = description
                    self.response = None
            
            desc = ""
            try:
                from upsonic.messages import ModelRequest, UserPromptPart
                for msg in reversed(messages):
                    if isinstance(msg, ModelRequest):
                        for part in msg.parts:
                            if isinstance(part, UserPromptPart):
                                desc = str(part.content)
                                raise StopIteration
            except StopIteration:
                pass
            return _ShimTask(desc)
    
    async def _handle_response_with_tools(
        self,
        response: "ModelResponse",
        messages: list["ModelMessage"],
        model_params: ModelRequestParameters
    ) -> "ModelResponse":
        """Handle tool calls in model response and execute them.
        
        Args:
            response: Initial model response
            messages: Current message history
            model_params: Model request parameters
            
        Returns:
            Final model response after tool execution
        """
        from upsonic.messages import ToolCallPart, ModelRequest
        import asyncio
        
        tool_calls = [
            part for part in response.parts 
            if isinstance(part, ToolCallPart)
        ]
        
        if not tool_calls:
            return response
        
        tool_results = await self._execute_tool_calls_for_invoke(tool_calls)
        
        messages.append(response)
        tool_request = ModelRequest(parts=tool_results)
        messages.append(tool_request)
        
        follow_up_response = await self.request(
            messages=messages,
            model_settings=self.settings,
            model_request_parameters=model_params
        )
        
        return await self._handle_response_with_tools(follow_up_response, messages, model_params)
    
    async def _execute_tool_calls_for_invoke(
        self, 
        tool_calls: list["ToolCallPart"]
    ) -> list["ToolReturnPart"]:
        """Execute tool calls and return results.
        
        Respects the tool call limit configured via bind_tools().
        
        Args:
            tool_calls: List of tool calls from model
            
        Returns:
            List of tool return parts
        """
        from upsonic.messages import ToolReturnPart
        from upsonic._utils import now_utc
        import asyncio
        
        if not tool_calls or not self._tool_manager:
            return []
        
        if self._tool_call_count >= self._tool_call_limit:
            error_results = []
            for tool_call in tool_calls:
                error_results.append(ToolReturnPart(
                    tool_name=tool_call.tool_name,
                    content=f"Tool call limit of {self._tool_call_limit} reached. Cannot execute more tools.",
                    tool_call_id=tool_call.tool_call_id,
                    timestamp=now_utc()
                ))
            return error_results
        
        if self._tool_call_count + len(tool_calls) > self._tool_call_limit:
            allowed_count = self._tool_call_limit - self._tool_call_count
            tool_calls_to_execute = tool_calls[:allowed_count]
            tool_calls_to_skip = tool_calls[allowed_count:]
            
            # Create error messages for skipped calls
            skipped_results = [
                ToolReturnPart(
                    tool_name=tc.tool_name,
                    content=f"Tool call limit of {self._tool_call_limit} would be exceeded. Tool call skipped.",
                    tool_call_id=tc.tool_call_id,
                    timestamp=now_utc()
                )
                for tc in tool_calls_to_skip
            ]
        else:
            tool_calls_to_execute = tool_calls
            skipped_results = []
        
        async def execute_single_tool(tool_call: "ToolCallPart") -> "ToolReturnPart":
            """Execute a single tool call."""
            try:
                result = await self._tool_manager.execute_tool(
                    tool_name=tool_call.tool_name,
                    args=tool_call.args_as_dict(),
                    tool_call_id=tool_call.tool_call_id
                )
                
                return ToolReturnPart(
                    tool_name=result.tool_name,
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                    timestamp=now_utc()
                )
            except Exception as e:
                return ToolReturnPart(
                    tool_name=tool_call.tool_name,
                    content=f"Error executing tool: {str(e)}",
                    tool_call_id=tool_call.tool_call_id,
                    timestamp=now_utc()
                )
        
        # Execute allowed tool calls in parallel
        executed_results = await asyncio.gather(
            *[execute_single_tool(tc) for tc in tool_calls_to_execute],
            return_exceptions=False
        )
        
        self._tool_call_count += len(tool_calls_to_execute)
        
        # Combine executed and skipped results
        return list(executed_results) + skipped_results
    
    def _extract_output_for_invoke(self, response: "ModelResponse") -> Any:
        """Extract output from model response.
        
        Handles both plain text output and structured Pydantic model output.
        Uses output parsers internally for modular parsing logic.
        
        Args:
            response: Model response
            
        Returns:
            Extracted output (string or Pydantic model instance)
        """
        from upsonic.uel.output_parser import StrOutputParser, PydanticOutputParser
        
        # Use StrOutputParser for string outputs
        if self._response_format == str or self._response_format is str:
            parser = StrOutputParser()
            return parser.parse(response)
        
        # Use PydanticOutputParser for structured outputs
        if self._response_format:
            try:
                parser = PydanticOutputParser(self._response_format)
                return parser.parse(response)
            except (ValueError, Exception):
                # If parsing fails, fall back to string output
                # This maintains backward compatibility
                parser = StrOutputParser()
                return parser.parse(response)
        
        # Default: return as string
        parser = StrOutputParser()
        return parser.parse(response)
    
    async def _save_to_memory(
        self, 
        messages: list["ModelMessage"],
        response: "ModelResponse"
    ) -> None:
        """Save messages and response to memory.
        
        Args:
            messages: Messages that were sent
            response: Model response to save
        """
        from upsonic.agent.context_managers import MemoryManager
        from upsonic.agent.run_result import RunResult
        
        if not self._memory:
            return
        
        memory_manager = MemoryManager(self._memory)
        async with memory_manager.manage_memory() as memory_handler:
            run_result = RunResult(output=None)
            
            # Add messages to run result
            # Note: We only add new messages (those not already in history)
            # The messages list already includes history, so we need to identify new ones
            history_length = len(memory_handler.get_message_history())
            if len(messages) > history_length:
                new_messages = messages[history_length:]
                run_result.add_messages(new_messages)
            
            run_result.add_message(response)
            
            memory_handler.process_response(run_result)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name."""
        raise NotImplementedError()

    @cached_property
    def profile(self) -> ModelProfile:
        """The model profile."""
        _profile = self._profile
        if callable(_profile):
            _profile = _profile(self.model_name)

        if _profile is None:
            return DEFAULT_PROFILE

        return _profile

    @property
    @abstractmethod
    def system(self) -> str:
        """The model provider, ex: openai.

        Use to populate the `gen_ai.system` OpenTelemetry semantic convention attribute,
        so should use well-known values listed in
        https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system
        when applicable.
        """
        raise NotImplementedError()

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None

    @staticmethod
    def _get_instructions(
        messages: list[ModelMessage], model_request_parameters: ModelRequestParameters | None = None
    ) -> str | None:
        """Get instructions from the first ModelRequest found when iterating messages in reverse.

        In the case that a "mock" request was generated to include a tool-return part for a result tool,
        we want to use the instructions from the second-to-most-recent request (which should correspond to the
        original request that generated the response that resulted in the tool-return part).
        """
        instructions = None

        last_two_requests: list[ModelRequest] = []
        for message in reversed(messages):
            if isinstance(message, ModelRequest):
                last_two_requests.append(message)
                if len(last_two_requests) == 2:
                    break
                if message.instructions is not None:
                    instructions = message.instructions
                    break

        # If we don't have two requests, and we didn't already return instructions, there are definitely not any:
        if instructions is None and len(last_two_requests) == 2:
            most_recent_request = last_two_requests[0]
            second_most_recent_request = last_two_requests[1]

            # If we've gotten this far and the most recent request consists of only tool-return parts or retry-prompt parts,
            # we use the instructions from the second-to-most-recent request. This is necessary because when handling
            # result tools, we generate a "mock" ModelRequest with a tool-return part for it, and that ModelRequest will not
            # have the relevant instructions from the agent.

            # While it's possible that you could have a message history where the most recent request has only tool returns,
            # I believe there is no way to achieve that would _change_ the instructions without manually crafting the most
            # recent message. That might make sense in principle for some usage pattern, but it's enough of an edge case
            # that I think it's not worth worrying about, since you can work around this by inserting another ModelRequest
            # with no parts at all immediately before the request that has the tool calls (that works because we only look
            # at the two most recent ModelRequests here).

            # If you have a use case where this causes pain, please open a GitHub issue and we can discuss alternatives.

            if all(p.part_kind == 'tool-return' or p.part_kind == 'retry-prompt' for p in most_recent_request.parts):
                instructions = second_most_recent_request.instructions

        if model_request_parameters and (output_instructions := model_request_parameters.prompted_output_instructions):
            if instructions:
                instructions = '\n\n'.join([instructions, output_instructions])
            else:
                instructions = output_instructions

        return instructions


@dataclass
class StreamedResponse(ABC):
    """Streamed response from an LLM when calling a tool."""

    model_request_parameters: ModelRequestParameters

    final_result_event: FinalResultEvent | None = field(default=None, init=False)

    provider_response_id: str | None = field(default=None, init=False)
    provider_details: dict[str, Any] | None = field(default=None, init=False)
    finish_reason: FinishReason | None = field(default=None, init=False)

    _parts_manager: ModelResponsePartsManager = field(default_factory=ModelResponsePartsManager, init=False)
    _event_iterator: AsyncIterator[ModelResponseStreamEvent] | None = field(default=None, init=False)
    _usage: RequestUsage = field(default_factory=RequestUsage, init=False)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream the response as an async iterable of [`ModelResponseStreamEvent`][upsonic.messages.ModelResponseStreamEvent]s.

        This proxies the `_event_iterator()` and emits all events, while also checking for matches
        on the result schema and emitting a [`FinalResultEvent`][upsonic.messages.FinalResultEvent] if/when the
        first match is found.
        """
        if self._event_iterator is None:

            async def iterator_with_final_event(
                iterator: AsyncIterator[ModelResponseStreamEvent],
            ) -> AsyncIterator[ModelResponseStreamEvent]:
                async for event in iterator:
                    yield event
                    if (
                        final_result_event := _get_final_result_event(event, self.model_request_parameters)
                    ) is not None:
                        self.final_result_event = final_result_event
                        yield final_result_event
                        break

                # If we broke out of the above loop, we need to yield the rest of the events
                # If we didn't, this will just be a no-op
                async for event in iterator:
                    yield event

            async def iterator_with_part_end(
                iterator: AsyncIterator[ModelResponseStreamEvent],
            ) -> AsyncIterator[ModelResponseStreamEvent]:
                last_start_event: PartStartEvent | None = None

                def part_end_event(next_part: ModelResponsePart | None = None) -> PartEndEvent | None:
                    if not last_start_event:
                        return None

                    index = last_start_event.index
                    part = self._parts_manager.get_parts()[index]
                    if not isinstance(part, TextPart | ThinkingPart | BaseToolCallPart):
                        # Parts other than these 3 don't have deltas, so don't need an end part.
                        return None

                    return PartEndEvent(
                        index=index,
                        part=part,
                        next_part_kind=next_part.part_kind if next_part else None,
                    )

                async for event in iterator:
                    if isinstance(event, PartStartEvent):
                        if last_start_event:
                            end_event = part_end_event(event.part)
                            if end_event:
                                yield end_event

                            event.previous_part_kind = last_start_event.part.part_kind
                        last_start_event = event

                    yield event

                end_event = part_end_event()
                if end_event:
                    yield end_event

            self._event_iterator = iterator_with_part_end(iterator_with_final_event(self._get_event_iterator()))
        return self._event_iterator

    @abstractmethod
    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Return an async iterator of [`ModelResponseStreamEvent`][upsonic.messages.ModelResponseStreamEvent]s.

        This method should be implemented by subclasses to translate the vendor-specific stream of events into
        upsonic-format events.

        It should use the `_parts_manager` to handle deltas, and should update the `_usage` attributes as it goes.
        """
        raise NotImplementedError()
        # noinspection PyUnreachableCode
        yield

    def get(self) -> ModelResponse:
        """Build a [`ModelResponse`][upsonic.messages.ModelResponse] from the data received from the stream so far."""
        return ModelResponse(
            parts=self._parts_manager.get_parts(),
            model_name=self.model_name,
            timestamp=self.timestamp,
            usage=self.usage(),
            provider_name=self.provider_name,
            provider_response_id=self.provider_response_id,
            provider_details=self.provider_details,
            finish_reason=self.finish_reason,
        )

    # TODO (v2): Make this a property
    def usage(self) -> RequestUsage:
        """Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
        return self._usage

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name of the response."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def provider_name(self) -> str | None:
        """Get the provider name."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        raise NotImplementedError()


ALLOW_MODEL_REQUESTS = True
"""Whether to allow requests to models.

This global setting allows you to disable request to most models, e.g. to make sure you don't accidentally
make costly requests to a model during tests.
"""


def check_allow_model_requests() -> None:
    """Check if model requests are allowed.

    If you're defining your own models that have costs or latency associated with their use, you should call this in
    [`Model.request`][upsonic.models.Model.request] and [`Model.request_stream`][upsonic.models.Model.request_stream].

    Raises:
        RuntimeError: If model requests are not allowed.
    """
    if not ALLOW_MODEL_REQUESTS:
        raise RuntimeError('Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False')


@contextmanager
def override_allow_model_requests(allow_model_requests: bool) -> Iterator[None]:
    """Context manager to temporarily override [`ALLOW_MODEL_REQUESTS`][upsonic.models.ALLOW_MODEL_REQUESTS].

    Args:
        allow_model_requests: Whether to allow model requests within the context.
    """
    global ALLOW_MODEL_REQUESTS
    old_value = ALLOW_MODEL_REQUESTS
    ALLOW_MODEL_REQUESTS = allow_model_requests  # pyright: ignore[reportConstantRedefinition]
    try:
        yield
    finally:
        ALLOW_MODEL_REQUESTS = old_value  # pyright: ignore[reportConstantRedefinition]



_OPENAI_CHAT_PROVIDERS = {
    'openai',
    'azure',
    'deepseek',
    'cerebras',
    'fireworks',
    'github',
    'grok',
    'heroku',
    'moonshotai',
    'ollama',
    'openrouter',
    'together',
    'vercel',
    'litellm',
    'nebius',
    'ovhcloud',
    'vllm',
    'nvidia',
}

_GOOGLE_PROVIDERS = {'google-gla', 'google-vertex'}


# --- Model ID Normalization ---

def _get_known_providers() -> set:
    """Extract all known provider names from KnownModelName."""
    providers = set()
    # Access the Literal type arguments from KnownModelName
    try:
        from typing import get_args
        known_names = get_args(KnownModelName.__value__)
        for name in known_names:
            if '/' in name and name != 'test':
                provider = name.split('/')[0]
                providers.add(provider)
    except Exception:
        # Fallback to common providers
        providers = {'openai', 'anthropic', 'bedrock', 'google-gla', 'google-vertex', 
                     'mistral', 'groq', 'cohere', 'grok', 'deepseek', 'cerebras'}
    return providers


def _build_model_alias_index() -> dict:
    """Build an index mapping simplified model names to full model IDs.
    
    For each known model, extracts a simplified version that users can use.
    Returns: {
        "provider/simplified-name": {
            "version": "provider/full-model-id",
            ...
        }
    }
    """
    import re
    from typing import get_args
    
    alias_index = {}
    
    try:
        known_names = get_args(KnownModelName.__value__)
    except Exception:
        return alias_index
    
    for full_id in known_names:
        if '/' not in full_id or full_id == 'test':
            continue
            
        provider, model_name = full_id.split('/', maxsplit=1)
        
        # Handle Bedrock models: anthropic.claude-3-5-sonnet-20241022-v2:0 -> claude-3-5-sonnet
        if provider == 'bedrock':
            # Pattern: [region.]vendor.model-name-date-version:revision
            # Example: anthropic.claude-3-5-sonnet-20241022-v2:0
            # Example: us.anthropic.claude-3-5-sonnet-20241022-v2:0 (cross-region profile)
            
            # Check if this is a regional prefix model (us., eu., global.)
            has_region_prefix = any(model_name.startswith(p) for p in ('us.', 'eu.', 'global.'))
            
            # Remove vendor prefix if present (e.g., anthropic. -> "")
            if '.' in model_name:
                parts = model_name.split('.')
                # Get the part after vendor (e.g., claude-3-5-sonnet-20241022-v2:0)
                model_part = parts[-1] if len(parts) > 1 else model_name
                # Extract base model name and version (remove date and version suffix)
                base_match = re.match(r'^(.+?)(?:-\d{8})?(-v\d+)?(?::\d+)?$', model_part)
                if base_match:
                    simplified = base_match.group(1)
                    version_match = base_match.group(2)
                    version = version_match.lstrip('-') if version_match else 'latest'
                else:
                    continue
            else:
                continue
            
            alias_key = f"{provider}/{simplified}"
            if alias_key not in alias_index:
                alias_index[alias_key] = {}
            
            # Prefer non-prefixed models (direct models) over regional inference profiles
            # Only add if: (1) version not set, or (2) this is non-prefixed and existing is prefixed
            if version not in alias_index[alias_key]:
                alias_index[alias_key][version] = full_id
            elif not has_region_prefix:
                # Non-prefixed model takes priority - overwrite prefixed one
                alias_index[alias_key][version] = full_id
            
            # Set as latest: prefer non-prefixed models
            if 'latest' not in alias_index[alias_key]:
                alias_index[alias_key]['latest'] = full_id
            elif not has_region_prefix:
                # Non-prefixed model takes priority for latest
                alias_index[alias_key]['latest'] = full_id
        else:
            # For other providers: extract base model name
            # Example: gpt-4o-2024-05-13 -> gpt-4o
            # Example: claude-3-5-sonnet-latest -> claude-3-5-sonnet
            
            # Try to extract date suffix
            date_match = re.match(r'^(.+?)(?:-\d{4}-\d{2}-\d{2})?(-\d{8})?$', model_name)
            if date_match and date_match.group(2):
                simplified = date_match.group(1)
                version = date_match.group(2).lstrip('-')
            elif '-latest' in model_name:
                simplified = model_name.replace('-latest', '')
                version = 'latest'
            else:
                simplified = model_name
                version = 'latest'
            
            alias_key = f"{provider}/{simplified}"
            if alias_key not in alias_index:
                alias_index[alias_key] = {}
            alias_index[alias_key][version] = full_id
            # Set as latest if not already set
            if 'latest' not in alias_index[alias_key]:
                alias_index[alias_key]['latest'] = full_id
    
    # Clean up internal tracking keys
    for key in alias_index:
        alias_index[key].pop('_max_version', None)
    
    return alias_index


# Cache the indexes
_KNOWN_PROVIDERS_CACHE = None
_MODEL_ALIAS_INDEX_CACHE = None


def _get_cached_known_providers() -> set:
    """Get cached known providers set."""
    global _KNOWN_PROVIDERS_CACHE
    if _KNOWN_PROVIDERS_CACHE is None:
        _KNOWN_PROVIDERS_CACHE = _get_known_providers()
    return _KNOWN_PROVIDERS_CACHE


def _get_cached_model_alias_index() -> dict:
    """Get cached model alias index."""
    global _MODEL_ALIAS_INDEX_CACHE
    if _MODEL_ALIAS_INDEX_CACHE is None:
        _MODEL_ALIAS_INDEX_CACHE = _build_model_alias_index()
    return _MODEL_ALIAS_INDEX_CACHE


def normalize_model_id(model_id: str) -> str:
    """Normalize a simplified model ID to its full format.
    
    Supports formats like:
    - bedrock/claude-3-5-sonnet:v2 -> bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
    - openai/gpt-4o:latest -> openai/gpt-4o
    - ollama/llama3.1:8b -> ollama/llama3.1:8b (pass-through, unknown provider)
    
    Args:
        model_id: The model ID to normalize (e.g., "bedrock/claude-3-5-sonnet:v2")
        
    Returns:
        The normalized full model ID, or the original if no alias found or provider unknown.
    """
    if '/' not in model_id:
        return model_id
    
    # Check if already a known model (exact match)
    try:
        from typing import get_args
        known_names = get_args(KnownModelName.__value__)
        if model_id in known_names:
            return model_id
    except Exception:
        pass
    
    # Parse provider and model:version
    provider, rest = model_id.split('/', maxsplit=1)
    
    # Check if provider is known
    known_providers = _get_cached_known_providers()
    if provider not in known_providers:
        # Unknown provider (like ollama), pass through as-is
        return model_id
    
    # Parse model name and version
    if ':' in rest:
        model_name, version = rest.rsplit(':', maxsplit=1)
    else:
        model_name = rest
        version = 'latest'  # Default to latest
    
    # Look up in alias index
    alias_index = _get_cached_model_alias_index()
    alias_key = f"{provider}/{model_name}"
    
    if alias_key in alias_index:
        versions = alias_index[alias_key]
        if version in versions:
            return versions[version]
        elif 'latest' in versions:
            return versions['latest']
    
    # No alias found, return original
    return model_id

def infer_model(  # noqa: C901
    model: Model | KnownModelName | str, provider_factory: Callable[[str], Provider[Any]] = infer_provider
) -> Model:
    """Infer the model from the name.

    Args:
        model:
            Model name to instantiate, in the format of `provider/model`. Use the string "test" to instantiate TestModel.
        provider_factory:
            Function that instantiates a provider object. The provider name is passed into the function parameter. Defaults to `provider.infer_provider`.
    """
    
    # Check for custom_provider from environment variable
    custom_provider = os.getenv("LLM_CUSTOM_PROVIDER") if os.getenv("LLM_CUSTOM_PROVIDER", None) else None
    
    env_set_model = os.getenv("LLM_MODEL_KEY") if os.getenv("LLM_MODEL_KEY", None) else "openai/gpt-4o"
    bypass_llm_model = None
    try:
        from celery import current_task

        task_id = current_task.request.id
        task_args = current_task.request.args
        task_kwargs = current_task.request.kwargs

        
        if task_kwargs.get("bypass_llm_model", None) is not None:
            bypass_llm_model = task_kwargs.get("bypass_llm_model")

    except Exception as e:
        pass


    if bypass_llm_model is not None:
        model = bypass_llm_model
        
    elif env_set_model is not None and env_set_model != "openai/gpt-4o":
        model = env_set_model

    if isinstance(model, Model):
        return model

    # Normalize simplified model IDs to full format
    # e.g., "bedrock/claude-3-5-sonnet:v2" -> "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    model = normalize_model_id(model)

    try:
        provider_name, model_name = model.split('/', maxsplit=1)
    except ValueError:
        provider_name = None
        model_name = model
        if model_name.startswith(('gpt', 'o1', 'o3')):
            provider_name = 'openai'
        elif model_name.startswith('claude'):
            provider_name = 'anthropic'
        elif model_name.startswith('gemini'):
            provider_name = 'google-gla'

        if provider_name is not None:
            warnings.warn(
                f"Specifying a model name without a provider prefix is deprecated. Instead of {model_name!r}, use '{provider_name}/{model_name}'.",
                DeprecationWarning,
            )
        else:
            raise UserError(f'Unknown model: {model}')

    if provider_name == 'vertexai':  # pragma: no cover
        warnings.warn(
            "The 'vertexai' provider name is deprecated. Use 'google-vertex' instead.",
            DeprecationWarning,
        )
        provider_name = 'google-vertex'

    # Use custom_provider if it's set from environment variable
    # custom_provider from os.getenv is always a string (provider name) or None
    if custom_provider is not None:
        # Use the custom provider name (override the inferred one)
        provider_name = custom_provider
        provider: Provider[Any] = provider_factory(provider_name)
    else:
        provider: Provider[Any] = provider_factory(provider_name)

    model_kind = provider_name
    if model_kind.startswith('gateway/'):
        from ..providers.gateway import normalize_gateway_provider

        model_kind = provider_name.removeprefix('gateway/')
        model_kind = normalize_gateway_provider(model_kind)

    if model_kind in _OPENAI_CHAT_PROVIDERS:
        model_kind = 'openai-chat'
    elif model_kind in _GOOGLE_PROVIDERS:
        model_kind = 'google'

    if model_kind == 'openai-chat':
        from .openai import OpenAIChatModel

        return OpenAIChatModel(model_name, provider=provider)
    elif model_kind == 'openai-responses':
        from .openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model_name, provider=provider)
    elif model_kind == 'google':
        from .google import GoogleModel

        return GoogleModel(model_name, provider=provider)
    elif model_kind == 'groq':
        from .groq import GroqModel

        return GroqModel(model_name, provider=provider)
    elif model_kind == 'cohere':
        from .cohere import CohereModel

        return CohereModel(model_name, provider=provider)
    elif model_kind == 'mistral':
        from .mistral import MistralModel

        return MistralModel(model_name, provider=provider)
    elif model_kind == 'anthropic':
        from .anthropic import AnthropicModel

        return AnthropicModel(model_name, provider=provider)
    elif model_kind == 'bedrock':
        from .bedrock import BedrockConverseModel

        return BedrockConverseModel(model_name, provider=provider)
    elif model_kind == 'huggingface':
        from .huggingface import HuggingFaceModel

        return HuggingFaceModel(model_name, provider=provider)
    else:
        raise UserError(f'Unknown model: {model}')  # pragma: no cover


def cached_async_http_client(*, provider: str | None = None, timeout: int = 600, connect: int = 5) -> httpx.AsyncClient:
    """Cached HTTPX async client that creates a separate client for each provider.

    The client is cached based on the provider parameter. If provider is None, it's used for non-provider specific
    requests (like downloading images). Multiple agents and calls can share the same client when they use the same provider.

    Each client will get its own transport with its own connection pool. The default pool size is defined by `httpx.DEFAULT_LIMITS`.

    There are good reasons why in production you should use a `httpx.AsyncClient` as an async context manager as
    described in [encode/httpx#2026](https://github.com/encode/httpx/pull/2026), but when experimenting or showing
    examples, it's very useful not to.

    The default timeouts match those of OpenAI,
    see <https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9>.
    """
    client = _cached_async_http_client(provider=provider, timeout=timeout, connect=connect)
    if client.is_closed:
        # This happens if the context manager is used, so we need to create a new client.
        # Since there is no API from `functools.cache` to clear the cache for a specific
        #  key, clear the entire cache here as a workaround.
        _cached_async_http_client.cache_clear()
        client = _cached_async_http_client(provider=provider, timeout=timeout, connect=connect)
    return client


@cache
def _cached_async_http_client(provider: str | None, timeout: int = 600, connect: int = 5) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


DataT = TypeVar('DataT', str, bytes)


class DownloadedItem(TypedDict, Generic[DataT]):
    """The downloaded data and its type."""

    data: DataT
    """The downloaded data."""

    data_type: str
    """The type of data that was downloaded.

    Extracted from header "content-type", but defaults to the media type inferred from the file URL if content-type is "application/octet-stream".
    """


@overload
async def download_item(
    item: FileUrl,
    data_format: Literal['bytes'],
    type_format: Literal['mime', 'extension'] = 'mime',
) -> DownloadedItem[bytes]: ...


@overload
async def download_item(
    item: FileUrl,
    data_format: Literal['base64', 'base64_uri', 'text'],
    type_format: Literal['mime', 'extension'] = 'mime',
) -> DownloadedItem[str]: ...


async def download_item(
    item: FileUrl,
    data_format: Literal['bytes', 'base64', 'base64_uri', 'text'] = 'bytes',
    type_format: Literal['mime', 'extension'] = 'mime',
) -> DownloadedItem[str] | DownloadedItem[bytes]:
    """Download an item by URL and return the content as a bytes object or a (base64-encoded) string.

    Args:
        item: The item to download.
        data_format: The format to return the content in:
            - `bytes`: The raw bytes of the content.
            - `base64`: The base64-encoded content.
            - `base64_uri`: The base64-encoded content as a data URI.
            - `text`: The content as a string.
        type_format: The format to return the media type in:
            - `mime`: The media type as a MIME type.
            - `extension`: The media type as an extension.

    Raises:
        UserError: If the URL points to a YouTube video or its protocol is gs://.
    """
    if item.url.startswith('gs://'):
        raise UserError('Downloading from protocol "gs://" is not supported.')
    elif isinstance(item, VideoUrl) and item.is_youtube:
        raise UserError('Downloading YouTube videos is not supported.')

    client = cached_async_http_client()
    response = await client.get(item.url, follow_redirects=True)
    response.raise_for_status()

    if content_type := response.headers.get('content-type'):
        content_type = content_type.split(';')[0]
        if content_type == 'application/octet-stream':
            content_type = None

    media_type = content_type or item.media_type

    data_type = media_type
    if type_format == 'extension':
        data_type = item.format

    data = response.content
    if data_format in ('base64', 'base64_uri'):
        data = base64.b64encode(data).decode('utf-8')
        if data_format == 'base64_uri':
            data = f'data:{media_type};base64,{data}'
        return DownloadedItem[str](data=data, data_type=data_type)
    elif data_format == 'text':
        return DownloadedItem[str](data=data.decode('utf-8'), data_type=data_type)
    else:
        return DownloadedItem[bytes](data=data, data_type=data_type)


@cache
def get_user_agent() -> str:
    """Get the user agent string for the HTTP client."""
    from .. import __version__

    return f'upsonic/{__version__}'


def _customize_tool_def(transformer: type[JsonSchemaTransformer], t: ToolDefinition):
    schema_transformer = transformer(t.parameters_json_schema, strict=t.strict)
    parameters_json_schema = schema_transformer.walk()
    return replace(
        t,
        parameters_json_schema=parameters_json_schema,
        strict=schema_transformer.is_strict_compatible if t.strict is None else t.strict,
    )


def _customize_output_object(transformer: type[JsonSchemaTransformer], o: OutputObjectDefinition):
    schema_transformer = transformer(o.json_schema, strict=o.strict)
    json_schema = schema_transformer.walk()
    return replace(
        o,
        json_schema=json_schema,
        strict=schema_transformer.is_strict_compatible if o.strict is None else o.strict,
    )


def _get_final_result_event(e: ModelResponseStreamEvent, params: ModelRequestParameters) -> FinalResultEvent | None:
    """Return an appropriate FinalResultEvent if `e` corresponds to a part that will produce a final result."""
    if isinstance(e, PartStartEvent):
        new_part = e.part
        if (isinstance(new_part, TextPart) and params.allow_text_output) or (
            isinstance(new_part, FilePart) and params.allow_image_output and isinstance(new_part.content, BinaryImage)
        ):
            return FinalResultEvent(tool_name=None, tool_call_id=None)
        elif isinstance(new_part, ToolCallPart) and (tool_def := params.tool_defs.get(new_part.tool_name)):
            if tool_def.kind == 'output':
                return FinalResultEvent(tool_name=new_part.tool_name, tool_call_id=new_part.tool_call_id)
            elif tool_def.defer:
                return FinalResultEvent(tool_name=None, tool_call_id=None)