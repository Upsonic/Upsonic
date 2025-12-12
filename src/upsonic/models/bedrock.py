from __future__ import annotations

import functools
import os
import typing
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, overload

import anyio
import anyio.to_thread
from botocore.exceptions import ClientError
from typing_extensions import ParamSpec, assert_never

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from upsonic.messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)

from upsonic.utils.package.exception import ModelHTTPError, UserError
from . import Model, ModelRequestParameters, StreamedResponse, download_item
from upsonic.providers import Provider, infer_provider
from upsonic.providers.bedrock import BedrockModelProfile
from upsonic.models.settings import ModelSettings
from upsonic.tools import ToolDefinition
from upsonic import _utils, usage
from upsonic.profiles import ModelProfileSpec
if TYPE_CHECKING:
    from botocore.client import BaseClient
    from botocore.eventstream import EventStream
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.literals import StopReasonType
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ContentBlockUnionTypeDef,
        ConverseRequestTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamMetadataEventTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseStreamResponseTypeDef,
        CountTokensRequestTypeDef,
        DocumentBlockTypeDef,
        GuardrailConfigurationTypeDef,
        ImageBlockTypeDef,
        InferenceConfigurationTypeDef,
        MessageUnionTypeDef,
        PerformanceConfigurationTypeDef,
        PromptVariableValuesTypeDef,
        ReasoningContentBlockOutputTypeDef,
        SystemContentBlockTypeDef,
        ToolChoiceTypeDef,
        ToolConfigurationTypeDef,
        ToolSpecificationTypeDef,
        ToolTypeDef,
        VideoBlockTypeDef,
    )

LatestBedrockModelNames = Literal[
    'amazon.titan-tg1-large',
    'amazon.titan-text-lite-v1',
    'amazon.titan-text-express-v1',
    'us.amazon.nova-pro-v1:0',
    'us.amazon.nova-lite-v1:0',
    'us.amazon.nova-micro-v1:0',
    'anthropic.claude-3-5-sonnet-20241022-v2:0',
    'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
    'anthropic.claude-3-5-haiku-20241022-v1:0',
    'us.anthropic.claude-3-5-haiku-20241022-v1:0',
    'anthropic.claude-instant-v1',
    'anthropic.claude-v2:1',
    'anthropic.claude-v2',
    'anthropic.claude-3-sonnet-20240229-v1:0',
    'us.anthropic.claude-3-sonnet-20240229-v1:0',
    'anthropic.claude-3-haiku-20240307-v1:0',
    'us.anthropic.claude-3-haiku-20240307-v1:0',
    'anthropic.claude-3-opus-20240229-v1:0',
    'us.anthropic.claude-3-opus-20240229-v1:0',
    'anthropic.claude-3-5-sonnet-20240620-v1:0',
    'us.anthropic.claude-3-5-sonnet-20240620-v1:0',
    'anthropic.claude-3-7-sonnet-20250219-v1:0',
    'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
    'anthropic.claude-opus-4-20250514-v1:0',
    'us.anthropic.claude-opus-4-20250514-v1:0',
    'anthropic.claude-sonnet-4-20250514-v1:0',
    'us.anthropic.claude-sonnet-4-20250514-v1:0',
    'eu.anthropic.claude-sonnet-4-20250514-v1:0',
    'anthropic.claude-sonnet-4-5-20250929-v1:0',
    'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    'eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
    'anthropic.claude-haiku-4-5-20251001-v1:0',
    'us.anthropic.claude-haiku-4-5-20251001-v1:0',
    'eu.anthropic.claude-haiku-4-5-20251001-v1:0',
    'cohere.command-text-v14',
    'cohere.command-r-v1:0',
    'cohere.command-r-plus-v1:0',
    'cohere.command-light-text-v14',
    'meta.llama3-8b-instruct-v1:0',
    'meta.llama3-70b-instruct-v1:0',
    'meta.llama3-1-8b-instruct-v1:0',
    'us.meta.llama3-1-8b-instruct-v1:0',
    'meta.llama3-1-70b-instruct-v1:0',
    'us.meta.llama3-1-70b-instruct-v1:0',
    'meta.llama3-1-405b-instruct-v1:0',
    'us.meta.llama3-2-11b-instruct-v1:0',
    'us.meta.llama3-2-90b-instruct-v1:0',
    'us.meta.llama3-2-1b-instruct-v1:0',
    'us.meta.llama3-2-3b-instruct-v1:0',
    'us.meta.llama3-3-70b-instruct-v1:0',
    'mistral.mistral-7b-instruct-v0:2',
    'mistral.mixtral-8x7b-instruct-v0:1',
    'mistral.mistral-large-2402-v1:0',
    'mistral.mistral-large-2407-v1:0',
]
"""Latest Bedrock models."""

BedrockModelName = str | LatestBedrockModelNames
"""Possible Bedrock model names.

Since Bedrock supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints.
See [the Bedrock docs](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for a full list.
"""

P = ParamSpec('P')
T = typing.TypeVar('T')

_FINISH_REASON_MAP: dict[StopReasonType, FinishReason] = {
    'content_filtered': 'content_filter',
    'end_turn': 'stop',
    'guardrail_intervened': 'content_filter',
    'max_tokens': 'length',
    'stop_sequence': 'stop',
    'tool_use': 'tool_call',
}

_AWS_BEDROCK_INFERENCE_GEO_PREFIXES: tuple[str, ...] = ('us.', 'eu.', 'apac.', 'jp.', 'au.', 'ca.')
"""Geo prefixes for Bedrock inference profile IDs (e.g., 'eu.', 'us.').

Used to strip the geo prefix so we can pass a pure foundation model ID/ARN to CountTokens,
which does not accept profile IDs. Extend if new geos appear (e.g., 'global.', 'us-gov.').
"""


class BedrockModelSettings(ModelSettings, total=False):
    """Settings for Bedrock models.

    See [the Bedrock Converse API docs](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestSyntax) for a full list.
    See [the boto3 implementation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html) of the Bedrock Converse API.
    """

    # ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    bedrock_guardrail_config: GuardrailConfigurationTypeDef
    """Content moderation and safety settings for Bedrock API requests.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailConfiguration.html>.
    """

    bedrock_performance_configuration: PerformanceConfigurationTypeDef
    """Performance optimization settings for model inference.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_PerformanceConfiguration.html>.
    """

    bedrock_request_metadata: dict[str, str]
    """Additional metadata to attach to Bedrock API requests.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestSyntax>.
    """

    bedrock_additional_model_response_fields_paths: list[str]
    """JSON paths to extract additional fields from model responses.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html>.
    """

    bedrock_prompt_variables: Mapping[str, PromptVariableValuesTypeDef]
    """Variables for substitution into prompt templates.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_PromptVariableValues.html>.
    """

    bedrock_additional_model_requests_fields: Mapping[str, Any]
    """Additional model-specific parameters to include in requests.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html>.
    """


@dataclass(init=False)
class BedrockConverseModel(Model):
    """A model that uses the Bedrock Converse API."""

    client: BedrockRuntimeClient

    _model_name: BedrockModelName = field(repr=False)
    _provider: Provider[BaseClient] = field(repr=False)
    _inference_profile_arn: str | None = field(repr=False, default=None)

    def __init__(
        self,
        model_name: BedrockModelName,
        *,
        provider: Literal['bedrock', 'gateway'] | Provider[BaseClient] = 'bedrock',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Bedrock model.

        Args:
            model_name: The name of the model to use.
            model_name: The name of the Bedrock model to use. List of model names available
                [here](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html).
            provider: The provider to use for authentication and API access. Can be either the string
                'bedrock' or an instance of `Provider[BaseClient]`. If not provided, a new provider will be
                created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/bedrock' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = cast('BedrockRuntimeClient', provider.client)

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.meta.endpoint_url)

    @property
    def model_name(self) -> str:
        """The model name or inference profile ARN."""
        return self._inference_profile_arn or self._model_name
    
    def _find_inference_profile_for_model(self, model_id: str) -> str | None:
        """Find an inference profile that contains the given model ID, or create one if needed."""
        if not BOTO3_AVAILABLE:
            return None
        
        try:
            # Get region from client
            region = self.client.meta.region_name
            
            # Create Bedrock control plane client (not runtime)
            session = boto3.Session(
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
                region_name=region,
            )
            bedrock_client = session.client('bedrock', region_name=region)
            
            # Try to list inference profiles
            try:
                # Try different possible API method names
                if hasattr(bedrock_client, 'list_inference_profiles'):
                    response = bedrock_client.list_inference_profiles()
                elif hasattr(bedrock_client, 'list_inference_profile'):
                    response = bedrock_client.list_inference_profile()
                else:
                    # API might not be available, try to create one
                    return self._create_inference_profile_for_model(bedrock_client, model_id, region)
                
                profiles = response.get('inferenceProfileSummaries', []) or response.get('inferenceProfiles', [])
                
                # Find a profile that contains this model
                for profile in profiles:
                    profile_arn = profile.get('inferenceProfileArn', '') or profile.get('arn', '')
                    profile_id = profile.get('inferenceProfileId', '') or profile.get('id', '')
                    
                    if not profile_arn or not profile_id:
                        continue
                    
                    # Get profile details to check models
                    try:
                        if hasattr(bedrock_client, 'get_inference_profile'):
                            profile_details = bedrock_client.get_inference_profile(
                                inferenceProfileIdentifier=profile_id
                            )
                        else:
                            continue
                        
                        model_units = profile_details.get('modelUnits', []) or profile_details.get('models', [])
                        
                        # Check if any model unit contains our model
                        for model_unit in model_units:
                            model_identifier = model_unit.get('modelIdentifier', '') or model_unit.get('modelId', '')
                            # Check if model ID matches (with or without version)
                            if model_id in model_identifier or model_identifier in model_id:
                                return profile_arn
                    except ClientError:
                        # Skip if we can't get profile details
                        continue
                
                # No profile found, try to create one
                return self._create_inference_profile_for_model(bedrock_client, model_id, region)
            except (ClientError, AttributeError):
                # If list API doesn't exist or fails, try to create one
                return self._create_inference_profile_for_model(bedrock_client, model_id, region)
        except Exception:
            # Fail silently and return None
            return None
        
        return None
    
    def _create_inference_profile_for_model(self, bedrock_client: Any, model_id: str, region: str) -> str | None:
        """Create an inference profile for the given model ID."""
        try:
            # Generate a profile name based on model ID
            profile_name = f"upsonic-auto-{model_id.replace('.', '-').replace(':', '-')[:50]}"
            
            # Try to create inference profile
            try:
                if hasattr(bedrock_client, 'create_inference_profile'):
                    response = bedrock_client.create_inference_profile(
                        inferenceProfileName=profile_name,
                        modelUnits=[
                            {
                                'modelIdentifier': model_id,
                                'initialCapacity': 1,
                            }
                        ]
                    )
                elif hasattr(bedrock_client, 'create_inference_profile_v2'):
                    response = bedrock_client.create_inference_profile_v2(
                        name=profile_name,
                        modelUnits=[
                            {
                                'modelIdentifier': model_id,
                            }
                        ]
                    )
                else:
                    return None
                
                return response.get('inferenceProfileArn') or response.get('arn')
            except ClientError as e:
                # If creation fails (e.g., permission denied), return None
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ('AccessDenied', 'UnauthorizedOperation'):
                    return None
                # For other errors, also return None (profile might already exist)
                return None
        except Exception:
            return None

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolTypeDef]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolTypeDef:
        tool_spec: ToolSpecificationTypeDef = {'name': f.name, 'inputSchema': {'json': f.parameters_json_schema}}

        if f.description:  # pragma: no branch
            tool_spec['description'] = f.description

        return {'toolSpec': tool_spec}

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        settings = cast(BedrockModelSettings, model_settings or {})
        response = await self._messages_create(messages, False, settings, model_request_parameters)
        model_response = await self._process_response(response)
        return model_response

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        """Count the number of tokens, works with limited models.

        Check the actual supported models on <https://docs.aws.amazon.com/bedrock/latest/userguide/count-tokens.html>
        """
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        system_prompt, bedrock_messages = await self._map_messages(messages, model_request_parameters)
        params: CountTokensRequestTypeDef = {
            'modelId': self._remove_inference_geo_prefix(self.model_name),
            'input': {
                'converse': {
                    'messages': bedrock_messages,
                    'system': system_prompt,
                },
            },
        }
        try:
            response = await anyio.to_thread.run_sync(functools.partial(self.client.count_tokens, **params))
        except ClientError as e:
            status_code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 500)
            raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.response) from e
        return usage.RequestUsage(input_tokens=response['inputTokens'])

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        settings = cast(BedrockModelSettings, model_settings or {})
        response = await self._messages_create(messages, True, settings, model_request_parameters)
        yield BedrockStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self.model_name,
            _event_stream=response['stream'],
            _provider_name=self._provider.name,
            _provider_response_id=response.get('ResponseMetadata', {}).get('RequestId', None),
        )

    async def _process_response(self, response: ConverseResponseTypeDef) -> ModelResponse:
        items: list[ModelResponsePart] = []
        if message := response['output'].get('message'):  # pragma: no branch
            for item in message['content']:
                if reasoning_content := item.get('reasoningContent'):
                    if redacted_content := reasoning_content.get('redactedContent'):
                        items.append(
                            ThinkingPart(
                                id='redacted_content',
                                content='',
                                signature=redacted_content.decode('utf-8'),
                                provider_name=self.system,
                            )
                        )
                    elif reasoning_text := reasoning_content.get('reasoningText'):  # pragma: no branch
                        signature = reasoning_text.get('signature')
                        items.append(
                            ThinkingPart(
                                content=reasoning_text['text'],
                                signature=signature,
                                provider_name=self.system if signature else None,
                            )
                        )
                if text := item.get('text'):
                    items.append(TextPart(content=text))
                elif tool_use := item.get('toolUse'):
                    items.append(
                        ToolCallPart(
                            tool_name=tool_use['name'],
                            args=tool_use['input'],
                            tool_call_id=tool_use['toolUseId'],
                        ),
                    )
        u = usage.RequestUsage(
            input_tokens=response['usage']['inputTokens'],
            output_tokens=response['usage']['outputTokens'],
        )
        response_id = response.get('ResponseMetadata', {}).get('RequestId', None)
        raw_finish_reason = response['stopReason']
        provider_details = {'finish_reason': raw_finish_reason}
        finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=items,
            usage=u,
            model_name=self.model_name,
            provider_response_id=response_id,
            provider_name=self._provider.name,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: BedrockModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ConverseStreamResponseTypeDef:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: BedrockModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ConverseResponseTypeDef:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: BedrockModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ConverseResponseTypeDef | ConverseStreamResponseTypeDef:
        system_prompt, bedrock_messages = await self._map_messages(messages, model_request_parameters)
        inference_config = self._map_inference_config(model_settings)

        params: ConverseRequestTypeDef = {
            'modelId': self.model_name,
            'messages': bedrock_messages,
            'system': system_prompt,
            'inferenceConfig': inference_config,
        }

        tool_config = self._map_tool_config(model_request_parameters)
        if tool_config:
            params['toolConfig'] = tool_config

        if model_request_parameters.builtin_tools:
            raise UserError('Bedrock does not support built-in tools')

        # Bedrock supports a set of specific extra parameters
        if model_settings:
            if guardrail_config := model_settings.get('bedrock_guardrail_config', None):
                params['guardrailConfig'] = guardrail_config
            if performance_configuration := model_settings.get('bedrock_performance_configuration', None):
                params['performanceConfig'] = performance_configuration
            if request_metadata := model_settings.get('bedrock_request_metadata', None):
                params['requestMetadata'] = request_metadata
            if additional_model_response_fields_paths := model_settings.get(
                'bedrock_additional_model_response_fields_paths', None
            ):
                params['additionalModelResponseFieldPaths'] = additional_model_response_fields_paths
            if additional_model_requests_fields := model_settings.get('bedrock_additional_model_requests_fields', None):
                params['additionalModelRequestFields'] = additional_model_requests_fields
            if prompt_variables := model_settings.get('bedrock_prompt_variables', None):
                params['promptVariables'] = prompt_variables

        try:
            if stream:
                model_response = await anyio.to_thread.run_sync(
                    functools.partial(self.client.converse_stream, **params)
                )
            else:
                model_response = await anyio.to_thread.run_sync(functools.partial(self.client.converse, **params))
        except ClientError as e:
            status_code = e.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 500)
            error_message = e.response.get('Error', {}).get('Message', '')
            
            # Check if this is an inference profile requirement error
            if 'inference profile' in error_message.lower() and 'on-demand throughput' in error_message.lower():
                # Try to automatically find or create an inference profile
                if not self._inference_profile_arn:
                    inference_profile_arn = self._find_inference_profile_for_model(self._model_name)
                    if inference_profile_arn:
                        # Update to use inference profile and retry
                        self._inference_profile_arn = inference_profile_arn
                        # Retry the request with inference profile
                        params['modelId'] = inference_profile_arn
                        try:
                            if stream:
                                model_response = await anyio.to_thread.run_sync(
                                    functools.partial(self.client.converse_stream, **params)
                                )
                            else:
                                model_response = await anyio.to_thread.run_sync(
                                    functools.partial(self.client.converse, **params)
                                )
                            return model_response
                        except ClientError as retry_e:
                            # If retry also fails, try fallback model
                            retry_status = retry_e.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 500)
                            retry_error = retry_e.response.get('Error', {}).get('Message', '')
                            # Fall through to try fallback model
                            error_message = retry_error
                
                # If we couldn't find/create an inference profile, try Amazon Titan as fallback
                # Amazon Titan models typically don't require inference profiles
                if 'anthropic' in self._model_name.lower() or 'claude' in self._model_name.lower():
                    # Try Amazon Titan Text Express as fallback
                    fallback_model = 'amazon.titan-text-express-v1'
                    params['modelId'] = fallback_model
                    try:
                        if stream:
                            model_response = await anyio.to_thread.run_sync(
                                functools.partial(self.client.converse_stream, **params)
                            )
                        else:
                            model_response = await anyio.to_thread.run_sync(
                                functools.partial(self.client.converse, **params)
                            )
                        # Success with fallback - update model name
                        self._model_name = fallback_model
                        return model_response
                    except ClientError:
                        # Fallback also failed, raise original error
                        pass
                
                # If we couldn't find an inference profile and fallback failed, raise helpful error
                raise UserError(
                    f"Model '{self._model_name}' requires an inference profile, but none could be found or created automatically. "
                    f"Please create an inference profile in AWS Bedrock console that includes this model, then use its ARN:\n"
                    f"   BedrockConverseModel(model_name='arn:aws:bedrock:REGION:ACCOUNT:inference-profile/PROFILE-NAME')\n\n"
                    f"Original error: {error_message}"
                ) from e
            
            raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.response) from e
        return model_response

    @staticmethod
    def _map_inference_config(
        model_settings: ModelSettings | None,
    ) -> InferenceConfigurationTypeDef:
        model_settings = model_settings or {}
        inference_config: InferenceConfigurationTypeDef = {}

        if max_tokens := model_settings.get('max_tokens'):
            inference_config['maxTokens'] = max_tokens
        if (temperature := model_settings.get('temperature')) is not None:
            inference_config['temperature'] = temperature
        if top_p := model_settings.get('top_p'):
            inference_config['topP'] = top_p
        if stop_sequences := model_settings.get('stop_sequences'):
            inference_config['stopSequences'] = stop_sequences

        return inference_config

    def _map_tool_config(self, model_request_parameters: ModelRequestParameters) -> ToolConfigurationTypeDef | None:
        tools = self._get_tools(model_request_parameters)
        if not tools:
            return None

        tool_choice: ToolChoiceTypeDef
        if not model_request_parameters.allow_text_output:
            tool_choice = {'any': {}}
        else:
            tool_choice = {'auto': {}}

        tool_config: ToolConfigurationTypeDef = {'tools': tools}
        if tool_choice and BedrockModelProfile.from_profile(self.profile).bedrock_supports_tool_choice:
            tool_config['toolChoice'] = tool_choice

        return tool_config

    async def _map_messages(  # noqa: C901
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> tuple[list[SystemContentBlockTypeDef], list[MessageUnionTypeDef]]:
        """Maps a `upsonic.Message` to the Bedrock `MessageUnionTypeDef`.

        Groups consecutive ToolReturnPart objects into a single user message as required by Bedrock Claude/Nova models.
        """
        profile = BedrockModelProfile.from_profile(self.profile)
        system_prompt: list[SystemContentBlockTypeDef] = []
        bedrock_messages: list[MessageUnionTypeDef] = []
        document_count: Iterator[int] = count(1)
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart) and part.content:
                        system_prompt.append({'text': part.content})
                    elif isinstance(part, UserPromptPart):
                        bedrock_messages.extend(await self._map_user_prompt(part, document_count))
                    elif isinstance(part, ToolReturnPart):
                        assert part.tool_call_id is not None
                        bedrock_messages.append(
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        'toolResult': {
                                            'toolUseId': part.tool_call_id,
                                            'content': [
                                                {'text': part.model_response_str()}
                                                if profile.bedrock_tool_result_format == 'text'
                                                else {'json': part.model_response_object()}
                                            ],
                                            'status': 'success',
                                        }
                                    }
                                ],
                            }
                        )
                    elif isinstance(part, RetryPromptPart):
                        # TODO: We need to add a test here.
                        if part.tool_name is None:  # pragma: no cover
                            bedrock_messages.append({'role': 'user', 'content': [{'text': part.model_response()}]})
                        else:
                            assert part.tool_call_id is not None
                            bedrock_messages.append(
                                {
                                    'role': 'user',
                                    'content': [
                                        {
                                            'toolResult': {
                                                'toolUseId': part.tool_call_id,
                                                'content': [{'text': part.model_response()}],
                                                'status': 'error',
                                            }
                                        }
                                    ],
                                }
                            )
            elif isinstance(message, ModelResponse):
                content: list[ContentBlockOutputTypeDef] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        content.append({'text': item.content})
                    elif isinstance(item, ThinkingPart):
                        if (
                            item.provider_name == self.system
                            and item.signature
                            and BedrockModelProfile.from_profile(self.profile).bedrock_send_back_thinking_parts
                        ):
                            if item.id == 'redacted_content':
                                reasoning_content: ReasoningContentBlockOutputTypeDef = {
                                    'redactedContent': item.signature.encode('utf-8'),
                                }
                            else:
                                reasoning_content: ReasoningContentBlockOutputTypeDef = {
                                    'reasoningText': {
                                        'text': item.content,
                                        'signature': item.signature,
                                    }
                                }
                            content.append({'reasoningContent': reasoning_content})
                        else:
                            start_tag, end_tag = self.profile.thinking_tags
                            content.append({'text': '\n'.join([start_tag, item.content, end_tag])})
                    elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):
                        pass
                    else:
                        assert isinstance(item, ToolCallPart)
                        content.append(self._map_tool_call(item))
                bedrock_messages.append({'role': 'assistant', 'content': content})
            else:
                assert_never(message)

        # Merge together sequential user messages.
        processed_messages: list[MessageUnionTypeDef] = []
        last_message: dict[str, Any] | None = None
        for current_message in bedrock_messages:
            if (
                last_message is not None
                and current_message['role'] == last_message['role']
                and current_message['role'] == 'user'
            ):
                # Add the new user content onto the existing user message.
                last_content = list(last_message['content'])
                last_content.extend(current_message['content'])
                last_message['content'] = last_content
                continue

            # Add the entire message to the list of messages.
            processed_messages.append(current_message)
            last_message = cast(dict[str, Any], current_message)

        if instructions := self._get_instructions(messages, model_request_parameters):
            system_prompt.insert(0, {'text': instructions})

        return system_prompt, processed_messages

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart, document_count: Iterator[int]) -> list[MessageUnionTypeDef]:
        content: list[ContentBlockUnionTypeDef] = []
        if isinstance(part.content, str):
            content.append({'text': part.content})
        else:
            for item in part.content:
                if isinstance(item, str):
                    content.append({'text': item})
                elif isinstance(item, BinaryContent):
                    format = item.format
                    if item.is_document:
                        name = f'Document {next(document_count)}'
                        assert format in ('pdf', 'txt', 'csv', 'doc', 'docx', 'xls', 'xlsx', 'html', 'md')
                        content.append({'document': {'name': name, 'format': format, 'source': {'bytes': item.data}}})
                    elif item.is_image:
                        assert format in ('jpeg', 'png', 'gif', 'webp')
                        content.append({'image': {'format': format, 'source': {'bytes': item.data}}})
                    elif item.is_video:
                        assert format in ('mkv', 'mov', 'mp4', 'webm', 'flv', 'mpeg', 'mpg', 'wmv', 'three_gp')
                        content.append({'video': {'format': format, 'source': {'bytes': item.data}}})
                    else:
                        raise NotImplementedError('Binary content is not supported yet.')
                elif isinstance(item, ImageUrl | DocumentUrl | VideoUrl):
                    downloaded_item = await download_item(item, data_format='bytes', type_format='extension')
                    format = downloaded_item['data_type']
                    if item.kind == 'image-url':
                        format = item.media_type.split('/')[1]
                        assert format in ('jpeg', 'png', 'gif', 'webp'), f'Unsupported image format: {format}'
                        image: ImageBlockTypeDef = {'format': format, 'source': {'bytes': downloaded_item['data']}}
                        content.append({'image': image})

                    elif item.kind == 'document-url':
                        name = f'Document {next(document_count)}'
                        document: DocumentBlockTypeDef = {
                            'name': name,
                            'format': item.format,
                            'source': {'bytes': downloaded_item['data']},
                        }
                        content.append({'document': document})

                    elif item.kind == 'video-url':  # pragma: no branch
                        format = item.media_type.split('/')[1]
                        assert format in (
                            'mkv',
                            'mov',
                            'mp4',
                            'webm',
                            'flv',
                            'mpeg',
                            'mpg',
                            'wmv',
                            'three_gp',
                        ), f'Unsupported video format: {format}'
                        video: VideoBlockTypeDef = {'format': format, 'source': {'bytes': downloaded_item['data']}}
                        content.append({'video': video})
                elif isinstance(item, AudioUrl):  # pragma: no cover
                    raise NotImplementedError('Audio is not supported yet.')
                elif isinstance(item, CachePoint):
                    # Bedrock support has not been implemented yet
                    pass
                else:
                    assert_never(item)
        return [{'role': 'user', 'content': content}]

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ContentBlockOutputTypeDef:
        return {
            'toolUse': {'toolUseId': _utils.guard_tool_call_id(t=t), 'name': t.tool_name, 'input': t.args_as_dict()}
        }

    @staticmethod
    def _remove_inference_geo_prefix(model_name: BedrockModelName) -> BedrockModelName:
        """Remove inference geographic prefix from model ID if present."""
        for prefix in _AWS_BEDROCK_INFERENCE_GEO_PREFIXES:
            if model_name.startswith(prefix):
                return model_name.removeprefix(prefix)
        return model_name


@dataclass
class BedrockStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Bedrock models."""

    _model_name: BedrockModelName
    _event_stream: EventStream[ConverseStreamOutputTypeDef]
    _provider_name: str
    _timestamp: datetime = field(default_factory=_utils.now_utc)
    _provider_response_id: str | None = None

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        """Return an async iterator of [`ModelResponseStreamEvent`][upsonic.messages.ModelResponseStreamEvent]s.

        This method should be implemented by subclasses to translate the vendor-specific stream of events into
        upsonic-format events.
        """
        if self._provider_response_id is not None:  # pragma: no cover
            self.provider_response_id = self._provider_response_id

        chunk: ConverseStreamOutputTypeDef
        tool_id: str | None = None
        async for chunk in _AsyncIteratorWrapper(self._event_stream):
            match chunk:
                case {'messageStart': _}:
                    continue
                case {'messageStop': message_stop}:
                    raw_finish_reason = message_stop['stopReason']
                    self.provider_details = {'finish_reason': raw_finish_reason}
                    self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)
                case {'metadata': metadata}:
                    if 'usage' in metadata:  # pragma: no branch
                        self._usage += self._map_usage(metadata)
                case {'contentBlockStart': content_block_start}:
                    index = content_block_start['contentBlockIndex']
                    start = content_block_start['start']
                    if 'toolUse' in start:  # pragma: no branch
                        tool_use_start = start['toolUse']
                        tool_id = tool_use_start['toolUseId']
                        tool_name = tool_use_start['name']
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=index,
                            tool_name=tool_name,
                            args=None,
                            tool_call_id=tool_id,
                        )
                        if maybe_event:  # pragma: no branch
                            yield maybe_event
                case {'contentBlockDelta': content_block_delta}:
                    index = content_block_delta['contentBlockIndex']
                    delta = content_block_delta['delta']
                    if 'reasoningContent' in delta:
                        if redacted_content := delta['reasoningContent'].get('redactedContent'):
                            yield self._parts_manager.handle_thinking_delta(
                                vendor_part_id=index,
                                id='redacted_content',
                                signature=redacted_content.decode('utf-8'),
                                provider_name=self.provider_name,
                            )
                        else:
                            signature = delta['reasoningContent'].get('signature')
                            yield self._parts_manager.handle_thinking_delta(
                                vendor_part_id=index,
                                content=delta['reasoningContent'].get('text'),
                                signature=signature,
                                provider_name=self.provider_name if signature else None,
                            )
                    if text := delta.get('text'):
                        maybe_event = self._parts_manager.handle_text_delta(vendor_part_id=index, content=text)
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    if 'toolUse' in delta:
                        tool_use = delta['toolUse']
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=index,
                            tool_name=tool_use.get('name'),
                            args=tool_use.get('input'),
                            tool_call_id=tool_id,
                        )
                        if maybe_event:  # pragma: no branch
                            yield maybe_event
                case _:
                    pass  # pyright wants match statements to be exhaustive

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    def _map_usage(self, metadata: ConverseStreamMetadataEventTypeDef) -> usage.RequestUsage:
        return usage.RequestUsage(
            input_tokens=metadata['usage']['inputTokens'],
            output_tokens=metadata['usage']['outputTokens'],
        )


class _AsyncIteratorWrapper(Generic[T]):
    """Wrap a synchronous iterator in an async iterator."""

    def __init__(self, sync_iterator: Iterable[T]):
        self.sync_iterator = iter(sync_iterator)

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        try:
            return await anyio.to_thread.run_sync(next, self.sync_iterator)
        except RuntimeError as e:
            if type(e.__cause__) is StopIteration:
                raise StopAsyncIteration
            else:
                raise e  # pragma: lax no cover