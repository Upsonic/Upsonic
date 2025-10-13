import json
from typing import Any

from pydantic_core import core_schema

class VectorDBError(Exception):
    """Base exception for all vector database provider errors."""
    pass

class VectorDBConnectionError(VectorDBError):
    """Raised when a connection to the database cannot be established or is lost."""
    pass

class CollectionDoesNotExistError(VectorDBError):
    """Raised when an operation is attempted on a collection that does not exist."""
    pass

class SearchError(VectorDBError):
    """Raised when a search or query operation fails."""
    pass

class UpsertError(VectorDBError):
    """Raised when a data ingestion (upsert) operation fails."""
    pass


class GuardrailValidationError(Exception):
    """Custom exception raised when a task fails validation after all retries."""
    pass

class NoAPIKeyException(Exception):
    """Raised when no API key is provided."""
    pass

class UnsupportedLLMModelException(Exception):
    """Raised when an unsupported LLM model is specified."""
    pass

class UnsupportedComputerUseModelException(Exception):
    """Raised when ComputerUse tools are used with an unsupported model."""
    pass

class ContextWindowTooSmallException(Exception):
    """Raised when the context window is too small for the input."""
    pass

class InvalidRequestException(Exception):
    """Raised when the request is invalid."""
    pass

class CallErrorException(Exception):
    """Raised when there is an error in making a call."""
    pass

class ServerStatusException(Exception):
    """Custom exception for server status check failures."""
    pass

class TimeoutException(Exception):
    """Custom exception for request timeout."""
    pass

class ToolError(Exception):
    """Raised when a tool encounters an error."""
    def __init__(self, message):
        self.message = message

# New exceptions for better error handling
class UupsonicError(Exception):
    """Base exception for all Upsonic-related errors."""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(message)

class AgentExecutionError(UupsonicError):
    """Raised when agent execution fails."""
    pass

class ModelConnectionError(UupsonicError):
    """Raised when there's an error connecting to the model."""
    pass

class TaskProcessingError(UupsonicError):
    """Raised when task processing fails."""
    pass

class ConfigurationError(UupsonicError):
    """Raised when there's a configuration error."""
    pass

class RetryExhaustedError(UupsonicError):
    """Raised when all retry attempts are exhausted."""
    pass


class ModelCapabilityError(UupsonicError):
    """
    Raised when a task requires a capability (e.g., video input)
    that the selected model does not support based on its registry entry.
    """
    def __init__(
        self,
        model_name: str,
        attachment_path: str,
        attachment_extension: str,
        required_capability: str,
        supported_extensions: list[str]
    ):
        # Base message
        message = (
            f"Model '{model_name}' does not support files with the extension '.{attachment_extension}' "
            f"(from attachment: '{attachment_path}').\n"
        )
        
        if supported_extensions:
            supported_str = ", ".join([f".{ext}" for ext in sorted(supported_extensions)])
            suggestion = f"Supported extensions for '{required_capability}' are: {supported_str}."
        else:
            suggestion = f"The model does not support any files for the '{required_capability}' capability."
            
        full_message = message + suggestion
        error_code = "MODEL_CAPABILITY_MISMATCH"
        super().__init__(message=full_message, error_code=error_code)

class AgentRunError(RuntimeError):
    """Base class for errors occurring during an agent run."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


class UsageLimitExceeded(AgentRunError):
    """Error raised when a Model's usage exceeds the specified limits."""


class UnexpectedModelBehavior(AgentRunError):
    """Error caused by unexpected Model behavior, e.g. an unexpected response code."""

    message: str
    """Description of the unexpected behavior."""
    body: str | None
    """The body of the response, if available."""

    def __init__(self, message: str, body: str | None = None):
        self.message = message
        if body is None:
            self.body: str | None = None
        else:
            try:
                self.body = json.dumps(json.loads(body), indent=2)
            except ValueError:
                self.body = body
        super().__init__(message)

    def __str__(self) -> str:
        if self.body:
            return f'{self.message}, body:\n{self.body}'
        else:
            return self.message

class UserError(RuntimeError):
    """Error caused by a usage mistake by the application developer — You!"""

    message: str
    """Description of the mistake."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ModelRetry(Exception):
    """Exception to raise when a tool function should be retried.

    The agent will return the message to the model and ask it to try calling the function/tool again.
    """

    message: str
    """The message to return to the model."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and other.message == self.message

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> core_schema.CoreSchema:
        """Pydantic core schema to allow `ModelRetry` to be (de)serialized."""
        schema = core_schema.typed_dict_schema(
            {
                'message': core_schema.typed_dict_field(core_schema.str_schema()),
                'kind': core_schema.typed_dict_field(core_schema.literal_schema(['model-retry'])),
            }
        )
        return core_schema.no_info_after_validator_function(
            lambda dct: ModelRetry(dct['message']),
            schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: {'message': x.message, 'kind': 'model-retry'},
                return_schema=schema,
            ),
        )

class ModelHTTPError(AgentRunError):
    """Raised when an model provider response has a status code of 4xx or 5xx."""

    status_code: int
    """The HTTP status code returned by the API."""

    model_name: str
    """The name of the model associated with the error."""

    body: object | None
    """The body of the response, if available."""

    message: str
    """The error message with the status code and response body, if available."""

    def __init__(self, status_code: int, model_name: str, body: object | None = None):
        self.status_code = status_code
        self.model_name = model_name
        self.body = body
        message = f'status_code: {status_code}, model_name: {model_name}, body: {body}'
        super().__init__(message)