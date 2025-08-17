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