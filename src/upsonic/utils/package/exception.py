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
