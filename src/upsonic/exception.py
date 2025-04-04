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
