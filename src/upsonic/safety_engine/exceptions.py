class DisallowedOperation(Exception):
    """Exception raised when an operation is not allowed by policy"""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)