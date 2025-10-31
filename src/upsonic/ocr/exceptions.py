from __future__ import annotations


class OCRError(Exception):
    """Base exception for all OCR-related errors."""

    def __init__(self, message: str, error_code: str | None = None, original_error: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.original_error = original_error

    def __str__(self) -> str:
        base = self.message
        if self.error_code:
            base = f"[{self.error_code}] {base}"
        if self.original_error:
            base = f"{base} (Original: {str(self.original_error)})"
        return base


class OCRProviderError(OCRError):
    """Exception raised when an OCR provider encounters an error."""

    pass


class OCRFileNotFoundError(OCRError):
    """Exception raised when the specified file is not found."""

    pass


class OCRUnsupportedFormatError(OCRError):
    """Exception raised when the file format is not supported."""

    pass


class OCRProcessingError(OCRError):
    """Exception raised when OCR processing fails."""

    pass

