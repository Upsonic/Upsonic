from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import time

from pydantic import BaseModel, Field, ConfigDict


@dataclass
class BoundingBox:
    """Bounding box for detected text."""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0


@dataclass
class OCRTextBlock:
    """A block of detected text with metadata."""
    text: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    page_number: int = 0
    language: Optional[str] = None


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    blocks: List[OCRTextBlock] = field(default_factory=list)
    confidence: float = 0.0
    page_count: int = 1
    processing_time_ms: float = 0.0
    provider: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "page_count": self.page_count,
            "processing_time_ms": self.processing_time_ms,
            "provider": self.provider,
            "blocks_count": len(self.blocks),
            "metadata": self.metadata,
        }


@dataclass
class OCRMetrics:
    """Metrics for OCR operations."""
    total_pages: int = 0
    total_characters: int = 0
    average_confidence: float = 0.0
    processing_time_ms: float = 0.0
    provider: str = ""
    files_processed: int = 0


class OCRConfig(BaseModel):
    """Configuration for OCR providers."""
    languages: List[str] = Field(default_factory=lambda: ['en'], description="Languages to detect")
    confidence_threshold: float = Field(default=0.0, description="Minimum confidence threshold")
    rotation_fix: bool = Field(default=False, description="Enable automatic rotation correction")
    enhance_contrast: bool = Field(default=False, description="Enhance image contrast before OCR")
    remove_noise: bool = Field(default=False, description="Apply noise reduction")
    pdf_dpi: int = Field(default=300, description="DPI for PDF rendering")
    preserve_formatting: bool = Field(default=True, description="Try to preserve text formatting")
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')


class OCRProvider(ABC):
    """
    Abstract base class for all OCR providers.
    
    This provides a unified interface for different OCR engines
    with support for various file formats and preprocessing options.
    """
    
    def __init__(self, config: Optional[OCRConfig] = None, **kwargs):
        """Initialize the OCR provider.
        
        Args:
            config: OCRConfig object with provider settings
            **kwargs: Additional provider-specific arguments
        """
        self.config = config or OCRConfig(**kwargs)
        self._metrics = OCRMetrics(provider=self.__class__.__name__)
        self._validate_dependencies()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The provider name."""
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """List of supported language codes."""
        raise NotImplementedError()
    
    @abstractmethod
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are installed."""
        raise NotImplementedError()
    
    @abstractmethod
    def _process_image(self, image, **kwargs) -> OCRResult:
        """Process a single image and extract text.
        
        Args:
            image: PIL Image object
            **kwargs: Additional provider-specific arguments
            
        Returns:
            OCRResult object
        """
        raise NotImplementedError()
    
    def get_text(self, file_path: Union[str, Path], **kwargs) -> str:
        """Extract text from an image or PDF file.
        
        This is the main public method for text extraction.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Extracted text as a string
        """
        result = self.process_file(file_path, **kwargs)
        return result.text
    
    def process_file(self, file_path: Union[str, Path], **kwargs) -> OCRResult:
        """Process a file and return detailed OCR results.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional provider-specific arguments
            
        Returns:
            OCRResult object with detailed information
        """
        from upsonic.ocr.utils import prepare_file_for_ocr
        
        start_time = time.time()
        
        # Merge kwargs with config
        processing_config = {
            'rotation_fix': kwargs.get('rotation_fix', self.config.rotation_fix),
            'enhance_contrast': kwargs.get('enhance_contrast', self.config.enhance_contrast),
            'remove_noise': kwargs.get('remove_noise', self.config.remove_noise),
            'pdf_dpi': kwargs.get('pdf_dpi', self.config.pdf_dpi),
        }
        
        # Prepare images from file
        images = prepare_file_for_ocr(file_path, **processing_config)
        
        # Process each image
        all_blocks = []
        all_text_parts = []
        total_confidence = 0.0
        
        for page_num, image in enumerate(images, start=1):
            page_result = self._process_image(image, **kwargs)
            
            # Adjust page numbers in blocks
            for block in page_result.blocks:
                block.page_number = page_num
                all_blocks.append(block)
            
            all_text_parts.append(page_result.text)
            total_confidence += page_result.confidence
        
        # Combine results
        combined_text = "\n\n".join(all_text_parts) if len(all_text_parts) > 1 else all_text_parts[0]
        avg_confidence = total_confidence / len(images) if images else 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        result = OCRResult(
            text=combined_text,
            blocks=all_blocks,
            confidence=avg_confidence,
            page_count=len(images),
            processing_time_ms=processing_time,
            provider=self.name,
            metadata={
                'file_path': str(file_path),
                'config': self.config.model_dump(),
            }
        )
        
        # Update metrics
        self._metrics.total_pages += len(images)
        self._metrics.total_characters += len(combined_text)
        self._metrics.processing_time_ms += processing_time
        self._metrics.files_processed += 1
        if all_blocks:
            self._metrics.average_confidence = sum(b.confidence for b in all_blocks) / len(all_blocks)
        
        return result
    
    def get_metrics(self) -> OCRMetrics:
        """Get current metrics for this OCR provider."""
        return self._metrics
    
    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self._metrics = OCRMetrics(provider=self.name)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this OCR provider."""
        return {
            'name': self.name,
            'supported_languages': self.supported_languages,
            'config': self.config.model_dump(),
            'metrics': {
                'files_processed': self._metrics.files_processed,
                'total_pages': self._metrics.total_pages,
                'total_characters': self._metrics.total_characters,
                'average_confidence': self._metrics.average_confidence,
            }
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

