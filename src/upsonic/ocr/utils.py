from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from upsonic.ocr.exceptions import (
    OCRFileNotFoundError,
    OCRUnsupportedFormatError,
    OCRProcessingError,
)

try:
    from PIL import Image
    import numpy as np
    _PIL_AVAILABLE = True
except ImportError:
    Image = None
    np = None
    _PIL_AVAILABLE = False

try:
    import pdf2image
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    pdf2image = None
    _PDF2IMAGE_AVAILABLE = False

# Track if we need to check for poppler
_POPPLER_CHECKED = False


SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
SUPPORTED_PDF_FORMATS = {'.pdf'}
ALL_SUPPORTED_FORMATS = SUPPORTED_IMAGE_FORMATS | SUPPORTED_PDF_FORMATS


def check_dependencies():
    """Check if required dependencies are installed."""
    if not _PIL_AVAILABLE:
        from upsonic.utils.printing import import_error
        import_error(
            package_name="Pillow",
            install_command='pip install Pillow',
            feature_name="OCR image processing"
        )


def check_pdf_dependencies():
    """Check if PDF processing dependencies are installed."""
    if not _PDF2IMAGE_AVAILABLE:
        from upsonic.utils.printing import import_error
        import_error(
            package_name="pdf2image",
            install_command='pip install pdf2image',
            feature_name="PDF OCR processing"
        )


def check_poppler_installed():
    """Check if poppler is installed and provide helpful installation instructions."""
    from upsonic.utils.printing import error_message
    import platform
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        install_cmd = "brew install poppler"
        detail = (
            "Poppler is not installed or not in PATH.\n\n"
            "Poppler is required for PDF processing.\n\n"
            "Installation instructions for macOS:\n"
            f"  {install_cmd}\n\n"
            "After installation, restart your terminal or IDE."
        )
    elif system == "Linux":
        install_cmd = "sudo apt-get install poppler-utils"
        detail = (
            "Poppler is not installed or not in PATH.\n\n"
            "Poppler is required for PDF processing.\n\n"
            "Installation instructions for Linux:\n"
            f"  Ubuntu/Debian: {install_cmd}\n"
            "  Fedora/RHEL: sudo dnf install poppler-utils\n"
            "  Arch: sudo pacman -S poppler\n\n"
            "After installation, restart your terminal or IDE."
        )
    elif system == "Windows":
        detail = (
            "Poppler is not installed or not in PATH.\n\n"
            "Poppler is required for PDF processing.\n\n"
            "Installation instructions for Windows:\n"
            "  1. Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases/\n"
            "  2. Extract the archive to a location (e.g., C:\\Program Files\\poppler)\n"
            "  3. Add the 'bin' folder to your system PATH:\n"
            "     - Search 'Environment Variables' in Windows\n"
            "     - Edit 'Path' in System Variables\n"
            "     - Add the path to poppler's bin folder\n"
            "  4. Restart your terminal or IDE\n\n"
            "Alternative: Install via conda:\n"
            "  conda install -c conda-forge poppler"
        )
    else:
        detail = (
            "Poppler is not installed or not in PATH.\n\n"
            "Poppler is required for PDF processing.\n\n"
            "Please install poppler-utils for your operating system.\n"
            "Visit: https://poppler.freedesktop.org/"
        )
    
    error_message(
        error_type="Poppler Not Installed",
        detail=detail
    )


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """Validate that the file exists and is readable.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Path object of the validated file
        
    Raises:
        OCRFileNotFoundError: If the file does not exist
    """
    path = Path(file_path)
    if not path.exists():
        raise OCRFileNotFoundError(
            f"File not found: {file_path}",
            error_code="FILE_NOT_FOUND"
        )
    if not path.is_file():
        raise OCRFileNotFoundError(
            f"Path is not a file: {file_path}",
            error_code="NOT_A_FILE"
        )
    return path


def get_file_format(file_path: Union[str, Path]) -> str:
    """Get the file format from the file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension in lowercase (e.g., '.pdf', '.png')
        
    Raises:
        OCRUnsupportedFormatError: If the format is not supported
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext not in ALL_SUPPORTED_FORMATS:
        raise OCRUnsupportedFormatError(
            f"Unsupported file format: {ext}. Supported formats: {', '.join(ALL_SUPPORTED_FORMATS)}",
            error_code="UNSUPPORTED_FORMAT"
        )
    
    return ext


def is_pdf(file_path: Union[str, Path]) -> bool:
    """Check if the file is a PDF.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a PDF, False otherwise
    """
    return get_file_format(file_path) in SUPPORTED_PDF_FORMATS


def is_image(file_path: Union[str, Path]) -> bool:
    """Check if the file is an image.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is an image, False otherwise
    """
    return get_file_format(file_path) in SUPPORTED_IMAGE_FORMATS


def load_image(file_path: Union[str, Path]) -> Image.Image:
    """Load an image from a file path.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        PIL Image object
        
    Raises:
        OCRProcessingError: If the image cannot be loaded
    """
    check_dependencies()
    
    try:
        path = validate_file_path(file_path)
        image = Image.open(path)
        
        # Convert palette images directly to RGB
        # This preserves better contrast for OCR compared to RGBA->RGB conversion
        if image.mode == 'P':
            # Suppress the transparency warning and convert directly
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                image = image.convert('RGB')
        # Convert RGBA to RGB if necessary
        elif image.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        # Convert to RGB if necessary (but keep grayscale as-is)
        elif image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        if isinstance(e, (OCRFileNotFoundError, OCRUnsupportedFormatError)):
            raise
        raise OCRProcessingError(
            f"Failed to load image: {file_path}",
            error_code="IMAGE_LOAD_FAILED",
            original_error=e
        )


def pdf_to_images(file_path: Union[str, Path], dpi: int = 300) -> List[Image.Image]:
    """Convert a PDF file to a list of images.
    
    Args:
        file_path: Path to the PDF file
        dpi: DPI for rendering the PDF pages
        
    Returns:
        List of PIL Image objects, one per page
        
    Raises:
        OCRProcessingError: If the PDF cannot be converted
    """
    check_pdf_dependencies()
    check_dependencies()
    
    global _POPPLER_CHECKED
    
    try:
        path = validate_file_path(file_path)
        images = pdf2image.convert_from_path(str(path), dpi=dpi)
        return images
    except Exception as e:
        if isinstance(e, (OCRFileNotFoundError, OCRUnsupportedFormatError)):
            raise
        
        # Check if this is a poppler installation error
        error_str = str(e).lower()
        if ('poppler' in error_str or 
            'pdftoppm' in error_str or 
            'pdfinfo' in error_str or
            'unable to get page count' in error_str or
            type(e).__name__ == 'PDFInfoNotInstalledError'):
            
            # Show helpful poppler installation message
            if not _POPPLER_CHECKED:
                _POPPLER_CHECKED = True
                check_poppler_installed()
            
            raise OCRProcessingError(
                "Poppler is required for PDF processing but is not installed or not in PATH. "
                "Please see the installation instructions above.",
                error_code="POPPLER_NOT_INSTALLED",
                original_error=e
            )
        
        raise OCRProcessingError(
            f"Failed to convert PDF to images: {file_path}",
            error_code="PDF_CONVERSION_FAILED",
            original_error=e
        )


def detect_rotation(image: Image.Image) -> float:
    """Detect the rotation angle of text in an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Rotation angle in degrees (0, 90, 180, or 270)
    """
    check_dependencies()
    
    try:
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Simple heuristic: check variance in different orientations
        variances = []
        for angle in [0, 90, 180, 270]:
            rotated = Image.fromarray(img_array).rotate(angle, expand=True)
            rotated_array = np.array(rotated)
            # Calculate variance of horizontal projections (text lines)
            projection = np.sum(rotated_array, axis=1)
            variance = np.var(projection)
            variances.append((angle, variance))
        
        # The correct orientation should have the highest variance
        # (more distinct text lines)
        best_angle = max(variances, key=lambda x: x[1])[0]
        return best_angle
    except Exception:
        return 0.0


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """Rotate an image by the specified angle.
    
    Args:
        image: PIL Image object
        angle: Rotation angle in degrees
        
    Returns:
        Rotated PIL Image object
    """
    check_dependencies()
    
    if angle == 0:
        return image
    
    try:
        return image.rotate(-angle, expand=True)
    except Exception as e:
        raise OCRProcessingError(
            f"Failed to rotate image by {angle} degrees",
            error_code="ROTATION_FAILED",
            original_error=e
        )


def preprocess_image(
    image: Image.Image,
    rotation_fix: bool = False,
    enhance_contrast: bool = False,
    remove_noise: bool = False,
    target_size: Tuple[int, int] | None = None
) -> Image.Image:
    """Preprocess an image for better OCR results.
    
    Args:
        image: PIL Image object
        rotation_fix: Whether to detect and fix rotation
        enhance_contrast: Whether to enhance image contrast
        remove_noise: Whether to apply noise reduction
        target_size: Optional target size (width, height) for resizing
        
    Returns:
        Preprocessed PIL Image object
    """
    check_dependencies()
    
    try:
        processed = image.copy()
        
        if rotation_fix:
            angle = detect_rotation(processed)
            if angle != 0:
                processed = rotate_image(processed, angle)
        
        if target_size:
            processed = processed.resize(target_size, Image.Resampling.LANCZOS)
        
        if enhance_contrast:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(1.5)
        
        if remove_noise:
            from PIL import ImageFilter
            processed = processed.filter(ImageFilter.MedianFilter(size=3))
        
        return processed
    except Exception as e:
        raise OCRProcessingError(
            "Failed to preprocess image",
            error_code="PREPROCESSING_FAILED",
            original_error=e
        )


def prepare_file_for_ocr(
    file_path: Union[str, Path],
    rotation_fix: bool = False,
    enhance_contrast: bool = False,
    remove_noise: bool = False,
    pdf_dpi: int = 300
) -> List[Image.Image]:
    """Prepare a file (image or PDF) for OCR processing.
    
    Args:
        file_path: Path to the file
        rotation_fix: Whether to detect and fix rotation
        enhance_contrast: Whether to enhance image contrast
        remove_noise: Whether to apply noise reduction
        pdf_dpi: DPI for PDF rendering
        
    Returns:
        List of preprocessed PIL Image objects
    """
    path = validate_file_path(file_path)
    
    if is_pdf(path):
        images = pdf_to_images(path, dpi=pdf_dpi)
    else:
        images = [load_image(path)]
    
    processed_images = []
    for image in images:
        processed = preprocess_image(
            image,
            rotation_fix=rotation_fix,
            enhance_contrast=enhance_contrast,
            remove_noise=remove_noise
        )
        processed_images.append(processed)
    
    return processed_images

