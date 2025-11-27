import pytest

import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from upsonic.ocr.ocr import OCR, infer_provider
from upsonic.ocr.base import OCRProvider, OCRConfig, OCRResult
from upsonic.ocr.exceptions import OCRError


class DummyProvider(OCRProvider):
    """Minimal OCR provider used for testing the high-level interface."""

    def __init__(self, config: OCRConfig | None = None, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.processed_files: list[tuple[str, dict]] = []
        self.dependencies_checked = False
        super().__init__(config=config, **kwargs)

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def supported_languages(self) -> list[str]:
        return ["en", "fr"]

    def _validate_dependencies(self) -> None:
        self.dependencies_checked = True

    def _process_image(self, image, **kwargs) -> OCRResult:  # pragma: no cover
        return OCRResult(text="image-text", confidence=1.0, provider=self.name)

    def process_file(self, file_path, **kwargs) -> OCRResult:  # noqa: D401 - override
        self.processed_files.append((str(file_path), dict(kwargs)))
        suffix = Path(file_path).suffix or ""
        return OCRResult(
            text=f"dummy-output{suffix}",
            blocks=[],
            confidence=0.95,
            page_count=1,
            provider=self.name,
        )


def test_ocr_initialization():
    ocr = OCR(
        DummyProvider,
        languages=["en", "fr"],
        rotation_fix=True,
        enhance_contrast=True,
        custom_flag=True,
    )

    assert isinstance(ocr.provider, DummyProvider)
    assert ocr.config.languages == ["en", "fr"]
    assert ocr.config.rotation_fix is True
    assert ocr.provider.dependencies_checked is True
    assert ocr.provider.init_kwargs["custom_flag"] is True


def test_ocr_extract_text():
    ocr = OCR(DummyProvider)
    mock_result = OCRResult(
        text="recognized", blocks=[], confidence=1.0, provider="dummy"
    )

    with patch.object(
        ocr.provider, "process_file", return_value=mock_result
    ) as mock_process:
        text = ocr.get_text("sample.png", quality="high")

    assert text == "recognized"
    mock_process.assert_called_once_with("sample.png", quality="high")


def test_ocr_extract_text_from_image(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake image data")

    ocr = OCR(DummyProvider)
    text = ocr.get_text(image_path)

    assert text == "dummy-output.png"
    assert ocr.provider.processed_files[-1][0].endswith("image.png")


def test_ocr_extract_text_from_pdf(tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    ocr = OCR(DummyProvider)
    result = ocr.process_file(pdf_path, pdf_dpi=200)

    assert isinstance(result, OCRResult)
    assert result.text == "dummy-output.pdf"
    assert ocr.provider.processed_files[-1][1]["pdf_dpi"] == 200


def test_ocr_provider_selection():
    with pytest.raises(OCRError, match="Unknown provider"):
        infer_provider("unknown-provider")


def _run_infer_provider_test(provider_name: str, module_path: str, class_name: str):
    fake_module = SimpleNamespace(**{class_name: DummyProvider})

    original_import = builtins.__import__
    target_calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_path:
            target_calls.append((name, tuple(fromlist)))
            return fake_module
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        with patch("upsonic.ocr.ocr.OCR") as mock_ocr:
            infer_provider(provider_name, languages=["en"])
            mock_ocr.assert_called_once_with(DummyProvider, languages=["en"])

    assert target_calls == [(module_path, (class_name,))]
