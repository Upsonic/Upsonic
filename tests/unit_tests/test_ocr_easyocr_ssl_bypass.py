"""Tests for the optional SSL-bypass behavior used by EasyOCREngine.

The previous code in ``easyocr.py`` unconditionally replaced
``ssl._create_default_https_context`` with the unverified variant for the
duration of ``easyocr.Reader(...)``. That is a process-wide side effect: any
concurrent HTTPS call in another thread inherits the unverified context for
the same window. The bypass is now opt-out via
``UPSONIC_OCR_DISABLE_SSL_BYPASS``; see ``_optional_ssl_bypass``.
"""

from __future__ import annotations

import logging
import ssl

import pytest

from upsonic.ocr.layer_1.engines.easyocr import (
    _optional_ssl_bypass,
    _truthy_env,
)


class TestTruthyEnv:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "Yes", "on", " 1 ", " yes "])
    def test_truthy_values(self, value, monkeypatch):
        monkeypatch.setenv("FOO", value)
        assert _truthy_env("FOO") is True

    @pytest.mark.parametrize("value", ["", "0", "no", "false", "off", "anything-else"])
    def test_falsy_values(self, value, monkeypatch):
        monkeypatch.setenv("FOO", value)
        assert _truthy_env("FOO") is False

    def test_unset_is_falsy(self, monkeypatch):
        monkeypatch.delenv("FOO", raising=False)
        assert _truthy_env("FOO") is False


class TestOptionalSslBypass:
    def test_disable_true_does_not_touch_global_context(self):
        original = ssl._create_default_https_context
        with _optional_ssl_bypass(disable=True):
            assert ssl._create_default_https_context is original
        assert ssl._create_default_https_context is original

    def test_disable_false_swaps_and_restores(self):
        original = ssl._create_default_https_context
        try:
            with _optional_ssl_bypass(disable=False):
                # During the window, default https context is unverified.
                assert ssl._create_default_https_context is ssl._create_unverified_context
                assert ssl._create_default_https_context is not original
            # After exit, the original context is restored.
            assert ssl._create_default_https_context is original
        finally:
            # Belt-and-suspenders: never leak unverified context across tests.
            ssl._create_default_https_context = original

    def test_disable_false_restores_on_exception(self):
        original = ssl._create_default_https_context

        class _Boom(Exception):
            pass

        try:
            with pytest.raises(_Boom):
                with _optional_ssl_bypass(disable=False):
                    raise _Boom()
            assert ssl._create_default_https_context is original
        finally:
            ssl._create_default_https_context = original

    def test_disable_false_emits_warning_log(self, caplog):
        original = ssl._create_default_https_context
        try:
            with caplog.at_level(logging.WARNING, logger="upsonic.ocr.layer_1.engines.easyocr"):
                with _optional_ssl_bypass(disable=False):
                    pass
            messages = [r.getMessage() for r in caplog.records]
            assert any("unverified" in msg for msg in messages), (
                f"expected an 'unverified' warning, got: {messages}"
            )
            assert any("UPSONIC_OCR_DISABLE_SSL_BYPASS" in msg for msg in messages), (
                f"expected env-var hint, got: {messages}"
            )
        finally:
            ssl._create_default_https_context = original

    def test_disable_true_emits_no_warning_log(self, caplog):
        with caplog.at_level(logging.WARNING, logger="upsonic.ocr.layer_1.engines.easyocr"):
            with _optional_ssl_bypass(disable=True):
                pass
        # No warning records from this logger when the bypass is disabled.
        assert [r for r in caplog.records if r.name == "upsonic.ocr.layer_1.engines.easyocr"] == []
