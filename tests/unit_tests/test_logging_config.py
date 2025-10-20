"""
Unit tests for logging_config.py module.

Tests the centralized logging and telemetry configuration system.
"""

import unittest
import logging
import os
import tempfile
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Import the module under test
from upsonic.utils.logging_config import (
    setup_logging,
    setup_sentry,
    get_logger,
    set_module_log_level,
    disable_logging,
    get_current_log_levels,
    get_env_log_level,
    get_env_bool,
)


class TestLoggingConfig(unittest.TestCase):
    """Test cases for logging configuration."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear any existing environment variables
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("UPSONIC_"):
                del os.environ[key]

    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_get_env_log_level_default(self):
        """Test get_env_log_level with default value."""
        level = get_env_log_level("NONEXISTENT_KEY", "WARNING")
        self.assertEqual(level, logging.WARNING)

    def test_get_env_log_level_from_env(self):
        """Test get_env_log_level reading from environment."""
        os.environ["TEST_LOG_LEVEL"] = "DEBUG"
        level = get_env_log_level("TEST_LOG_LEVEL")
        self.assertEqual(level, logging.DEBUG)

    def test_get_env_log_level_invalid(self):
        """Test get_env_log_level with invalid level."""
        os.environ["TEST_LOG_LEVEL"] = "INVALID"
        level = get_env_log_level("TEST_LOG_LEVEL")
        self.assertEqual(level, logging.INFO)  # Should default to INFO

    def test_get_env_bool_true_values(self):
        """Test get_env_bool with true values."""
        for value in ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]:
            os.environ["TEST_BOOL"] = value
            result = get_env_bool("TEST_BOOL")
            self.assertTrue(result, f"Failed for value: {value}")

    def test_get_env_bool_false_values(self):
        """Test get_env_bool with false values."""
        for value in ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"]:
            os.environ["TEST_BOOL"] = value
            result = get_env_bool("TEST_BOOL")
            self.assertFalse(result, f"Failed for value: {value}")

    def test_get_env_bool_default(self):
        """Test get_env_bool with default value."""
        result = get_env_bool("NONEXISTENT_KEY", default=True)
        self.assertTrue(result)

    def test_get_logger_creates_logger(self):
        """Test get_logger creates a logger instance."""
        logger = get_logger("test.module")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test.module")

    def test_get_logger_autoconfigures(self):
        """Test get_logger auto-configures on first call."""
        # This should trigger setup_logging automatically
        logger = get_logger("test.autoconfigure")
        self.assertIsInstance(logger, logging.Logger)

    @patch('upsonic.utils.logging_config.sentry_sdk')
    def test_setup_sentry_disabled(self, mock_sentry):
        """Test setup_sentry when telemetry is disabled."""
        os.environ["UPSONIC_TELEMETRY"] = "false"

        # Force reconfiguration
        from upsonic.utils import logging_config
        logging_config._SENTRY_CONFIGURED = False

        setup_sentry()

        # Sentry should be initialized with empty DSN
        mock_sentry.init.assert_called_once()
        call_kwargs = mock_sentry.init.call_args[1]
        self.assertEqual(call_kwargs['dsn'], "")

    @patch('upsonic.utils.logging_config.sentry_sdk')
    @patch('upsonic.utils.logging_config.atexit.register')
    def test_setup_sentry_enabled(self, mock_atexit, mock_sentry):
        """Test setup_sentry when telemetry is enabled."""
        os.environ["UPSONIC_TELEMETRY"] = "https://test@sentry.io/123"

        # Force reconfiguration
        from upsonic.utils import logging_config
        logging_config._SENTRY_CONFIGURED = False

        setup_sentry()

        # Sentry should be initialized with DSN
        mock_sentry.init.assert_called_once()
        call_kwargs = mock_sentry.init.call_args[1]
        self.assertEqual(call_kwargs['dsn'], "https://test@sentry.io/123")

        # atexit handler should be registered
        mock_atexit.assert_called_once()

    def test_setup_logging_basic(self):
        """Test basic setup_logging functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            setup_logging(
                level="DEBUG",
                log_file=str(log_file),
                force_reconfigure=True,
                enable_console=False
            )

            logger = get_logger("upsonic.test.basic")  # Use upsonic namespace
            logger.info("Test message")

            # Flush handlers to ensure write
            for handler in logger.handlers:
                handler.flush()
            for handler in logging.getLogger("upsonic").handlers:
                handler.flush()

            # Check log file was created
            self.assertTrue(log_file.exists())

            # Check log content
            with open(log_file, 'r') as f:
                content = f.read()
                self.assertIn("Test message", content)

    def test_setup_logging_disable(self):
        """Test setup_logging with UPSONIC_DISABLE_LOGGING."""
        os.environ["UPSONIC_DISABLE_LOGGING"] = "true"

        setup_logging(force_reconfigure=True)

        logger = logging.getLogger("upsonic")
        # Should have NullHandler
        self.assertTrue(any(isinstance(h, logging.NullHandler) for h in logger.handlers))

    def test_setup_logging_console_disabled(self):
        """Test setup_logging with console disabled."""
        setup_logging(
            level="INFO",
            enable_console=False,
            force_reconfigure=True
        )

        logger = logging.getLogger("upsonic")
        # Should not have StreamHandler
        self.assertFalse(any(isinstance(h, logging.StreamHandler) for h in logger.handlers))

    def test_set_module_log_level(self):
        """Test setting log level for specific module."""
        # Setup logging first
        setup_logging(force_reconfigure=True)

        # Set specific module level
        set_module_log_level("upsonic.loaders", "WARNING")

        loader_logger = logging.getLogger("upsonic.loaders")
        self.assertEqual(loader_logger.level, logging.WARNING)

    def test_set_module_log_level_invalid(self):
        """Test set_module_log_level with invalid level."""
        with self.assertRaises(ValueError):
            set_module_log_level("test.module", "INVALID_LEVEL")

    def test_disable_logging(self):
        """Test disable_logging functionality."""
        # Setup logging first
        setup_logging(force_reconfigure=True)

        # Disable it
        disable_logging()

        logger = logging.getLogger("upsonic")
        # Should have NullHandler
        self.assertTrue(any(isinstance(h, logging.NullHandler) for h in logger.handlers))
        # Should be at CRITICAL+1 (effectively disabled)
        self.assertGreater(logger.level, logging.CRITICAL)

    def test_get_current_log_levels(self):
        """Test get_current_log_levels returns correct levels."""
        setup_logging(level="INFO", force_reconfigure=True)
        set_module_log_level("upsonic.loaders", "DEBUG")

        levels = get_current_log_levels()

        self.assertIn("upsonic", levels)
        self.assertEqual(levels["upsonic"], "INFO")
        self.assertIn("upsonic.loaders", levels)
        self.assertEqual(levels["upsonic.loaders"], "DEBUG")

    def test_setup_logging_formats(self):
        """Test different log formats."""
        for fmt in ["simple", "detailed", "json"]:
            with self.subTest(format=fmt):
                with tempfile.TemporaryDirectory() as tmpdir:
                    log_file = Path(tmpdir) / f"test_{fmt}.log"

                    setup_logging(
                        level="INFO",
                        log_format=fmt,
                        log_file=str(log_file),
                        force_reconfigure=True,
                        enable_console=False
                    )

                    logger = get_logger("test.format")
                    logger.info("Test message")

                    # Check log file was created
                    self.assertTrue(log_file.exists())

    def test_setup_logging_file_permission_error(self):
        """Test setup_logging handles file permission errors gracefully."""
        # Try to write to invalid path
        setup_logging(
            level="INFO",
            log_file="/invalid/path/test.log",
            force_reconfigure=True,
            enable_console=False
        )

        # Should not raise exception, just skip file handler
        logger = logging.getLogger("upsonic")
        self.assertIsNotNone(logger)

    def test_module_logger_inheritance(self):
        """Test that module loggers inherit from upsonic logger."""
        setup_logging(level="WARNING", force_reconfigure=True)

        # Child logger should inherit level
        child_logger = get_logger("upsonic.agent.test")
        parent_logger = logging.getLogger("upsonic")

        # Child should propagate to parent
        self.assertTrue(child_logger.propagate)


class TestSentryIntegration(unittest.TestCase):
    """Test cases for Sentry integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up after tests."""
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch('upsonic.utils.logging_config.sentry_sdk')
    def test_sentry_environment_config(self, mock_sentry):
        """Test Sentry environment configuration."""
        os.environ["UPSONIC_ENVIRONMENT"] = "development"
        os.environ["UPSONIC_SENTRY_SAMPLE_RATE"] = "0.5"

        from upsonic.utils import logging_config
        logging_config._SENTRY_CONFIGURED = False

        setup_sentry()

        call_kwargs = mock_sentry.init.call_args[1]
        self.assertEqual(call_kwargs['environment'], "development")
        self.assertEqual(call_kwargs['traces_sample_rate'], 0.5)

    @patch('upsonic.utils.logging_config.sentry_sdk')
    @patch('upsonic.utils.package.system_id.get_system_id')
    def test_sentry_user_id_tracking(self, mock_get_system_id, mock_sentry):
        """Test Sentry user ID tracking."""
        mock_get_system_id.return_value = "test-system-id-123"

        from upsonic.utils import logging_config
        logging_config._SENTRY_CONFIGURED = False

        setup_sentry()

        # Should set user ID
        mock_sentry.set_user.assert_called_once_with({"id": "test-system-id-123"})

    @patch('upsonic.utils.logging_config.sentry_sdk')
    @patch('upsonic.utils.package.system_id.get_system_id')
    def test_sentry_user_id_failure_graceful(self, mock_get_system_id, mock_sentry):
        """Test Sentry handles system ID failure gracefully."""
        mock_get_system_id.side_effect = Exception("System ID error")

        from upsonic.utils import logging_config
        logging_config._SENTRY_CONFIGURED = False

        # Should not raise exception
        setup_sentry()

        # Sentry should still be initialized
        mock_sentry.init.assert_called_once()


if __name__ == '__main__':
    unittest.main()
