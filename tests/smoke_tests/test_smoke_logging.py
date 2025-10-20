"""
Smoke tests for logging and telemetry system.

These tests verify that the logging and telemetry system works end-to-end
in real-world scenarios without mocking.
"""

import unittest
import os
import tempfile
import time
from pathlib import Path

from upsonic import Agent, Task
from upsonic.utils.logging_config import (
    setup_logging,
    get_logger,
    get_current_log_levels,
    disable_logging,
)


class TestLoggingSmoke(unittest.TestCase):
    """Smoke tests for logging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Disable telemetry for smoke tests to avoid sending test data
        os.environ["UPSONIC_TELEMETRY"] = "false"

    def test_basic_logging_flow(self):
        """Test basic logging works end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            # Setup logging
            setup_logging(
                level="INFO",
                log_file=str(log_file),
                force_reconfigure=True,
                enable_console=False
            )

            # Get logger and write messages (use upsonic namespace)
            logger = get_logger("upsonic.test.smoke")
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")

            # Flush handlers to ensure write
            import logging
            for handler in logging.getLogger("upsonic").handlers:
                handler.flush()

            # Verify log file exists and contains messages
            self.assertTrue(log_file.exists())

            with open(log_file, 'r') as f:
                content = f.read()
                self.assertIn("Test info message", content)
                self.assertIn("Test warning message", content)
                self.assertIn("Test error message", content)

    def test_logging_with_agent_execution(self):
        """Test logging during agent execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "agent.log"

            # Setup logging
            setup_logging(
                level="INFO",
                log_file=str(log_file),
                force_reconfigure=True,
                enable_console=False
            )

            # Create and execute simple agent task
            try:
                agent = Agent(model="openai/gpt-4o-mini")
                task = Task("What is 2+2? Answer with just the number.")

                # Execute task (may fail if no API key, that's ok for logging test)
                try:
                    agent.do(task)
                except Exception:
                    pass  # We're testing logging, not agent functionality

                # Give time for async writes
                time.sleep(0.1)

                # Verify log file was created
                self.assertTrue(log_file.exists())

                # Log file should have content (even if agent failed)
                self.assertGreater(log_file.stat().st_size, 0)

            except Exception as e:
                # If agent creation fails, that's ok - we're testing logging
                self.assertTrue(log_file.exists(), f"Log file should exist even if agent fails: {e}")

    def test_module_specific_log_levels(self):
        """Test module-specific log level configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "module.log"

            # Setup with environment variables
            os.environ["UPSONIC_LOG_LEVEL"] = "WARNING"
            os.environ["UPSONIC_LOG_LEVEL_AGENT"] = "DEBUG"

            setup_logging(
                log_file=str(log_file),
                force_reconfigure=True,
                enable_console=False
            )

            # Get current levels
            levels = get_current_log_levels()

            # Main logger should be WARNING
            self.assertEqual(levels["upsonic"], "WARNING")

            # Agent logger should be DEBUG
            if "upsonic.agent" in levels:
                self.assertEqual(levels["upsonic.agent"], "DEBUG")

    def test_telemetry_disabled(self):
        """Test that telemetry can be disabled."""
        # Telemetry is disabled via setUp
        self.assertEqual(os.environ.get("UPSONIC_TELEMETRY"), "false")

        # Setup should work without errors
        setup_logging(force_reconfigure=True, enable_console=False)

        # Logger should still work
        logger = get_logger("test.telemetry")
        logger.info("Test with telemetry disabled")

    def test_logging_disable_and_reenable(self):
        """Test disabling and re-enabling logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "toggle.log"

            # Enable logging
            setup_logging(
                level="INFO",
                log_file=str(log_file),
                force_reconfigure=True,
                enable_console=False
            )

            import logging
            logger = get_logger("upsonic.test.toggle")
            logger.info("Message 1")
            for handler in logging.getLogger("upsonic").handlers:
                handler.flush()

            # Disable logging
            disable_logging()

            logger.info("Message 2 (should not appear)")

            # Re-enable
            setup_logging(
                level="INFO",
                log_file=str(log_file),
                force_reconfigure=True,
                enable_console=False
            )

            logger = get_logger("upsonic.test.toggle")
            logger.info("Message 3")
            for handler in logging.getLogger("upsonic").handlers:
                handler.flush()

            # Check log content
            with open(log_file, 'r') as f:
                content = f.read()
                self.assertIn("Message 1", content)
                self.assertNotIn("Message 2", content)
                self.assertIn("Message 3", content)

    def test_concurrent_logging(self):
        """Test logging from multiple loggers concurrently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "concurrent.log"

            setup_logging(
                level="INFO",
                log_file=str(log_file),
                force_reconfigure=True,
                enable_console=False
            )

            # Create multiple loggers
            loggers = [
                get_logger(f"upsonic.test.concurrent.{i}")
                for i in range(5)
            ]

            # Log from each
            for i, logger in enumerate(loggers):
                logger.info(f"Message from logger {i}")

            # Flush handlers to ensure write
            import logging
            for handler in logging.getLogger("upsonic").handlers:
                handler.flush()

            # Verify all messages present
            with open(log_file, 'r') as f:
                content = f.read()
                for i in range(5):
                    self.assertIn(f"Message from logger {i}", content)

    def test_log_rotation_with_large_output(self):
        """Test logging with large output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "large.log"

            setup_logging(
                level="INFO",
                log_file=str(log_file),
                force_reconfigure=True,
                enable_console=False
            )

            logger = get_logger("upsonic.test.large")

            # Write many log messages
            for i in range(100):
                logger.info(f"Log message {i}: " + "x" * 100)

            # Flush handlers to ensure write
            import logging
            for handler in logging.getLogger("upsonic").handlers:
                handler.flush()

            # Verify file exists and has substantial size
            self.assertTrue(log_file.exists())
            self.assertGreater(log_file.stat().st_size, 10000)

    def test_logging_with_different_formats(self):
        """Test different log formats."""
        for log_format in ["simple", "detailed", "json"]:
            with self.subTest(format=log_format):
                with tempfile.TemporaryDirectory() as tmpdir:
                    log_file = Path(tmpdir) / f"format_{log_format}.log"

                    setup_logging(
                        level="INFO",
                        log_format=log_format,
                        log_file=str(log_file),
                        force_reconfigure=True,
                        enable_console=False
                    )

                    logger = get_logger("upsonic.test.format")
                    logger.info("Test format message")

                    # Flush handlers to ensure write
                    import logging
                    for handler in logging.getLogger("upsonic").handlers:
                        handler.flush()

                    # Verify log file exists and has content
                    self.assertTrue(log_file.exists())
                    self.assertGreater(log_file.stat().st_size, 0)

                    # Check format-specific content
                    with open(log_file, 'r') as f:
                        content = f.read()
                        self.assertIn("Test format message", content)

                        if log_format == "json":
                            # JSON format should have JSON structure
                            self.assertIn('"message":', content)
                            self.assertIn('"level":', content)


class TestTelemetrySmoke(unittest.TestCase):
    """Smoke tests for telemetry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Disable telemetry for most tests
        os.environ["UPSONIC_TELEMETRY"] = "false"

    def test_telemetry_disabled_flow(self):
        """Test complete flow with telemetry disabled."""
        # Setup logging with telemetry disabled
        setup_logging(force_reconfigure=True, enable_console=False)

        # Create logger
        logger = get_logger("test.telemetry.disabled")

        # Log messages of different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Should complete without errors
        time.sleep(0.1)

    def test_environment_variable_parsing(self):
        """Test various environment variable formats."""
        test_cases = [
            ("false", "false"),
            ("False", "false"),
            ("FALSE", "false"),
            ("true", None),  # Should use default DSN when not "false"
        ]

        for env_value, expected in test_cases:
            with self.subTest(env_value=env_value):
                os.environ["UPSONIC_TELEMETRY"] = env_value

                # Should not raise exception
                try:
                    setup_logging(force_reconfigure=True, enable_console=False)
                except Exception as e:
                    self.fail(f"setup_logging raised exception with UPSONIC_TELEMETRY={env_value}: {e}")


class TestLoggingPerformance(unittest.TestCase):
    """Performance smoke tests for logging."""

    def setUp(self):
        """Set up test fixtures."""
        os.environ["UPSONIC_TELEMETRY"] = "false"

    def test_logging_performance_overhead(self):
        """Test that logging doesn't add significant overhead."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "perf.log"

            setup_logging(
                level="INFO",
                log_file=str(log_file),
                force_reconfigure=True,
                enable_console=False
            )

            logger = get_logger("test.perf")

            # Measure time to write many log messages
            start_time = time.time()

            for i in range(1000):
                logger.info(f"Performance test message {i}")

            elapsed = time.time() - start_time

            # Should complete in reasonable time (< 1 second for 1000 messages)
            self.assertLess(elapsed, 1.0, f"Logging 1000 messages took {elapsed}s, too slow!")

    def test_get_logger_caching(self):
        """Test that get_logger caches logger instances."""
        logger1 = get_logger("test.cache")
        logger2 = get_logger("test.cache")

        # Should return same instance
        self.assertIs(logger1, logger2)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
