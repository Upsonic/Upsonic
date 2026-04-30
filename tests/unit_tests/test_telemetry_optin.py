"""
TECH-1428: Tests for opt-in telemetry behavior.

The previous behavior had three problems:
  1. `setup_sentry()` ran at import time, calling `sentry_sdk.init()` and
     replacing the host application's global Sentry client.
  2. The DSN defaulted to upsonic's own Sentry project, so any host that
     didn't explicitly opt out shipped error logs to a third party.
  3. `LoggingIntegration` was registered globally, monkey-patching
     `logging.Logger.callHandlers` and breaking sandboxed environments
     such as Temporal workflows.

Fix surface:
  * Importing `upsonic.utils.logging_config` MUST NOT call `sentry_sdk.init`.
  * `setup_sentry()` is now a no-op when `UPSONIC_TELEMETRY` is unset; there
    is no default DSN.
  * `enable_telemetry(dsn=...)` is the explicit opt-in entry point and
    constructs an isolated `sentry_sdk.Client` bound to a local `Hub` —
    it never calls `sentry_sdk.init()`.
  * No `LoggingIntegration` is registered.
  * `capture_exception()` routes through the isolated hub, never the global
    one.
"""

import importlib
import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


def _reset_logging_config_state() -> None:
    """Reset module-level flags so each test starts from a clean slate."""
    from upsonic.utils import logging_config

    logging_config._SENTRY_CONFIGURED = False
    logging_config._upsonic_client = None
    logging_config._upsonic_scope = None


class TestNoImportTimeSideEffects(unittest.TestCase):
    """Importing the module must not touch the global Sentry SDK."""

    def setUp(self) -> None:
        self.original_env = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_importing_module_does_not_call_sentry_init(self) -> None:
        """Re-importing logging_config must not invoke sentry_sdk.init().

        The original bug shipped a hard-coded default DSN, so even a host
        that never set UPSONIC_TELEMETRY had its global Sentry client
        silently replaced. We force-clear the env var (and stub load_dotenv
        so the repo's .env can't put it back) to reproduce that scenario.
        """
        os.environ.pop("UPSONIC_TELEMETRY", None)
        sys.modules.pop("upsonic.utils.logging_config", None)

        with patch("dotenv.load_dotenv"), patch("sentry_sdk.init") as mock_init:
            importlib.import_module("upsonic.utils.logging_config")
            mock_init.assert_not_called()

    def test_importing_module_does_not_register_logging_integration(self) -> None:
        """No LoggingIntegration may be installed at import time."""
        os.environ.pop("UPSONIC_TELEMETRY", None)
        sys.modules.pop("upsonic.utils.logging_config", None)

        with patch("dotenv.load_dotenv"), patch(
            "sentry_sdk.integrations.logging.LoggingIntegration"
        ) as mock_integration:
            importlib.import_module("upsonic.utils.logging_config")
            mock_integration.assert_not_called()


class TestNoDefaultDSN(unittest.TestCase):
    """The library must not ship a hard-coded fallback DSN."""

    def setUp(self) -> None:
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("UPSONIC_"):
                del os.environ[key]
        _reset_logging_config_state()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original_env)
        _reset_logging_config_state()

    def test_setup_sentry_noop_without_env_var(self) -> None:
        """setup_sentry() is a no-op when UPSONIC_TELEMETRY is unset."""
        from upsonic.utils import logging_config

        with patch("sentry_sdk.init") as mock_init, patch(
            "sentry_sdk.Client"
        ) as mock_client:
            logging_config.setup_sentry()
            mock_init.assert_not_called()
            mock_client.assert_not_called()

    def test_setup_sentry_noop_when_telemetry_false(self) -> None:
        """Explicit `false` keeps telemetry disabled."""
        os.environ["UPSONIC_TELEMETRY"] = "false"
        from upsonic.utils import logging_config

        with patch("sentry_sdk.init") as mock_init, patch(
            "sentry_sdk.Client"
        ) as mock_client:
            logging_config.setup_sentry()
            mock_init.assert_not_called()
            mock_client.assert_not_called()

    def test_module_source_contains_no_hardcoded_upsonic_dsn(self) -> None:
        """No literal upsonic ingest DSN may remain in the source."""
        import upsonic.utils.logging_config as mod

        source = open(mod.__file__, "r", encoding="utf-8").read()
        self.assertNotIn("ingest.us.sentry.io", source)
        self.assertNotIn("o4508336623583232", source)


class TestIsolatedClientNotGlobalInit(unittest.TestCase):
    """enable_telemetry() must use an isolated Client + Hub, never global init."""

    _DSN = "https://abc@example.ingest.sentry.io/1"

    def setUp(self) -> None:
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("UPSONIC_"):
                del os.environ[key]
        _reset_logging_config_state()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original_env)
        _reset_logging_config_state()

    def test_enable_telemetry_function_exists(self) -> None:
        """The public opt-in API must be exported."""
        from upsonic.utils import logging_config

        self.assertTrue(hasattr(logging_config, "enable_telemetry"))
        self.assertTrue(callable(logging_config.enable_telemetry))

    def test_enable_telemetry_does_not_call_sentry_init(self) -> None:
        """Even when enabled, sentry_sdk.init() must never be called."""
        from upsonic.utils import logging_config

        with patch("sentry_sdk.init") as mock_init, patch(
            "sentry_sdk.Client"
        ) as mock_client, patch("sentry_sdk.Scope"):
            mock_client.return_value = MagicMock()
            result = logging_config.enable_telemetry(dsn=self._DSN)
            self.assertTrue(result)
            mock_init.assert_not_called()
            mock_client.assert_called_once()

    def test_enable_telemetry_does_not_construct_hub(self) -> None:
        """sentry_sdk.Hub must NOT be used — its 2.x ctor mutates global state.

        Reading sentry_sdk 2.48 source: Hub.__init__ calls
        get_global_scope().set_client(client), which is exactly the global
        mutation we are trying to avoid. The fix uses a bare Scope instead.
        """
        from upsonic.utils import logging_config

        with patch("sentry_sdk.Client", return_value=MagicMock()), patch(
            "sentry_sdk.Hub"
        ) as mock_hub, patch("sentry_sdk.Scope", return_value=MagicMock()):
            logging_config.enable_telemetry(dsn=self._DSN)
            mock_hub.assert_not_called()

    def test_enable_telemetry_constructs_isolated_scope(self) -> None:
        """A bare Scope bound to the isolated Client must be created."""
        from upsonic.utils import logging_config

        fake_client = MagicMock(name="Client")
        fake_scope = MagicMock(name="Scope")
        with patch("sentry_sdk.Client", return_value=fake_client), patch(
            "sentry_sdk.Scope", return_value=fake_scope
        ) as mock_scope_ctor, patch("sentry_sdk.init") as mock_init:
            logging_config.enable_telemetry(dsn=self._DSN)
            mock_init.assert_not_called()
            mock_scope_ctor.assert_called_once_with()
            fake_scope.set_client.assert_called_once_with(fake_client)
            self.assertIs(logging_config._upsonic_client, fake_client)
            self.assertIs(logging_config._upsonic_scope, fake_scope)

    def test_enable_telemetry_does_not_install_logging_integration(self) -> None:
        """LoggingIntegration must never be passed to the isolated Client."""
        from upsonic.utils import logging_config

        with patch("sentry_sdk.Client") as mock_client, patch("sentry_sdk.Scope"):
            logging_config.enable_telemetry(dsn=self._DSN)
            kwargs = mock_client.call_args.kwargs
            integrations = kwargs.get("integrations") or []
            for integration in integrations:
                name = type(integration).__name__
                self.assertNotEqual(name, "LoggingIntegration")

    def test_enable_telemetry_returns_false_without_dsn(self) -> None:
        """enable_telemetry() with no DSN and no env var is a no-op."""
        from upsonic.utils import logging_config

        with patch("sentry_sdk.Client") as mock_client:
            result = logging_config.enable_telemetry()
            self.assertFalse(result)
            mock_client.assert_not_called()

    def test_enable_telemetry_reads_env_when_dsn_omitted(self) -> None:
        """If no DSN argument is given, fall back to UPSONIC_TELEMETRY env var."""
        os.environ["UPSONIC_TELEMETRY"] = self._DSN
        from upsonic.utils import logging_config

        with patch("sentry_sdk.Client") as mock_client, patch("sentry_sdk.Scope"):
            result = logging_config.enable_telemetry()
            self.assertTrue(result)
            self.assertEqual(mock_client.call_args.kwargs["dsn"], self._DSN)


class TestHostSentryNotHijacked(unittest.TestCase):
    """If the host already configured Sentry, upsonic must not overwrite it."""

    _DSN = "https://abc@example.ingest.sentry.io/1"

    def setUp(self) -> None:
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("UPSONIC_"):
                del os.environ[key]
        _reset_logging_config_state()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original_env)
        _reset_logging_config_state()

    def test_global_hub_client_unchanged_after_enable(self) -> None:
        """The global Hub's client must be untouched after enable_telemetry().

        We snapshot the current global hub's client identity before and
        after the call. enable_telemetry() must build an isolated Client +
        Hub pair without ever swapping the global hub's bound client.
        """
        import sentry_sdk
        from upsonic.utils import logging_config

        before = sentry_sdk.Hub.current.client
        logging_config.enable_telemetry(dsn=self._DSN)
        after = sentry_sdk.Hub.current.client
        self.assertIs(before, after)


class TestCaptureExceptionUsesIsolatedHub(unittest.TestCase):
    """capture_exception() must route through the isolated hub, not the global one."""

    _DSN = "https://abc@example.ingest.sentry.io/1"

    def setUp(self) -> None:
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("UPSONIC_"):
                del os.environ[key]
        _reset_logging_config_state()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original_env)
        _reset_logging_config_state()

    def test_capture_exception_noop_when_disabled(self) -> None:
        """Without enable_telemetry(), capture_exception is a silent no-op."""
        from upsonic.utils import logging_config

        # Should not raise even though no hub is configured.
        logging_config.capture_exception(RuntimeError("boom"))

    def test_capture_exception_uses_isolated_scope(self) -> None:
        """Exceptions captured through upsonic's helper hit the isolated scope."""
        from upsonic.utils import logging_config

        fake_scope = MagicMock(name="IsolatedScope")
        with patch("sentry_sdk.Client", return_value=MagicMock()), patch(
            "sentry_sdk.Scope", return_value=fake_scope
        ):
            logging_config.enable_telemetry(dsn=self._DSN)

        err = RuntimeError("boom")
        logging_config.capture_exception(err)
        fake_scope.capture_exception.assert_called_once_with(err)


class TestDisableTelemetry(unittest.TestCase):
    """disable_telemetry() should reset the isolated client/hub."""

    _DSN = "https://abc@example.ingest.sentry.io/1"

    def setUp(self) -> None:
        _reset_logging_config_state()

    def tearDown(self) -> None:
        _reset_logging_config_state()

    def test_disable_telemetry_clears_state(self) -> None:
        from upsonic.utils import logging_config

        with patch("sentry_sdk.Client", return_value=MagicMock()), patch(
            "sentry_sdk.Scope", return_value=MagicMock()
        ):
            logging_config.enable_telemetry(dsn=self._DSN)

        self.assertTrue(logging_config.is_telemetry_enabled())
        logging_config.disable_telemetry()
        self.assertFalse(logging_config.is_telemetry_enabled())
        self.assertIsNone(logging_config._upsonic_client)
        self.assertIsNone(logging_config._upsonic_scope)


if __name__ == "__main__":
    unittest.main()
