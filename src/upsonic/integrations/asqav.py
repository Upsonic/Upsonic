"""Asqav governance integration for Upsonic agents.

Signs every tool call and agent action with ML-DSA-65 (quantum-safe)
cryptographic signatures, creating a tamper-evident audit trail.

Usage::

    from upsonic import Agent, Task
    from upsonic.integrations.asqav import AsqavGovernance

    gov = AsqavGovernance(api_key="sk_...")
    agent = Agent("openai/gpt-4o", instrument=gov)
    agent.print_do("Analyze quarterly revenue data")

    # Export audit trail
    gov.export_audit("json")
"""

from __future__ import annotations

import os
from typing import Any, Optional, TYPE_CHECKING

from upsonic.integrations.tracing import TracingProvider

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter as _SpanExporter


class AsqavGovernance(TracingProvider):
    """Asqav governance integration for Upsonic agents.

    Extends the standard tracing pipeline with cryptographic signing.
    Every span (tool call, agent step, LLM invocation) gets an ML-DSA-65
    signature chained to the previous action, creating a tamper-evident
    audit trail suitable for EU AI Act Article 12 compliance.

    Args:
        api_key: Asqav API key (``sk_...``).
            Falls back to ``ASQAV_API_KEY`` env var.
        agent_name: Name for the asqav agent identity.
            Defaults to ``"upsonic-agent"``.
        endpoint: Asqav API endpoint.
            Falls back to ``ASQAV_API_URL`` env var.
        sign_tool_calls: Sign individual tool call spans.
        sign_llm_calls: Sign LLM invocation spans.
        sign_agent_steps: Sign agent reasoning steps.
        include_content: Include prompt/response content in traces.
        service_name: Service name reported in traces.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        agent_name: str = "upsonic-agent",
        endpoint: Optional[str] = None,
        sign_tool_calls: bool = True,
        sign_llm_calls: bool = True,
        sign_agent_steps: bool = True,
        include_content: bool = True,
        service_name: str = "upsonic",
    ) -> None:
        self._api_key = api_key or os.environ.get("ASQAV_API_KEY", "")
        self._agent_name = agent_name
        self._endpoint = endpoint or os.environ.get(
            "ASQAV_API_URL", "https://api.asqav.com"
        )
        self._sign_tool_calls = sign_tool_calls
        self._sign_llm_calls = sign_llm_calls
        self._sign_agent_steps = sign_agent_steps

        self._asqav = None
        self._agent = None
        self._session = None

        super().__init__(
            service_name=service_name,
            include_content=include_content,
        )

        self._init_asqav()

    def _init_asqav(self) -> None:
        """Initialize asqav client and agent."""
        try:
            import asqav

            asqav.init(api_key=self._api_key, api_url=self._endpoint)
            self._asqav = asqav
            self._agent = asqav.Agent.create(self._agent_name)
            self._session = self._agent.start_session(
                name="upsonic-session",
                metadata={"framework": "upsonic"},
            )
        except ImportError:
            raise ImportError(
                "asqav is required for AsqavGovernance. "
                "Install it with: pip install asqav"
            )
        except Exception:
            # If asqav API is unreachable, continue without signing
            # but still provide tracing
            pass

    def _create_exporter(self) -> _SpanExporter:
        """Create an OTLP exporter that also signs spans via asqav."""
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory import InMemorySpanExporter

        # Use in-memory exporter as base - we process spans for signing
        exporter = InMemorySpanExporter()
        return _AsqavSigningExporter(
            inner=exporter,
            session=self._session,
            sign_tool_calls=self._sign_tool_calls,
            sign_llm_calls=self._sign_llm_calls,
            sign_agent_steps=self._sign_agent_steps,
        )

    def export_audit(self, format: str = "json") -> Optional[bytes]:
        """Export the audit trail as JSON or CSV.

        Args:
            format: Export format (``"json"`` or ``"csv"``).

        Returns:
            Audit trail data as bytes, or None if export fails.
        """
        if self._asqav is None:
            return None
        try:
            return self._asqav.export_audit(format)
        except Exception:
            return None

    def shutdown(self) -> None:
        """End the asqav session and shut down tracing."""
        if self._session is not None:
            try:
                self._session.end()
            except Exception:
                pass
        super().shutdown()


class _AsqavSigningExporter:
    """Wraps a SpanExporter to sign spans via asqav before export."""

    def __init__(
        self,
        inner: Any,
        session: Any,
        sign_tool_calls: bool = True,
        sign_llm_calls: bool = True,
        sign_agent_steps: bool = True,
    ) -> None:
        self._inner = inner
        self._session = session
        self._sign_tool_calls = sign_tool_calls
        self._sign_llm_calls = sign_llm_calls
        self._sign_agent_steps = sign_agent_steps

    def export(self, spans: Any) -> Any:
        """Sign relevant spans and forward to inner exporter."""
        if self._session is not None:
            for span in spans:
                self._maybe_sign(span)
        return self._inner.export(spans)

    def _maybe_sign(self, span: Any) -> None:
        """Sign a span if it matches the configured span types."""
        try:
            name = span.name or ""
            attrs = dict(span.attributes or {})

            if "tool" in name.lower() and self._sign_tool_calls:
                self._session.log(
                    f"tool:{name}",
                    metadata={k: str(v) for k, v in attrs.items()},
                )
            elif "llm" in name.lower() and self._sign_llm_calls:
                self._session.log(
                    f"llm:{name}",
                    metadata={k: str(v) for k, v in attrs.items()},
                )
            elif self._sign_agent_steps:
                self._session.log(
                    f"agent:{name}",
                    metadata={k: str(v) for k, v in attrs.items()},
                )
        except Exception:
            pass  # Never break the tracing pipeline

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
