"""
Upsonic-maintained prebuilt autonomous agents.

Each class in this module wires
:class:`~upsonic.agent.prebuilt_autonomous_agent.PrebuiltAutonomousAgent` to a
specific upstream template repo/folder and exposes a high-level,
template-aware API (``new_experiment()``, ``new_task()``, etc.) so users do
not have to remember the placeholders in ``first_message.md``.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING, Union

from upsonic.agent.prebuilt_autonomous_agent.prebuilt_autonomous_agent import (
    PrebuiltAutonomousAgent,
)

if TYPE_CHECKING:
    from upsonic.models import Model


# --------------------------------------------------------------------------- #
# Experiment — the "about to run / running / finished run" object
# --------------------------------------------------------------------------- #


class Experiment:
    """
    A prepared experiment run for :class:`AppliedScientist`.

    Holds the template parameters and optional input paths for a single run.
    The actual work is deferred until :meth:`run`, :meth:`run_in_background`,
    :meth:`run_async`, or :meth:`run_stream` is called, so the same agent
    instance can prepare and launch multiple experiments.
    """

    def __init__(
        self,
        agent: "AppliedScientist",
        name: str,
        template_params: Dict[str, Any],
        inputs: Optional[List[str]] = None,
    ) -> None:
        self._agent = agent
        self._name = name
        self._template_params = template_params
        self._inputs = inputs
        self._thread: Optional[threading.Thread] = None
        self._raw_output: Any = None
        self._error: Optional[BaseException] = None
        self._done: bool = False
        self._started: bool = False
        self._stop_requested: bool = False

    @property
    def name(self) -> str:
        """The experiment name (also the folder name inside `experiments/`)."""
        return self._name

    @property
    def template_params(self) -> Dict[str, Any]:
        """The rendered template parameters that will be sent to the agent."""
        return dict(self._template_params)

    @property
    def inputs(self) -> Optional[List[str]]:
        """User paths that will be copied into the workspace at run time."""
        return list(self._inputs) if self._inputs is not None else None

    # ------------------------------------------------------------------ #
    # Foreground runs
    # ------------------------------------------------------------------ #

    def run(
        self,
        *,
        verbose: bool = False,
        preview_chars: int = 400,
    ) -> str:
        """
        Execute the experiment with pretty terminal output (calls
        :meth:`PrebuiltAutonomousAgent.run_console`). Blocks until the run
        completes and returns the concatenated assistant text.
        """
        if self._started:
            raise RuntimeError(
                "Experiment has already been started. Create a new one with "
                "scientist.new_experiment(...) to run again."
            )
        self._started = True
        try:
            result = self._agent.run_console(
                inputs=self._inputs,
                verbose=verbose,
                preview_chars=preview_chars,
                **self._template_params,
            )
            self._raw_output = result
            self._done = True
            return result
        except BaseException as e:
            self._error = e
            self._done = True
            raise

    async def run_async(
        self,
        *,
        verbose: bool = False,
        return_output: bool = False,
        timeout: Optional[float] = None,
        partial_on_timeout: bool = False,
    ) -> Any:
        """Async variant: run to completion without TTY formatting."""
        if self._started:
            raise RuntimeError(
                "Experiment has already been started. Create a new one with "
                "scientist.new_experiment(...) to run again."
            )
        self._started = True
        try:
            result = await self._agent.run_async(
                inputs=self._inputs,
                verbose=verbose,
                return_output=return_output,
                timeout=timeout,
                partial_on_timeout=partial_on_timeout,
                **self._template_params,
            )
            self._raw_output = result
            self._done = True
            return result
        except BaseException as e:
            self._error = e
            self._done = True
            raise

    def run_stream(
        self,
        *,
        verbose: bool = False,
        events: bool = False,
    ):
        """Sync streaming iterator — yields text chunks (or events)."""
        if self._started:
            raise RuntimeError(
                "Experiment has already been started. Create a new one with "
                "scientist.new_experiment(...) to run again."
            )
        self._started = True
        return self._agent.run_stream(
            inputs=self._inputs,
            verbose=verbose,
            events=events,
            **self._template_params,
        )

    # ------------------------------------------------------------------ #
    # Background run (Jupyter-friendly — no printing)
    # ------------------------------------------------------------------ #

    def run_in_background(self) -> "Experiment":
        """
        Start the experiment in a background thread and return immediately.

        Silences all agent printing so the call is safe inside a Jupyter cell
        without flooding the notebook with output. Poll :attr:`is_running`,
        :attr:`is_done`, :attr:`result`, and :attr:`error`, or block with
        :meth:`wait` to retrieve the final result.
        """
        if self._started:
            raise RuntimeError(
                "Experiment has already been started. Create a new one with "
                "scientist.new_experiment(...) to run again."
            )
        self._started = True

        agent = self._agent
        template_params = self._template_params
        inputs = self._inputs

        def worker() -> None:
            from upsonic.agent.agent import _run_in_bg_loop

            saved_print_param = agent._print_param
            saved_print = agent.print
            saved_show_tool_calls = agent.show_tool_calls
            saved_debug = agent.debug
            try:
                agent._print_param = False
                agent.print = False
                agent.show_tool_calls = False
                agent.debug = False
                result = _run_in_bg_loop(
                    agent.run_async(
                        inputs=inputs,
                        verbose=False,
                        **template_params,
                    )
                )
                self._raw_output = result
            except BaseException as e:
                self._error = e
            finally:
                agent._print_param = saved_print_param
                agent.print = saved_print
                agent.show_tool_calls = saved_show_tool_calls
                agent.debug = saved_debug
                self._done = True

        self._thread = threading.Thread(
            target=worker,
            name=f"Experiment-{self._template_params.get('research_paper', 'run')}",
            daemon=True,
        )
        self._thread.start()
        return self

    def wait(self, timeout: Optional[float] = None) -> "Optional[ExperimentResult]":
        """
        Block until the background run completes (or ``timeout`` elapses).
        Re-raises any exception that the run raised. Returns :attr:`result`.
        """
        if self._thread is None:
            raise RuntimeError("Experiment has not been started in background.")
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError(
                f"Experiment did not complete within {timeout} seconds."
            )
        if self._error is not None:
            raise self._error
        return self.result

    def stop(self, wait_for_run_id: float = 5.0) -> bool:
        """
        Cancel a running experiment.

        Uses the framework's :func:`upsonic.run.cancel.cancel_run`, which marks
        the in-flight run so the agent raises at its next pipeline checkpoint.
        Does not kill the thread outright — cancellation is cooperative.

        Args:
            wait_for_run_id: Poll up to this many seconds for the agent's
                ``run_id`` to appear (it's set once the run actually starts).

        Returns:
            ``True`` if a cancellation was requested, ``False`` otherwise
            (experiment never started, already done, or no ``run_id`` found).
        """
        import time
        from upsonic.run.cancel import cancel_run

        self._stop_requested = True
        if not self._started or self._done:
            return False

        deadline = time.time() + wait_for_run_id
        run_id = getattr(self._agent, "run_id", None)
        while not run_id and time.time() < deadline:
            time.sleep(0.05)
            run_id = getattr(self._agent, "run_id", None)

        if not run_id:
            return False
        return cancel_run(run_id)

    @property
    def is_running(self) -> bool:
        """True while the background thread is alive and not finished."""
        return self._thread is not None and not self._done

    @property
    def is_done(self) -> bool:
        """True once the run has finished (successfully or with error)."""
        return self._done

    @property
    def stop_requested(self) -> bool:
        """True if :meth:`stop` has been called on this experiment."""
        return self._stop_requested

    @property
    def record(self) -> Optional["ExperimentRecord"]:
        """
        The on-disk :class:`ExperimentRecord` for this experiment.

        Because the caller passes an explicit ``name`` to
        :meth:`AppliedScientist.new_experiment`, this is a direct lookup. We
        first check ``experiments.json`` via the registry; if the agent has
        created the experiment folder but hasn't registered it yet (a common
        mid-Phase-0 state), we synthesize a record from the folder itself so
        ``progress.json`` can be polled immediately.

        Returns ``None`` only while the experiment folder itself doesn't
        exist yet.
        """
        registry = self._agent.experiments
        try:
            return registry[self._name]
        except KeyError:
            pass
        except Exception:
            return None

        try:
            experiments_dir = registry._resolve_dir()
        except Exception:
            return None

        folder = experiments_dir / self._name
        if not folder.exists():
            return None

        return ExperimentRecord({"name": self._name}, experiments_dir)

    @property
    def result(self) -> Optional["ExperimentResult"]:
        """
        Structured :class:`ExperimentResult` parsed from ``result.json`` once
        the run has completed and written the file. Returns ``None`` if the
        run is still in progress or never produced a ``result.json``.
        Raises any exception the run raised.
        """
        if self._error is not None:
            raise self._error
        rec = self.record
        if rec is None:
            return None
        if rec.result is None:
            return None
        return ExperimentResult(rec)

    @property
    def output(self) -> Any:
        """
        The agent's raw return value (text / AgentRunOutput) from the
        underlying ``run_async`` call. Most users want :attr:`result` instead;
        this is exposed for debugging or power-user access.
        """
        if not self._done:
            return None
        if self._error is not None:
            raise self._error
        return self._raw_output

    @property
    def error(self) -> Optional[BaseException]:
        """The exception raised by the run, if any."""
        return self._error

    @property
    def progress_bar(self) -> Any:
        """
        Rich HTML progress bar for Jupyter.

        Returns an ``IPython.display.HTML`` snapshot built from the on-disk
        ``progress.json``. Re-evaluate the property (or re-run the cell) to
        refresh while a background run is in flight.
        """
        from IPython.display import HTML

        rec = self.record
        if rec is None:
            if self.is_running:
                msg = (
                    f"<strong>{self._name}</strong> — agent is starting up; "
                    "Phase 0 hasn't created the experiment folder yet. Re-run "
                    "this cell in a few seconds."
                )
            elif self._done and self._error is not None:
                msg = f"<strong>{self._name}</strong> errored before creating a folder: {self._error!r}"
            elif not self._started:
                msg = f"<strong>{self._name}</strong> has not been started yet."
            else:
                msg = f"No experiment folder found on disk for <strong>{self._name}</strong>."
            return HTML(f"<div style='color:#888; font-family:-apple-system,sans-serif'>{msg}</div>")
        phases = rec.phases or []
        total = len(phases)
        done = sum(1 for p in phases if p.get("status") == "done")
        failed = any(p.get("status") == "failed" for p in phases)
        pct = int(100 * done / total) if total else 0

        progress_status = None
        if isinstance(rec.progress, dict):
            progress_status = rec.progress.get("status")
        status = progress_status or rec.status or "?"
        activity = rec.current_activity or ""

        icon = {"done": "✓", "current": "●", "pending": "○", "failed": "✗"}
        color = {
            "done": "#2ecc71",
            "current": "#3498db",
            "pending": "#bdc3c7",
            "failed": "#e74c3c",
        }
        status_color = (
            "#e74c3c" if failed
            else "#2ecc71" if status == "COMPLETED"
            else "#3498db"
        )

        rows = "".join(
            f"<li style='color:{color.get(p.get('status'), '#555')}'>"
            f"{icon.get(p.get('status'), '·')} {p.get('name')}"
            f"{' — ' + p['summary'] if p.get('summary') else ''}"
            f"</li>"
            for p in phases
        )
        html = f"""
        <div style='font-family: -apple-system, sans-serif; max-width: 520px'>
          <div style='margin-bottom:6px'>
            <strong>{rec.name}</strong>
            <span style='color:{status_color}; font-weight:600; margin-left:8px'>{status}</span>
          </div>
          <progress value='{done}' max='{max(total, 1)}'
                    style='width:100%; height:12px'></progress>
          <div style='font-size:0.85em; color:#555; margin:2px 0 6px'>
            {done}/{total} phases ({pct}%)
          </div>
          <ul style='margin:4px 0; padding-left:20px; list-style:none; font-size:0.9em'>
            {rows}
          </ul>
          {f"<div style='font-size:0.85em; color:#333; margin-top:4px'><em>{activity}</em></div>" if activity else ""}
        </div>
        """
        return HTML(html)

    def last_logs(self, n: int = 5) -> Any:
        """
        Jupyter-renderable view of the last ``n`` entries from ``log.json``.

        Useful while a background run is in progress — re-run the cell to see
        newer entries as the agent appends them. Returns an
        :class:`IPython.display.HTML` panel with one card per phase entry
        (timestamp, action, status, details).
        """
        from IPython.display import HTML
        import html as html_mod

        rec = self.record
        if rec is None or not isinstance(rec.log, dict):
            return HTML(
                "<em style='color:#888'>No log.json on disk yet.</em>"
            )
        phases = rec.log.get("phases")
        if not isinstance(phases, list) or not phases:
            return HTML(
                "<em style='color:#888'>log.json has no phase entries yet.</em>"
            )

        tail = phases[-max(n, 1):]
        status_color = {
            "completed": "#2ecc71",
            "done": "#2ecc71",
            "success": "#2ecc71",
            "in_progress": "#3498db",
            "running": "#3498db",
            "current": "#3498db",
            "pending": "#bdc3c7",
            "failed": "#e74c3c",
            "error": "#e74c3c",
        }

        def fmt_details(details: Any) -> str:
            if details is None:
                return ""
            if isinstance(details, str):
                return html_mod.escape(details)
            try:
                pretty = json.dumps(details, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                pretty = str(details)
            if len(pretty) > 800:
                pretty = pretty[:800] + "\n…"
            return f"<pre style='margin:4px 0; padding:6px; background:#f7f7f7; border-radius:4px; font-size:0.8em; overflow-x:auto'>{html_mod.escape(pretty)}</pre>"

        cards = []
        for entry in tail:
            if not isinstance(entry, dict):
                continue
            ts = entry.get("timestamp") or entry.get("completed_at") or ""
            action = entry.get("action") or entry.get("name") or f"Phase {entry.get('phase', '?')}"
            status = entry.get("status") or ""
            color = status_color.get(str(status).lower(), "#555")
            details_keys = ("details", "summary", "data", "notes")
            details = next((entry.get(k) for k in details_keys if k in entry), None)
            if details is None:
                extra = {k: v for k, v in entry.items()
                         if k not in {"phase", "timestamp", "completed_at", "status", "action", "name", "index"}}
                details = extra or None
            cards.append(
                "<div style='border-left:3px solid " + color + "; padding:6px 10px; margin:6px 0; background:#fafafa'>"
                f"<div style='font-size:0.8em; color:#888'>{html_mod.escape(ts)}</div>"
                f"<div><strong>{html_mod.escape(str(action))}</strong>"
                f" <span style='color:{color}; font-size:0.85em'>[{html_mod.escape(str(status))}]</span></div>"
                f"{fmt_details(details)}"
                "</div>"
            )

        return HTML(
            f"<div style='font-family:-apple-system,sans-serif; max-width:720px'>"
            f"<div style='font-size:0.85em; color:#555; margin-bottom:4px'>"
            f"Last {len(cards)} of {len(phases)} log entries for "
            f"<strong>{html_mod.escape(self._name)}</strong></div>"
            + "".join(cards) + "</div>"
        )

    def __repr__(self) -> str:
        if self._done:
            state = "done"
        elif self.is_running:
            state = "running"
        else:
            state = "pending"
        return (
            f"Experiment(name={self._name!r}, "
            f"agent={type(self._agent).__name__}, state={state})"
        )


# --------------------------------------------------------------------------- #
# ExperimentResult — structured view of result.json
# --------------------------------------------------------------------------- #


class ExperimentResult:
    """
    Structured view of a completed experiment's ``result.json``.

    Exposes the four things callers actually want after a run:

    - :attr:`verdict`  — one of ``"BETTER"`` / ``"WORSE"`` / ``"INCONCLUSIVE"`` / ``"FAILED"``
    - :attr:`summary`  — 2-3 paragraphs describing the new method
    - :attr:`explanation` — 2-3 sentences explaining *why* this verdict
    - :attr:`table`    — list of comparison metric dicts
    """

    def __init__(self, record: "ExperimentRecord") -> None:
        self._record = record
        data = record.result
        self._data: Dict[str, Any] = data if isinstance(data, dict) else {}

    @property
    def record(self) -> "ExperimentRecord":
        """The underlying :class:`ExperimentRecord`."""
        return self._record

    @property
    def name(self) -> Optional[str]:
        return self._data.get("name") or self._record.name

    @property
    def verdict(self) -> Optional[str]:
        """``"BETTER"`` | ``"WORSE"`` | ``"INCONCLUSIVE"`` | ``"FAILED"``."""
        return self._data.get("verdict")

    @property
    def summary(self) -> Optional[str]:
        """Short prose description of the new method (2-3 paragraphs)."""
        return self._data.get("summary")

    @property
    def explanation(self) -> Optional[str]:
        """Short prose explaining why this verdict was reached."""
        return self._data.get("explanation")

    @property
    def table(self) -> List[Dict[str, Any]]:
        """
        List of metric comparison rows — each a dict with keys
        ``name``, ``current``, ``new``, ``diff``, ``diff_display``, ``unit``,
        ``higher_is_better``, ``better``.
        """
        comp = self._data.get("comparison")
        if not isinstance(comp, dict):
            return []
        metrics = comp.get("metrics")
        return metrics if isinstance(metrics, list) else []

    @property
    def file_locations(self) -> Dict[str, Any]:
        """``file_locations`` mapping from ``result.json``."""
        loc = self._data.get("file_locations")
        return loc if isinstance(loc, dict) else {}

    def to_dict(self) -> Dict[str, Any]:
        """Return the full parsed ``result.json`` as a plain dict."""
        return dict(self._data)

    def _repr_html_(self) -> str:
        """Pretty HTML in Jupyter: header, verdict badge, metric table, summary."""
        verdict = (self.verdict or "UNKNOWN").upper()
        color = {
            "BETTER": "#2ecc71",
            "WORSE": "#e74c3c",
            "INCONCLUSIVE": "#f39c12",
            "FAILED": "#7f8c8d",
        }.get(verdict, "#7f8c8d")

        rows = "".join(
            "<tr>"
            f"<td style='padding:4px 10px'>{m.get('name', '')}</td>"
            f"<td style='padding:4px 10px; text-align:right'>{m.get('current', '')}</td>"
            f"<td style='padding:4px 10px; text-align:right'>{m.get('new', '')}</td>"
            f"<td style='padding:4px 10px; text-align:right'>{m.get('diff_display', '')}</td>"
            f"<td style='padding:4px 10px; text-align:center'>{m.get('better', '') or ''}</td>"
            "</tr>"
            for m in self.table
        )
        table_html = (
            f"<table style='border-collapse:collapse; margin:10px 0'>"
            f"<thead><tr style='background:#f5f5f5'>"
            f"<th style='padding:4px 10px; text-align:left'>Metric</th>"
            f"<th style='padding:4px 10px; text-align:right'>Current</th>"
            f"<th style='padding:4px 10px; text-align:right'>New</th>"
            f"<th style='padding:4px 10px; text-align:right'>Diff</th>"
            f"<th style='padding:4px 10px'>Better</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
            if self.table else "<em style='color:#888'>No comparison metrics.</em>"
        )
        summary = self.summary or ""
        explanation = self.explanation or ""
        return f"""
        <div style='font-family: -apple-system, sans-serif; max-width: 720px'>
          <div style='margin-bottom:8px'>
            <strong style='font-size:1.05em'>{self.name or 'Experiment'}</strong>
            <span style='background:{color}; color:white; padding:2px 10px;
                         border-radius:10px; margin-left:8px; font-size:0.85em;
                         font-weight:600'>{verdict}</span>
          </div>
          {table_html}
          {f"<div style='margin-top:6px'><strong>Why:</strong> {explanation}</div>" if explanation else ""}
          {f"<div style='margin-top:6px; color:#555'>{summary}</div>" if summary else ""}
        </div>
        """

    def __repr__(self) -> str:
        return (
            f"ExperimentResult(name={self.name!r}, "
            f"verdict={self.verdict!r}, metrics={len(self.table)})"
        )


# --------------------------------------------------------------------------- #
# ExperimentRecord + ExperimentRegistry — disk-backed view of finished runs
# --------------------------------------------------------------------------- #


class ExperimentRecord:
    """
    Read-only view of an experiment on disk.

    Backed by ``experiments.json`` plus the per-experiment JSON files
    (``progress.json``, ``log.json``, ``result.json``). Every property
    re-reads from disk so you always see the latest state — useful for
    polling while a background run is in progress.
    """

    def __init__(self, data: Dict[str, Any], experiments_dir: Path) -> None:
        self._data = data
        self._experiments_dir = experiments_dir

    # Plain registry fields ------------------------------------------------ #

    @property
    def name(self) -> Optional[str]:
        return self._data.get("name")

    @property
    def status(self) -> Optional[str]:
        """``"in_progress"`` | ``"completed"`` | ``"failed"`` — from experiments.json."""
        return self._data.get("status")

    @property
    def verdict(self) -> Optional[str]:
        """``"BETTER"`` | ``"WORSE"`` | ``"INCONCLUSIVE"`` | ``"FAILED"`` — set on finalize."""
        return self._data.get("verdict")

    @property
    def key_metric(self) -> Optional[Dict[str, Any]]:
        return self._data.get("key_metric")

    @property
    def baseline_model(self) -> Optional[str]:
        return self._data.get("baseline_model")

    @property
    def new_method(self) -> Optional[str]:
        return self._data.get("new_method")

    @property
    def paper(self) -> Optional[str]:
        return self._data.get("paper")

    @property
    def date(self) -> Optional[str]:
        return self._data.get("date")

    @property
    def path(self) -> Path:
        """Absolute path to the experiment folder on disk."""
        name = self.name or ""
        return self._experiments_dir / name

    # File readers --------------------------------------------------------- #

    def _read_json(self, filename: str) -> Optional[Any]:
        f = self.path / filename
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    @property
    def progress(self) -> Optional[Dict[str, Any]]:
        """Parsed ``progress.json`` (live snapshot) — ``None`` if missing/invalid."""
        return self._read_json("progress.json")

    @property
    def log(self) -> Optional[Dict[str, Any]]:
        """Parsed ``log.json`` (phase-by-phase structured log)."""
        return self._read_json("log.json")

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        """Parsed ``result.json`` (final report — only present when done)."""
        return self._read_json("result.json")

    # Derived views from result.json -------------------------------------- #

    @property
    def summary(self) -> Optional[str]:
        """The ``summary`` field from ``result.json`` — the new method description."""
        r = self.result
        return r.get("summary") if isinstance(r, dict) else None

    @property
    def explanation(self) -> Optional[str]:
        """The ``explanation`` field from ``result.json`` — why this verdict."""
        r = self.result
        return r.get("explanation") if isinstance(r, dict) else None

    @property
    def comparison(self) -> Optional[List[Dict[str, Any]]]:
        """
        List of comparison metric dicts from ``result.json.comparison.metrics``.

        Each entry has: ``name``, ``current``, ``new``, ``diff``, ``diff_display``,
        ``unit``, ``higher_is_better``, ``better``.
        """
        r = self.result
        if not isinstance(r, dict):
            return None
        comp = r.get("comparison")
        if not isinstance(comp, dict):
            return None
        metrics = comp.get("metrics")
        return metrics if isinstance(metrics, list) else None

    # Derived views from progress.json ------------------------------------ #

    @property
    def phases(self) -> Optional[List[Dict[str, Any]]]:
        """
        Normalised list of phase dicts from ``progress.json`` — for progress
        bars. Accepts both the documented list schema and the dict-keyed
        variant some agents emit. Each returned dict has at minimum
        ``name``, ``status``, and (if present) ``summary`` / ``index``.

        Status values are normalised so UI code can rely on a fixed
        vocabulary: ``"done"`` | ``"current"`` | ``"pending"`` | ``"failed"``.
        """
        p = self.progress
        if not isinstance(p, dict):
            return None
        raw = p.get("phases")
        if raw is None:
            return None

        status_aliases = {
            "completed": "done",
            "complete": "done",
            "finished": "done",
            "success": "done",
            "ok": "done",
            "running": "current",
            "in_progress": "current",
            "in-progress": "current",
            "active": "current",
            "todo": "pending",
            "waiting": "pending",
            "queued": "pending",
            "error": "failed",
            "errored": "failed",
        }

        def normalise(entry: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
            out = dict(entry)
            raw_status = out.get("status")
            if isinstance(raw_status, str):
                out["status"] = status_aliases.get(raw_status.lower(), raw_status.lower())
            if "index" not in out:
                out["index"] = fallback_index
            if "summary" not in out:
                out["summary"] = None
            return out

        if isinstance(raw, list):
            return [
                normalise(entry if isinstance(entry, dict) else {}, i)
                for i, entry in enumerate(raw)
            ]

        if isinstance(raw, dict):
            ordered: List[Dict[str, Any]] = []
            for i, (key, entry) in enumerate(raw.items()):
                if not isinstance(entry, dict):
                    entry = {}
                with_name = dict(entry)
                with_name.setdefault("name", key)
                ordered.append(normalise(with_name, i))
            current_idx = p.get("current_phase")
            if isinstance(current_idx, int):
                for i, ph in enumerate(ordered):
                    if ph.get("status") == "pending" and i == current_idx:
                        ph["status"] = "current"
                    elif i < current_idx and ph.get("status") == "pending":
                        ph["status"] = "done"
            return ordered

        return None

    @property
    def current_activity(self) -> Optional[str]:
        """The ``current_activity`` line from ``progress.json``."""
        p = self.progress
        return p.get("current_activity") if isinstance(p, dict) else None

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying experiments.json entry as a plain dict."""
        return dict(self._data)

    def __repr__(self) -> str:
        return (
            f"ExperimentRecord(name={self.name!r}, status={self.status!r}, "
            f"verdict={self.verdict!r})"
        )


class ExperimentRegistry:
    """
    Dict-like, always-fresh view of ``experiments.json``.

    Iteration order matches the file. Every access re-reads the file so
    entries reflect the live state (e.g. a background run flipping from
    ``in_progress`` to ``completed``).
    """

    def __init__(
        self,
        agent: "AppliedScientist",
        experiments_subdir: str = "experiments",
    ) -> None:
        self._agent = agent
        self._subdir = experiments_subdir

    def _resolve_dir(self) -> Path:
        if self._agent.autonomous_workspace is None:
            raise RuntimeError(
                "Agent has no workspace; set one at construction to browse experiments."
            )
        sub = Path(self._subdir)
        if sub.is_absolute():
            return sub
        return self._agent.autonomous_workspace / sub

    def _load(self) -> List[Dict[str, Any]]:
        root = self._resolve_dir()
        ej = root / "experiments.json"
        if not ej.exists():
            return []
        try:
            data = json.loads(ej.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        raw = data.get("experiments", [])
        return [e for e in raw if isinstance(e, dict)]

    def __iter__(self) -> Iterator[str]:
        return iter(e["name"] for e in self._load() if "name" in e)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, name: object) -> bool:
        return any(e.get("name") == name for e in self._load())

    def __getitem__(self, name: str) -> ExperimentRecord:
        root = self._resolve_dir()
        for e in self._load():
            if e.get("name") == name:
                return ExperimentRecord(e, root)
        raise KeyError(name)

    def get(
        self,
        name: str,
        default: Optional[ExperimentRecord] = None,
    ) -> Optional[ExperimentRecord]:
        try:
            return self[name]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        return [e["name"] for e in self._load() if "name" in e]

    def values(self) -> List[ExperimentRecord]:
        root = self._resolve_dir()
        return [ExperimentRecord(e, root) for e in self._load()]

    def items(self) -> List[tuple[str, ExperimentRecord]]:
        root = self._resolve_dir()
        return [
            (e["name"], ExperimentRecord(e, root))
            for e in self._load()
            if "name" in e
        ]

    def __repr__(self) -> str:
        try:
            names = self.keys()
        except Exception:
            names = []
        return f"ExperimentRegistry({len(names)} experiment(s): {names})"


# --------------------------------------------------------------------------- #
# AppliedScientist
# --------------------------------------------------------------------------- #


class AppliedScientist(PrebuiltAutonomousAgent):
    """
    Prebuilt "applied scientist" agent backed by
    ``Upsonic/AutonomousAgents/applied_scientist``.

    The agent repo and folder are hard-coded, so callers only supply model and
    workspace. Each experiment is created via :meth:`new_experiment` and then
    launched with ``experiment.run()`` (pretty TTY), ``experiment.run_in_background()``
    (quiet, Jupyter-friendly), or ``experiment.run_async()``.

    After experiments finish, read historical state through :attr:`experiments`:

        ```python
        scientist.experiments                      # ExperimentRegistry
        scientist.experiments["tabpfn_adult"].status
        scientist.experiments["tabpfn_adult"].summary
        scientist.experiments["tabpfn_adult"].comparison
        ```
    """

    AGENT_REPO: str = "https://github.com/Upsonic/Upsonic"
    AGENT_FOLDER: str = "prebuilt_autonomous_agents/applied_scientist"

    def __init__(
        self,
        *,
        model: Union[str, "Model"] = "openai/gpt-4o",
        workspace: Optional[str] = None,
        experiments_directory: str = "experiments",
        **kwargs: Any,
    ) -> None:
        kwargs.pop("agent_repo", None)
        kwargs.pop("agent_folder", None)
        super().__init__(
            model=model,
            workspace=workspace,
            agent_repo=self.AGENT_REPO,
            agent_folder=self.AGENT_FOLDER,
            **kwargs,
        )
        self._experiments_directory: str = experiments_directory

    def new_experiment(
        self,
        name: str,
        *,
        research_paper: str,
        current_notebook: str,
        current_data: str,
        experiments_directory: Optional[str] = None,
        inputs: Optional[List[str]] = None,
    ) -> Experiment:
        """
        Prepare an experiment for this scientist agent. Does not run the
        experiment; call :meth:`Experiment.run`, :meth:`Experiment.run_in_background`,
        :meth:`Experiment.run_async`, or :meth:`Experiment.run_stream` to execute.

        ``name`` is the exact folder name / ``"name"`` field used by the agent
        in every JSON file it writes. The agent is instructed to use it
        verbatim — no derivation from the paper title, no suffixes — so
        ``scientist.experiments[name]`` always points at this run.

        ``experiments_directory`` defaults to the value supplied at
        construction (``"experiments"`` unless overridden). It is always
        relative to the agent's workspace.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "new_experiment(name=...) requires a non-empty string."
            )
        exp_dir = experiments_directory or self._experiments_directory
        if experiments_directory is not None:
            self._experiments_directory = experiments_directory
        return Experiment(
            agent=self,
            name=name,
            template_params={
                "research_name": name,
                "research_paper": research_paper,
                "current_notebook": current_notebook,
                "current_data": current_data,
                "experiments_directory": exp_dir,
            },
            inputs=inputs,
        )

    @property
    def experiments(self) -> ExperimentRegistry:
        """
        Live, dict-like view over ``{workspace}/{experiments_directory}/experiments.json``.

        Every access re-reads the file, so entries reflect the current state
        of running or completed experiments on disk.
        """
        return ExperimentRegistry(self, experiments_subdir=self._experiments_directory)

    def progress_bar_live(
        self,
        experiment: "Experiment",
        interval: float = 5.0,
    ) -> None:
        """
        Block the current Jupyter cell and live-refresh ``experiment.progress_bar``
        every ``interval`` seconds until the experiment finishes.

        The cell output is replaced in place (``IPython.display.clear_output``)
        so the progress bar animates without scrolling. Interrupting the kernel
        (◼️) stops the watcher but leaves the background run untouched.

        Args:
            experiment: The :class:`Experiment` to follow. Must have been
                started via :meth:`Experiment.run_in_background` (otherwise
                there's nothing to refresh).
            interval: Poll interval in seconds. Defaults to 5.
        """
        import time
        from IPython.display import clear_output, display

        if not experiment.is_running and not experiment.is_done:
            display(experiment.progress_bar)
            return

        while experiment.is_running:
            clear_output(wait=True)
            display(experiment.progress_bar)
            time.sleep(interval)

        clear_output(wait=True)
        display(experiment.progress_bar)
        if experiment.error is not None:
            print(f"\nFinished with error: {experiment.error!r}")
        else:
            print("\nFinished.")

    def list_experiments(
        self,
        *,
        status: Optional[str] = None,
        descending: bool = True,
    ) -> None:
        """
        Print every experiment in the registry as a formatted table.

        Each row shows: ``date``, ``name``, ``status``, ``verdict``, and
        ``new_method vs baseline_model``. Sorted by date (newest first by
        default). Re-reads ``experiments.json`` on every call.

        Args:
            status: Optional filter — ``"in_progress"``, ``"completed"``, or
                ``"failed"``. ``None`` shows all.
            descending: Reverse sort order (newest first by default).
        """
        rows: List[Dict[str, Any]] = []
        for name, rec in self.experiments.items():
            if status is not None and rec.status != status:
                continue
            rows.append({
                "name": name,
                "date": rec.date,
                "status": rec.status,
                "verdict": rec.verdict,
                "baseline_model": rec.baseline_model,
                "new_method": rec.new_method,
            })
        rows.sort(key=lambda r: (r.get("date") or ""), reverse=descending)

        if not rows:
            filt = f" with status={status!r}" if status else ""
            print(f"No experiments registered{filt}.")
            return

        header = f"{'DATE':<12} {'NAME':>30}  {'STATUS':<12}  {'VERDICT':<14}  NEW vs BASELINE"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['date'] or '????-??-??':<12} {r['name']:>30}  "
                f"{r['status'] or '-':<12}  {r['verdict'] or '-':<14}  "
                f"{r['new_method'] or '-'} vs {r['baseline_model'] or '-'}"
            )

    def __repr__(self) -> str:
        ws = str(self.autonomous_workspace) if self.autonomous_workspace else None
        return (
            f"AppliedScientist(model={self.model_name!r}, workspace={ws!r}, "
            f"experiments_directory={self._experiments_directory!r})"
        )
