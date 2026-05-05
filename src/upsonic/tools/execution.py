"""Tool execution wrapper — replaces ``create_behavioral_wrapper``.

Holds the 9-aspect pipeline (KB setup, before-hook, 3 pause checks,
cache check, timeout-wrapped retry loop, metrics, cache write,
show-result, after-hook, stop-after-call). Reads KB-related state
from a ``ToolRegistry`` reference.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, TYPE_CHECKING

from upsonic.tools.base import Tool
from upsonic.tools.config import ToolConfig
from upsonic.tools.hitl import (
    ConfirmationPause,
    ExternalExecutionPause,
    UserInputPause,
)

if TYPE_CHECKING:
    from upsonic.tools.registry import ToolRegistry


class ToolWrapper:
    """Build a behavioral wrapper around a ``Tool``.

    Same aspect order and exception semantics as the legacy
    ``create_behavioral_wrapper`` on ``ToolProcessor``. Reads
    knowledge-base state from a ``ToolRegistry`` reference passed at
    construction.
    """

    def __init__(self, registry: "ToolRegistry") -> None:
        self._registry = registry

    def wrap(self, tool: Tool) -> Callable[..., Awaitable[Any]]:
        """Return an async callable that runs the tool through the
        9-aspect behavioral pipeline.
        """
        registry = self._registry

        @functools.wraps(tool.execute)
        async def wrapper(**kwargs: Any) -> Any:
            from upsonic.utils.printing import console, spacing

            config = getattr(tool, 'config', ToolConfig())

            # ── KB setup ──────────────────────────────────────────
            if registry.knowledge_base_instances:
                for kb_id, kb in registry.knowledge_base_instances.items():
                    if tool.name in (registry.class_instance_to_tools.get(kb_id) or []):
                        try:
                            await kb.setup_async()
                        except Exception as e:
                            from upsonic.utils.printing import warning_log
                            warning_log(
                                f"Could not ensure KnowledgeBase setup for tool '{tool.name}': {e}",
                                "ToolWrapper",
                            )
                        break

            func_dict: Dict[str, Any] = {}

            # ── Before hook ───────────────────────────────────────
            if config.tool_hooks and config.tool_hooks.before:
                try:
                    result = config.tool_hooks.before(**kwargs)
                    if result is not None:
                        func_dict["func_before"] = result
                except Exception as e:
                    console.print(f"[red]Before hook error: {e}[/red]")
                    raise

            # ── Pause checks ──────────────────────────────────────
            if config.requires_confirmation:
                raise ConfirmationPause()

            if config.requires_user_input:
                raise UserInputPause()

            if config.external_execution:
                raise ExternalExecutionPause()

            # ── Cache check ───────────────────────────────────────
            cache_key = None
            if config.cache_results:
                cache_key = self._get_cache_key(tool.name, kwargs)
                cached = self._get_cached_result(cache_key, config)
                if cached is not None:
                    console.print(f"[green]✓ Cache hit for {tool.name}[/green]")
                    func_dict["func_cache"] = cached
                    return func_dict

            # ── Retry loop with timeout ──────────────────────────
            start_time = time.time()
            max_retries = config.max_retries
            result = None
            execution_success = False

            for attempt in range(max_retries + 1):
                try:
                    if config.timeout:
                        result = await asyncio.wait_for(
                            tool.execute(**kwargs),
                            timeout=config.timeout,
                        )
                    else:
                        result = await tool.execute(**kwargs)

                    execution_success = True
                    break

                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        console.print(
                            f"[yellow]Tool '{tool.name}' timed out, retrying in {wait_time}s... "
                            f"(attempt {attempt + 1}/{max_retries + 1})[/yellow]"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise TimeoutError(
                            f"Tool '{tool.name}' timed out after {config.timeout}s "
                            f"and {max_retries} retries"
                        )

                except (ExternalExecutionPause, ConfirmationPause, UserInputPause):
                    raise

                except Exception as e:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        console.print(
                            f"[yellow]Tool '{tool.name}' failed, retrying in {wait_time}s... "
                            f"(attempt {attempt + 1}/{max_retries + 1})[/yellow]"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        console.print(
                            f"[bold red]Tool error after {max_retries} retries: {e}[/bold red]"
                        )
                        raise

            execution_time = time.time() - start_time

            # ── Metrics ────────────────────────────────────────────
            tool.record_execution(
                execution_time=execution_time,
                args=kwargs,
                result=result,
                success=execution_success,
            )

            # ── Cache write ────────────────────────────────────────
            if config.cache_results and cache_key:
                self._cache_result(cache_key, result, config)

            # ── Show result ────────────────────────────────────────
            if config.show_result:
                console.print(f"[bold green]Tool Result:[/bold green] {result}")
                spacing()

            # ── After hook ─────────────────────────────────────────
            if config.tool_hooks and config.tool_hooks.after:
                try:
                    hook_result = config.tool_hooks.after(result)
                    if hook_result is not None:
                        func_dict["func_after"] = hook_result
                except Exception as e:
                    console.print(f"[bold red]After hook error: {e}[/bold red]")

            func_dict["func"] = result

            # ── Stop after call ────────────────────────────────────
            if config.stop_after_tool_call:
                console.print("[bold yellow]Stopping after tool call[/bold yellow]")
                func_dict["_stop_execution"] = True

            return func_dict

        return wrapper

    # ------------------------------------------------------------------
    # Cache helpers (private)
    # ------------------------------------------------------------------

    def _get_cache_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        key_data = json.dumps(
            {"tool": tool_name, "args": args},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str, config: ToolConfig) -> Any:
        cache_dir = Path(config.cache_dir or Path.home() / '.upsonic' / 'cache')
        cache_file = cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)

            if config.cache_ttl:
                age = time.time() - data.get('timestamp', 0)
                if age > config.cache_ttl:
                    cache_file.unlink()
                    return None

            return data.get('result')

        except Exception:
            return None

    def _cache_result(self, cache_key: str, result: Any, config: ToolConfig) -> None:
        cache_dir = Path(config.cache_dir or Path.home() / '.upsonic' / 'cache')
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / f"{cache_key}.json"

        try:
            data = {
                'timestamp': time.time(),
                'result': result,
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not cache result: {e}", "ToolWrapper")
