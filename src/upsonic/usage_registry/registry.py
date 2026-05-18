"""In-memory usage registry — Phase 0 backend.

Cross-process persistence will be added in Phase 4 via the existing
``Storage`` layer. The public API of this class will not change at that
point — only the internal ``_entries`` list will gain a storage flush.
"""
from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Optional

from upsonic.usage_registry.aggregated import AggregatedUsage
from upsonic.usage_registry.entry import UsageEntry


class UsageRegistry:
    """Append-only ledger keyed by ``entry_id``.

    Thread-safe. Idempotent on ``entry_id`` — recording the same id twice
    replaces the previous row instead of double-counting, which is what
    makes the registry retry-safe and resume-safe without the baseline
    arithmetic the old ``TaskUsage.snapshot/subtract`` flow needed.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, UsageEntry] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def record(self, entry: UsageEntry) -> None:
        """Insert or replace ``entry`` by its ``entry_id``."""
        with self._lock:
            self._entries[entry.entry_id] = entry

    def record_many(self, entries: Iterable[UsageEntry]) -> None:
        with self._lock:
            for e in entries:
                self._entries[e.entry_id] = e

    def remove(self, entry_id: str) -> bool:
        with self._lock:
            return self._entries.pop(entry_id, None) is not None

    def clear(self) -> None:
        """Drop every entry. Primarily for tests."""
        with self._lock:
            self._entries.clear()

    # ------------------------------------------------------------------
    # Read — entries
    # ------------------------------------------------------------------
    def entries(
        self,
        *,
        chat_usage_id: Optional[str] = None,
        agent_usage_id: Optional[str] = None,
        task_usage_id: Optional[str] = None,
        team_usage_id: Optional[str] = None,
        workflow_usage_id: Optional[str] = None,
        system_usage_id: Optional[str] = None,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> List[UsageEntry]:
        """Return entries matching every non-``None`` filter (AND semantics).

        A filter set to ``None`` is ignored. To match "entries with no
        such scope set" use a sentinel like ``""`` and filter manually.
        """
        with self._lock:
            rows = list(self._entries.values())

        def keep(e: UsageEntry) -> bool:
            if chat_usage_id is not None and e.chat_usage_id != chat_usage_id:
                return False
            if agent_usage_id is not None and e.agent_usage_id != agent_usage_id:
                return False
            if task_usage_id is not None and e.task_usage_id != task_usage_id:
                return False
            if team_usage_id is not None and e.team_usage_id != team_usage_id:
                return False
            if workflow_usage_id is not None and e.workflow_usage_id != workflow_usage_id:
                return False
            if system_usage_id is not None and e.system_usage_id != system_usage_id:
                return False
            if run_id is not None and e.run_id != run_id:
                return False
            if user_id is not None and e.user_id != user_id:
                return False
            if kind is not None and e.kind != kind:
                return False
            return True

        return [e for e in rows if keep(e)]

    def get(self, entry_id: str) -> Optional[UsageEntry]:
        with self._lock:
            return self._entries.get(entry_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Read — aggregated
    # ------------------------------------------------------------------
    def aggregate(self, **scope) -> AggregatedUsage:
        """Roll up every entry matching ``scope`` into a single view.

        Convenience for the most common query path. Passes ``scope``
        straight to :meth:`entries`.
        """
        return AggregatedUsage.from_entries(self.entries(**scope))

    # Convenience read-shortcuts so callers don't have to memorize the
    # kwarg name when they only care about one scope.
    def by_chat(self, chat_usage_id: str) -> AggregatedUsage:
        return self.aggregate(chat_usage_id=chat_usage_id)

    def by_agent(self, agent_usage_id: str) -> AggregatedUsage:
        return self.aggregate(agent_usage_id=agent_usage_id)

    def by_task(self, task_usage_id: str) -> AggregatedUsage:
        return self.aggregate(task_usage_id=task_usage_id)

    def by_team(self, team_usage_id: str) -> AggregatedUsage:
        return self.aggregate(team_usage_id=team_usage_id)

    def by_workflow(self, workflow_usage_id: str) -> AggregatedUsage:
        return self.aggregate(workflow_usage_id=workflow_usage_id)


# ----------------------------------------------------------------------
# Default registry — process-wide singleton for in-memory mode.
# ----------------------------------------------------------------------
_default_registry: Optional[UsageRegistry] = None
_default_lock = threading.Lock()


def get_default_registry() -> UsageRegistry:
    """Return the process-wide default registry, creating it on first call.

    Tests should call :meth:`UsageRegistry.clear` between cases rather
    than swap the singleton, so that production wiring keeps working
    when the test fixture tears down.
    """
    global _default_registry
    if _default_registry is None:
        with _default_lock:
            if _default_registry is None:
                _default_registry = UsageRegistry()
    return _default_registry
