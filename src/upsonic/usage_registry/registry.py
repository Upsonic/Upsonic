"""In-memory usage registry — Phase 0 backend.

Cross-process persistence will be added in Phase 4 via the existing
``Storage`` layer. The public API of this class will not change at that
point — only the internal ``_entries`` list will gain a storage flush.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from upsonic.usage_registry.aggregated import AggregatedUsage
from upsonic.usage_registry.entry import UsageEntry

if TYPE_CHECKING:
    from upsonic.storage.base import Storage


class UsageRegistry:
    """Append-only ledger keyed by ``entry_id``.

    Thread-safe. Idempotent on ``entry_id`` — recording the same id twice
    replaces the previous row instead of double-counting, which is what
    makes the registry retry-safe and resume-safe without the baseline
    arithmetic the old ``TaskUsage.snapshot/subtract`` flow needed.

    Optional storage backend (Phase 4): when ``storage`` is attached every
    :meth:`record` performs a write-through into the storage backend's
    ``usage_entries`` table, and :meth:`load_from_storage` rehydrates the
    in-memory dict from a prior process — so ``chat.total_cost`` keeps
    accumulating across restarts and across workers that share the same
    storage URL.
    """

    def __init__(self, storage: Optional["Storage"] = None) -> None:
        self._entries: Dict[str, UsageEntry] = {}
        self._lock = threading.RLock()
        self._storage: Optional["Storage"] = storage

    # ------------------------------------------------------------------
    # Storage wiring (Phase 4)
    # ------------------------------------------------------------------
    def attach_storage(self, storage: "Storage") -> None:
        """Bind a storage backend so subsequent :meth:`record` calls
        also persist. Calling this on a registry that already has data
        does NOT back-fill storage — call :meth:`flush_to_storage` for
        that."""
        self._storage = storage

    def detach_storage(self) -> None:
        self._storage = None

    @property
    def storage(self) -> Optional["Storage"]:
        return self._storage

    def flush_to_storage(self, storage: Optional["Storage"] = None) -> int:
        """Bulk-write every in-memory entry to ``storage`` (or the attached
        one). Idempotent thanks to ``entry_id`` upserts. Returns count."""
        target = storage if storage is not None else self._storage
        if target is None:
            return 0
        with self._lock:
            rows = list(self._entries.values())
        for e in rows:
            try:
                target.upsert_usage_entry(e.to_dict())
            except NotImplementedError:
                return 0   # Backend doesn't support persistence; bail silently.
        return len(rows)

    def load_from_storage(
        self,
        storage: Optional["Storage"] = None,
        **scope,
    ) -> int:
        """Pull every row matching ``scope`` from storage into memory.

        Returns the count loaded. Typical use: when a Chat opens, call
        ``registry.load_from_storage(storage, chat_usage_id=chat.chat_usage_id)``
        so ``chat.total_cost`` reflects historical spend.
        """
        target = storage if storage is not None else self._storage
        if target is None:
            return 0
        try:
            rows = target.query_usage_entries(**scope)
        except NotImplementedError:
            return 0
        count = 0
        with self._lock:
            for row in rows:
                entry = UsageEntry.from_dict(row)
                self._entries[entry.entry_id] = entry
                count += 1
        return count

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def record(self, entry: UsageEntry) -> None:
        """Insert or replace ``entry`` by its ``entry_id`` (and persist
        through to the attached storage backend when one is bound)."""
        with self._lock:
            self._entries[entry.entry_id] = entry
        if self._storage is not None:
            try:
                self._storage.upsert_usage_entry(entry.to_dict())
            except NotImplementedError:
                # Storage backend hasn't been ported; keep the in-memory
                # write — the rest of the system still works, only cross-
                # process resume is unavailable on this backend.
                pass

    def record_many(self, entries: Iterable[UsageEntry]) -> None:
        # Materialise once — the input may be a generator.
        materialised = list(entries)
        with self._lock:
            for e in materialised:
                self._entries[e.entry_id] = e
        if self._storage is not None:
            try:
                for e in materialised:
                    self._storage.upsert_usage_entry(e.to_dict())
            except NotImplementedError:
                pass

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
