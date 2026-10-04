"""Durable invocation snapshots retain identity, inputs and completed receipts."""

import pickle
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from upsonic.graphv2.checkpoint import Checkpoint, MemorySaver, SqliteCheckpointer
from upsonic.graphv2.primitives import Command


def make_checkpoint(pending_tasks):
    return Checkpoint(
        checkpoint_id="checkpoint", thread_id="thread", state={"count": 0},
        next_nodes=["worker"], pending_tasks=pending_tasks,
    )


def test_sqlite_pending_tasks_survive_fresh_connection(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    tasks = [{"id": "invocation", "node": "worker", "input": {"item": [1, 2]},
              "status": "COMPLETED", "result": Command(update={"receipt": "paid"}, goto="finish")}]
    with closing(sqlite3.connect(path)) as connection:
        SqliteCheckpointer(connection).put(make_checkpoint(tasks))
    with closing(sqlite3.connect(path)) as connection:
        recovered = SqliteCheckpointer(connection).get("thread")
        assert recovered.pending_tasks == tasks
        assert recovered.pending_tasks[0]["result"].goto == "finish"


def test_sqlite_upgrades_existing_table_without_losing_legacy_checkpoint(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("""CREATE TABLE checkpoints (
            checkpoint_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL,
            parent_checkpoint_id TEXT, state BLOB NOT NULL, next_nodes TEXT NOT NULL,
            timestamp TEXT NOT NULL, metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        connection.execute("""INSERT INTO checkpoints
            (checkpoint_id, thread_id, state, next_nodes, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)""",
            ("legacy", "thread", pickle.dumps({"paid": True}), '["worker"]',
             "2026-01-01T00:00:00", "{}"))
        connection.commit()
        saver = SqliteCheckpointer(connection)
        legacy = saver.get("thread")
        assert legacy.state == {"paid": True}
        assert legacy.pending_tasks is None
        saver.put(make_checkpoint([]))
        assert saver.get("thread").pending_tasks == []


def test_checkpoint_dictionary_distinguishes_legacy_from_settled_plan():
    legacy = Checkpoint("legacy", "thread", {}, ["worker"])
    assert Checkpoint.from_dict(legacy.to_dict()).pending_tasks is None
    old_dictionary = legacy.to_dict()
    old_dictionary.pop("pending_tasks", None)
    assert Checkpoint.from_dict(old_dictionary).pending_tasks is None
    assert Checkpoint.from_dict(make_checkpoint([]).to_dict()).pending_tasks == []


def test_memory_saver_detaches_dispatch_snapshots():
    saver = MemorySaver()
    checkpoint = make_checkpoint([{"id": "invocation", "status": "DISPATCH_RECORDED",
                                   "input": {"items": [1]}}])
    saver.put(checkpoint)
    checkpoint.pending_tasks[0]["status"] = "COMPLETED"
    checkpoint.pending_tasks[0]["input"]["items"].append(2)
    recovered = saver.get("thread")
    assert recovered.pending_tasks[0]["status"] == "DISPATCH_RECORDED"
    assert recovered.pending_tasks[0]["input"] == {"items": [1]}
    recovered.pending_tasks[0]["status"] = "UNKNOWN"
    assert saver.get("thread").pending_tasks[0]["status"] == "DISPATCH_RECORDED"
    history = list(saver.list("thread"))
    history[0].pending_tasks.clear()
    assert len(saver.get("thread").pending_tasks) == 1


def test_sqlite_latest_is_durable_write_order_not_wall_clock(tmp_path):
    with closing(sqlite3.connect(tmp_path / "clock.sqlite")) as connection:
        saver = SqliteCheckpointer(connection)
        plan = make_checkpoint([{"id": "invocation", "status": "NOT_STARTED"}])
        plan.timestamp = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
        saver.put(plan)
        dispatch = make_checkpoint([{"id": "invocation", "status": "DISPATCH_RECORDED"}])
        dispatch.checkpoint_id = "dispatch"
        dispatch.timestamp = datetime(2026, 10, 4, 11, tzinfo=timezone.utc)
        saver.put(dispatch)
        assert saver.get("thread").checkpoint_id == "dispatch"
        assert [c.checkpoint_id for c in saver.list("thread")] == ["dispatch", "checkpoint"]
