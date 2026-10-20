import json
from pathlib import Path
from typing import Any

import pytest

from src.infrastructure.sessions.jsonl_session_store import JsonlSessionStore


def _save_plan(
    store: JsonlSessionStore,
    status: str,
    *,
    attempts: int = 0,
    max_attempts: int = 3,
) -> dict[str, Any]:
    plan = {
        "goal": "Terminer la tâche",
        "tasks": [
            {
                "description": "Étape en attente",
                "status": status,
                "attempts": attempts,
                "max_attempts": max_attempts,
            }
        ],
    }
    store.append_event("plan_snapshot", {"plan": plan})
    return plan


def test_session_store_restores_recent_messages_and_respects_limit(
    tmp_path: Path,
) -> None:
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    store.append_event("user_message", {"content": "ancienne question"})
    store.append_event("assistant_message", {"content": "ancienne réponse"})
    store.append_event("tool_started", {"tool": "echo"})
    store.append_event("user_message", {"content": "question récente"})

    restarted = JsonlSessionStore(store.path)

    assert restarted.load_messages(limit=2) == [
        {"role": "assistant", "content": "ancienne réponse"},
        {"role": "user", "content": "question récente"},
    ]
    assert restarted.load_messages(limit=0) == []
    assert restarted.load_messages(limit=-1) == []


def test_new_session_does_not_restore_previous_session_messages(
    tmp_path: Path,
) -> None:
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    store.append_event("user_message", {"content": "message privé"})
    previous_session_id = store.session_id

    new_session_id = store.new_session()
    restarted = JsonlSessionStore(store.path)

    assert new_session_id != previous_session_id
    assert restarted.session_id == new_session_id
    assert restarted.load_messages() == []


@pytest.mark.parametrize(
    ("status", "attempts", "expected"),
    [
        ("pending", 0, True),
        ("in_progress", 0, True),
        ("failed", 2, True),
        ("failed", 3, False),
        ("completed", 0, False),
    ],
)
def test_session_store_restores_only_resumable_plans(
    tmp_path: Path,
    status: str,
    attempts: int,
    expected: bool,
) -> None:
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    plan = _save_plan(store, status, attempts=attempts)

    assert (store.load_unfinished_plan() == plan) is expected


def test_completed_turn_does_not_restore_older_unfinished_snapshot(
    tmp_path: Path,
) -> None:
    store = JsonlSessionStore(tmp_path / "sessions.jsonl")
    _save_plan(store, "pending")
    store.append_event("turn_finished", {"status": "completed"})

    assert store.load_unfinished_plan() is None


def test_session_store_reports_corrupt_journal_without_silently_resetting(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "sessions.jsonl"
    journal.write_text('{"session_id":"valid"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="aucun événement n'a été supprimé"):
        JsonlSessionStore(journal)


def test_session_store_ignores_blank_lines_and_rejects_non_object_events(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "sessions.jsonl"
    journal.write_text(
        '\n{"session_id":"valid","type":"session_started","data":{}}\n',
        encoding="utf-8",
    )
    store = JsonlSessionStore(journal)
    assert store.session_id == "valid"

    journal.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="aucun événement n'a été supprimé"):
        JsonlSessionStore(journal)
