import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlSessionStore:
    """Journal local append-only pour reprendre les échanges après redémarrage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.session_id = self._find_latest_session() or str(uuid.uuid4())

    def load_messages(self, limit: int = 20) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for event in self._read_events():
            if event.get("session_id") != self.session_id:
                continue
            if event.get("type") not in {"user_message", "assistant_message"}:
                continue
            role = "user" if event["type"] == "user_message" else "assistant"
            content = event.get("data", {}).get("content")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
        return messages[-max(0, limit) :] if limit else []

    def append_event(self, event_type: str, data: dict[str, Any]) -> None:
        event = {
            "session_id": self.session_id,
            "type": event_type,
            "at": datetime.now(UTC).isoformat(),
            "data": data,
        }
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def new_session(self) -> str:
        self.session_id = str(uuid.uuid4())
        self.append_event("session_started", {})
        return self.session_id

    def load_unfinished_plan(self) -> dict[str, Any] | None:
        events = [
            event
            for event in self._read_events()
            if event.get("session_id") == self.session_id
        ]
        plan_index = next(
            (
                i
                for i in range(len(events) - 1, -1, -1)
                if events[i].get("type") == "plan_snapshot"
            ),
            None,
        )
        if plan_index is None:
            return None
        last_finished = max(
            (
                i
                for i, event in enumerate(events)
                if event.get("type") == "turn_finished"
            ),
            default=-1,
        )
        plan = events[plan_index].get("data", {}).get("plan")
        if not isinstance(plan, dict):
            return None
        finished = events[last_finished] if last_finished >= 0 else None
        if (
            plan_index < last_finished
            and finished
            and finished.get("data", {}).get("status") == "completed"
        ):
            return None
        tasks = plan.get("tasks", [])
        if not isinstance(tasks, list) or not any(
            isinstance(task, dict)
            and (
                task.get("status") in {"pending", "in_progress"}
                or (
                    task.get("status") == "failed"
                    and int(task.get("attempts", 0)) < int(task.get("max_attempts", 3))
                )
            )
            for task in tasks
        ):
            return None
        return plan

    def _find_latest_session(self) -> str | None:
        latest = None
        for event in self._read_events():
            session_id = event.get("session_id")
            if isinstance(session_id, str):
                latest = session_id
        return latest

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("événement non objet")
                events.append(event)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"Journal de sessions illisible ({self.path}); "
                "aucun événement n'a été supprimé."
            ) from exc
        return events
