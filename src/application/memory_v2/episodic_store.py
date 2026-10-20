# episodic_store.py
from datetime import UTC, datetime
from typing import Any

from .models import MemoryItem, new_id


class EpisodicMemory:
    def __init__(self) -> None:
        self.store: list[MemoryItem] = []

    def add(
        self,
        input: str,
        output: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store.append(
            MemoryItem(
                id=new_id(),
                content=f"Q:{input} | A:{output}",
                metadata=metadata or {},
                timestamp=datetime.now(UTC),
            )
        )

    def recent(self, limit: int = 10) -> list[MemoryItem]:
        return self.store[-limit:]

    def clear(self) -> None:
        self.store.clear()
