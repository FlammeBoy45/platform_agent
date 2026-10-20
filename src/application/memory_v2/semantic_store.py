# semantic_store.py

from datetime import UTC, datetime

from .models import MemoryItem, new_id


class SemanticMemory:
    def __init__(self) -> None:
        self.store: list[MemoryItem] = []

    def add_fact(self, content: str, importance: float = 0.5) -> None:
        self.store.append(
            MemoryItem(
                id=new_id(),
                content=content,
                metadata={"type": "fact"},
                timestamp=datetime.now(UTC),
                importance=importance,
            )
        )

    def all(self) -> list[MemoryItem]:
        return self.store

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """Recherche par mot-clé dans les faits stockés."""
        q = query.lower()
        scored: list[tuple[float, MemoryItem]] = []

        for item in self.store:
            score = 0.0
            if q in item.content.lower():
                score = 1.0
            else:
                # matching partiel par mots
                query_terms = set(q.split())
                content_terms = set(item.content.lower().split())
                overlap = query_terms & content_terms
                if overlap:
                    score = len(overlap) / len(query_terms)

            if score > 0:
                scored.append((score * item.importance, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def clear(self) -> None:
        self.store.clear()
