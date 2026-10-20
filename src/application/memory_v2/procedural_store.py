# procedural_store.py

from datetime import UTC, datetime

from .models import MemoryItem, new_id


class ProceduralMemory:
    def __init__(self) -> None:
        self.store: list[MemoryItem] = []

    def add_pattern(self, task: str, steps: list[str], success: bool) -> None:
        self.store.append(
            MemoryItem(
                id=new_id(),
                content=task,
                metadata={"steps": steps, "success": success},
                timestamp=datetime.now(UTC),
            )
        )

    def all(self) -> list[MemoryItem]:
        return self.store

    def best_strategy(self, task: str) -> list[MemoryItem]:
        candidates = [m for m in self.store if task in m.content]
        return sorted(
            candidates, key=lambda x: x.metadata.get("success", 0), reverse=True
        )

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """Recherche les patterns par mot-clé, triés par succès."""
        q = query.lower()
        scored: list[tuple[float, MemoryItem]] = []

        for item in self.store:
            content = item.content.lower()
            steps_text = " ".join(item.metadata.get("steps", [])).lower()

            score = 0.0
            if q in content or q in steps_text:
                score = 1.0
            else:
                query_terms = set(q.split())
                all_terms = set(content.split()) | set(steps_text.split())
                overlap = query_terms & all_terms
                if overlap:
                    score = len(overlap) / len(query_terms)

            if score > 0:
                success_bonus = 0.3 if item.metadata.get("success") else 0.0
                scored.append((score + success_bonus, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def clear(self) -> None:
        self.store.clear()
