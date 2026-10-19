from typing import Any, Protocol, runtime_checkable

from src.domain.entities.memory import MemoryFact


class LongTermMemory(Protocol):
    def remember(self, key: str, value: str, text: str) -> None:
        ...

    def get(self, key: str) -> MemoryFact | None:
        ...

    def search(self, query: str, limit: int = 5) -> list[MemoryFact]:
        ...

    def all(self) -> list[MemoryFact]:
        ...

    def forget(self, query: str | None = None) -> int:
        """Supprime les souvenirs correspondants, ou tous si query vaut None."""
        ...


@runtime_checkable
class AgentMemory(Protocol):
    def store_episode(self, user_input: str, output: Any = None) -> None:
        ...

    def store_fact(self, fact: str, importance: float = 0.5) -> None:
        ...

    def store_skill(
        self, task: str, steps: list[str], success: bool
    ) -> None:
        ...

    def retrieve(self, query: str) -> list[MemoryFact]:
        ...

    def retrieve_skills(self, query: str) -> list[MemoryFact]:
        ...
