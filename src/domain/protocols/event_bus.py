from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class EventBus(Protocol):
    """Protocole pour le bus d'événements de l'agent."""

    def subscribe(
        self, event_type: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """S'abonne à un type d'événement."""
        ...

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Émet un événement vers tous les abonnés."""
        ...
