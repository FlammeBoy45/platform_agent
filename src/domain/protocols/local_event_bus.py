from collections.abc import Awaitable, Callable
from typing import Any

from src.domain.protocols.event_bus import EventBus


class LocalEventBus(EventBus):
    """Implémentation locale simple du bus d'événements."""

    def __init__(self) -> None:
        self._handlers: dict[
            str, list[Callable[[dict[str, Any]], Awaitable[None]]]
        ] = {}

    def subscribe(
        self, event_type: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                await handler(data)
