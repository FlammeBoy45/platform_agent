from collections.abc import Callable

from src.domain.entities.agent import AgentState
from src.domain.entities.message import Message


class ConversationManager:
    """Gère l'historique des messages et l'état de la session."""

    def __init__(
        self,
        max_messages: int = 10,
        on_message: Callable[[str, str], None] | None = None,
    ) -> None:
        self.state = AgentState()
        self.max_messages = max_messages
        self.on_message = on_message

    def add_user_message(self, content: str) -> None:
        self.state.messages.append(Message(role="user", content=content))
        self.trim_history()
        if self.on_message:
            self.on_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.state.messages.append(Message(role="assistant", content=content))
        self.trim_history()
        if self.on_message:
            self.on_message("assistant", content)

    def restore_messages(self, messages: list[dict[str, str]]) -> None:
        self.state.messages = [
            Message(role=item["role"], content=item["content"])
            for item in messages
            if item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
        ]
        self.trim_history()

    def get_messages_as_dicts(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.state.messages]

    def trim_history(self) -> None:
        """Évite l'explosion du contexte LLM."""
        self.state.messages = self.state.messages[-self.max_messages :]

    def reset(self) -> None:
        self.state = AgentState()
