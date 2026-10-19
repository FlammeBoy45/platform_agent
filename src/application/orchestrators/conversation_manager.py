from src.domain.entities.agent import AgentState
from src.domain.entities.message import Message


class ConversationManager:
    """Gère l'historique des messages et l'état de la session."""
    
    def __init__(self, max_messages: int = 10) -> None:
        self.state = AgentState()
        self.max_messages = max_messages

    def add_user_message(self, content: str) -> None:
        self.state.messages.append(Message(role="user", content=content))
        self.trim_history()

    def add_assistant_message(self, content: str) -> None:
        self.state.messages.append(Message(role="assistant", content=content))
        self.trim_history()

    def get_messages_as_dicts(self) -> list[dict[str, str]]:
        return [
            {"role": m.role, "content": m.content}
            for m in self.state.messages
        ]

    def trim_history(self) -> None:
        """Évite l'explosion du contexte LLM."""
        self.state.messages = self.state.messages[-self.max_messages:]

    def reset(self) -> None:
        self.state = AgentState()
