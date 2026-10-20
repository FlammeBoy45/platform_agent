from dataclasses import dataclass, field

from src.domain.entities.message import Message


@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
