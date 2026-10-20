from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
