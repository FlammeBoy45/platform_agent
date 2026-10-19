from datetime import UTC, datetime

from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    """Entité représentant un fait stocké en mémoire long terme."""
    key: str
    value: str
    text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))