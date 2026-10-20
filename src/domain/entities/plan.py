from uuid import uuid4

from pydantic import BaseModel, Field


class Task(BaseModel):
    """Une étape spécifique d'un plan complexe."""

    id: str = Field(default_factory=lambda: str(uuid4()))

    description: str = Field(..., description="Description de la tâche")

    status: str = Field("pending", pattern="^(pending|in_progress|completed|failed)$")

    result: str | None = None

    attempts: int = 0

    max_attempts: int = 3

    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """Un ensemble de tâches pour atteindre un objectif persistant."""

    goal: str = Field(..., description="Objectif final de l'agent")

    tasks: list[Task] = Field(default_factory=list)

    current_step: int = 0

    revision_count: int = 0

    max_revisions: int = 5
