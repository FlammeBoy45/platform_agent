from typing import Any

from pydantic import BaseModel, Field

from src.domain.entities.plan import Plan


class ToolCall(BaseModel):
    """Structure stricte d'un appel d'outil."""

    tool: str = Field(..., description="Nom de l'outil à appeler")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Arguments de l'outil"
    )
    plan: Plan | None = Field(
        None, description="Mise à jour optionnelle du plan d'action"
    )


class FinalResponse(BaseModel):
    """Structure d'une réponse finale à l'utilisateur."""

    content: str = Field(..., description="Message final adressé à l'utilisateur")
