from typing import Any

from src.domain.protocols.memory import LongTermMemory


class MemoryTool:
    name = "memory"
    description = (
        "Consulte les souvenirs persistants de l'utilisateur. "
        "Actions disponibles: search, list."
    )
    args_schema = {
        "action": "Action: search ou list.",
        "query": "Texte à rechercher pour l'action search.",
    }
    optional_args = ("query",)
    return_direct = True
    trigger_words: tuple[str, ...] = (
        "souviens", "souvenir", "mémoire", "memoire", "memory", "retiens", "oublie",
    )

    def __init__(self, memory: LongTermMemory | None) -> None:
        self.memory = memory

    async def execute(self, **kwargs: Any) -> str:
        if self.memory is None:
            return "La mémoire persistante est désactivée."

        action = str(kwargs.get("action", "")).casefold().strip()
        query = str(kwargs.get("query", "")).strip()

        if action == "search":
            if not query:
                return "Précise ce que je dois rechercher dans mes souvenirs."
            facts = self.memory.search(query, limit=8)
            if not facts:
                return "Je n'ai trouvé aucun souvenir correspondant."
            return "Souvenirs retrouvés :\n" + "\n".join(
                f"- {fact.text}" for fact in facts
            )

        if action == "list":
            facts = self.memory.all()[-20:]
            if not facts:
                return "Je n'ai encore rien mémorisé."
            return "Souvenirs récents :\n" + "\n".join(
                f"- {fact.text}" for fact in reversed(facts)
            )

        if action in {"remember", "forget", "effacer", "supprimer", "write"}:
            return (
                "Les souvenirs ne peuvent pas être modifiés par un appel du modèle. "
                "L'utilisateur doit demander directement l'enregistrement ou l'oubli."
            )

        return (
            "Action mémoire invalide. Actions disponibles : search, list."
        )
