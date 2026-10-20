from typing import Any


class TextStatsTool:
    name = "text_stats"
    description = (
        "Compte les caracteres, mots et lignes d'un texte fourni par l'utilisateur."
    )
    args_schema = {
        "text": "Texte a analyser.",
    }
    return_direct = True
    trigger_words: tuple[str, ...] = (
        "compte",
        "nombre de mots",
        "nombre de caracteres",
        "nombre de caractères",
        "statistiques",
        "longueur",
    )

    def infer_args(self, user_input: str, args: dict[str, Any]) -> dict[str, Any]:
        """Extrait le texte à analyser si absent des arguments."""
        if args.get("text"):
            return args

        text = user_input
        for separator in (":", "->"):
            if separator in text:
                text = text.split(separator, 1)[1]
                break

        cleaned = text.strip()
        prefixes = ("compte", "statistiques", "longueur")
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        return {**args, "text": cleaned}

    async def execute(self, **kwargs: Any) -> str:
        text = str(kwargs.get("text", ""))
        words = text.split()
        lines = text.splitlines() or ([text] if text else [])

        return f"Caracteres: {len(text)}; Mots: {len(words)}; Lignes: {len(lines)}"
