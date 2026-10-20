from typing import Any


class EchoTool:
    name = "echo"
    description = (
        "Retourne exactement le texte fourni. A utiliser seulement si "
        "l'utilisateur demande explicitement de repeter ou d'echo."
    )
    chat_safe = False
    args_schema = {
        "text": "Texte exact a retourner.",
    }
    return_direct = True
    trigger_words: tuple[str, ...] = (
        "echo",
        "repete",
        "répète",
        "retourne exactement",
        "redis exactement",
    )

    def infer_args(self, user_input: str, args: dict[str, Any]) -> dict[str, Any]:
        """Extrait le texte à répéter si absent des arguments."""
        if args.get("text"):
            return args

        text = user_input
        for separator in (":", "->"):
            if separator in text:
                text = text.split(separator, 1)[1]
                break

        cleaned = text.strip()
        prefixes = ("echo", "repete exactement", "répète exactement")
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        return {**args, "text": cleaned}

    async def execute(self, **kwargs: Any) -> str:
        return str(kwargs.get("text", ""))
