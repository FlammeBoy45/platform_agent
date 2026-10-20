# reflector.py

from src.domain.protocols.llm_provider import LLMProvider


class Reflector:
    """
    Évalue le résultat d'une action et décide si l'objectif est atteint
    ou s'il faut réessayer avec une approche différente.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def evaluate(
        self,
        goal: str,
        action_taken: str,
        result: str,
        success: bool,
    ) -> dict[str, str]:
        """
        Évalue le résultat et retourne :
        {
            "status": "ok" | "retry" | "replan" | "give_up",
            "analysis": "pourquoi ça a marché/échoué",
            "suggestion": "quoi faire différemment si retry/replan"
        }
        """
        if success and "ERREUR" not in result and "Échec" not in result:
            # Succès apparent → on vérifie quand même avec le LLM
            pass
        elif not success:
            # Échec → analyse obligatoire
            pass
        else:
            return {"status": "ok", "analysis": "Succès", "suggestion": ""}

        prompt = f"""
Tu es un évaluateur d'actions. Analyse le résultat et décide de la suite.

OBJECTIF : {goal}
ACTION TENTÉE : {action_taken}
RÉSULTAT : {result}
SUCCÈS TECHNIQUE : {success}

Choisis UNIQUEMENT parmi ces statuts :
- "ok" : l'objectif est atteint, on peut passer à la suite
- "retry" : l'action a échoué mais on peut réessayer différemment
- "replan" : l'approche entière est à revoir
- "give_up" : impossible, l'utilisateur doit intervenir

Retourne UNIQUEMENT ce JSON :
{{"status": "ok|retry|replan|give_up", "analysis": "ton analyse courte",
  "suggestion": "quoi faire si retry/replan"}}
"""

        try:
            raw = await self.llm.chat(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Goal: {goal}\nAction: {action_taken}\nResult: {result}"
                        ),
                    },
                ]
            )

            import json

            # Extraire le JSON
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`").strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            parsed = json.loads(text)

            return {
                "status": parsed.get("status", "ok"),
                "analysis": parsed.get("analysis", ""),
                "suggestion": parsed.get("suggestion", ""),
            }
        except Exception:
            # Fallback : si succès → ok, sinon retry
            return {
                "status": "ok" if success else "retry",
                "analysis": "Évaluation automatique",
                "suggestion": "Réessayer avec des paramètres différents"
                if not success
                else "",
            }
