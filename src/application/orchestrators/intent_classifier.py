import re
from typing import Any


def is_plan_resume_request(user_input: str) -> bool:
    normalized = re.sub(r"[^\w\s]", " ", user_input.casefold())
    normalized = " ".join(normalized.split())
    resume_phrases = (
        "continue",
        "continuer",
        "continue la tâche",
        "continue la tache",
        "poursuis",
        "poursuivre",
        "reprends",
        "reprendre",
        "reprends la tâche",
        "reprends la tache",
        "termine",
        "termine la tâche",
        "termine la tache",
        "fais la suite",
        "fais la suite du plan",
        "où en étais tu",
        "ou en etais tu",
    )
    return any(
        normalized == phrase or normalized.startswith(f"{phrase} ")
        for phrase in resume_phrases
    )


def has_action_intent(user_input: str) -> bool:
    action_keywords = (
        "créer",
        "crée",
        "cree",
        "creer",
        "fais",
        "faire",
        "prépare",
        "préparer",
        "prepare",
        "preparer",
        "ponds",
        "implémente",
        "implemente",
        "supprimer",
        "supprime",
        "effacer",
        "efface",
        "modifier",
        "modifie",
        "ajouter",
        "ajoute",
        "écrire",
        "ecrire",
        "lis",
        "lire",
        "lit",
        "déplacer",
        "copier",
        "calcule",
        "cherche",
        "recherche",
        "google",
        "internet",
        "web",
        "prix",
        "cours",
        "valeur",
        "script",
        "code",
        "python",
        "programme",
        "fonction",
        "algorithme",
        "heure",
        "date",
        "aujourd'hui",
        "demain",
    )
    normalized = user_input.lower()
    return any(keyword in normalized for keyword in action_keywords)


def get_forced_tool_for_intent(user_input: str) -> tuple[str, dict[str, Any]]:
    normalized = user_input.lower()

    file_targets = (
        "fichier",
        "file",
        "module",
        "modou",
        "dossier",
        "folder",
        "répertoire",
        "repertoire",
    )
    file_actions = (
        "crée",
        "créer",
        "cree",
        "creer",
        "fais",
        "faire",
        "prépare",
        "préparer",
        "prepare",
        "preparer",
        "ajoute",
        "ajouter",
        "écris",
        "ecris",
        "écrire",
        "ecrire",
        "lis",
        "lire",
        "ouvre",
        "ouvrir",
        "modifie",
        "modifier",
        "supprime",
        "supprimer",
        "efface",
        "effacer",
    )
    if any(word in normalized for word in file_targets) and any(
        word in normalized for word in file_actions
    ):
        return ("file_crud", {"action": "", "path": "", "content": ""})
    if re.search(r"\.[a-z0-9]{1,8}\b", normalized) and any(
        word in normalized for word in file_actions
    ):
        return ("file_crud", {"action": "", "path": "", "content": ""})

    calc_keywords = (
        "calcule",
        "calcul",
        "combien fait",
        "additionne",
        "multiplie",
        "divise",
        "soustrait",
        "pourcentage",
    )
    if any(keyword in normalized for keyword in calc_keywords):
        return ("calculator", {"expression": user_input})

    crypto_keywords = (
        "bitcoin",
        "ethereum",
        "crypto",
        "prix",
        "cours",
        "btc",
        "eth",
        "valeur",
        "acheter",
        "vendre",
    )
    if any(keyword in normalized for keyword in crypto_keywords):
        return ("crypto_price", {"symbol": "BTC"})

    time_keywords = (
        "heure",
        "date",
        "aujourd'hui",
        "demain",
        "quel jour",
        "quel jour de",
        "quel mois",
        "quelle année",
    )
    if any(keyword in normalized for keyword in time_keywords) or re.search(
        r"\b(?:time|day|month|year)\b", normalized
    ):
        offset = re.search(r"\butc\s*([+-]\s*\d{1,2}(?::?\d{2})?)", normalized)
        timezone = f"UTC{offset.group(1).replace(' ', '')}" if offset else "UTC"
        return ("datetime", {"timezone": timezone})

    search_keywords = ("cherche", "recherche", "google", "internet", "web")
    if any(keyword in normalized for keyword in search_keywords):
        if any(
            phrase in normalized
            for phrase in (
                "dans le code",
                "dans les fichiers",
                "dans le projet",
                "où est défini",
                "ou est defini",
            )
        ):
            return ("workspace_search", {"query": user_input, "path": "."})
        return ("google_search", {"query": user_input})

    code_keywords = (
        "script",
        "code",
        "python",
        "programme",
        "fonction",
        "algorithme",
    )
    if any(keyword in normalized for keyword in code_keywords):
        return ("python_code", {"code": user_input})

    return ("final", {"content": f"Action demandée: {user_input}"})
