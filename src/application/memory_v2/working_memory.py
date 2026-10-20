# working_memory.py

from datetime import UTC, datetime, timedelta
from typing import Any


class WorkingMemory:
    """
    Mémoire de travail volatile pour le contexte immédiat d'une tâche.
    Stocke les résultats intermédiaires pendant l'exécution d'un plan.
    Vidée à chaque nouvelle session.
    """

    def __init__(self, ttl_seconds: int = 300):
        self._store: dict[str, Any] = {}
        self._timestamps: dict[str, datetime] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    def put(self, key: str, value: Any) -> None:
        """Stocke une valeur avec TTL."""
        self._store[key] = value
        self._timestamps[key] = datetime.now(UTC)

    def get(self, key: str) -> Any | None:
        """Récupère une valeur si elle n'a pas expiré."""
        if key not in self._store:
            return None

        age = datetime.now(UTC) - self._timestamps[key]
        if age > self._ttl:
            del self._store[key]
            del self._timestamps[key]
            return None

        return self._store[key]

    def get_all(self) -> dict[str, Any]:
        """Récupère toutes les valeurs non expirées."""
        self._cleanup()
        return dict(self._store)

    def get_context_for_prompt(self) -> str:
        """Formatte le contenu pour injection dans le prompt."""
        self._cleanup()
        if not self._store:
            return ""

        lines = ["\nMÉMOIRE DE TRAVAIL (contexte immédiat):"]
        for key, value in self._store.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Vide complètement la mémoire de travail."""
        self._store.clear()
        self._timestamps.clear()

    def _cleanup(self) -> None:
        """Supprime les entrées expirées."""
        now = datetime.now(UTC)
        expired = [key for key, ts in self._timestamps.items() if now - ts > self._ttl]
        for key in expired:
            del self._store[key]
            del self._timestamps[key]
