import hashlib
import json
import os
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.domain.entities.memory import MemoryFact


class JsonMemoryStore:
    MAX_ENTRIES = 2_000
    MAX_EPISODES = 500
    MAX_SKILLS = 200
    MAX_TEXT_CHARS = 8_000
    FACT_RETENTION = timedelta(days=365)
    EPISODE_RETENTION = timedelta(days=30)

    def __init__(self, path: str | Path, max_entries: int = MAX_ENTRIES) -> None:
        self.path = Path(path)
        self.max_entries = max(1, max_entries)
        self._facts: dict[str, MemoryFact] = {}
        self._load()

    def remember(self, key: str, value: str, text: str) -> None:
        self._purge_expired()
        fact = MemoryFact(
            key=key[:200],
            value=value[: self.MAX_TEXT_CHARS],
            text=text[: self.MAX_TEXT_CHARS],
        )
        self._facts[fact.key] = fact
        self._trim()
        self._save()

    def get(self, key: str) -> MemoryFact | None:
        self._purge_expired()
        return self._facts.get(key)

    def search(self, query: str, limit: int = 5) -> list[MemoryFact]:
        self._purge_expired()
        query_terms = self._terms(query)
        if not query_terms or limit <= 0:
            return []

        scored: list[tuple[float, MemoryFact]] = []

        for fact in self._facts.values():
            fact_terms = self._terms(f"{fact.key} {fact.value} {fact.text}")
            overlap = query_terms & fact_terms
            score = len(overlap) / len(query_terms)
            if query_terms <= fact_terms:
                score += 0.5
            if score:
                scored.append((score, fact))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [fact for _, fact in scored[:limit]]

    def all(self) -> list[MemoryFact]:
        self._purge_expired()
        return list(self._facts.values())

    def forget(self, query: str | None = None) -> int:
        self._purge_expired()
        if query is None or not query.strip():
            removed = len(self._facts)
            self._facts.clear()
        else:
            terms = self._terms(query)
            if not terms:
                return 0
            matching_keys = [
                key
                for key, fact in self._facts.items()
                if terms <= self._terms(f"{fact.key} {fact.value} {fact.text}")
            ]
            for key in matching_keys:
                del self._facts[key]
            removed = len(matching_keys)
        if removed:
            self._save()
        return removed

    def list_recent(self, limit: int = 20) -> list[MemoryFact]:
        self._purge_expired()
        if limit <= 0:
            return []
        return list(reversed(list(self._facts.values())[-limit:]))

    def store_episode(self, user_input: str, output: Any = None) -> None:
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False, default=str)
        try:
            output_data = json.loads(output)
        except json.JSONDecodeError:
            output_data = output
        if isinstance(output_data, dict):
            response = (
                output_data.get("output")
                or output_data.get("result")
                or output_data.get("content")
                or output
            )
        else:
            response = output_data
        self.remember(
            key=f"episode.{uuid.uuid4()}",
            value=output,
            text=f"Utilisateur: {user_input}\nAgent: {response}",
        )

    def store_fact(self, fact: str, importance: float = 0.5) -> None:
        key = hashlib.sha256(fact.casefold().encode("utf-8")).hexdigest()
        self.remember(
            key=f"fact.{key}",
            value=str(importance),
            text=fact,
        )

    def store_skill(
        self, task: str, steps: list[str], success: bool
    ) -> None:
        value = json.dumps(
            {"steps": steps, "success": success},
            ensure_ascii=False,
        )
        self.remember(
            key=f"skill.{uuid.uuid4()}",
            value=value,
            text=f"Tâche: {task} | Étapes: {', '.join(steps)}",
        )

    def retrieve(self, query: str) -> list[MemoryFact]:
        return self.search(query, limit=20)

    def retrieve_skills(self, query: str) -> list[MemoryFact]:
        matching = [
            fact for fact in self.search(query, limit=len(self._facts))
            if fact.key.startswith("skill.")
        ]
        return matching[:10]

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Never treat a damaged existing store as an empty one: the next
            # write would otherwise silently destroy the user's saved memory.
            raise RuntimeError(
                f"Impossible de charger la mémoire persistante {self.path}; "
                "le fichier a été conservé intact."
            ) from exc

        if not isinstance(raw, list):
            raise RuntimeError(
                f"Format de mémoire invalide dans {self.path}; "
                "le fichier a été conservé intact."
            )

        needs_timestamp_migration = False
        for item in raw:
            if not isinstance(item, dict):
                raise RuntimeError(
                    f"Entrée mémoire invalide dans {self.path}; "
                    "le fichier a été conservé intact."
                )

            try:
                if "created_at" not in item:
                    needs_timestamp_migration = True
                fact = MemoryFact(
                    key=str(item["key"]),
                    value=str(item["value"]),
                    text=str(item["text"]),
                    **(
                        {"created_at": item["created_at"]}
                        if "created_at" in item
                        else {}
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Entrée mémoire invalide dans {self.path}; "
                    "le fichier a été conservé intact."
                ) from exc

            self._facts[fact.key] = fact

        expired = self._purge_expired(save=False)
        original_count = len(self._facts)
        self._trim()
        if expired or len(self._facts) < original_count or needs_timestamp_migration:
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [fact.model_dump(mode="json") for fact in self._facts.values()]
        temporary_path = self.path.with_name(
            f"{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for attempt in range(5):
                try:
                    os.replace(temporary_path, self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        finally:
            temporary_path.unlink(missing_ok=True)

    def _trim(self) -> None:
        def remove_oldest(prefix: str, limit: int) -> None:
            keys = [key for key in self._facts if key.startswith(prefix)]
            for key in keys[:-limit] if limit else keys:
                self._facts.pop(key, None)

        remove_oldest("episode.", self.MAX_EPISODES)
        remove_oldest("skill.", self.MAX_SKILLS)
        while len(self._facts) > self.max_entries:
            oldest_key = next(
                (
                    key
                    for key in self._facts
                    if key.startswith(("episode.", "skill."))
                ),
                next(iter(self._facts)),
            )
            del self._facts[oldest_key]

    def _purge_expired(self, save: bool = True) -> int:
        now = datetime.now(UTC)
        expired_keys: list[str] = []
        for key, fact in self._facts.items():
            created_at = fact.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            retention = (
                self.EPISODE_RETENTION
                if key.startswith("episode.")
                else self.FACT_RETENTION
            )
            if now - created_at >= retention:
                expired_keys.append(key)

        for key in expired_keys:
            del self._facts[key]
        if expired_keys and save:
            self._save()
        return len(expired_keys)

    def _terms(self, text: str) -> set[str]:
        stop_words = {"est", "suis", "sont", "dans", "avec", "pour", "quoi", "quel", "quelle", "quels", "quelles", "mon", "ma", "mes", "ton", "ta", "tes", "le", "la", "les", "un", "une", "des", "du", "de", "et", "à", "au", "aux", "je", "tu", "il", "elle", "me", "te", "moi", "toi", "que", "qui", "comment"}
        return {term for term in re.findall(r"[\wÀ-ÿ]+", text.casefold()) if len(term) > 1 and term not in stop_words}
