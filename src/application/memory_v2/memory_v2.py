# memory_v2.py

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from src.domain.entities.memory import MemoryFact
from src.infrastructure.memory.chroma_memory_store import ChromaMemoryStore

from .episodic_store import EpisodicMemory
from .models import MemoryItem, new_id
from .procedural_store import ProceduralMemory
from .retriever import MemoryRetriever
from .semantic_store import SemanticMemory


class MemoryV2:
    def __init__(self, persist_path: str = "./data/chroma_db"):
        # Stores persistants
        self.episodic_store = ChromaMemoryStore(
            path=persist_path, collection_name="episodic_memory"
        )
        self.semantic_store = ChromaMemoryStore(
            path=persist_path, collection_name="semantic_memory"
        )
        self.procedural_store = ChromaMemoryStore(
            path=persist_path, collection_name="procedural_memory"
        )

        # Stores in-memory pour le cache rapide
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.retriever = MemoryRetriever()

    # =========================
    # WRITE API
    # =========================

    def store_episode(self, user_input: str, output: Any = None) -> None:
        if not isinstance(output, str):
            try:
                output = json.dumps(output, ensure_ascii=False)
            except Exception:
                output = str(output)

        self.episodic_store.remember(
            key=str(uuid.uuid4()), value=output, text=f"Q:{user_input} | A:{output}"
        )
        self.episodic.add(user_input, output)

    def store_fact(self, fact: str, importance: float = 0.5) -> None:
        self.semantic_store.remember(key=fact, value=str(importance), text=fact)
        self.semantic.add_fact(fact, importance)

    def store_skill(self, task: str, steps: list[str], success: bool) -> None:
        self.procedural_store.remember(
            key=str(uuid.uuid4()),
            value=json.dumps({"steps": steps, "success": success}),
            text=f"Tâche: {task} | Étapes: {', '.join(steps)}",
        )
        self.procedural.add_pattern(task, steps, success)

    # =========================
    # READ API (RAG CORE)
    # =========================

    def retrieve(self, query: str) -> list[MemoryItem]:
        episodic_results = self.episodic_store.search(query, limit=10)
        semantic_results = self.semantic_store.search(query, limit=10)
        procedural_results = self.procedural_store.search(query, limit=10)

        episodic_items = self._facts_to_items(episodic_results)
        semantic_items = self._facts_to_items(semantic_results)
        procedural_items = self._facts_to_items(procedural_results)

        if not episodic_items:
            episodic_items = self.episodic.recent(20)
        if not semantic_items:
            semantic_items = self.semantic.all()
        if not procedural_items:
            procedural_items = self.procedural.all()

        pool = episodic_items + semantic_items + procedural_items
        return self.retriever.retrieve(pool, query)

    # =========================
    # HELPERS
    # =========================

    def retrieve_skills(self, query: str) -> list[MemoryItem]:
        chroma_results = self.procedural_store.search(query, limit=10)
        if chroma_results:
            return self._facts_to_items(chroma_results)
        return self.procedural.search(query)

    def retrieve_facts(self, query: str) -> list[MemoryItem]:
        chroma_results = self.semantic_store.search(query, limit=10)
        if chroma_results:
            return self._facts_to_items(chroma_results)
        return self.semantic.search(query)

    def clear(self) -> None:
        self.episodic.clear()
        self.semantic.clear()
        self.procedural.clear()

    def _facts_to_items(
        self, facts: Sequence[MemoryFact | MemoryItem]
    ) -> list[MemoryItem]:
        """Convertit une liste de MemoryFact en MemoryItem."""
        items: list[MemoryItem] = []
        for f in facts:
            if isinstance(f, MemoryItem):
                items.append(f)
            else:
                items.append(
                    MemoryItem(
                        id=new_id(),
                        content=getattr(f, "text", str(f)),
                        metadata={
                            "key": getattr(f, "key", ""),
                            "value": getattr(f, "value", ""),
                            "source": "chromadb",  # ✅ Marqueur pour le bonus vectoriel
                        },
                        timestamp=datetime.now(UTC),
                    )
                )
        return items
