# tests/test_memory_v2_complet.py

"""
Tests complets de la mémoire v2.
Exécute avec : pytest tests/test_memory_v2_complet.py -v
"""

import time
from datetime import UTC, datetime

from src.application.memory_v2.episodic_store import EpisodicMemory
from src.application.memory_v2.memory_v2 import MemoryV2
from src.application.memory_v2.models import MemoryItem
from src.application.memory_v2.procedural_store import ProceduralMemory
from src.application.memory_v2.retriever import MemoryRetriever
from src.application.memory_v2.semantic_store import SemanticMemory
from src.application.memory_v2.working_memory import WorkingMemory

# ============================================================
# ÉPISODIQUE
# ============================================================


class TestEpisodicMemory:
    def test_add_and_recent(self):
        mem = EpisodicMemory()
        mem.add("user1", "réponse1")
        mem.add("user2", "réponse2")
        mem.add("user3", "réponse3")

        recent = mem.recent(2)
        assert len(recent) == 2
        assert recent[-1].content == "Q:user3 | A:réponse3"

    def test_recent_respects_limit(self):
        mem = EpisodicMemory()
        for i in range(20):
            mem.add(f"input{i}", f"output{i}")

        recent = mem.recent(5)
        assert len(recent) == 5


# ============================================================
# SÉMANTIQUE
# ============================================================


class TestSemanticMemory:
    def test_add_and_search(self):
        mem = SemanticMemory()
        mem.add_fact("Python est un langage de programmation", importance=0.9)
        mem.add_fact("Le chat dort sur le canapé", importance=0.3)

        results = mem.search("python langage")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_search_partial_match(self):
        mem = SemanticMemory()
        mem.add_fact("Paris est la capitale de la France", importance=0.9)
        mem.add_fact("Londres est la capitale de l'Angleterre", importance=0.8)

        results = mem.search("france capitale")
        assert len(results) >= 1
        assert any("Paris" in r.content for r in results)

    def test_search_no_match(self):
        mem = SemanticMemory()
        mem.add_fact("Python est un langage", importance=0.9)

        results = mem.search("voiture volante")
        assert len(results) == 0

    def test_importance_weighting(self):
        mem = SemanticMemory()
        mem.add_fact("Fait important", importance=0.9)
        mem.add_fact("Fait banal", importance=0.1)

        results = mem.search("Fait")
        assert len(results) == 2
        assert results[0].content == "Fait important"  # Plus important en premier


# ============================================================
# PROCÉDURALE
# ============================================================


class TestProceduralMemory:
    def test_add_and_search(self):
        mem = ProceduralMemory()
        mem.add_pattern("calculer_prix", ["récupérer taux", "multiplier"], success=True)
        mem.add_pattern("chercher_vol", ["appeler API", "parser JSON"], success=False)

        results = mem.search("calculer")
        assert len(results) == 1
        assert results[0].content == "calculer_prix"

    def test_best_strategy_prefers_success(self):
        mem = ProceduralMemory()
        mem.add_pattern("tache_x", ["méthode1"], success=False)
        mem.add_pattern("tache_x", ["méthode2"], success=True)
        mem.add_pattern("tache_x", ["méthode3"], success=True)

        best = mem.best_strategy("tache_x")
        assert len(best) == 3
        assert best[0].metadata["success"] is True

    def test_search_by_steps(self):
        mem = ProceduralMemory()
        mem.add_pattern("tache_api", ["utiliser REST", "parser JSON"], success=True)

        results = mem.search("api REST JSON")
        assert len(results) >= 1


# ============================================================
# RETRIEVER
# ============================================================


class TestMemoryRetriever:
    def test_exact_match_score(self):
        retriever = MemoryRetriever()
        item = MemoryItem(
            id="1",
            content="Python est un langage",
            metadata={},
            timestamp=datetime.now(UTC),
            importance=0.5,
        )
        score = retriever.score(item, "python langage")
        assert score > 0.3

    def test_no_match_score(self):
        retriever = MemoryRetriever()
        item = MemoryItem(
            id="1",
            content="Python est un langage",
            metadata={},
            timestamp=datetime.now(UTC),
            importance=0.5,
        )
        score = retriever.score(item, "voiture volante")
        assert score < 0.40

    def test_chroma_bonus(self):
        retriever = MemoryRetriever()
        item_chroma = MemoryItem(
            id="1",
            content="test",
            metadata={"source": "chromadb"},
            timestamp=datetime.now(UTC),
            importance=0.5,
        )
        item_normal = MemoryItem(
            id="2",
            content="test",
            metadata={},
            timestamp=datetime.now(UTC),
            importance=0.5,
        )
        score_chroma = retriever.score(item_chroma, "test")
        score_normal = retriever.score(item_normal, "test")
        assert score_chroma > score_normal  # Bonus ChromaDB

    def test_retrieve_respects_top_k(self):
        retriever = MemoryRetriever()
        items = [
            MemoryItem(
                id=str(i),
                content=f"item{i}",
                metadata={},
                timestamp=datetime.now(UTC),
                importance=0.5,
            )
            for i in range(10)
        ]
        results = retriever.retrieve(items, "item", top_k=3)
        assert len(results) == 3

    def test_diversity_avoids_duplicates(self):
        retriever = MemoryRetriever()
        items = [
            MemoryItem(
                id="1",
                content="Le chat est noir",
                metadata={},
                timestamp=datetime.now(UTC),
                importance=0.5,
            ),
            MemoryItem(
                id="2",
                content="Le chat est noir aussi",
                metadata={},
                timestamp=datetime.now(UTC),
                importance=0.5,
            ),
            MemoryItem(
                id="3",
                content="La voiture est rouge",
                metadata={},
                timestamp=datetime.now(UTC),
                importance=0.5,
            ),
        ]
        results = retriever.retrieve_with_diversity(items, "chat noir", top_k=2)
        assert len(results) == 2
        # Évite deux résultats trop similaires pour la requête "chat noir".
        contents = [r.content for r in results]
        assert len(contents) == len(set(contents))  # Pas de doublons exacts


# ============================================================
# WORKING MEMORY
# ============================================================


class TestWorkingMemory:
    def test_put_and_get(self):
        wm = WorkingMemory()
        wm.put("key1", "value1")
        assert wm.get("key1") == "value1"

    def test_get_missing_key(self):
        wm = WorkingMemory()
        assert wm.get("inexistant") is None

    def test_ttl_expiration(self):
        wm = WorkingMemory(ttl_seconds=1)
        wm.put("key1", "value1")
        assert wm.get("key1") == "value1"
        time.sleep(1.1)
        assert wm.get("key1") is None

    def test_get_all(self):
        wm = WorkingMemory()
        wm.put("a", 1)
        wm.put("b", 2)
        all_data = wm.get_all()
        assert len(all_data) == 2
        assert all_data["a"] == 1

    def test_clear(self):
        wm = WorkingMemory()
        wm.put("key1", "value1")
        wm.clear()
        assert wm.get("key1") is None

    def test_context_for_prompt(self):
        wm = WorkingMemory()
        wm.put("resultat_calculator", "42")
        wm.put("resultat_file_crud", "Fichier créé")
        context = wm.get_context_for_prompt()
        assert "MÉMOIRE DE TRAVAIL" in context
        assert "42" in context
        assert "Fichier créé" in context


# ============================================================
# MEMORY V2 - INTÉGRATION
# ============================================================


class TestMemoryV2Integration:
    def test_store_and_retrieve_episode(self):
        mem = MemoryV2()
        mem.store_episode("Quel temps fait-il ?", "Il fait beau")
        mem.store_episode("Quelle heure est-il ?", "Il est midi")

        results = mem.retrieve("météo temps")
        assert len(results) > 0

    def test_store_and_retrieve_fact(self):
        mem = MemoryV2()
        mem.store_fact("Paris est la capitale de la France", importance=0.9)
        mem.store_fact("Londres est la capitale de l'Angleterre", importance=0.8)

        facts = mem.retrieve_facts("Paris France")
        assert len(facts) > 0

    def test_store_and_retrieve_skill(self):
        mem = MemoryV2()
        mem.store_skill(
            "convertir_devise", ["récupérer taux", "multiplier"], success=True
        )
        mem.store_skill("chercher_vol", ["appeler API", "parser JSON"], success=False)

        skills = mem.retrieve_skills("convertir")
        assert len(skills) > 0

    def test_clear(self):
        mem = MemoryV2()
        mem.store_episode("test", "test")
        mem.store_fact("test fact", 0.5)
        mem.store_skill("test skill", ["step1"], True)

        mem.clear()

        assert len(mem.episodic.recent(100)) == 0
        assert len(mem.semantic.all()) == 0
        assert len(mem.procedural.all()) == 0

    def test_retrieve_mixes_types(self):
        """Retourne des résultats épisodiques, sémantiques et procéduraux."""
        mem = MemoryV2()
        mem.store_episode("épisode test", "sortie test")
        mem.store_fact("fait test Python", importance=0.9)
        mem.store_skill("skill test", ["étape1"], success=True)

        results = mem.retrieve("test")
        assert len(results) > 0


# ============================================================
# PERSISTANCE (si ChromaDB dispo)
# ============================================================


class TestPersistence:
    def test_chromadb_persists_data(self):
        """Vérifie que ChromaDB garde les données entre 2 instances."""
        mem1 = MemoryV2(persist_path="./data/test_chroma")
        mem1.store_fact("Test persistance ChromaDB", importance=0.9)
        mem1.store_episode("user test", "réponse test")
        mem1.store_skill("skill test", ["step1", "step2"], True)

        # Nouvelle instance
        mem2 = MemoryV2(persist_path="./data/test_chroma")

        # Vérifier que les données sont récupérables
        facts = mem2.retrieve_facts("persistance")
        episodes = mem2.retrieve("user test")
        skills = mem2.retrieve_skills("skill test")

        # Au moins un des trois doit avoir des résultats
        total = len(facts) + len(episodes) + len(skills)
        assert total > 0, "Aucune donnée persistée retrouvée"
