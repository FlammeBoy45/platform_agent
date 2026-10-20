import chromadb

from src.domain.entities.memory import MemoryFact


class ChromaMemoryStore:
    """
    Implémentation d'une mémoire vectorielle utilisant ChromaDB.
    Permet une recherche sémantique des faits passés pour une meilleure pertinence.
    """

    def __init__(
        self, path: str = "./data/chroma_db", collection_name: str = "agent_memory"
    ):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def remember(self, key: str, value: str, text: str) -> None:
        """Persiste ou met à jour un fait avec indexation vectorielle."""
        self.collection.upsert(
            ids=[key], documents=[text], metadatas=[{"key": key, "value": value}]
        )

    def get(self, key: str) -> MemoryFact | None:
        """Récupère un fait spécifique par sa clé."""
        result = self.collection.get(ids=[key])
        if not result["ids"]:
            return None
        ids = result["ids"]
        metadatas = result["metadatas"]
        documents = result["documents"]
        if not ids or not metadatas or not documents:
            return None
        return MemoryFact(
            key=str(ids[0]), value=str(metadatas[0]["value"]), text=str(documents[0])
        )

    def search(self, query: str, limit: int = 5) -> list[MemoryFact]:
        """Effectue une recherche sémantique (similarité) sur les faits mémorisés."""
        results = self.collection.query(query_texts=[query], n_results=limit)

        facts: list[MemoryFact] = []
        if not results["ids"] or not results["ids"][0]:
            return facts

        ids_list = results["ids"][0]
        metadatas_list = results["metadatas"][0] if results["metadatas"] else []
        documents_list = results["documents"][0] if results["documents"] else []
        if not ids_list or not metadatas_list or not documents_list:
            return facts

        for i in range(len(ids_list)):
            facts.append(
                MemoryFact(
                    key=str(ids_list[i]),
                    value=str(metadatas_list[i]["value"]),
                    text=str(documents_list[i]),
                )
            )
        return facts

    def all(self) -> list[MemoryFact]:
        """Récupère tous les faits mémorisés (utile pour le contexte global)."""
        results = self.collection.get()
        facts: list[MemoryFact] = []
        ids = results["ids"]
        metadatas = results["metadatas"]
        documents = results["documents"]
        if not ids or not metadatas or not documents:
            return facts
        for i in range(len(ids)):
            facts.append(
                MemoryFact(
                    key=str(ids[i]),
                    value=str(metadatas[i]["value"]),
                    text=str(documents[i]),
                )
            )
        return facts
