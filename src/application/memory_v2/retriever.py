# retriever.py

from .models import MemoryItem


class MemoryRetriever:
    def score(self, item: MemoryItem, query: str) -> float:
        """
        Score hybride combinant :
        - similarité textuelle (mots-clés)
        - recency (favorise les souvenirs récents)
        - importance (faits importants)
        - bonus ChromaDB (items venant de la recherche vectorielle)
        """
        q = query.lower()
        c = item.content.lower()

        # 1. Similarité par mots
        query_terms = set(q.split())
        content_terms = set(c.split())

        if q in c:
            similarity = 1.0
        elif query_terms & content_terms:
            overlap = len(query_terms & content_terms)
            similarity = 0.3 + (0.5 * overlap / len(query_terms))
        else:
            similarity = 0.05  # score minimal

        # 2. Recency (les souvenirs récents sont plus pertinents)
        recency = 1.0 / (1.0 + (item.access_count * 0.05))

        # 3. Importance du fait
        importance = max(0.1, item.importance)

        # ChromaDB results have already passed a semantic filter.
        is_from_chroma = item.metadata.get("source") == "chromadb"
        chroma_bonus = 0.2 if is_from_chroma else 0.0

        # Score final pondéré
        return (
            0.45 * similarity + 0.25 * recency + 0.15 * importance + 0.15 * chroma_bonus
        )

    def retrieve(
        self, items: list[MemoryItem], query: str, top_k: int = 5
    ) -> list[MemoryItem]:
        """Récupère les meilleurs souvenirs avec score hybride."""
        if not items:
            return []

        ranked = sorted(items, key=lambda x: self.score(x, query), reverse=True)

        for item in ranked:
            item.access_count += 1

        return ranked[:top_k]

    def retrieve_with_diversity(
        self, items: list[MemoryItem], query: str, top_k: int = 5
    ) -> list[MemoryItem]:
        """
        Récupération avec diversité : évite les doublons sémantiques
        en pénalisant les items trop similaires entre eux.
        """
        if not items or len(items) <= top_k:
            return self.retrieve(items, query, top_k)

        ranked = sorted(items, key=lambda x: self.score(x, query), reverse=True)

        diverse: list[MemoryItem] = [ranked[0]]

        for item in ranked[1:]:
            if len(diverse) >= top_k:
                break

            # Vérifier si cet item est trop similaire à ceux déjà sélectionnés
            is_diverse = True
            item_terms = set(item.content.lower().split())

            for selected in diverse:
                selected_terms = set(selected.content.lower().split())
                overlap = len(item_terms & selected_terms) / max(len(item_terms), 1)
                if overlap > 0.7:  # 70% de similarité → doublon
                    is_diverse = False
                    break

            if is_diverse:
                diverse.append(item)
                item.access_count += 1

        return diverse
