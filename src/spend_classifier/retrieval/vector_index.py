"""Nearest-neighbour retrieval over the coded-item corpus."""

from __future__ import annotations

import numpy as np

from ..schemas import Candidate, MasterItem
from .embeddings import TextVectorizer


class VectorIndex:
    """Flat cosine index. Vectors are assumed L2-normalised, so a dot product
    is the cosine similarity. For a few hundred-thousand items this is fine; a
    production system would swap in an ANN index behind the same call."""

    def __init__(self, vectors: np.ndarray, items: list[MasterItem]) -> None:
        if len(items) != vectors.shape[0]:
            raise ValueError("vectors and items length mismatch")
        self.vectors = vectors
        self.items = items

    def query(self, qvec: np.ndarray, top_k: int) -> list[tuple[MasterItem, float]]:
        if self.vectors.shape[0] == 0:
            return []
        sims = self.vectors @ qvec
        k = min(top_k, sims.shape[0])
        # argpartition for top-k, then sort that slice by score (descending)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.items[j], float(sims[j])) for j in top]


class Retriever:
    """Builds the index from the coded-item corpus and answers queries."""

    def __init__(self, vectorizer: TextVectorizer, master_items: list[MasterItem]) -> None:
        self.vectorizer = vectorizer
        self.master_items = master_items
        texts = [m.embedding_text() for m in master_items]
        matrix = vectorizer.fit_transform(texts)
        self.index = VectorIndex(matrix, master_items)

    def retrieve(self, query_text: str, top_k: int) -> list[Candidate]:
        qvec = self.vectorizer.transform([query_text])[0]
        hits = self.index.query(qvec, top_k)
        return [
            Candidate(
                item_id=item.item_id,
                description=item.description,
                code=item.code,
                similarity=sim,
            )
            for item, sim in hits
        ]
