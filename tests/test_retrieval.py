"""Retrieval: vectorizer properties and nearest-neighbour ordering."""

from __future__ import annotations

import numpy as np

from spend_classifier.config import ClassifierConfig
from spend_classifier.retrieval import build_retriever
from spend_classifier.retrieval.embeddings import HashingTfidfVectorizer


def test_vectors_are_l2_normalised():
    vec = HashingTfidfVectorizer(dim=512)
    mat = vec.fit_transform(["gate valve carbon steel", "ball valve stainless steel"])
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_empty_text_yields_zero_vector():
    vec = HashingTfidfVectorizer(dim=128)
    vec.fit(["something"])
    out = vec.transform([""])
    assert np.linalg.norm(out[0]) == 0.0


def test_similar_text_scores_higher_than_dissimilar():
    vec = HashingTfidfVectorizer(dim=2048)
    corpus = ["gate valve carbon steel 2in", "power cable xlpe 25sqmm"]
    mat = vec.fit_transform(corpus)
    q = vec.transform(["valve gate steel"])[0]
    sims = mat @ q
    assert sims[0] > sims[1]  # the valve doc wins


def test_retriever_topk_ordering_and_count(corpus):
    cfg = ClassifierConfig(top_k=3, embedding_dim=2048)
    retriever = build_retriever(cfg, corpus)
    cands = retriever.retrieve("gate valve carbon steel 2in 150lb", top_k=3)
    assert len(cands) == 3
    # similarities must be sorted descending
    sims = [c.similarity for c in cands]
    assert sims == sorted(sims, reverse=True)
    # the closest item should be a gate valve
    assert cands[0].code.startswith("40141607")


def test_offcatalogue_query_has_low_similarity(corpus):
    cfg = ClassifierConfig(top_k=3, embedding_dim=2048)
    retriever = build_retriever(cfg, corpus)
    cands = retriever.retrieve("office chair ergonomic mesh", top_k=3)
    assert max(c.similarity for c in cands) < 0.1
