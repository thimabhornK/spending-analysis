"""Retrieval package: builds the configured vectorizer + retriever."""

from __future__ import annotations

from ..config import ClassifierConfig
from ..schemas import MasterItem
from .embeddings import HashingTfidfVectorizer, OpenAIEmbeddings, TextVectorizer
from .vector_index import Retriever, VectorIndex


def build_vectorizer(config: ClassifierConfig) -> TextVectorizer:
    if config.embedding_backend == "openai":
        return OpenAIEmbeddings(config.openai_embedding_model)
    return HashingTfidfVectorizer(dim=config.embedding_dim)


def build_retriever(config: ClassifierConfig, master_items: list[MasterItem]) -> Retriever:
    return Retriever(build_vectorizer(config), master_items)


__all__ = [
    "build_vectorizer",
    "build_retriever",
    "Retriever",
    "VectorIndex",
    "TextVectorizer",
    "HashingTfidfVectorizer",
    "OpenAIEmbeddings",
]
