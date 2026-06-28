"""Text vectorizers used for retrieval.

Two implementations sit behind one interface:

- ``HashingTfidfVectorizer`` — fully offline and deterministic. Bag-of-words
  TF-IDF hashed into a fixed-width vector. Two descriptions of the same item
  share tokens, so cosine similarity is high; no model or network needed. This
  is what makes the offline evaluation meaningful and reproducible.
- ``OpenAIEmbeddings`` — real dense embeddings, selected with
  ``SPEND_EMBEDDING_BACKEND=openai``. Same interface, so nothing else changes.

In production you would use dense embeddings; the offline TF-IDF backend keeps
the whole system runnable in CI and in a clone with no credentials.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


class TextVectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]) -> "TextVectorizer": ...

    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray: ...

    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        return self.fit(corpus).transform(corpus)


class HashingTfidfVectorizer(TextVectorizer):
    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self.idf: dict[str, float] = {}
        self.default_idf: float = 1.0
        self.n_docs: int = 0

    def _bucket(self, token: str) -> int:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        return h % self.dim

    def fit(self, corpus: list[str]) -> "HashingTfidfVectorizer":
        self.n_docs = len(corpus)
        df: dict[str, int] = {}
        for text in corpus:
            for tok in set(tokenize(text)):
                df[tok] = df.get(tok, 0) + 1
        # smoothed idf so a token seen in every doc still has a small positive weight
        self.idf = {t: math.log((self.n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.default_idf = math.log((self.n_docs + 1) / 1) + 1.0  # OOV ~ df=1
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            if not toks:
                continue
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            n = len(toks)
            for t, c in counts.items():
                weight = (c / n) * self.idf.get(t, self.default_idf)
                out[i, self._bucket(t)] += weight
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


class OpenAIEmbeddings(TextVectorizer):
    """Real dense embeddings. Requires `openai` and OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "openai not installed. `pip install -e \".[openai]\"`"
                ) from exc
            self._client = OpenAI()
        return self._client

    def fit(self, corpus: list[str]) -> "OpenAIEmbeddings":
        return self  # no corpus fitting needed for dense embeddings

    def transform(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs API
        client = self._client_lazy()
        vecs: list[list[float]] = []
        for start in range(0, len(texts), 128):
            batch = texts[start : start + 128]
            resp = client.embeddings.create(model=self.model, input=batch)
            vecs.extend([d.embedding for d in resp.data])
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
