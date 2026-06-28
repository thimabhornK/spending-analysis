"""Runtime configuration.

Every tunable that affects a classification decision lives here so it can be
reviewed, versioned, and swept in the evaluation harness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClassifierConfig:
    # --- backend selection ------------------------------------------------- #
    # "tfidf" runs fully offline & deterministic (hashed TF-IDF retrieval).
    # "openai" uses real dense embeddings for retrieval.
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("SPEND_EMBEDDING_BACKEND", "tfidf")
    )
    # "mock" picks the code by similarity-weighted vote over neighbours.
    # "openai" asks a real LLM to choose, grounded on the retrieved candidates.
    classifier_backend: str = field(
        default_factory=lambda: os.getenv("SPEND_CLASSIFIER_BACKEND", "mock")
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv("SPEND_OPENAI_EMBED_MODEL", "text-embedding-3-small")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("SPEND_OPENAI_MODEL", "gpt-4o-mini")
    )

    # --- retrieval --------------------------------------------------------- #
    top_k: int = 5                 # neighbours retrieved per query
    embedding_dim: int = 4096      # hashed TF-IDF dimensionality (offline backend)

    # --- decision gating --------------------------------------------------- #
    # An item is auto-classified only if its confidence clears this bar. This is
    # the dial that trades automation rate against precision.
    auto_classify_threshold: float = field(
        default_factory=lambda: float(os.getenv("SPEND_AUTO_THRESHOLD", "0.70"))
    )
    # Below this top-neighbour similarity an item is treated as out-of-catalogue
    # (likely non-contract / novel spend) and always routed to a human.
    min_similarity_floor: float = 0.18

    # Confidence is a blend of how similar the best neighbour is (absolute) and
    # how strongly the neighbours agree on one code (consensus). This weights the
    # consensus term; (1 - w) weights the absolute-similarity term.
    consensus_weight: float = 0.6
    # Cosine similarities from sparse TF-IDF are modest in absolute terms; a top
    # similarity at/above this counts as fully adequate when scoring confidence.
    sim_reference: float = 0.45

    # --- reproducibility --------------------------------------------------- #
    seed: int = 7


DEFAULT_CONFIG = ClassifierConfig()
