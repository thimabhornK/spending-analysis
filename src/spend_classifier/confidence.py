"""Confidence scoring for a semantic classification.

Confidence blends two signals:

- **absolute similarity** — how close the single best neighbour is. A high score
  means "we have seen something very like this before".
- **consensus** — the share of the retrieved similarity mass that agrees on the
  chosen code. A high score means "the neighbours don't disagree".

A correct-looking code supported by one weak, lonely neighbour should *not* be
auto-applied; requiring both signals is what keeps the auto-classified set clean.
"""

from __future__ import annotations

from .config import ClassifierConfig
from .schemas import Candidate
from .taxonomy import safe_parse


def compute_confidence(
    candidates: list[Candidate], selected_code: str, config: ClassifierConfig
) -> tuple[float, float, float]:
    """Return (confidence, consensus, top_similarity).

    Consensus is measured at the commodity level — the granularity retrieval
    actually determines — so it isn't diluted by the several internal variants a
    commodity may have in the corpus."""
    if not candidates:
        return 0.0, 0.0, 0.0
    total = sum(c.similarity for c in candidates) or 1e-9
    sel = safe_parse(selected_code)
    sel_comm = sel.commodity if sel else None

    def same_commodity(code: str) -> bool:
        p = safe_parse(code)
        return bool(p and sel_comm and p.commodity == sel_comm)

    consensus = sum(c.similarity for c in candidates if same_commodity(c.code)) / total
    top_sim = max(c.similarity for c in candidates)
    sim_adequacy = min(1.0, top_sim / config.sim_reference)
    base = config.consensus_weight * consensus + (1.0 - config.consensus_weight) * sim_adequacy
    # Catalogue coverage: if even the closest item is below the similarity floor,
    # we have no real grounding, so confidence is pulled down regardless of how
    # much the (equally weak) neighbours happen to agree. At/above the floor this
    # factor is 1.0 and has no effect.
    coverage = min(1.0, top_sim / config.min_similarity_floor)
    conf = coverage * base
    return max(0.0, min(1.0, conf)), consensus, top_sim
