"""Offline, deterministic code selector.

Picks the code by a similarity-weighted vote over the retrieved neighbours: each
neighbour contributes its cosine similarity to its code's tally, and the highest
tally wins. This rewards both *strong* matches (high similarity) and *consistent*
matches (several neighbours agreeing), which is exactly what the confidence score
later reads off. It mirrors what a real LLM tends to do when grounded on
candidates, while staying fully reproducible.
"""

from __future__ import annotations

from collections import defaultdict

from ..schemas import Candidate, POItem
from ..taxonomy import safe_parse
from .base import CodeSelector, SelectionResult


class MockCodeSelector(CodeSelector):
    def select(self, po_item: POItem, candidates: list[Candidate]) -> SelectionResult:
        if not candidates:
            return SelectionResult(code="", in_master=False, rationale="no candidates retrieved")

        # Vote at the *commodity* level (the part retrieval determines reliably);
        # the 4-digit internal extension is inherited from the single closest item.
        comm_tally: dict[str, float] = defaultdict(float)
        for cand in candidates:
            parsed = safe_parse(cand.code)
            if parsed:
                comm_tally[parsed.commodity] += cand.similarity
        if not comm_tally:
            return SelectionResult(code="", in_master=False, rationale="candidate codes unparseable")

        best_comm = max(comm_tally, key=lambda c: comm_tally[c])
        within = [c for c in candidates if (safe_parse(c.code) and safe_parse(c.code).commodity == best_comm)]
        chosen = max(within, key=lambda c: c.similarity)

        share = comm_tally[best_comm] / sum(comm_tally.values())
        rationale = (
            f"{len(within)}/{len(candidates)} nearest items share commodity "
            f"{safe_parse(chosen.code).standard} ({share:.0%} of similarity mass); "
            f"internal code inherited from closest match"
        )
        return SelectionResult(code=chosen.code, in_master=True, rationale=rationale)
