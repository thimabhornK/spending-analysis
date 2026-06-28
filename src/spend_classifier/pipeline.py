"""End-to-end orchestration: exact-code match -> retrieval -> selection -> gate.

``SpendClassifier`` is the single object the demo, the API, and the evaluation
harness all use. Built once from the coded-item corpus, it classifies any number
of PO items.
"""

from __future__ import annotations

from .classifier import build_selector
from .confidence import compute_confidence
from .config import DEFAULT_CONFIG, ClassifierConfig
from .retrieval import build_retriever
from .schemas import (
    Candidate,
    Classification,
    ClassifyDecision,
    MasterItem,
    MatchMethod,
    POItem,
)
from .taxonomy import safe_parse


class SpendClassifier:
    def __init__(self, master_items: list[MasterItem], config: ClassifierConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.master_items = master_items
        self.by_material = {m.material_no: m for m in master_items if m.material_no}
        self.retriever = build_retriever(config, master_items)
        self.selector = build_selector(config)

    # ----------------------------------------------------------------------- #
    def classify(self, po: POItem) -> Classification:
        # STEP 1 — exact code match against the master.
        if po.material_no and po.material_no in self.by_material:
            m = self.by_material[po.material_no]
            return self._result(
                po, code=m.code, in_master=True, confidence=0.99,
                method=MatchMethod.EXACT_CODE, decision=ClassifyDecision.AUTO_CLASSIFY,
                candidates=[], matched_material_no=po.material_no,
                rationale=f"Exact material-number match to master item {m.item_id}.",
            )

        # STEP 2 — semantic retrieval-augmented classification.
        candidates = self.retriever.retrieve(po.embedding_text(), self.config.top_k)
        sel = self.selector.select(po, candidates)
        conf, _consensus, top_sim = compute_confidence(candidates, sel.code, self.config)

        if not candidates or top_sim < self.config.min_similarity_floor:
            decision = ClassifyDecision.NEEDS_REVIEW
            rationale = (
                f"No sufficiently similar contract item (top similarity {top_sim:.2f} "
                f"< floor {self.config.min_similarity_floor:.2f}); likely non-contract "
                f"spend, routed to a buyer."
            )
        elif conf >= self.config.auto_classify_threshold:
            decision = ClassifyDecision.AUTO_CLASSIFY
            rationale = f"{sel.rationale}; confidence {conf:.2f} >= {self.config.auto_classify_threshold:.2f}."
        else:
            decision = ClassifyDecision.NEEDS_REVIEW
            rationale = (
                f"{sel.rationale}; confidence {conf:.2f} < "
                f"{self.config.auto_classify_threshold:.2f}; routed to a buyer."
            )

        return self._result(
            po, code=(sel.code or None), in_master=sel.in_master, confidence=conf,
            method=MatchMethod.SEMANTIC, decision=decision,
            candidates=candidates, rationale=rationale,
        )

    # ----------------------------------------------------------------------- #
    def _result(
        self, po: POItem, *, code, in_master, confidence, method, decision,
        candidates: list[Candidate], rationale: str, matched_material_no=None,
    ) -> Classification:
        parsed = safe_parse(code) if code else None
        return Classification(
            po_item_id=po.po_item_id,
            decision=decision,
            match_method=method,
            code=parsed.digits if parsed else None,
            confidence=confidence,
            in_master=in_master,
            matched_material_no=matched_material_no,
            candidates=candidates,
            rationale=rationale,
            code_standard=parsed.standard if parsed else None,
            code_dotted=parsed.dotted if parsed else None,
        )
