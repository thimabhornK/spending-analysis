"""Pipeline: the exact-match, semantic-auto, and review paths end to end."""

from __future__ import annotations

import json

from conftest import make_po

from spend_classifier import SpendClassifier
from spend_classifier.config import ClassifierConfig
from spend_classifier.schemas import ClassifyDecision, MatchMethod


def _clf(corpus, **overrides):
    cfg = ClassifierConfig(embedding_dim=2048, **overrides)
    return SpendClassifier(corpus, config=cfg)


def test_exact_material_match_autoclassifies(corpus):
    clf = _clf(corpus)
    res = clf.classify(make_po("PO-1", "anything at all", material_no="MAT-3"))
    assert res.match_method is MatchMethod.EXACT_CODE
    assert res.decision is ClassifyDecision.AUTO_CLASSIFY
    assert res.code == "401416090001"          # copied from the master row
    assert res.confidence >= 0.99
    assert res.matched_material_no == "MAT-3"


def test_semantic_match_autoclassifies_correct_commodity(corpus):
    clf = _clf(corpus)
    res = clf.classify(make_po("PO-2", "valve gate steel carbon 2 inch 150lb"))
    assert res.match_method is MatchMethod.SEMANTIC
    assert res.decision is ClassifyDecision.AUTO_CLASSIFY
    assert res.code.startswith("40141607")     # gate-valve commodity
    assert res.candidates                       # retrieval populated


def test_non_contract_item_routes_to_review(corpus):
    clf = _clf(corpus)
    res = clf.classify(make_po("PO-3", "office chair ergonomic mesh"))
    assert res.decision is ClassifyDecision.NEEDS_REVIEW
    assert "non-contract" in res.rationale.lower() or "routed" in res.rationale.lower()


def test_unknown_material_no_falls_through_to_semantic(corpus):
    clf = _clf(corpus)
    res = clf.classify(make_po("PO-4", "gate valve 2in carbon steel", material_no="MAT-DOES-NOT-EXIST"))
    assert res.match_method is MatchMethod.SEMANTIC  # exact lookup missed


def test_high_threshold_sends_borderline_to_review(corpus):
    strict = _clf(corpus, auto_classify_threshold=0.999)
    res = strict.classify(make_po("PO-5", "valve gate steel"))
    assert res.decision is ClassifyDecision.NEEDS_REVIEW


def test_classification_is_json_serialisable(corpus):
    clf = _clf(corpus)
    res = clf.classify(make_po("PO-6", "ball valve stainless 6in"))
    audit = res.to_audit_dict()
    json.dumps(audit)  # must not raise
    assert audit["po_item_id"] == "PO-6"
    assert "candidates" in audit
    assert res.code_dotted is not None and res.code_dotted.count("-") == 4
