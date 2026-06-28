"""Classifier: commodity-consensus selection and confidence behaviour."""

from __future__ import annotations

from conftest import make_candidate, make_po

from spend_classifier.classifier.mock_llm import MockCodeSelector
from spend_classifier.confidence import compute_confidence


def test_selector_picks_majority_commodity_and_inherits_internal(config):
    # three gate-valve neighbours (two internals) outweigh one ball-valve
    cands = [
        make_candidate("401416070001", "gate valve 2in", 0.60),
        make_candidate("401416070002", "gate valve 4in", 0.50),
        make_candidate("401416090001", "ball valve 2in", 0.45),
    ]
    res = MockCodeSelector().select(make_po("PO-1", "gate valve"), cands)
    assert res.code.startswith("40141607")          # gate-valve commodity won
    assert res.code == "401416070001"               # internal from closest neighbour
    assert res.in_master is True


def test_selector_handles_no_candidates():
    res = MockCodeSelector().select(make_po("PO-1", "x"), [])
    assert res.code == ""
    assert res.in_master is False


def test_confidence_high_when_neighbours_agree_and_close(config):
    cands = [
        make_candidate("401416070001", "gate valve 2in", 0.60),
        make_candidate("401416070002", "gate valve 4in", 0.55),
        make_candidate("401416070001", "gate valve 2in", 0.50),
    ]
    conf, consensus, top_sim = compute_confidence(cands, "401416070001", config)
    assert consensus == 1.0            # all neighbours share the commodity
    assert top_sim == 0.60
    assert conf > 0.8


def test_confidence_low_when_commodities_split(config):
    cands = [
        make_candidate("401416070001", "gate valve", 0.30),
        make_candidate("401416090001", "ball valve", 0.29),
        make_candidate("401712050001", "tee fitting", 0.28),
    ]
    conf, consensus, _ = compute_confidence(cands, "401416070001", config)
    assert consensus < 0.5             # selected commodity is a minority
    assert conf < 0.6


def test_confidence_collapses_below_similarity_floor(config):
    # neighbours agree, but everything is far away -> coverage gate pulls conf down
    cands = [
        make_candidate("401416070001", "gate valve", 0.02),
        make_candidate("401416070002", "gate valve", 0.02),
    ]
    conf, consensus, top_sim = compute_confidence(cands, "401416070001", config)
    assert consensus == 1.0
    assert top_sim < config.min_similarity_floor
    assert conf < 0.2                  # coverage gate dominates


def test_confidence_zero_without_candidates(config):
    assert compute_confidence([], "401416070001", config) == (0.0, 0.0, 0.0)
