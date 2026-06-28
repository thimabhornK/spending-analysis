"""Shared fixtures: a tiny in-memory coded-item corpus and builders."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spend_classifier import MasterItem, POItem  # noqa: E402
from spend_classifier.config import DEFAULT_CONFIG  # noqa: E402
from spend_classifier.schemas import Candidate  # noqa: E402


def make_master(item_id, description, code, *, material_no=None, unit="EA",
                contract_name="Test Contract", contract_type="Materials") -> MasterItem:
    return MasterItem(
        item_id=item_id, material_no=material_no, description=description,
        unit=unit, contract_name=contract_name, contract_type=contract_type, code=code,
    )


def make_po(po_item_id, description, *, material_no=None, unit="EA",
            contract_name=None, contract_type=None) -> POItem:
    return POItem(
        po_item_id=po_item_id, material_no=material_no, description=description,
        unit=unit, contract_name=contract_name, contract_type=contract_type,
    )


def make_candidate(code, description, similarity) -> Candidate:
    return Candidate(item_id=f"M-{code}", description=description, code=code, similarity=similarity)


@pytest.fixture
def config():
    return DEFAULT_CONFIG


@pytest.fixture
def corpus():
    """Three commodities with distinctive vocabulary, two internal variants each.

    Codes: gate valve 40-14-16-07, ball valve 40-14-16-09, power cable 39-12-10-02.
    """
    return [
        make_master("M-1", "gate valve 2in carbon steel 150lb", "401416070001", material_no="MAT-1"),
        make_master("M-2", "gate valve 4in stainless steel 300lb", "401416070002", material_no="MAT-2"),
        make_master("M-3", "ball valve 2in carbon steel 150lb", "401416090001", material_no="MAT-3"),
        make_master("M-4", "ball valve 6in stainless steel 600lb", "401416090002", material_no="MAT-4"),
        make_master("M-5", "power cable 25sqmm xlpe 415v", "391210020001", material_no="MAT-5"),
        make_master("M-6", "power cable 95sqmm armoured 11kv", "391210020002", material_no="MAT-6"),
    ]
