"""Load the coded-item corpus (the RAG knowledge base) and the taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import MasterItem


def load_master_items(path: str | Path) -> list[MasterItem]:
    raw = json.loads(Path(path).read_text())
    return [MasterItem(**row) for row in raw]


def load_taxonomy(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
