"""Minimal FastAPI surface for the classifier.

    GET  /health
    POST /classify   body: a PO item -> a Classification

The coded-item corpus path is read from SPEND_MASTER_PATH (defaults to the
generated synthetic corpus).
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI

from ..config import DEFAULT_CONFIG
from ..knowledge_base import load_master_items
from ..pipeline import SpendClassifier
from ..schemas import Classification, POItem

app = FastAPI(title="Spending Analysis", version="1.0.0")

MASTER_PATH = os.getenv("SPEND_MASTER_PATH", "data/knowledge_base/master_items.json")


@lru_cache(maxsize=1)
def _classifier() -> SpendClassifier:
    return SpendClassifier(load_master_items(MASTER_PATH), config=DEFAULT_CONFIG)


@app.get("/health")
def health() -> dict:
    clf = _classifier()
    return {"status": "ok", "corpus_size": len(clf.master_items), "backend": DEFAULT_CONFIG.embedding_backend}


@app.post("/classify", response_model=Classification)
def classify(item: POItem) -> Classification:
    return _classifier().classify(item)
