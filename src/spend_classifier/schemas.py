"""Typed data model for the classification pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ClassifyDecision(str, Enum):
    AUTO_CLASSIFY = "AUTO_CLASSIFY"   # confident enough to apply without review
    NEEDS_REVIEW = "NEEDS_REVIEW"     # routed to a human (low confidence / novel)


class MatchMethod(str, Enum):
    EXACT_CODE = "Exact (Code Match)"   # material number found in the master
    SEMANTIC = "Semantic (RAG)"         # nearest-neighbour retrieval + selection
    NONE = "None"


# --------------------------------------------------------------------------- #
# Reference data (the retrieval corpus) and inputs
# --------------------------------------------------------------------------- #
class MasterItem(BaseModel):
    """A previously-coded contract line: the unit of the retrieval corpus."""

    item_id: str
    material_no: Optional[str] = None
    description: str
    unit: str
    contract_name: str
    contract_type: str
    code: str  # 12-digit internal UNSPSC

    def embedding_text(self) -> str:
        """The text used for retrieval — grounds on description + unit + contract."""
        return f"{self.description} | unit:{self.unit} | {self.contract_name} | {self.contract_type}"


class POItem(BaseModel):
    """An item to be classified."""

    po_item_id: str
    material_no: Optional[str] = None
    description: str
    unit: str
    contract_name: Optional[str] = None
    contract_type: Optional[str] = None

    def embedding_text(self) -> str:
        parts = [self.description, f"unit:{self.unit}"]
        if self.contract_name:
            parts.append(self.contract_name)
        if self.contract_type:
            parts.append(self.contract_type)
        return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Retrieval + classification outputs
# --------------------------------------------------------------------------- #
class Candidate(BaseModel):
    """A retrieved neighbour and its similarity to the query."""

    item_id: str
    description: str
    code: str
    similarity: float


class Classification(BaseModel):
    """The full result for one PO item."""

    po_item_id: str
    decision: ClassifyDecision
    match_method: MatchMethod
    code: Optional[str] = None            # predicted 12-digit code (None if unresolved)
    confidence: float = 0.0
    in_master: bool = False               # was the chosen code present in the corpus?
    matched_material_no: Optional[str] = None
    candidates: list[Candidate] = Field(default_factory=list)
    rationale: str = ""

    # convenience views filled in by the classifier for reporting / output
    code_standard: Optional[str] = None   # 8-digit dotted
    code_dotted: Optional[str] = None     # 12-digit dotted

    def to_audit_dict(self) -> dict:
        return {
            "po_item_id": self.po_item_id,
            "decision": self.decision.value,
            "match_method": self.match_method.value,
            "code": self.code,
            "code_dotted": self.code_dotted,
            "confidence": round(self.confidence, 4),
            "in_master": self.in_master,
            "matched_material_no": self.matched_material_no,
            "candidates": [c.model_dump() for c in self.candidates],
            "rationale": self.rationale,
        }
