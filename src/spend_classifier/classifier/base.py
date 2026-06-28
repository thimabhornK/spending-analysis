"""Interface for choosing a UNSPSC code from retrieved candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..schemas import Candidate, POItem


@dataclass
class SelectionResult:
    code: str
    in_master: bool
    rationale: str


class CodeSelector(ABC):
    """Given a PO item and its nearest coded neighbours, choose the code."""

    @abstractmethod
    def select(self, po_item: POItem, candidates: list[Candidate]) -> SelectionResult: ...
