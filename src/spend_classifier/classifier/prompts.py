"""Prompt construction for the grounded LLM code selector.

The model is given the item plus the retrieved coded neighbours and asked to pick
the best UNSPSC code. It is allowed to return a standard code that is *not* among
the neighbours when the item is genuinely new (``in_master: false``) — but it must
never invent a malformed code: correctness beats conformity, and retrieval
grounds the decision.
"""

from __future__ import annotations

import json

from ..schemas import Candidate, POItem

SYSTEM_PROMPT = (
    "You are a procurement taxonomy expert. You assign a 12-digit internal UNSPSC "
    "code (8-digit standard commodity + 4-digit internal extension) to a purchase "
    "order line item. UNSPSC describes WHAT is being bought, never the commercial "
    "conditions (time of work, overtime, wage, quantity, price). You are given the "
    "item and the most similar already-coded items retrieved from the contract "
    "catalogue; use them as grounding. Prefer an existing catalogue code when the "
    "item's essence matches one; otherwise return the correct standard code and set "
    "in_master to false. Never invent a malformed code. Respond with JSON only."
)

RESPONSE_SCHEMA = {
    "code": "12-digit code, digits only or dotted",
    "in_master": "true if the code is one of the candidate codes, else false",
    "reason": "one sentence",
}


def build_messages(po_item: POItem, candidates: list[Candidate]) -> list[dict]:
    cand_lines = [
        {"code": c.code, "description": c.description, "similarity": round(c.similarity, 3)}
        for c in candidates
    ]
    user = (
        "ITEM TO CLASSIFY:\n"
        f"  description: {po_item.description}\n"
        f"  unit: {po_item.unit}\n"
        f"  contract: {po_item.contract_name or 'n/a'} ({po_item.contract_type or 'n/a'})\n\n"
        "RETRIEVED CANDIDATES (nearest coded items):\n"
        f"{json.dumps(cand_lines, indent=2)}\n\n"
        f"Return JSON with this shape: {json.dumps(RESPONSE_SCHEMA)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
