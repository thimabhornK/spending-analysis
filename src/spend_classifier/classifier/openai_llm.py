"""Real LLM code selector (grounded on retrieved candidates).

Selected with SPEND_CLASSIFIER_BACKEND=openai. Kept deliberately thin: it builds
the grounded prompt, calls the model at temperature 0 for determinism, parses the
JSON, and validates the code. Any failure falls back to the top retrieved
candidate so the pipeline degrades gracefully rather than crashing.
"""

from __future__ import annotations

import json

from ..schemas import Candidate, POItem
from ..taxonomy import safe_parse
from .base import CodeSelector, SelectionResult
from .prompts import build_messages


class OpenAICodeSelector(CodeSelector):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "openai not installed. `pip install -e \".[openai]\"`"
                ) from exc
            self._client = OpenAI()
        return self._client

    def select(self, po_item: POItem, candidates: list[Candidate]) -> SelectionResult:  # pragma: no cover - needs API
        if not candidates:
            return SelectionResult(code="", in_master=False, rationale="no candidates retrieved")
        try:
            resp = self._client_lazy().chat.completions.create(
                model=self.model,
                messages=build_messages(po_item, candidates),
                temperature=0,
                response_format={"type": "json_object"},
            )
            payload = json.loads(resp.choices[0].message.content)
            parsed = safe_parse(str(payload.get("code", "")))
            if parsed is None:
                raise ValueError("model returned a malformed code")
            candidate_codes = {c.code for c in candidates}
            in_master = parsed.digits in {safe_parse(c).digits for c in candidate_codes if safe_parse(c)}
            return SelectionResult(
                code=parsed.digits,
                in_master=bool(payload.get("in_master", in_master)),
                rationale=str(payload.get("reason", "LLM selection")),
            )
        except Exception as exc:  # fall back to strongest neighbour
            top = candidates[0]
            return SelectionResult(
                code=top.code,
                in_master=True,
                rationale=f"fell back to top candidate after LLM error: {exc}",
            )
