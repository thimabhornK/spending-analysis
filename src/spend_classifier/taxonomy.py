"""UNSPSC code handling.

The standard UNSPSC code is 8 digits, a 4-level hierarchy:

    SS FF CC PP
    │  │  │  └── Commodity  (the leaf in the public standard)
    │  │  └───── Class
    │  └──────── Family
    └─────────── Segment

This project extends it the way the production system did: four extra internal
digits append a company-specific sub-commodity (e.g. a material grade or a
contract-specific variant) that the public standard doesn't capture, giving a
12-digit code:

    SS FF CC PP NNNN   ->   e.g. 40-14-16-07-0003

`UNSPSCCode` parses, validates, and exposes each level so the evaluation can
score accuracy at the family / class / commodity / full-code granularity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DIGITS = re.compile(r"\d")


def _digits_only(code: str) -> str:
    return "".join(_DIGITS.findall(code or ""))


@dataclass(frozen=True)
class UNSPSCCode:
    """A 12-digit internal UNSPSC code (8 standard + 4 internal)."""

    digits: str  # canonical 12-digit string, no separators

    def __post_init__(self) -> None:
        d = _digits_only(self.digits)
        if len(d) != 12:
            raise ValueError(f"expected 12 digits, got {len(d)!r} from {self.digits!r}")
        object.__setattr__(self, "digits", d)

    # --- hierarchy levels -------------------------------------------------- #
    @property
    def segment(self) -> str:
        return self.digits[0:2]

    @property
    def family(self) -> str:
        return self.digits[0:4]

    @property
    def klass(self) -> str:
        return self.digits[0:6]

    @property
    def commodity(self) -> str:
        """The 8-digit public UNSPSC commodity code."""
        return self.digits[0:8]

    @property
    def internal(self) -> str:
        """The 4-digit company-internal extension."""
        return self.digits[8:12]

    # --- formatting -------------------------------------------------------- #
    @property
    def standard(self) -> str:
        """8-digit public code, dotted: SS-FF-CC-PP."""
        d = self.digits
        return f"{d[0:2]}-{d[2:4]}-{d[4:6]}-{d[6:8]}"

    @property
    def dotted(self) -> str:
        """Full 12-digit code, dotted: SS-FF-CC-PP-NNNN."""
        return f"{self.standard}-{self.internal}"

    def matches(self, other: "UNSPSCCode", level: str = "commodity") -> bool:
        """Whether two codes agree down to the given level."""
        getter = {
            "segment": lambda c: c.segment,
            "family": lambda c: c.family,
            "class": lambda c: c.klass,
            "commodity": lambda c: c.commodity,
            "full": lambda c: c.digits,
        }[level]
        return getter(self) == getter(other)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.dotted


def parse(code: str) -> UNSPSCCode:
    return UNSPSCCode(code)


def safe_parse(code: str) -> UNSPSCCode | None:
    try:
        return UNSPSCCode(code)
    except (ValueError, TypeError):
        return None
