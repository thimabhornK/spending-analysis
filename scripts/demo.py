#!/usr/bin/env python3
"""Demo: classify representative PO items and show the retrieval-grounded
decision for each — an exact match, a confident semantic match, a low-confidence
review, and an off-catalogue (non-contract) item.

Run:  python scripts/demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spend_classifier import SpendClassifier, POItem  # noqa: E402
from spend_classifier.config import DEFAULT_CONFIG  # noqa: E402
from spend_classifier.knowledge_base import load_master_items  # noqa: E402
from spend_classifier.schemas import ClassifyDecision, MatchMethod  # noqa: E402

KB = ROOT / "data" / "knowledge_base" / "master_items.json"
PO = ROOT / "data" / "synthetic" / "po_items.json"


def pick(clf, dataset):
    picks = {}
    for entry in dataset:
        po = POItem(**entry["po"])
        res = clf.classify(po)
        kind = entry["kind"]
        if kind == "exact" and "exact" not in picks:
            picks["exact"] = (po, res, entry)
        elif kind == "non_contract" and "non_contract" not in picks:
            picks["non_contract"] = (po, res, entry)
        elif res.match_method is MatchMethod.SEMANTIC:
            if res.decision is ClassifyDecision.AUTO_CLASSIFY and "auto" not in picks:
                picks["auto"] = (po, res, entry)
            elif res.decision is ClassifyDecision.NEEDS_REVIEW and entry["is_contract"] and "review" not in picks:
                picks["review"] = (po, res, entry)
        if len(picks) == 4:
            break
    return picks


def main() -> None:
    if not KB.exists() or not PO.exists():
        sys.exit("Data not found. Run: python scripts/generate_synthetic_data.py")

    clf = SpendClassifier(load_master_items(KB), config=DEFAULT_CONFIG)
    dataset = json.loads(PO.read_text())
    picks = pick(clf, dataset)

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        sys.exit("This demo needs `rich`. Install with: pip install rich")

    console = Console()
    console.print(
        Panel.fit(
            "[bold]Spending Analysis[/bold] — RAG-based UNSPSC spend classification\n"
            f"corpus: {len(clf.master_items)} coded items   "
            f"backend: {DEFAULT_CONFIG.embedding_backend}/{DEFAULT_CONFIG.classifier_backend}   "
            f"threshold: {DEFAULT_CONFIG.auto_classify_threshold}",
            border_style="cyan",
        )
    )

    order = [
        ("exact", "EXACT MATCH (material number found in contract master)", "green"),
        ("auto", "SEMANTIC MATCH (auto-classified)", "green"),
        ("review", "SEMANTIC MATCH (low confidence -> review)", "yellow"),
        ("non_contract", "OFF-CATALOGUE (non-contract -> review)", "red"),
    ]
    for key, title, color in order:
        if key not in picks:
            continue
        po, res, entry = picks[key]
        console.rule(f"[bold {color}]{title}[/bold {color}]")
        console.print(f'[bold]{po.po_item_id}[/bold]  "{po.description}"  (unit={po.unit})')
        truth = entry["truth_code"] or "n/a (non-contract)"
        console.print(f"ground-truth code: {truth}")

        if res.candidates:
            t = Table(title="retrieved candidates (nearest coded items)")
            t.add_column("similarity", justify="right")
            t.add_column("code")
            t.add_column("description")
            for c in res.candidates:
                sty = "green" if c.similarity >= DEFAULT_CONFIG.sim_reference else "yellow"
                t.add_row(f"[{sty}]{c.similarity:.2f}[/{sty}]", c.code, c.description)
            console.print(t)
        else:
            console.print("[dim]no retrieval needed (exact code match)[/dim]")

        console.print(
            Panel.fit(
                f"decision: [bold {color}]{res.decision.value}[/bold {color}]   "
                f"confidence: [bold]{res.confidence:.3f}[/bold]   "
                f"method: {res.match_method.value}\n"
                f"predicted code: {res.code_dotted or '(none)'}"
                f"{'' if res.in_master else '  (proposed, not in master)'}\n"
                f"{res.rationale}",
                border_style=color,
            )
        )
        console.print()


if __name__ == "__main__":
    main()
