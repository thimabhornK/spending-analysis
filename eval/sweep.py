"""Sweep the auto-classify confidence threshold to show the precision/automation
trade-off — the dial an operator turns. Run: python -m eval.sweep
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from spend_classifier.config import DEFAULT_CONFIG

from eval.harness import evaluate

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SWEEP_JSON = Path("eval/sweep.json")


def run_sweep(thresholds=THRESHOLDS) -> list[dict]:
    rows = []
    for t in thresholds:
        cfg = dataclasses.replace(DEFAULT_CONFIG, auto_classify_threshold=t)
        r = evaluate(config=cfg)
        rows.append(
            {
                "threshold": t,
                "auto_classify_rate": round(r.auto_classify_rate, 4),
                "auto_classify_precision": round(r.auto_classify_precision, 4),
                "review_recall": round(r.review_recall, 4),
                "incorrect_autoclassify": r.confusion["incorrect_autoclassify"],
            }
        )
    return rows


def print_sweep(rows: list[dict]) -> None:
    SWEEP_JSON.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_JSON.write_text(json.dumps(rows, indent=2))
    try:
        from rich.console import Console
        from rich.table import Table

        t = Table(title="Auto-classify threshold sweep", title_style="bold")
        t.add_column("threshold", justify="right")
        t.add_column("auto rate", justify="right")
        t.add_column("auto precision", justify="right")
        t.add_column("review recall", justify="right")
        t.add_column("incorrect auto", justify="right")
        for r in rows:
            t.add_row(
                f"{r['threshold']:.2f}",
                f"{r['auto_classify_rate']*100:.1f}%",
                f"{r['auto_classify_precision']*100:.1f}%",
                f"{r['review_recall']*100:.1f}%",
                str(r["incorrect_autoclassify"]),
            )
        Console().print(t)
    except ImportError:
        for r in rows:
            print(r)
    print(f"sweep written to {SWEEP_JSON}")


if __name__ == "__main__":  # pragma: no cover
    print_sweep(run_sweep())
