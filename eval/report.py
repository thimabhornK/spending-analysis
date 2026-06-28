"""Render evaluation results to the console and write eval/results.json."""

from __future__ import annotations

import json
from pathlib import Path

from eval.harness import EvalResults

RESULTS_JSON = Path("eval/results.json")


def print_report(results: EvalResults) -> dict:
    d = results.to_dict()
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(d, indent=2))

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        at = Table(title="Classification accuracy (contract items)", title_style="bold")
        at.add_column("level")
        at.add_column("accuracy", justify="right")
        for lvl in ["family", "class", "commodity", "full"]:
            label = f"[bold]{lvl}[/bold]" if lvl == "commodity" else lvl
            val = d["level_accuracy"][lvl]
            val_s = f"[bold]{val*100:.1f}%[/bold]" if lvl == "commodity" else f"{val*100:.1f}%"
            at.add_row(label, val_s)
        at.add_section()
        at.add_row("top-k retrieval hit", f"{d['topk_retrieval']*100:.1f}%")
        at.add_row("commodity acc | auto-classified", f"{d['commodity_acc_autoclassified']*100:.1f}%")
        console.print(at)

        st = Table(title="Automation safety", title_style="bold")
        st.add_column("metric")
        st.add_column("value", justify="right")
        st.add_row("auto-classify rate", f"{d['auto_classify_rate']*100:.1f}%")
        st.add_row("[bold]auto-classify precision[/bold]", f"[bold green]{d['auto_classify_precision']*100:.1f}%[/bold green]")
        st.add_row("review recall (of items needing review)", f"{d['review_recall']*100:.1f}%")
        console.print(st)

        c = d["confusion"]
        ct = Table(title="should auto?  x  did auto?", title_style="bold")
        ct.add_column("")
        ct.add_column("auto-classified", justify="right")
        ct.add_column("review", justify="right")
        ct.add_row("correct / should-review-no", str(c["correct_autoclassify"]), str(c["over_review"]))
        ct.add_row("wrong / should-review-yes", f"[red]{c['incorrect_autoclassify']}[/red]", str(c["good_review"]))
        console.print(ct)

        console.print(
            Panel.fit(
                f"items: {d['n_items']}  (contract {d['n_contract']})\n"
                f"decisions: {d['decisions']}\n"
                f"p50 latency: {d['p50_ms']:.3f} ms / item\n"
                f"results written to {RESULTS_JSON}",
                title="run summary",
            )
        )
    except ImportError:
        print(json.dumps(d, indent=2))
        print(f"results written to {RESULTS_JSON}")

    return d
