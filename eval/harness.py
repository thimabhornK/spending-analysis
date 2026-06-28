"""Evaluate the classifier on the labelled PO set.

Two families of metric:

1. **Accuracy** — at each taxonomy level (family / class / commodity / full code),
   measured over the contract items (the ones with a ground-truth code). The
   commodity-level number is the headline "classification accuracy".
2. **Automation safety** — treating it as a decision system: of the items we
   auto-classify, how many are right (precision); of the items we *should* have
   sent to a human (non-contract, or a wrong code), how many we caught (recall).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from spend_classifier import SpendClassifier, POItem
from spend_classifier.config import DEFAULT_CONFIG, ClassifierConfig
from spend_classifier.knowledge_base import load_master_items
from spend_classifier.schemas import ClassifyDecision
from spend_classifier.taxonomy import safe_parse

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "data" / "knowledge_base" / "master_items.json"
PO_PATH = ROOT / "data" / "synthetic" / "po_items.json"

LEVELS = ["family", "class", "commodity", "full"]


@dataclass
class EvalResults:
    n_items: int
    n_contract: int
    level_accuracy: dict           # over contract items
    top1_commodity: float          # == level_accuracy["commodity"]
    topk_retrieval: float          # truth commodity present among retrieved
    commodity_acc_autoclassified: float
    auto_classify_rate: float
    auto_classify_precision: float
    review_recall: float
    confusion: dict
    decisions: dict
    p50_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def _level_match(pred_code: str | None, truth_code: str | None, level: str) -> bool:
    if not pred_code or not truth_code:
        return False
    p, t = safe_parse(pred_code), safe_parse(truth_code)
    if p is None or t is None:
        return False
    return p.matches(t, level)


def evaluate(config: ClassifierConfig = DEFAULT_CONFIG, *, kb_path=KB_PATH, po_path=PO_PATH) -> EvalResults:
    master = load_master_items(kb_path)
    dataset = json.loads(Path(po_path).read_text())
    clf = SpendClassifier(master, config=config)

    level_hits = {lvl: 0 for lvl in LEVELS}
    n_contract = 0
    topk_hits = 0
    auto_correct_comm = 0
    auto_total = 0
    confusion = {
        "correct_autoclassify": 0,
        "incorrect_autoclassify": 0,   # risk cell
        "good_review": 0,
        "over_review": 0,
    }
    decisions = {"AUTO_CLASSIFY": 0, "NEEDS_REVIEW": 0}
    should_review_total = 0
    latencies = []

    for entry in dataset:
        po = POItem(**entry["po"])
        truth = entry["truth_code"]
        is_contract = entry["is_contract"]

        t0 = time.perf_counter()
        res = clf.classify(po)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        decisions[res.decision.value] += 1
        commodity_correct = _level_match(res.code, truth, "commodity") if is_contract else False

        if is_contract:
            n_contract += 1
            for lvl in LEVELS:
                if _level_match(res.code, truth, lvl):
                    level_hits[lvl] += 1
            # retrieval hit: truth commodity among candidates (exact-match path counts as hit)
            if res.match_method.value.startswith("Exact"):
                topk_hits += 1
            else:
                t = safe_parse(truth)
                cand_comms = {safe_parse(c.code).commodity for c in res.candidates if safe_parse(c.code)}
                if t and t.commodity in cand_comms:
                    topk_hits += 1

        is_auto = res.decision is ClassifyDecision.AUTO_CLASSIFY
        should_review = (not is_contract) or (not commodity_correct)
        if should_review:
            should_review_total += 1

        if is_auto:
            auto_total += 1
            if is_contract and commodity_correct:
                confusion["correct_autoclassify"] += 1
                auto_correct_comm += 1
            else:
                confusion["incorrect_autoclassify"] += 1
        else:  # NEEDS_REVIEW
            if should_review:
                confusion["good_review"] += 1
            else:
                confusion["over_review"] += 1

    n = len(dataset)
    level_acc = {lvl: (level_hits[lvl] / n_contract if n_contract else 0.0) for lvl in LEVELS}
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0

    return EvalResults(
        n_items=n,
        n_contract=n_contract,
        level_accuracy=level_acc,
        top1_commodity=level_acc["commodity"],
        topk_retrieval=(topk_hits / n_contract if n_contract else 0.0),
        commodity_acc_autoclassified=(auto_correct_comm / auto_total if auto_total else 0.0),
        auto_classify_rate=(decisions["AUTO_CLASSIFY"] / n if n else 0.0),
        auto_classify_precision=(
            confusion["correct_autoclassify"] / auto_total if auto_total else 0.0
        ),
        review_recall=(
            confusion["good_review"] / should_review_total if should_review_total else 0.0
        ),
        confusion=confusion,
        decisions=decisions,
        p50_ms=p50,
    )


if __name__ == "__main__":  # pragma: no cover
    from eval.report import print_report

    print_report(evaluate())
