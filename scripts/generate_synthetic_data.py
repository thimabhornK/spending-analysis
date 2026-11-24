#!/usr/bin/env python3
"""Generate a synthetic UNSPSC taxonomy, a coded-item corpus (the RAG knowledge
base), and a labelled set of PO items to classify.

Everything here is invented. No company data is used. Codes are synthetic and do
not correspond to the real UNSPSC database; they only reproduce its *structure*
(8-digit standard commodity + a 4-digit internal extension = 12 digits).

The PO set is labelled by `kind` so the evaluation can measure the system the way
it actually runs:

    exact         item carries a material number present in the master   -> Step 1
    semantic      a paraphrase of a contract item, no usable material no -> Step 2
    new_variant   a new variant of an existing commodity (new internal)  -> Step 2
    non_contract  off-catalogue spend (office/travel/software/...)        -> review

Most items are under contract (exact + semantic + new_variant), as in a typical
procurement book.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 7
ROOT = Path(__file__).resolve().parents[1]
TAX_PATH = ROOT / "data" / "taxonomy" / "unspsc_taxonomy.json"
KB_PATH = ROOT / "data" / "knowledge_base" / "master_items.json"
PO_PATH = ROOT / "data" / "synthetic" / "po_items.json"

VARIANTS_PER_COMMODITY = 4

# --------------------------------------------------------------------------- #
# Catalogue: segment -> family -> class -> commodities.
# Each commodity has a distinctive head term plus attribute pools that are shared
# within a class (so within-class items are genuinely confusable, as in reality).
# --------------------------------------------------------------------------- #
VALVE_ATTRS = {
    "size": ["1in", "2in", "3in", "4in", "6in", "8in"],
    "material": ["carbon steel", "stainless steel", "cast iron", "bronze"],
    "rating": ["150lb", "300lb", "600lb", "900lb"],
}
FITTING_ATTRS = {
    "size": ["1in", "2in", "3in", "4in", "6in"],
    "material": ["carbon steel", "stainless steel", "alloy 20"],
    "schedule": ["sch40", "sch80", "sch160"],
}
INSTR_ATTRS = {
    "range": ["0-10bar", "0-25bar", "0-100bar", "4-20ma"],
    "conn": ["flanged", "threaded", "wafer"],
    "material": ["316ss", "hastelloy", "monel"],
}
PUMP_ATTRS = {
    "duty": ["5kw", "15kw", "30kw", "75kw"],
    "material": ["cast iron", "stainless steel", "duplex"],
    "seal": ["mechanical seal", "packing", "magnetic drive"],
}
ELEC_ATTRS = {
    "rating": ["240v", "415v", "11kv"],
    "size": ["2.5sqmm", "6sqmm", "25sqmm", "95sqmm"],
    "type": ["xlpe", "pvc", "armoured"],
}
MRO_ATTRS = {
    "size": ["m8", "m12", "m16", "6203", "6205", "dn50"],
    "material": ["carbon steel", "stainless steel", "ptfe", "graphite"],
    "grade": ["grade a", "grade b", "class 8.8"],
}
CHEM_ATTRS = {
    "grade": ["technical grade", "industrial grade", "catalyst grade"],
    "pack": ["25kg bag", "200l drum", "1000l ibc"],
    "purity": ["98 percent", "99 percent", "99.9 percent"],
}
SVC_ATTRS = {
    "scope": ["annual", "quarterly", "shutdown"],
    "asset": ["pressure vessel", "rotating equipment", "field instrument"],
}
MANPOWER_ATTRS = {
    "shift": ["static", "rotating"],
    "confine": ["confine", "non-confine"],
}

CATALOG = [
    {
        "seg": ("40", "Industrial equipment and components"),
        "families": [
            {
                "fam": ("14", "Fluid handling - valves"),
                "classes": [
                    {
                        "cls": ("16", "Industrial valves"),
                        "commodities": [
                            ("07", "Gate valve", "gate valve", VALVE_ATTRS, ["EA"]),
                            ("08", "Globe valve", "globe valve", VALVE_ATTRS, ["EA"]),
                            ("09", "Ball valve", "ball valve", VALVE_ATTRS, ["EA"]),
                            ("10", "Check valve", "check valve", VALVE_ATTRS, ["EA"]),
                            ("11", "Butterfly valve", "butterfly valve", VALVE_ATTRS, ["EA"]),
                        ],
                    },
                    {
                        "cls": ("18", "Valve actuators"),
                        "commodities": [
                            ("05", "Pneumatic actuator", "pneumatic actuator", VALVE_ATTRS, ["EA"]),
                            ("06", "Electric actuator", "electric actuator", VALVE_ATTRS, ["EA"]),
                        ],
                    },
                ],
            },
            {
                "fam": ("17", "Pipe fittings"),
                "classes": [
                    {
                        "cls": ("12", "Pipe fittings"),
                        "commodities": [
                            ("04", "Elbow fitting", "elbow 90 degree", FITTING_ATTRS, ["EA"]),
                            ("05", "Tee fitting", "tee fitting", FITTING_ATTRS, ["EA"]),
                            ("06", "Reducer fitting", "concentric reducer", FITTING_ATTRS, ["EA"]),
                            ("07", "Weld neck flange", "weld neck flange", FITTING_ATTRS, ["EA"]),
                        ],
                    },
                ],
            },
        ],
    },
    {
        "seg": ("41", "Laboratory and measuring instruments"),
        "families": [
            {
                "fam": ("11", "Instruments"),
                "classes": [
                    {
                        "cls": ("15", "Process instruments"),
                        "commodities": [
                            ("03", "Pressure transmitter", "pressure transmitter", INSTR_ATTRS, ["EA"]),
                            ("04", "Flow meter", "electromagnetic flow meter", INSTR_ATTRS, ["EA"]),
                            ("05", "Temperature transmitter", "temperature transmitter", INSTR_ATTRS, ["EA"]),
                            ("06", "Level gauge", "level gauge", INSTR_ATTRS, ["EA"]),
                            ("07", "Control valve positioner", "valve positioner", INSTR_ATTRS, ["EA"]),
                        ],
                    },
                ],
            },
        ],
    },
    {
        "seg": ("40", "Industrial equipment and components"),
        "families": [
            {
                "fam": ("10", "Rotating equipment"),
                "classes": [
                    {
                        "cls": ("16", "Pumps"),
                        "commodities": [
                            ("02", "Centrifugal pump", "centrifugal pump", PUMP_ATTRS, ["EA"]),
                            ("03", "Gear pump", "gear pump", PUMP_ATTRS, ["EA"]),
                            ("04", "Diaphragm pump", "diaphragm pump", PUMP_ATTRS, ["EA"]),
                        ],
                    },
                ],
            },
        ],
    },
    {
        "seg": ("39", "Electrical systems and lighting"),
        "families": [
            {
                "fam": ("12", "Power distribution"),
                "classes": [
                    {
                        "cls": ("10", "Cables and breakers"),
                        "commodities": [
                            ("02", "Power cable", "power cable", ELEC_ATTRS, ["M", "ROLL"]),
                            ("03", "Circuit breaker", "circuit breaker", ELEC_ATTRS, ["EA"]),
                            ("04", "Cable gland", "cable gland", ELEC_ATTRS, ["EA"]),
                        ],
                    },
                ],
            },
        ],
    },
    {
        "seg": ("31", "Manufacturing components and supplies"),
        "families": [
            {
                "fam": ("16", "Hardware and consumables"),
                "classes": [
                    {
                        "cls": ("15", "MRO consumables"),
                        "commodities": [
                            ("02", "Hex bolt", "hex bolt", MRO_ATTRS, ["EA", "BOX"]),
                            ("03", "Ball bearing", "ball bearing", MRO_ATTRS, ["EA"]),
                            ("04", "Gasket", "spiral wound gasket", MRO_ATTRS, ["EA"]),
                            ("05", "Mechanical seal", "mechanical seal", MRO_ATTRS, ["EA"]),
                            ("06", "Industrial lubricant", "industrial lubricant", MRO_ATTRS, ["L", "DRUM"]),
                        ],
                    },
                ],
            },
        ],
    },
    {
        "seg": ("12", "Chemicals including bio and gas materials"),
        "families": [
            {
                "fam": ("16", "Process chemicals"),
                "classes": [
                    {
                        "cls": ("12", "Catalysts and additives"),
                        "commodities": [
                            ("02", "Polymerization catalyst", "polymerization catalyst", CHEM_ATTRS, ["KG", "BAG"]),
                            ("03", "Antioxidant additive", "antioxidant additive", CHEM_ATTRS, ["KG", "BAG"]),
                            ("04", "Process solvent", "process solvent", CHEM_ATTRS, ["L", "DRUM"]),
                            ("05", "Corrosion inhibitor", "corrosion inhibitor", CHEM_ATTRS, ["L", "IBC"]),
                        ],
                    },
                ],
            },
        ],
    },
    {
        "seg": ("81", "Engineering and research services"),
        "families": [
            {
                "fam": ("10", "Industrial services"),
                "classes": [
                    {
                        "cls": ("15", "Inspection and calibration"),
                        "commodities": [
                            ("02", "Calibration service", "instrument calibration service", SVC_ATTRS, ["JOB"]),
                            ("03", "Inspection service", "ndt inspection service", SVC_ATTRS, ["JOB"]),
                            ("04", "Overhaul service", "equipment overhaul service", SVC_ATTRS, ["JOB"]),
                        ],
                    },
                ],
            },
        ],
    },
    {
        "seg": ("80", "Management and business professionals"),
        "families": [
            {
                "fam": ("11", "Contract staffing"),
                "classes": [
                    {
                        "cls": ("16", "Technical manpower"),
                        "commodities": [
                            ("02", "Manpower technician", "manpower supply technician", MANPOWER_ATTRS, ["MD"]),
                            ("03", "Manpower operator", "manpower supply plant operator", MANPOWER_ATTRS, ["MD"]),
                            ("04", "Manpower supervisor", "manpower supply field supervisor", MANPOWER_ATTRS, ["MD"]),
                        ],
                    },
                ],
            },
        ],
    },
]

CONTRACT_TYPE_BY_SEG = {
    "40": "Materials", "41": "Materials", "39": "Materials", "31": "Materials",
    "12": "Materials", "81": "Services", "80": "Manpower",
}
CONTRACT_NAME_BY_SEG = {
    "40": "Valves & Rotating Frame Agreement",
    "41": "Field Instruments Contract",
    "39": "Electrical Bulk Supply Contract",
    "31": "MRO Consumables Contract",
    "12": "Process Chemicals Supply Contract",
    "81": "Inspection & Calibration Services Contract",
    "80": "Technical Manpower Contract",
}

# off-catalogue (non-contract) spend: deliberately disjoint vocabulary
NON_CONTRACT = [
    ("office chair ergonomic mesh", "EA"),
    ("laptop computer 14 inch business", "EA"),
    ("printer toner cartridge black", "EA"),
    ("marketing campaign social media", "JOB"),
    ("air ticket bangkok singapore economy", "EA"),
    ("hotel accommodation conference", "NIGHT"),
    ("catering lunch buffet meeting", "PAX"),
    ("software license productivity suite annual", "USER"),
    ("courier parcel delivery domestic", "JOB"),
    ("stationery notebook a4 ream", "BOX"),
    ("first aid kit refill", "EA"),
    ("coffee beans pantry supply", "KG"),
    ("uniform polo shirt embroidered", "EA"),
    ("mobile phone plan corporate", "LINE"),
    ("billboard advertising rental", "MONTH"),
]

PHRASE_SYNONYMS = {
    "carbon steel": "cs",
    "stainless steel": "ss",
    "1in": "1 inch", "2in": "2 inch", "3in": "3 inch", "4in": "4 inch",
    "6in": "6 inch", "8in": "8 inch",
    "150lb": "150 lb class", "300lb": "300 lb class",
    "90 degree": "90deg",
    "electromagnetic": "mag",
    "316ss": "316 stainless",
    "non-confine": "non confine",
}


def code12(seg, fam, cls, com, internal):
    return f"{seg}{fam}{cls}{com}{internal:04d}"


def make_description(head, attrs, rng):
    parts = [head]
    for _label, options in attrs.items():
        parts.append(rng.choice(options))
    return " ".join(parts)


def build_taxonomy():
    tax = {"segments": {}}
    for seg_block in CATALOG:
        seg, seg_name = seg_block["seg"]
        seg_node = tax["segments"].setdefault(seg, {"name": seg_name, "families": {}})
        for fam_block in seg_block["families"]:
            fam, fam_name = fam_block["fam"]
            fam_node = seg_node["families"].setdefault(fam, {"name": fam_name, "classes": {}})
            for cls_block in fam_block["classes"]:
                cls, cls_name = cls_block["cls"]
                cls_node = fam_node["classes"].setdefault(cls, {"name": cls_name, "commodities": {}})
                for com, com_name, _head, _attrs, _units in cls_block["commodities"]:
                    cls_node["commodities"][com] = {"name": com_name}
    return tax


def build_master(rng):
    master = []
    mat_counter = 1000
    for seg_block in CATALOG:
        seg, _ = seg_block["seg"]
        for fam_block in seg_block["families"]:
            fam, _ = fam_block["fam"]
            for cls_block in fam_block["classes"]:
                cls, _ = cls_block["cls"]
                for com, _com_name, head, attrs, units in cls_block["commodities"]:
                    seen = set()
                    v = 0
                    tries = 0
                    while v < VARIANTS_PER_COMMODITY and tries < 40:
                        tries += 1
                        desc = make_description(head, attrs, rng)
                        if desc in seen:
                            continue
                        seen.add(desc)
                        v += 1
                        mat_counter += 1
                        master.append({
                            "item_id": f"M-{mat_counter}",
                            "material_no": f"MAT-{mat_counter}",
                            "description": desc,
                            "unit": rng.choice(units),
                            "contract_name": CONTRACT_NAME_BY_SEG[seg],
                            "contract_type": CONTRACT_TYPE_BY_SEG[seg],
                            "code": code12(seg, fam, cls, com, v),
                        })
    return master


def paraphrase(desc, head, rng):
    """Reword a description: synonym swaps, token drop, generic prefix, reorder.
    Bag-of-words retrieval ignores order, so the realistic error pressure comes
    from swapping/dropping distinctive tokens — occasionally even the head term."""
    text = desc
    for phrase, alt in PHRASE_SYNONYMS.items():
        if phrase in text and rng.random() < 0.5:
            text = text.replace(phrase, alt)

    tokens = text.split()
    # drop a non-head token sometimes (loses an attribute)
    if len(tokens) > 3 and rng.random() < 0.4:
        i = rng.randrange(1, len(tokens))
        tokens.pop(i)

    # occasionally drop the distinctive leading modifier of the head, leaving the
    # generic noun -> creates genuine within-class ambiguity (the main error mode)
    head_words = head.split()
    if len(head_words) >= 2 and rng.random() < 0.15:
        distinctive = head_words[0]
        if distinctive in tokens:
            tokens.remove(distinctive)

    rng.shuffle(tokens)
    if rng.random() < 0.5:
        tokens = [rng.choice(["supply of", "procurement of", "purchase"])] + tokens
    return " ".join(tokens)


def build_po_items(master, rng):
    items = []
    n = 0

    def add(po, truth_code, is_contract, kind):
        nonlocal n
        n += 1
        items.append({
            "po": po,
            "truth_code": truth_code,
            "is_contract": is_contract,
            "kind": kind,
        })

    # exact (~24%) and semantic (~40%) derive from sampled master items
    for m in master:
        r = rng.random()
        if r < 0.32:  # exact
            add(
                {
                    "po_item_id": f"PO-{n:05d}",
                    "material_no": m["material_no"],
                    "description": m["description"],
                    "unit": m["unit"],
                    "contract_name": m["contract_name"],
                    "contract_type": m["contract_type"],
                },
                truth_code=m["code"], is_contract=True, kind="exact",
            )
        elif r < 0.85:  # semantic paraphrase
            head = m["description"].split()  # not used directly; head recomputed below
            # recover head from the commodity by stripping known attribute tokens is
            # overkill; the paraphraser only needs the leading words, so pass first two
            head_guess = " ".join(m["description"].split()[:2])
            add(
                {
                    "po_item_id": f"PO-{n:05d}",
                    "material_no": None,
                    "description": paraphrase(m["description"], head_guess, rng),
                    "unit": m["unit"],
                    "contract_name": m["contract_name"],
                    "contract_type": m["contract_type"],
                },
                truth_code=m["code"], is_contract=True, kind="semantic",
            )

    # new_variant (~14% of total): new internal under an existing commodity
    n_variants = int(0.14 * len(items) / (1 - 0.14)) if items else 0
    for _ in range(max(8, n_variants)):
        m = rng.choice(master)
        seg, fam, cls, com = m["code"][0:2], m["code"][2:4], m["code"][4:6], m["code"][6:8]
        head_guess = " ".join(m["description"].split()[:2])
        # build a fresh description for the same commodity with a new attribute mix
        new_desc = paraphrase(m["description"], head_guess, rng)
        add(
            {
                "po_item_id": f"PO-{n:05d}",
                "material_no": None,
                "description": new_desc,
                "unit": m["unit"],
                "contract_name": m["contract_name"],
                "contract_type": m["contract_type"],
            },
            truth_code=code12(seg, fam, cls, com, 9001), is_contract=True, kind="new_variant",
        )

    # non_contract: tuned so most items are under contract
    target_total = len(items) / 0.78
    n_non = int(round(target_total - len(items)))
    for _ in range(max(1, n_non)):
        desc, unit = rng.choice(NON_CONTRACT)
        add(
            {
                "po_item_id": f"PO-{n:05d}",
                "material_no": None,
                "description": desc,
                "unit": unit,
                "contract_name": None,
                "contract_type": None,
            },
            truth_code=None, is_contract=False, kind="non_contract",
        )

    rng.shuffle(items)
    # renumber ids after shuffle for stable, readable ordering
    for i, entry in enumerate(items):
        entry["po"]["po_item_id"] = f"PO-{i:05d}"
    return items


def main():
    rng = random.Random(SEED)
    taxonomy = build_taxonomy()
    master = build_master(rng)
    po_items = build_po_items(master, rng)

    TAX_PATH.parent.mkdir(parents=True, exist_ok=True)
    KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    PO_PATH.parent.mkdir(parents=True, exist_ok=True)
    TAX_PATH.write_text(json.dumps(taxonomy, indent=2))
    KB_PATH.write_text(json.dumps(master, indent=2))
    PO_PATH.write_text(json.dumps(po_items, indent=2))

    kinds: dict[str, int] = {}
    for e in po_items:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    contract = sum(1 for e in po_items if e["is_contract"])
    n_comm = sum(
        len(c["commodities"])
        for s in taxonomy["segments"].values()
        for f in s["families"].values()
        for c in f["classes"].values()
    )
    print(f"taxonomy: {len(taxonomy['segments'])} segments, {n_comm} commodities")
    print(f"master corpus: {len(master)} coded items")
    print(f"PO items: {len(po_items)}  ({contract} contract = {contract/len(po_items):.0%})")
    print(f"  by kind: {kinds}")


if __name__ == "__main__":
    main()
