"""Development-only fitting and predeclared operating-point selection."""
from collections import Counter
import json
from pathlib import Path
import time

import joblib

from rag_screen.common import file_hash, read_jsonl, write_json
from rag_screen.metrics import input_metrics
from .models import ContextDetector, local_repair

BUDGETS = {"max_benign_char_loss": .01, "max_quote_damage": .10,
           "min_support_retention": .99, "min_attack_char_removal": .50}


def load_split(data, split):
    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name in [f"{split}.cases.jsonl", f"{split}.labels.jsonl"]:
        if file_hash(data / name) != manifest["files"][name]:
            raise ValueError(f"Frozen data changed: {name}")
    cases = read_jsonl(data / f"{split}.cases.jsonl")
    label_rows = read_jsonl(data / f"{split}.labels.jsonl")
    labels = {r["id"]: r for r in label_rows}
    if len(labels) != len(label_rows) or len({c["id"] for c in cases}) != len(cases) or set(labels) != {c["id"] for c in cases}:
        raise ValueError("Mismatched or duplicate case/label IDs")
    for case in cases:
        docs = {d["id"]: d["text"] for d in case["documents"]}
        for key in ["attack_spans", "quotation_spans"]:
            for did, ranges in labels[case["id"]][key].items():
                if did not in docs or any(not (0 <= a < b <= len(docs[did])) for a, b in ranges):
                    raise ValueError("Invalid annotation offsets")
        for fact in labels[case["id"]]["support"]:
            if fact["doc_id"] in docs:
                if "".join(docs[fact["doc_id"]][a:b] for a, b in fact["ranges"]) != fact["sentence"]:
                    raise ValueError("Support offsets no longer reconstruct original sentence")
    return cases, labels


def tune(detector, cases, labels):
    options = []
    for threshold in [round(i / 20, 2) for i in range(1, 20)]:
        sums = Counter()
        quote_cases = 0
        for case in cases:
            kept, audit = local_repair(case["question"], case["documents"], detector.scan, threshold)
            sums.update(input_metrics(case, labels[case["id"]], audit, kept))
            quote_cases += case["condition"] == "quotation"
        benign_loss = sums["benign_chars_removed_or_quarantined"] / sums["benign_chars"]
        damage = sums["quotation_damaged"] / quote_cases
        support = sums["support_after"] / sums["support_before"]
        removal = 1 - sums["attack_chars_retained"] / sums["attack_chars"]
        violation = (max(0, benign_loss-.01) + max(0, damage-.10) + max(0, .99-support) + max(0, .50-removal))
        options.append({"threshold": threshold, "benign_char_loss": benign_loss, "quotation_damage": damage,
                        "support_retention": support, "attack_char_removal": removal,
                        "feasible": violation < 1e-12, "violation": violation})
        if round(threshold*100) % 20 == 0:
            print(f"  threshold={threshold:.2f} removal={removal:.3f} quote_damage={damage:.3f} support={support:.3f}", flush=True)
    feasible = [p for p in options if p["feasible"]]
    selected = min(feasible or options, key=lambda p: (p["violation"], -p["attack_char_removal"], p["benign_char_loss"], -p["threshold"]))
    return {**selected, "interpretation": "feasible validation operating point" if feasible else "INFEASIBLE: least-violation diagnostic policy; not a successful budget-compliant defence"}, options


def train(data="data/followup", output="artifacts/followup_v2"):
    data, output = Path(data), Path(output)
    if (output / "policy.json").exists():
        raise ValueError("Checkpoint exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    training, tl = load_split(data, "train")
    calibration, cl = load_split(data, "calibration")
    validation, vl = load_split(data, "validation")
    sets = [{r["group"] for r in rows} for rows in [training, calibration, validation]]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Source-group leakage")
    policy, fit_reports = {}, {}
    write_json(output / "selection_protocol.json", {"budgets": BUDGETS, "thresholds": [i/20 for i in range(1,20)],
               "criterion": "Maximise injected-character removal subject to all budgets; ties lower benign loss then higher cutoff. If infeasible retain least violation for explicit diagnostic evaluation.",
               "models": "Fixed C=1; context word/bigram + focus character ngrams; focus-only input ablation uses same data and calibration protocol",
               "generator_outputs_used_for_selection": False})
    for name, context in [("context", True), ("focus", False)]:
        start = time.perf_counter()
        print(f"Fitting {name} detector", flush=True)
        model = ContextDetector(context)
        fit_reports[name] = model.fit(training, tl, calibration, cl)
        joblib.dump(model, output / f"{name}.joblib")
        print(f"Selecting {name} policy on 40 validation questions", flush=True)
        policy[name], options = tune(model, validation, vl)
        write_json(output / f"{name}_tradeoffs.json", options)
        fit_reports[name]["fit_and_select_seconds"] = time.perf_counter()-start
        print(name, policy[name], flush=True)
    policy["budgets"] = BUDGETS
    policy["data_manifest_sha256"] = file_hash(data / "manifest.json")
    policy["checkpoints"] = {name: file_hash(output / f"{name}.joblib") for name in ["context", "focus"]}
    policy["fit_reports"] = fit_reports
    write_json(output / "policy.json", policy)
    return policy
