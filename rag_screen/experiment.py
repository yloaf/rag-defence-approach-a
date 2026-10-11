from __future__ import annotations

from collections import defaultdict
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

from .common import canonical, digest, file_hash, read_jsonl, source_hash, write_json, write_jsonl
from .defense import ARMS, Judge, defend
from .detector import Detector, reliability
from .metrics import achieved, input_metrics, paired_bootstrap, score, summarize
from .ollama import ANSWER_SCHEMA, Ollama, PromptBudget, parse_answer


def load_split(data, split):
    data = Path(data)
    cases = read_jsonl(data / f"{split}.cases.jsonl")
    labels = {r["id"]: r for r in read_jsonl(data / f"{split}.labels.jsonl")}
    if len({c["id"] for c in cases}) != len(cases) or set(labels) != {c["id"] for c in cases}:
        raise ValueError("Case IDs must be unique and match labels exactly")
    return cases, labels


def train(data, output, seed=42):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "detector.pkl").exists():
        raise ValueError("Checkpoint directory already exists; use a new output directory")
    training, tl = load_split(data, "train")
    calibration, cl = load_split(data, "calibration")
    validation, vl = load_split(data, "validation")
    splits = [training, calibration, validation]
    group_sets = [{c["group"] for c in s} for s in splits]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Source-group leakage between fitting partitions")
    detector = Detector(seed)
    print("Fitting contextual segment detector and separate calibration mappings", flush=True)
    fit_info = detector.fit(training, tl, calibration, cl)
    detector.save(output / "detector.pkl")
    print("Selecting deletion policy on validation only (input-handling proxy; no QA tuning)", flush=True)
    tradeoffs = []
    for threshold in [round(i / 20, 2) for i in range(1, 20)]:
        for document_threshold in [.50, .70, .85, .95]:
            policy = {"span_threshold": threshold, "document_threshold": document_threshold}
            totals = defaultdict(float)
            quotes = 0
            for case in validation:
                kept, audit = defend(case["question"], case["documents"], "learned_repair", detector, policy)
                measured = input_metrics(case, vl[case["id"]], audit, kept)
                for key, value in measured.items():
                    totals[key] += value
                quotes += case["condition"] == "quotation"
            benign_loss = totals["benign_chars_removed_or_quarantined"] / max(1, totals["benign_chars"])
            quote_damage = totals["quotation_damaged"] / max(1, quotes)
            support = totals["support_after"] / max(1, totals["support_before"])
            residual = totals["attack_chars_retained"] / max(1, totals["attack_chars"])
            violation = max(0, benign_loss - .01) + max(0, quote_damage - .10) + max(0, .99 - support)
            tradeoffs.append({**policy, "benign_char_loss": benign_loss, "quotation_damage_rate": quote_damage,
                              "candidate_support_retention": support, "attack_chars_retained_fraction": residual,
                              "feasible_input_budgets": violation <= 1e-12, "violation": violation})
    best = min(tradeoffs, key=lambda p: (not p["feasible_input_budgets"], p["violation"],
                                        p["attack_chars_retained_fraction"], p["benign_char_loss"], -p["span_threshold"]))
    policy = {**best, "selection": "validation residual attack characters, subject to character-loss/quotation/support budgets; QA non-inferiority NOT tested",
              "detector_sha256": file_hash(output / "detector.pkl"),
              "data_manifest_sha256": file_hash(data / "manifest.json"), "seed": seed,
              "source_sha256": source_hash(), "fit_info": fit_info}
    write_json(output / "policy.json", policy)
    write_json(output / "validation_tradeoffs.json", tradeoffs)
    write_json(output / "training_manifest.json", {
        "source_sha256": source_hash(), "seed": seed,
        "fitting_files": {p.name: file_hash(p) for s in ["train", "calibration", "validation"]
                          for p in [data / f"{s}.cases.jsonl", data / f"{s}.labels.jsonl"]},
        "method": "TF-IDF contextual word/bigram features; class-balanced logistic regression; held-out sigmoid calibration"})
    return {k: v for k, v in policy.items() if k != "fit_info"}


def calibration_audit(cases, labels, detector):
    sp, sy, dp, dy = [], [], [], []
    strata = defaultdict(lambda: [[], []])
    for case in cases:
        for doc in case["documents"]:
            scan = detector.scan(case["question"], doc["text"])
            attacks = labels[case["id"]]["attack_spans"].get(doc["id"], [])
            dp.append(scan["risk"])
            dy.append(int(bool(attacks)))
            strata[case["condition"]][0].append(scan["risk"])
            strata[case["condition"]][1].append(int(bool(attacks)))
            for a, b, probability in scan["spans"]:
                sp.append(probability)
                sy.append(int(any(a < d and c < b for c, d in attacks)))
    return {"segment": reliability(sp, sy), "document": reliability(dp, dy),
            "document_by_condition": {k: reliability(*v) for k, v in strata.items()}}


def recover_results(path):
    """Recover only an interrupted final write; internal corruption is fatal."""
    path = Path(path)
    if not path.exists():
        return []
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        boundary = content.rfind(b"\n") + 1
        path.with_suffix(".interrupted-tail").write_bytes(content[boundary:])
        path.write_bytes(content[:boundary])
    return read_jsonl(path)


def run(data, checkpoint, tokenizer, output, split="pilot", groups=10, arms=None,
        model="qwen2.5:7b", base_url="http://127.0.0.1:11434", cache="artifacts/ollama.sqlite", uncached=False):
    data, checkpoint, output = Path(data), Path(checkpoint), Path(output)
    arms = list(arms or ["none", "lexical_reject", "learned_reject", "learned_repair"])
    if len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("Unknown or duplicate arms")
    cases, labels = load_split(data, split)
    group_ids = sorted({c["group"] for c in cases})
    if groups:
        group_ids = group_ids[:groups]
    cases = [c for c in cases if c["group"] in group_ids]
    if not cases:
        raise ValueError("Empty selected partition")
    if split == "test":
        for case in cases:
            label = labels[case["id"]]
            if label.get("annotation_status") != "reviewed" or len(set(label.get("reviewed_by", []))) < 2:
                raise ValueError("Final test requires two independent named reviewers per label; use pilot for unreviewed development runs")
    policy = json.loads((checkpoint / "policy.json").read_text())
    if policy["detector_sha256"] != file_hash(checkpoint / "detector.pkl"):
        raise ValueError("Checkpoint hash mismatch")
    if policy["data_manifest_sha256"] != file_hash(data / "manifest.json"):
        raise ValueError("Prepared data manifest differs from frozen training data")
    data_manifest = json.loads((data / "manifest.json").read_text())
    for filename, expected in data_manifest["files"].items():
        # Reviewed test labels are permitted to change status/annotations after preparation.
        if filename.endswith("test.labels.jsonl"):
            continue
        if file_hash(data / filename) != expected:
            raise ValueError(f"Prepared data changed: {filename}")
    detector = Detector.load(checkpoint / "detector.pkl")
    budget = PromptBudget(tokenizer)
    client = Ollama(cache, model, base_url, cache=not uncached)
    judge = Judge(client, budget)
    manifest = {"source_sha256": source_hash(), "policy_sha256": file_hash(checkpoint / "policy.json"),
                "detector_sha256": file_hash(checkpoint / "detector.pkl"), "tokenizer_sha256": budget.sha256,
                "cases_sha256": file_hash(data / f"{split}.cases.jsonl"), "labels_sha256": file_hash(data / f"{split}.labels.jsonl"),
                "data_manifest_sha256": file_hash(data / "manifest.json"), "split": split,
                "group_ids": group_ids, "case_ids": [c["id"] for c in cases], "arms": arms,
                "ollama": client.identity, "python": sys.version, "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ["numpy", "scikit-learn", "tokenizers"]},
                "budget": 6000, "temperature": 0, "seed": 42, "num_ctx": 8192,
                "answer_output_limit": 128, "judge_output_limit": 768, "cache_enabled": not uncached,
                "status": "development diagnostic" if split != "test" else "reviewed held-out evaluation"}
    run_hash = digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Run manifest changed: refuse unsafe resume; use a new output directory")
    else:
        write_json(manifest_path, manifest)
        source_dir = output / "frozen_source"
        source_dir.mkdir(exist_ok=True)
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(path, source_dir / path.name)
        frozen = output / "frozen_inputs"
        frozen.mkdir(exist_ok=True)
        write_jsonl(frozen / "cases.jsonl", cases)
        write_jsonl(frozen / "labels.jsonl", [labels[c["id"]] for c in cases])
        for name in ["detector.pkl", "policy.json", "training_manifest.json"]:
            if (checkpoint / name).exists():
                shutil.copyfile(checkpoint / name, frozen / name)
        shutil.copyfile(tokenizer, frozen / "qwen_tokenizer.json")
        shutil.copyfile(data / "manifest.json", frozen / "data_manifest.json")
        if Path("requirements.lock.txt").exists():
            shutil.copyfile("requirements.lock.txt", frozen / "requirements.lock.txt")
    results_path = output / "results.jsonl"
    previous = recover_results(results_path)
    expected_keys = {(c["id"], a) for c in cases for a in arms}
    done = set()
    for row in previous:
        key = (row["id"], row["arm"])
        if row["run_hash"] != run_hash or key not in expected_keys or key in done:
            raise ValueError("Result manifest mismatch or duplicate result")
        done.add(key)
    with results_path.open("a", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            for arm in arms:
                if (case["id"], arm) in done:
                    continue
                start = time.perf_counter()
                call_start = len(client.calls)
                kept, audit = defend(case["question"], case["documents"], arm, detector, policy, judge)
                selected, prompt, skipped = budget.select(case["question"], kept)
                if selected:
                    raw, _ = client.generate(prompt, ANSWER_SCHEMA, budget)
                    answer, parse_error = parse_answer(raw, {d["id"] for d in selected})
                    abstention_reason = "parse_failure" if parse_error else "model" if answer["abstain"] else None
                else:
                    raw, parse_error = "", None
                    answer = {"answer": "", "citations": [], "abstain": True}
                    abstention_reason = "no_usable_context"
                result = {"id": case["id"], "pair_id": case["pair_id"], "group": case["group"],
                          "condition": case["condition"], "arm": arm, "run_hash": run_hash,
                          "answer": answer, "raw_response": raw, "parse_error": parse_error,
                          "abstention_reason": abstention_reason, "audit": audit,
                          "selected": selected, "skipped": skipped, "prompt": prompt,
                          "prompt_sha256": digest(prompt), "prompt_tokens": budget.count(prompt),
                          "calls": client.calls[call_start:], "wall_seconds": time.perf_counter() - start}
                # Evaluator labels enter only here, after all inference is finished.
                result["metrics"] = score(case, labels[case["id"]], result)
                handle.write(canonical(result) + "\n")
                handle.flush()
                done.add((case["id"], arm))
                print(f"[{len(done)}/{len(expected_keys)}] {case['id']} {arm} f1={result['metrics']['f1']:.3f} attack={result['metrics']['attack_success']}", flush=True)
    write_json(output / "calibration.json", calibration_audit(cases, labels, detector))
    return report(output, labels)


def report(output, labels, resamples=10000):
    output = Path(output)
    rows = read_jsonl(output / "results.jsonl")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    indexed = {(r["pair_id"], r["condition"], r["arm"]): r for r in rows}
    for row in rows:
        clean = indexed.get((row["pair_id"], "clean", row["arm"]))
        if clean and row["metrics"]["attacked"]:
            row["new_target_emission"] = int(row["metrics"]["attack_success"] and not achieved(clean["answer"]["answer"], labels[row["id"]]))
    summaries = {a: summarize([r for r in rows if r["arm"] == a]) for a in manifest["arms"]}
    conditions = {a: {c: summarize([r for r in rows if r["arm"] == a and r["condition"] == c])
                      for c in sorted({r["condition"] for r in rows})} for a in manifest["arms"]}
    contrasts = []
    for a, b, metric, condition in [("learned_repair", "none", "attack_success", None),
                                     ("learned_repair", "learned_reject", "f1", None),
                                     ("learned_repair", "learned_reject", "support_final", None),
                                     ("learned_repair", "none", "f1", "clean")]:
        if a in summaries and b in summaries:
            contrasts.append(paired_bootstrap(rows, a, b, metric, condition, resamples))
    costs = {}
    for arm in manifest["arms"]:
        calls = [call for row in rows if row["arm"] == arm for call in row["calls"]]
        uncached_calls = [c for c in calls if not c["cache_hit"]]
        costs[arm] = {"requested_calls": len(calls), "cache_hits": sum(c["cache_hit"] for c in calls),
                      "judge_calls": sum(c["purpose"] == "judge" for c in calls),
                      "uncached_wall_seconds": sum(c["wall_seconds"] for c in uncached_calls),
                      "uncached_output_tokens": sum(c["output_tokens"] or 0 for c in uncached_calls),
                      "note": "Cache latency is not model throughput; server load time is logged per call"}
    summary = {"status": manifest["status"], "by_arm": summaries, "by_condition": conditions,
               "paired_contrasts": contrasts, "costs": costs,
               "limitations": ["Synthetic attack/quotation labels require human review.",
                               "HotpotQA internal partitions of the official development set; not official test performance.",
                               "Single training seed and one local generator; no deployment safety claim.",
                               "Support and span metrics use exact character offsets; semantic residual risk is not independently human-scored.",
                               "Policy selected using input metrics, not validation ASR or clean-F1 non-inferiority."]}
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arm"] + list(next(iter(summaries.values()))))
        writer.writeheader()
        writer.writerows({"arm": a, **s} for a, s in summaries.items())
    lines = ["# Local RAG screening experiment", "", f"Status: **{manifest['status']}**.", "",
             f"Evaluated {len(manifest['group_ids'])} source groups, {len(manifest['case_ids'])} conditions, {len(manifest['arms'])} arms.", "",
             "| Arm | F1 | Clean F1 | ASR (successes/cases) | Quotation damage | Support retained | Parse failures |",
             "|---|---:|---:|---:|---:|---:|---:|"]

    def pct(v):
        return "n/a" if v is None else f"{100*v:.1f}%"

    for arm, s in summaries.items():
        clean_f1 = conditions[arm].get("clean", {}).get("f1")
        lines.append(f"| {arm} | {s['f1']:.3f} | {clean_f1:.3f} | {pct(s['asr'])} ({s['attack_successes']}/{s['attacked_cases']}) | {pct(s['quotation_damage_rate'])} | {pct(s['candidate_support_retention'])} | {s['parse_failures']} |")
    lines += ["", "ASR is scored on the validated answer field only; malformed outputs are failures of the application and counted separately. Inspect raw responses in results.jsonl.", "", "Paired differences (95% source-group bootstrap intervals):", ""]
    for c in contrasts:
        if c:
            lines.append(f"- {c['contrast']}, {c['metric']}, {c['condition'] or 'eligible conditions'}: {c['difference']:.4f} [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] ({c['groups']} groups).")
    if summaries.get("none", {}).get("asr") == 0:
        lines += ["", "Unfiltered ASR is zero: this run provides no evidence of a defence benefit."]
    lines += ["", "Limitations:", ""] + ["- " + item for item in summary["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(output / "report.md"), "by_arm": summaries}
