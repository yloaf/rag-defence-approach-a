"""Frozen, resumable common-harness evaluation of the six methods."""
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import time

from rag_screen.common import canonical, digest, file_hash, write_json
from rag_screen.experiment import recover_results
from rag_screen.ollama import ANSWER_SCHEMA, Ollama, PromptBudget, parse_answer
from .evaluation import score
from .models import ARMS, Methods
from .training import load_split


def seal(row):
    return {**row, "record_sha256": digest(row)}


def verify_rows(rows, expected, run_hash):
    done = set()
    for row in rows:
        key = (row["id"], row["arm"])
        payload = {k: v for k, v in row.items() if k != "record_sha256"}
        if row.get("record_sha256") != digest(payload):
            raise ValueError("Result record checksum mismatch")
        if row["run_hash"] != run_hash or key not in expected or key in done:
            raise ValueError("Result identity mismatch or duplicate")
        done.add(key)
    return done


def run(split="development", output=None, data="data/followup", checkpoint="artifacts/followup_v2"):
    if split not in {"development", "test"}:
        raise ValueError("Generator evaluation is restricted to development or test")
    output = Path(output or ("runs/followup_development_v2" if split == "development" else "runs/followup_test"))
    data, checkpoint = Path(data), Path(checkpoint)
    cases, labels = load_split(data, split)
    policy = json.loads((checkpoint / "policy.json").read_text())
    if policy["data_manifest_sha256"] != file_hash(data / "manifest.json"):
        raise ValueError("Training data manifest changed")
    for name, expected in policy["checkpoints"].items():
        if file_hash(checkpoint / f"{name}.joblib") != expected:
            raise ValueError("Learned checkpoint changed")
    methods = Methods(checkpoint)
    if methods.old_policy["detector_sha256"] != file_hash("artifacts/detector_seed42/detector.pkl"):
        raise ValueError("Original current checkpoint changed")
    budget = PromptBudget("data/raw/qwen_tokenizer.json")
    client = Ollama("artifacts/followup/ollama.sqlite")
    paths = [p for folder in ["rag_study", "rag_screen"] for p in sorted(Path(folder).glob("*.py"))]
    paths += [Path("FOLLOWUP_PROTOCOL.md"), Path("requirements.lock.txt"), Path("pyproject.toml"),
              data / "manifest.json", data / f"{split}.cases.jsonl", data / f"{split}.labels.jsonl",
              Path("data/raw/qwen_tokenizer.json"), Path("artifacts/detector_seed42/detector.pkl"),
              Path("artifacts/detector_seed42/policy.json")]
    paths += sorted(checkpoint.glob("*.json")) + sorted(checkpoint.glob("*.joblib"))
    paths += [Path(p) for p in methods.previous.identity["source"]]
    previous_checkpoint = Path(methods.previous.identity["v6_checkpoint"])
    paths += [previous_checkpoint, previous_checkpoint.parent / "thresholds.json", previous_checkpoint.parent / "training.json"]
    # Include the V4 dependencies actually loaded by the previous V6 adapter.
    metadata = json.loads((previous_checkpoint.parent / "training.json").read_text())
    old_run = Path(metadata["source"]["source_run"])
    paths += sorted(old_run.glob("*.joblib")) + sorted(old_run.glob("*.json"))
    if split == "test":
        dev = Path("runs/followup_development_v2/completion.json")
        if not dev.exists():
            raise ValueError("Complete development before freezing final evaluation")
        paths.append(dev)
    files = {str(p.resolve()): file_hash(p) for p in paths}
    manifest = {"files": files, "split": split, "arms": list(ARMS), "case_ids": [c["id"] for c in cases],
                "groups": sorted({c["group"] for c in cases}), "previous_v6": methods.previous.identity,
                "ollama": client.identity, "python": sys.version, "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ["numpy", "scikit-learn", "tokenizers", "joblib"]},
                "input_limit": 6000, "output_limit": 128, "num_ctx": 8192, "temperature": 0, "seed": 42,
                "method_order_seed": 731, "human_reviewed": False,
                "status": "held-out synthetic evaluation" if split == "test" else "development diagnostic"}
    run_hash = digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Frozen manifest changed; unsafe resume refused")
    else:
        write_json(manifest_path, manifest)
        snapshot = output / "snapshot"
        snapshot.mkdir()
        inventory = {}
        for i, (path, expected) in enumerate(files.items()):
            target = snapshot / f"{i:03d}_{Path(path).name}"
            shutil.copyfile(path, target)
            inventory[target.name] = {"original": path, "sha256": expected}
        write_json(snapshot / "inventory.json", inventory)
    results = output / "results.jsonl"
    if (output / "completion.json").exists():
        completion = json.loads((output / "completion.json").read_text())
        if completion["results_sha256"] != file_hash(results):
            raise ValueError("Completed results changed")
    expected_keys = {(c["id"], a) for c in cases for a in ARMS}
    done = verify_rows(recover_results(results), expected_keys, run_hash)
    rng = random.Random(731)
    start = time.perf_counter()
    with results.open("a", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            arms = list(ARMS)
            rng.shuffle(arms)
            for arm in arms:
                if (case["id"], arm) in done:
                    continue
                tick, call_start = time.perf_counter(), len(client.calls)
                kept, audit = methods.defend(case["question"], case["documents"], arm)
                selected, prompt, skipped = budget.select(case["question"], kept)
                if selected:
                    raw, _ = client.generate(prompt, ANSWER_SCHEMA, budget)
                    answer, error = parse_answer(raw, {d["id"] for d in selected})
                else:
                    raw, error = "", None
                    answer = {"answer": "", "citations": [], "abstain": True}
                row = {"id": case["id"], "pair_id": case["pair_id"], "group": case["group"],
                       "condition": case["condition"], "arm": arm, "run_hash": run_hash,
                       "answer": answer, "raw_response": raw, "parse_error": error,
                       "audit": audit, "selected": selected, "skipped": skipped, "prompt": prompt,
                       "prompt_sha256": digest(prompt), "prompt_tokens": budget.count(prompt),
                       "calls": client.calls[call_start:], "wall_seconds": time.perf_counter()-tick}
                # Labels are supplied only to the evaluator after inference.
                row["metrics"] = score(case, labels[case["id"]], row)
                handle.write(canonical(seal(row)) + "\n")
                handle.flush()
                done.add((case["id"], arm))
                if len(done) % 12 == 0:
                    print(f"{split}: {len(done)}/{len(expected_keys)} outcomes; {time.perf_counter()-start:.0f}s this session", flush=True)
    identity_end = Ollama("artifacts/followup/ollama.sqlite").identity
    if identity_end != manifest["ollama"]:
        raise ValueError("Ollama identity changed during evaluation")
    if any(file_hash(p) != h for p, h in files.items()):
        raise ValueError("Frozen file changed during evaluation")
    if len(done) != len(expected_keys):
        raise ValueError("Incomplete evaluation")
    completion_path = output / "completion.json"
    if not completion_path.exists():
        write_json(completion_path, {"run_hash": run_hash, "results_sha256": file_hash(results), "outcomes": len(done),
                                   "groups": len(manifest["groups"]), "model_identity_verified": True})
    from .analysis import report
    return report(output)
