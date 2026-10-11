"""Read-only post-run integrity and evidence-map audit, independent of generation."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_screen.common import digest, file_hash, read_jsonl, write_json
from rag_screen.metrics import index_set
from rag_screen.ollama import answer_prompt, PromptBudget
from rag_study.evaluation import score
from rag_study.study import verify_rows
from rag_study.training import load_split


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    completion = json.loads((output / "completion.json").read_text())
    cases, labels = load_split("data/followup", manifest["split"])
    cases = {c["id"]: c for c in cases}
    expected = {(cid, arm) for cid in manifest["case_ids"] for arm in manifest["arms"]}
    rows = read_jsonl(output / "results.jsonl")
    checked = verify_rows(rows, expected, digest(manifest))
    assert checked == expected
    assert completion["results_sha256"] == file_hash(output / "results.jsonl")
    assert completion["run_hash"] == digest(manifest)
    inventory = json.loads((output / "snapshot/inventory.json").read_text())
    for name, entry in inventory.items():
        assert file_hash(output / "snapshot" / name) == entry["sha256"]
        assert manifest["files"][entry["original"]] == entry["sha256"]
    budget = PromptBudget("data/raw/qwen_tokenizer.json")
    count = Counter()
    for row in rows:
        case, label = cases[row["id"]], labels[row["id"]]
        docs = {d["id"]: d for d in case["documents"]}
        audit = {d["id"]: d for d in row["audit"]}
        assert len(audit) == len(row["audit"]) == len(docs)
        for selected in row["selected"]:
            original = docs[selected["id"]]["text"]
            a = audit[selected["id"]]
            assert not a["quarantined"]
            reconstructed = "".join(original[i] for i in sorted(index_set(a["retained_original_ranges"])))
            assert selected["text"] == reconstructed == a["repaired_text"]
            count["selected_passage_maps"] += 1
        prompt = answer_prompt(case["question"], row["selected"])
        assert row["prompt"] == prompt and row["prompt_sha256"] == digest(prompt)
        assert row["prompt_tokens"] == budget.count(prompt) <= 6000
        assert row["metrics"] == score(case, label, row)
        count["recomputed_rows"] += 1
    result = {"status": "passed", "expected_outcomes": len(expected), "source_groups": len(manifest["groups"]),
              "snapshot_files": len(inventory), **count, "results_sha256": completion["results_sha256"],
              "verifier_sha256": file_hash(__file__), "scope": "Mechanical integrity, exact retained characters, prompts and metric reproduction; not independent human semantic review."}
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify(sys.argv[1] if len(sys.argv) > 1 else "runs/followup_test")
