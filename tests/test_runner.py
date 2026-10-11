import json
from pathlib import Path

import pytest

from rag_screen.common import file_hash, read_jsonl, write_json, write_jsonl
from rag_screen import experiment


class CleanDetector:
    def scan(self, question, text):
        return {"risk": .01, "spans": [[0, len(text), .01]]}


class FakeOllama:
    count = 0
    def __init__(self, *args, **kwargs):
        self.identity = {"model": "test", "digest": "fixed"}
        self.calls = []

    def generate(self, prompt, schema, budget):
        assert "GOLD_ONLY_NEVER_IN_INFERENCE" not in prompt
        type(self).count += 1
        info = {"cache_hit": False, "purpose": "answer", "wall_seconds": .1, "output_tokens": 4}
        self.calls.append(info)
        return json.dumps({"answer": "Paris", "citations": ["d0"], "abstain": False}), info


def fixture_data(tmp_path):
    data, checkpoint, output = [tmp_path / name for name in ["data", "checkpoint", "run"]]
    case = {"id": "q:clean", "pair_id": "q", "group": "g", "condition": "clean", "split": "pilot",
            "question": "Capital?", "documents": [{"id": "d0", "title": "France", "text": "Paris is the capital.", "rank": 1}]}
    label = {"id": "q:clean", "answer": "GOLD_ONLY_NEVER_IN_INFERENCE", "support": [],
             "attack_spans": {}, "quotation_spans": {}, "target": None, "objective": None}
    write_jsonl(data / "pilot.cases.jsonl", [case])
    write_jsonl(data / "pilot.labels.jsonl", [label])
    write_json(data / "manifest.json", {"files": {p.name: file_hash(p) for p in data.glob("*.jsonl")}})
    checkpoint.mkdir()
    (checkpoint / "detector.pkl").write_bytes(b"fixture")
    write_json(checkpoint / "policy.json", {"span_threshold": .5, "document_threshold": .5,
        "detector_sha256": file_hash(checkpoint / "detector.pkl"), "data_manifest_sha256": file_hash(data / "manifest.json")})
    return data, checkpoint, output


def test_runner_resume_hash_checks_and_gold_boundary(tmp_path, monkeypatch):
    data, checkpoint, output = fixture_data(tmp_path)
    monkeypatch.setattr(experiment, "Ollama", FakeOllama)
    monkeypatch.setattr(experiment.Detector, "load", lambda _: CleanDetector())
    FakeOllama.count = 0
    kwargs = dict(data=data, checkpoint=checkpoint, tokenizer="data/raw/qwen_tokenizer.json", output=output,
                  groups=1, arms=["none"], cache=tmp_path / "cache.sqlite")
    experiment.run(**kwargs)
    assert FakeOllama.count == 1
    first = (output / "results.jsonl").read_bytes()
    experiment.run(**kwargs)
    assert FakeOllama.count == 1
    assert (output / "results.jsonl").read_bytes() == first
    assert read_jsonl(output / "results.jsonl")[0]["metrics"]["f1"] == 0
    with pytest.raises(ValueError, match="Run manifest changed"):
        experiment.run(**{**kwargs, "arms": ["none", "lexical_reject"]})
    (data / "pilot.labels.jsonl").write_text('{}\n')
    with pytest.raises((ValueError, KeyError)):
        experiment.run(**kwargs)


def test_reserved_test_requires_review_before_model_calls(tmp_path):
    data, checkpoint, output = fixture_data(tmp_path)
    for kind in ["cases", "labels"]:
        (data / f"test.{kind}.jsonl").write_bytes((data / f"pilot.{kind}.jsonl").read_bytes())
    with pytest.raises(ValueError, match="two independent"):
        experiment.run(data, checkpoint, "data/raw/qwen_tokenizer.json", output, split="test")
