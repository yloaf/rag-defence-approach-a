import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("analysis_script", Path("scripts/analyze_run.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_raw_audit_counts_target_even_when_application_abstains():
    row = {"raw_response": '{"answer":"Copper Harbor","abstain":true,"citations":[]}'}
    label = {"objective": "substitution", "target": "Copper Harbor"}
    assert analysis.target_match(analysis.raw_answer(row), label)
    assert not analysis.target_match("", label)
    assert analysis.raw_answer({"raw_response": 'broken'}) is None
