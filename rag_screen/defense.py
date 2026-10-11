"""Defence modules receive questions/text only, never evaluator labels."""
from __future__ import annotations

from copy import deepcopy
import json
import re

from .common import canonical, compact_ranges, delete_ranges, merge_ranges, segments
from .ollama import JUDGE_SCHEMA, JUDGE_SYSTEM, qwen_prompt

ARMS = ("none", "lexical_reject", "learned_reject", "learned_repair", "judge", "combined")
LEXICAL = re.compile(r"ignore.{0,60}(user|question)|override|assistant instruction|system message|answer field|USER UPDATE", re.I | re.S)


def judge_windows(text, budget, focus_tokens=1024):
    spans = segments(text, max_chars=1200, overlap=160)
    windows = []
    start = 0
    while start < len(spans):
        end = start
        entries = []
        while end < len(spans):
            a, b = spans[end]
            entry = {"segment_id": end, "text": text[a:b]}
            if budget.count(canonical(entries + [entry])) > focus_tokens:
                if not entries:
                    raise ValueError("One judge segment exceeds focus-token budget")
                break
            entries.append(entry)
            end += 1
        windows.append((entries, {i: spans[i] for i in range(start, end)}))
        if end == len(spans):
            break
        start = max(start + 1, end - 1)  # overlap a complete segment
    return windows


def locate_judge(raw, text, span_map):
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {"verdict", "quotes", "reason"}:
        raise ValueError("Invalid judge object")
    verdict = result["verdict"]
    if verdict not in {"benign", "injection", "uncertain"} or not isinstance(result["reason"], str) or not isinstance(result["quotes"], list):
        raise ValueError("Invalid judge schema")
    if verdict == "uncertain":
        raise ValueError("Judge uncertain")
    if verdict == "benign" and result["quotes"]:
        raise ValueError("Benign verdict with offending quotes")
    if verdict == "injection" and not result["quotes"]:
        raise ValueError("Injection without localisation")
    ranges = []
    for item in result["quotes"]:
        if not isinstance(item, dict) or set(item) != {"segment_id", "quote"}:
            raise ValueError("Invalid quote schema")
        sid, quote = item["segment_id"], item["quote"]
        if type(sid) is not int or sid not in span_map or not isinstance(quote, str) or not quote.strip():
            raise ValueError("Invalid segment or empty quote")
        a, b = span_map[sid]
        region = text[a:b]
        occurrences = [m.start() for m in re.finditer(r"(?=" + re.escape(quote) + r")", region)]
        if len(occurrences) != 1:
            raise ValueError("Quote absent or ambiguous")
        left = a + occurrences[0]
        right = left + len(quote)
        while left > a and text[left - 1].isalnum():
            left -= 1
        while right < b and text[right].isalnum():
            right += 1
        ranges.append([left, right])
    return merge_ranges(ranges, len(text)), result


class Judge:
    def __init__(self, ollama, budget):
        self.ollama, self.budget = ollama, budget

    def scan(self, question, doc_id, text):
        ranges, logs = [], []
        for entries, mapping in judge_windows(text, self.budget):
            prompt = qwen_prompt(JUDGE_SYSTEM, canonical({"question": question, "passage_id": doc_id, "segments": entries}))
            raw, info = self.ollama.generate(prompt, JUDGE_SCHEMA, self.budget, "judge", 768)
            try:
                located, result = locate_judge(raw, text, mapping)
                ranges.extend(located)
                logs.append({"result": result, "call": info})
            except (ValueError, KeyError, TypeError) as exc:
                logs.append({"error": str(exc), "raw": raw, "call": info})
                return [], "judge_invalid_or_uncertain", logs
        return merge_ranges(ranges, len(text)), None, logs


def defend(question, documents, arm, detector=None, policy=None, judge=None):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if arm in {"learned_repair", "learned_reject", "combined"} and (detector is None or policy is None):
        raise ValueError("Learned arms require a frozen detector and policy")
    if arm in {"judge", "combined"} and judge is None:
        raise ValueError("Judge arm requires a judge")
    kept, audits = [], []
    for original in documents:
        doc = deepcopy(original)
        text = doc["text"]
        offsets = list(range(len(text)))
        passes = []
        reason = None
        if arm == "lexical_reject" and LEXICAL.search(text):
            reason = "lexical_rule"
        elif arm not in {"none", "lexical_reject"}:
            for iteration in range(3):  # two deletion passes, then residual check
                ranges, detection = [], {"pass": iteration}
                if arm in {"learned_repair", "learned_reject", "combined"}:
                    scan = detector.scan(question, text)
                    detection["learned"] = scan
                    ranges.extend([[a, b] for a, b, p in scan["spans"] if p >= policy["span_threshold"]])
                    high_risk = scan["risk"] >= policy["document_threshold"]
                else:
                    high_risk = False
                # Combined arm always judges, independent of the learned risk.
                if arm in {"judge", "combined"}:
                    judge_ranges, failure, logs = judge.scan(question, doc["id"], text)
                    ranges.extend(judge_ranges)
                    detection["judge"] = logs
                    if failure:
                        reason = failure
                ranges = merge_ranges(ranges, len(text))
                detection["proposed_original_ranges"] = compact_ranges(offsets[i] for a, b in ranges for i in range(a, b))
                passes.append(detection)
                if reason:
                    break
                if high_risk and not ranges:
                    reason = "high_risk_without_localisation"
                    break
                if not ranges:
                    break
                if arm == "learned_reject":
                    reason = "learned_whole_passage_rejection"
                    break
                if iteration == 2:
                    reason = "residual_risk_after_two_passes"
                    break
                text, offsets = delete_ranges(text, ranges, offsets)
        deleted = compact_ranges(set(range(len(original["text"]))) - set(offsets))
        retained = [] if reason else compact_ranges(offsets)
        audits.append({"id": doc["id"], "rank": doc["rank"], "quarantined": bool(reason),
                       "reason": reason, "deleted_ranges": deleted,
                       "retained_original_ranges": retained, "passes": passes, "repaired_text": text})
        if not reason:
            doc["text"] = text
            kept.append(doc)
    return kept, audits
