"""Small learned models and matched baselines; all inference is label-free."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from rag_screen.common import compact_ranges, delete_ranges, file_hash, merge_ranges, segments, words
from rag_screen.defense import defend as original_defend
from rag_screen.detector import Detector
from .data import PREVIOUS

ARMS = ("none", "keyword_repair", "old_current_repair", "previous_v6", "learned_reject", "learned_repair")


def units(text, context=True):
    spans = segments(text)
    records = []
    for i, (a, b) in enumerate(spans):
        left = text[spans[max(0, i-2)][0]:a][-500:] if context else ""
        right = text[b:spans[min(len(spans)-1, i+2)][1]][:500] if context else ""
        records.append((text[a:b], left, right))
    return spans, records


def focus(records):
    return [r[0] for r in records]


def lexical(record):
    out = []
    for prefix, text in zip(("focus=", "left=", "right="), record):
        tokens = words(text)
        out.extend(prefix + t for t in tokens)
        out.extend(prefix + a + "_" + b for a, b in zip(tokens, tokens[1:]))
    return out or ["empty"]


def training_units(cases, labels, context):
    examples = {}
    for case in cases:
        label = labels[case["id"]]
        for doc in case["documents"]:
            # Deduplicate using identical contextual examples for BOTH models.
            # Project away neighbors only afterwards so the ablation preserves
            # every training observation and its weight.
            spans, records = units(doc["text"], True)
            attack = label["attack_spans"].get(doc["id"], [])
            for (a, b), record in zip(spans, records):
                target = int(any(a < d and c < b for c, d in attack))
                if record in examples and examples[record] != target:
                    # A focus-only ablation necessarily has contradictory paired labels.
                    # Retain both observations, not an arbitrary majority decision.
                    examples[(record, target)] = target
                else:
                    examples[record] = target
    records, labels_out = [], []
    for key, label in examples.items():
        record = key[0] if len(key) == 2 and isinstance(key[0], tuple) else key
        records.append(record if context else (record[0], "", ""))
        labels_out.append(label)
    return records, np.asarray(labels_out)


class ContextDetector:
    def __init__(self, context=True):
        self.context = context
        self.vectorizer = FeatureUnion([
            ("words", TfidfVectorizer(analyzer=lexical, max_features=45000, sublinear_tf=True)),
            ("characters", Pipeline([("focus", FunctionTransformer(focus)),
                                      ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=25000, sublinear_tf=True))]))])
        self.classifier = LogisticRegression(C=1, class_weight="balanced", max_iter=800, random_state=42)
        self.calibrator = LogisticRegression(C=1, max_iter=300)

    def fit(self, training, training_labels, calibration, calibration_labels):
        x, y = training_units(training, training_labels, self.context)
        cx, cy = training_units(calibration, calibration_labels, self.context)
        self.classifier.fit(self.vectorizer.fit_transform(x), y)
        logits = self.classifier.decision_function(self.vectorizer.transform(cx))
        self.calibrator.fit(logits.reshape(-1, 1), cy)
        return {"context": self.context, "training_units": len(x), "positive_units": int(y.sum()),
                "calibration_units": len(cx), "calibration_positive_units": int(cy.sum()),
                "converged": int(max(self.classifier.n_iter_)) < self.classifier.max_iter}

    @lru_cache(maxsize=20000)
    def scan(self, question, text):
        spans, x = units(text, self.context)
        if not spans:
            return {"risk": 0, "spans": []}
        logits = self.classifier.decision_function(self.vectorizer.transform(x))
        p = self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        # This is a max-span score, not an independently calibrated document probability.
        return {"risk": float(max(p)), "spans": [[a, b, float(v)] for (a, b), v in zip(spans, p)]}


# A general directive rule fixed before development. No attack target or exact
# template strings are used, and there is deliberately no quotation allowlist.
KEYWORD = re.compile(
    r"\b(ignore|disregard|override|abandon|replace|substitute|append|suffix)\b|"
    r"\b(answer|respond|reply|return|write|state|give|supply)\b.{0,70}\b(only|exact|just|instead|nothing else)\b|"
    r"\b(assistant|model|user|administrator|supervisor|rubric|instruction|directive|response rule)\b.{0,90}\b(must|should|requires?|directs?|priority|latest|answer|reply|response|state|add|ensure)\b|"
    r"\b(add|include|insert|attach|ensure|finish|end)\b.{0,70}\b(answer|response|reply|tag|literal|string|text)\b",
    re.I | re.S)


def local_repair(question, documents, scanner, threshold, whole=False):
    kept, audits = [], []
    for original in documents:
        text = original["text"]
        mapping = list(range(len(text)))
        passes, reason = [], None
        for i in range(3):
            scan = scanner(question, text)
            ranges = merge_ranges([[a, b] for a, b, p in scan["spans"] if p >= threshold], len(text))
            passes.append({"pass": i, "max_span_score": scan["risk"], "scores": scan["spans"],
                           "proposed_original_ranges": compact_ranges(mapping[k] for a, b in ranges for k in range(a, b))})
            if not ranges:
                break
            if whole:
                reason = "whole_passage_rejection"
                break
            if i == 2:
                reason = "residual_risk_after_two_deletions"
                break
            text, mapping = delete_ranges(text, ranges, mapping)
        removed = compact_ranges(set(range(len(original["text"]))) - set(mapping))
        if not text.strip():
            reason = "empty_repair"
        audits.append({"id": original["id"], "rank": original["rank"], "quarantined": bool(reason),
                       "reason": reason, "deleted_ranges": removed,
                       "retained_original_ranges": [] if reason else compact_ranges(mapping),
                       "passes": passes, "repaired_text": text})
        if not reason:
            kept.append({**original, "text": text})
    return kept, audits


def keyword_scan(question, text):
    scored = [[a, b, float(bool(KEYWORD.search(text[a:b])))] for a, b in segments(text)]
    return {"risk": max([p for _, _, p in scored], default=0), "spans": scored}


class PreviousV6:
    def __init__(self):
        # Read existing research artifacts only; do not create upstream bytecode.
        sys.dont_write_bytecode = True
        root = PREVIOUS / "experiment_v6"
        sys.path.insert(0, str(root))
        from v6_models import Defence
        from v6_reference import load_scorer
        run = root / "runs/span_context200"
        metadata = json.loads((run / "training.json").read_text(encoding="utf-8"))
        checkpoint = run / "contextual.joblib"
        if file_hash(checkpoint) != metadata["model_sha256"]:
            raise ValueError("Previous V6 checkpoint hash mismatch")
        old, fingerprint, _ = load_scorer(metadata["source"]["source_run"])
        settings = json.loads((run / "thresholds.json").read_text(encoding="utf-8"))
        self.model = Defence(old, joblib.load(checkpoint), settings["thresholds"])
        self.identity = {"v6_checkpoint": str(checkpoint), "v6_sha256": file_hash(checkpoint),
                         "thresholds_sha256": file_hash(run / "thresholds.json"), "v4": fingerprint,
                         "source": {str(p): file_hash(p) for v in [4, 5, 6] for p in sorted((PREVIOUS / f"experiment_v{v}").glob("*.py"))}}

    @lru_cache(maxsize=10000)
    def filter(self, text):
        return self.model.filter_text(text, "contextual", .75, .05)

    def defend(self, documents):
        kept, audits = [], []
        for doc in documents:
            result = self.filter(doc["text"])
            retained = result["retained_ranges"]
            quarantine = "rejected" in result["status"]
            initial_removed = compact_ranges(i for span in result["spans"] if span["removed"] for i in range(span["start"], span["end"]))
            audits.append({"id": doc["id"], "rank": doc["rank"], "quarantined": quarantine,
                           "reason": result["status"] if quarantine else None,
                           "deleted_ranges": initial_removed, "retained_original_ranges": retained,
                           "passes": [{"previous_v6": result}], "repaired_text": result["text"]})
            if not quarantine:
                kept.append({**doc, "text": result["text"]})
        return kept, audits


class Methods:
    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        self.policy = json.loads((checkpoint / "policy.json").read_text(encoding="utf-8"))
        self.new = joblib.load(checkpoint / "context.joblib")
        self.focus = joblib.load(checkpoint / "focus.joblib")
        self.old = Detector.load("artifacts/detector_seed42/detector.pkl")
        self.old_policy = json.loads(Path("artifacts/detector_seed42/policy.json").read_text(encoding="utf-8"))
        self.previous = PreviousV6()

    def defend(self, question, documents, arm):
        if arm == "none":
            return original_defend(question, documents, "none")
        if arm == "old_current_repair":
            return original_defend(question, documents, "learned_repair", self.old, self.old_policy)
        if arm == "previous_v6":
            return self.previous.defend(documents)
        if arm == "keyword_repair":
            return local_repair(question, documents, keyword_scan, .5)
        if arm in {"learned_repair", "learned_reject"}:
            return local_repair(question, documents, self.new.scan, self.policy["context"]["threshold"], arm == "learned_reject")
        if arm == "focus_ablation":
            return local_repair(question, documents, self.focus.scan, self.policy["focus"]["threshold"])
        raise ValueError(arm)
