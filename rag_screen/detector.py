"""CPU-friendly contextual segment classifier; not a pretrained encoder."""
from __future__ import annotations

from functools import lru_cache
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .common import segments, words


def features(record):
    question, text, a, b = record
    focus = words(text[a:b])
    out = ["focus=" + t for t in focus]
    out += ["bigram=" + x + "_" + y for x, y in zip(focus, focus[1:])]
    # Learned contextual cues, rather than hardcoded quotation exemptions.
    for prefix, region in [("before=", text[max(0, a - 240):a]),
                           ("after=", text[b:b + 160])]:
        tokens = words(region)
        out += [prefix + t for t in tokens]
        out += [prefix + x + "_" + y for x, y in zip(tokens, tokens[1:])]
    q = set(words(question))
    out += ["question_overlap=" + t for t in focus if t in q]
    return out or ["empty"]


def examples(cases, labels, deduplicate=False):
    records, targets, doc_ends, doc_targets = [], [], [], []
    seen = set()
    for case in cases:
        label = labels[case["id"]]
        for doc in case["documents"]:
            attack = label["attack_spans"].get(doc["id"], [])
            for a, b in segments(doc["text"]):
                record = (case["question"], doc["text"], a, b)
                y = int(any(a < d and c < b for c, d in attack))
                key = (record, y)
                if deduplicate and key in seen:
                    continue
                seen.add(key)
                records.append(record)
                targets.append(y)
            doc_ends.append(len(records))
            doc_targets.append(int(bool(attack)))
    return records, np.asarray(targets), doc_ends, np.asarray(doc_targets)


def reliability(probabilities, labels):
    p, y = np.asarray(probabilities), np.asarray(labels)
    bins = []
    ece = 0.0
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        n = int(mask.sum())
        row = {"lower": i / 10, "upper": (i + 1) / 10, "n": n}
        if n:
            confidence, frequency = float(p[mask].mean()), float(y[mask].mean())
            row.update(confidence=confidence, frequency=frequency)
            ece += n / len(p) * abs(confidence - frequency)
        bins.append(row)
    return {"n": len(p), "brier": float(np.mean((p - y) ** 2)), "ece": ece, "bins": bins}


class Detector:
    def __init__(self, seed=42):
        self.vectorizer = TfidfVectorizer(analyzer=features, max_features=80000, sublinear_tf=True)
        self.classifier = LogisticRegression(C=4.0, class_weight="balanced", max_iter=400, random_state=seed)
        self.segment_calibrator = LogisticRegression(C=1.0, max_iter=200)
        self.document_calibrator = LogisticRegression(C=1.0, max_iter=200)
        self.seed = seed

    def fit(self, train, train_labels, calibration, calibration_labels):
        records, y, _, _ = examples(train, train_labels, deduplicate=True)
        if len(set(y)) != 2:
            raise ValueError("Training requires both attack and benign labels")
        self.classifier.fit(self.vectorizer.fit_transform(records), y)
        cal, cy, ends, dy = examples(calibration, calibration_labels)
        logits = self.classifier.decision_function(self.vectorizer.transform(cal))
        self.segment_calibrator.fit(logits.reshape(-1, 1), cy)
        maxima, start = [], 0
        for end in ends:
            maxima.append(float(max(logits[start:end], default=-30)))
            start = end
        self.document_calibrator.fit(np.array(maxima).reshape(-1, 1), dy)
        return {"training_segments": len(y), "positive_training_segments": int(y.sum()),
                "features": len(self.vectorizer.vocabulary_), "seed": self.seed,
                "calibration_fit_diagnostics_not_test": {
                    "segment": reliability(self.segment_calibrator.predict_proba(logits.reshape(-1, 1))[:, 1], cy),
                    "document": reliability(self.document_calibrator.predict_proba(np.array(maxima).reshape(-1, 1))[:, 1], dy)}}

    @lru_cache(maxsize=30000)
    def scan(self, question, text):
        spans = segments(text)
        if not spans:
            return {"risk": 0.0, "spans": []}
        records = [(question, text, a, b) for a, b in spans]
        logits = self.classifier.decision_function(self.vectorizer.transform(records))
        probabilities = self.segment_calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        risk = self.document_calibrator.predict_proba([[float(max(logits))]])[0, 1]
        return {"risk": float(risk), "spans": [[a, b, float(p)] for (a, b), p in zip(spans, probabilities)]}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        # Load only locally generated, trusted checkpoints. Pickle executes code.
        with open(path, "rb") as f:
            return pickle.load(f)
