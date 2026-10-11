from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(canonical(record) + "\n")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def words(text):
    return re.findall(r"\w+", text.lower())


def merge_ranges(ranges, length):
    merged = []
    for a, b in sorted(ranges):
        if not 0 <= a < b <= length:
            raise ValueError(f"Invalid character range {(a, b)} for length {length}")
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def delete_ranges(text, ranges, offsets=None):
    """Exact deletion with a reversible surviving-character -> original map."""
    offsets = list(range(len(text))) if offsets is None else list(offsets)
    if len(offsets) != len(text):
        raise ValueError("Text and offset map length disagree")
    keep = [True] * len(text)
    for a, b in merge_ranges(ranges, len(text)):
        keep[a:b] = [False] * (b - a)
    return ("".join(ch for ch, k in zip(text, keep) if k),
            [i for i, k in zip(offsets, keep) if k])


def compact_ranges(indices):
    result = []
    for i in sorted(set(indices)):
        if result and result[-1][1] == i:
            result[-1][1] += 1
        else:
            result.append([i, i + 1])
    return result


def segments(text, max_chars=600, overlap=100):
    """Sentence/line spans, splitting long units with overlapping word windows.

    Every non-whitespace character is scanned, including suffixes and long lines.
    This deliberately modest detector uses character spans, not BPE token labels.
    """
    if max_chars <= overlap or overlap < 0:
        raise ValueError("Require 0 <= overlap < max_chars")
    boundaries = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n+", text)] + [len(text)]
    result = []
    for left, right in zip(boundaries, boundaries[1:]):
        while left < right:
            end = min(left + max_chars, right)
            if end < right:
                space = text.rfind(" ", left + max_chars // 2, end)
                if space > left:
                    end = space + 1
                else:
                    # Never cut a word; exceptionally long words remain one unit.
                    while end < right and not text[end].isspace():
                        end += 1
            a, b = left, end
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1
            if a < b:
                result.append((a, b))
            if end == right:
                break
            nxt = max(left + 1, end - overlap)
            while nxt < end and not text[nxt - 1].isspace():
                nxt += 1
            left = nxt
    return result


def source_hash():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})
