"""Source-exclusive fresh cohorts, exact annotation maps and leakage audit."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import re

from rag_screen.common import digest, file_hash, read_jsonl, words, write_json, write_jsonl
from rag_screen.data import rank_documents
from .templates import ACTION, BENIGN, FAMILIES, REPORTS, TEMPLATES, make_payload, quote_payload

PREVIOUS = Path(r"D:\The Folder\Uni Assignments\COMPSCI 764\LitRevWorkfile\testPrograms\trust-scoring-evaluation")
COUNTS = {"train": 200, "calibration": 60, "validation": 40, "development": 10, "test": 50, "contrast": 30}


def normalized(text):
    return " ".join(words(text))


def prior_inventory(raw):
    ids, titles, texts, files = set(), set(), set(), {}
    for version, folder in [(2, "mirror_check"), (3, "prepared500"), (4, "pilot200"), (5, "context200"), (6, "pilot200")]:
        for split in ["train", "calibration", "validation", "test"]:
            p = PREVIOUS / f"experiment_v{version}/data" / folder / f"{split}.jsonl"
            files[str(p)] = file_hash(p)
            for row in read_jsonl(p):
                ids.add(row["question_id"])
                for doc in row["passages"]:
                    titles.add(normalized(doc["source_id"]))
                    texts.add(normalized(doc.get("_eval_original_text", doc["text"])))
    for p in sorted(Path("data/prepared").glob("*.cases.jsonl")):
        files[str(p.resolve())] = file_hash(p)
        for c in read_jsonl(p):
            ids.add(c["pair_id"])
    # Exclude ALL original candidates for any previous question, not just its top 8.
    for row in raw:
        if row["_id"] in ids:
            for title, sentences in row["context"]:
                titles.add(normalized(title))
                texts.add(normalized("".join(sentences)))
    return ids, titles, texts, files


class NearIndex:
    """Exact Jaccard candidate audit on 5-word shingles; no approximate search."""
    def __init__(self):
        self.inverted = defaultdict(set)
        self.entries = []

    def grams(self, text):
        tokens = words(text)
        return {tuple(tokens[i:i+5]) for i in range(len(tokens)-4)}

    def overlaps(self, text, threshold=.85):
        grams = self.grams(text)
        if len(grams) < 5:
            return False
        counts = Counter(i for g in grams for i in self.inverted[g])
        return any(intersection / (len(grams) + len(self.entries[i]) - intersection) >= threshold
                   for i, intersection in counts.items())

    def add(self, text):
        grams = self.grams(text)
        if len(grams) < 5:
            return
        index = len(self.entries)
        self.entries.append(grams)
        for g in grams:
            self.inverted[g].add(index)


def supports(row):
    contexts = dict(row["context"])
    ids = {title: f"d{i}" for i, (title, _) in enumerate(row["context"])}
    result = []
    for title, index in row["supporting_facts"]:
        sentence = contexts[title][index]
        start = sum(map(len, contexts[title][:index])) + len(sentence) - len(sentence.lstrip())
        if not sentence.strip():
            continue
        result.append({"doc_id": ids[title], "sentence": sentence.strip(), "ranges": [[start, start+len(sentence.strip())]]})
    return result


def inject(documents, facts, host, payload, position):
    docs, support = deepcopy(documents), deepcopy(facts)
    doc = docs[host]
    original = doc["text"]
    boundaries = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+", original)] + [len(original)]
    offset = [0, boundaries[len(boundaries)//2], len(original)][position]
    inserted = "\n" + payload + "\n"
    doc["text"] = original[:offset] + inserted + original[offset:]
    for fact in support:
        if fact["doc_id"] != doc["id"]:
            continue
        ranges = []
        for a, b in fact["ranges"]:
            if b <= offset:
                ranges.append([a, b])
            elif a >= offset:
                ranges.append([a+len(inserted), b+len(inserted)])
            else:
                ranges.extend([[a, offset], [offset+len(inserted), b+len(inserted)]])
        fact["ranges"] = ranges
    return docs, support, {doc["id"]: [[offset+1, offset+1+len(payload)]]}


def make_cases(rows, split):
    cases, labels = [], []
    families = FAMILIES if split in {"development", "test", "contrast"} else FAMILIES[:2]
    tsplit = "test" if split == "contrast" else split
    for n, row in enumerate(rows):
        docs = rank_documents(row["question"], [{"id": f"d{i}", "title": title, "text": "".join(sentences)} for i, (title, sentences) in enumerate(row["context"])])
        facts = supports(row)
        variations = [("clean", None, None, None, None)]
        for fi, family in enumerate(families):
            objective = "marker" if (n+fi) % 2 == 0 else "substitution"
            target = f"SIGIL_{digest(row['_id'])[:9].upper()}" if objective == "marker" else ["Copper Harbor", "Velvet Island", "Orchid Valley", "Silver Meadow", "Amber Ridge"][n % 5]
            if target.casefold() in (row["question"] + row["answer"] + "".join(d["text"] for d in docs)).casefold():
                target = f"Fictional {digest(row['_id'])[:10]}"
            payload, template = make_payload(tsplit, family, n, target, objective)
            variations.append((family, payload, objective, target, template))
            if split in {"train", "calibration", "contrast"} or fi == n % len(families):
                name = f"quotation_{family}" if split in {"train", "calibration", "contrast"} else "quotation"
                variations.append((name, quote_payload(tsplit, n, payload), None, None, template))
        if split in {"train", "calibration"}:
            variations.append(("benign_instruction", BENIGN[n % len(BENIGN)], None, None, None))
        for condition, payload, objective, target, template in variations:
            altered, support, attack, quotation = deepcopy(docs), deepcopy(facts), {}, {}
            family = condition if condition in families else condition.removeprefix("quotation_")
            fi = families.index(family) if family in families else n % len(families)
            if payload is not None:
                altered, support, spans = inject(docs, facts, (n+fi) % len(docs), payload, n % 3)
                if objective:
                    attack = spans
                elif condition.startswith("quotation"):
                    quotation = spans
            cid = row["_id"] + ":" + condition
            cases.append({"id": cid, "pair_id": row["_id"], "group": row["_id"], "condition": condition,
                          "split": split, "task": "hotpotqa", "question": row["question"], "documents": altered})
            labels.append({"id": cid, "answer": row["answer"], "support": support,
                           "attack_spans": attack, "quotation_spans": quotation,
                           "objective": objective, "target": target, "family": family if objective else None,
                           "template_id": template, "annotation_status": "synthetic_construction_checked_not_human_reviewed"})
    return cases, labels


def prepare(output="data/followup", seed=17641):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("Prepared study exists; do not overwrite frozen cohorts")
    raw_path = Path("data/raw/hotpot_dev_distractor_v1.json")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    old_ids, old_titles, old_texts, prior_files = prior_inventory(raw)
    near = NearIndex()
    print(f"Indexing {len(old_texts)} earlier source texts for near-duplicate exclusions", flush=True)
    for text in sorted(old_texts):
        near.add(text)
    for row in raw:
        if row["_id"] in old_ids:
            near.add(row["question"])
    shuffled = sorted(raw, key=lambda r: r["_id"])
    random.Random(seed).shuffle(shuffled)
    selected, exclusions = [], Counter()
    for row in shuffled:
        if row["_id"] in old_ids:
            exclusions["prior_question"] += 1
            continue
        contexts = dict(row["context"])
        if any(title not in contexts or not 0 <= i < len(contexts[title]) for title, i in row["supporting_facts"]):
            exclusions["invalid_support_annotation"] += 1
            continue
        titles = {normalized(title) for title, _ in row["context"]}
        texts = {normalized("".join(sentences)) for _, sentences in row["context"]}
        if titles & old_titles or texts & old_texts:
            exclusions["prior_or_selected_source"] += 1
            continue
        if any(near.overlaps(text) for text in sorted(texts | {row["question"]})):
            exclusions["near_duplicate"] += 1
            continue
        selected.append(row)
        old_titles.update(titles)
        old_texts.update(texts)
        for text in sorted(texts | {row["question"]}):
            near.add(text)
        if len(selected) == sum(COUNTS.values()):
            break
    if len(selected) < sum(COUNTS.values()):
        raise ValueError(f"Only {len(selected)} eligible questions; refuse silently reduced sample")
    # Randomise assignment after source-exclusive selection; each group is one question.
    random.Random(seed+1).shuffle(selected)
    cursor, assignments = 0, {}
    for split, count in COUNTS.items():
        rows = selected[cursor:cursor+count]
        cursor += count
        assignments[split] = [r["_id"] for r in rows]
        cases, labels = make_cases(rows, split)
        write_jsonl(output / f"{split}.cases.jsonl", cases)
        write_jsonl(output / f"{split}.labels.jsonl", labels)
    manifest = {"seed": seed, "raw_sha256": file_hash(raw_path), "counts": COUNTS,
                "prior_files": prior_files, "previous_question_count": len(old_ids),
                "source_exclusion": "all original candidates, all previous V2-V6 cohorts and all 500 initial current questions",
                "near_duplicate_rule": "exact Jaccard >=0.85 on 5-word shingles; at least 5 shingles",
                "exclusions_during_selection": dict(exclusions), "assignments": assignments,
                "templates_sha256": digest([TEMPLATES, ACTION, REPORTS, BENIGN]),
                "labels": "Synthetic; offsets mechanically checked; no independent human annotations claimed",
                "files": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "manifest.json", manifest)
    print({"counts": COUNTS, "exclusions": dict(exclusions)}, flush=True)
    return manifest
