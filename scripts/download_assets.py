"""Download public research assets; record origin and content hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request
import argparse

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
ASSETS = {
    "qwen_tokenizer.json": (
        f"https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/{REVISION}/tokenizer.json",
        "Apache-2.0", "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["mirror", "original"], default="mirror")
    args = parser.parse_args()
    dest = ROOT / "data/raw"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if args.source == "original":
        ASSETS["hotpot_dev_distractor_v1.json"] = (
            "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            "CC BY-SA 4.0", "https://hotpotqa.github.io/")
    for name, (url, license_name, attribution) in ASSETS.items():
        path = dest / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "RAG-research/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                content = response.read()
            json.loads(content)  # Do not accept HTML error pages.
            path.write_bytes(content)
        manifest[name] = {"url": url, "license": license_name,
                          "attribution": attribution, "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if args.source == "mirror":
        import pyarrow.parquet as pq
        url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
        expected = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
        parquet = dest / "hotpot_validation.parquet"
        if not parquet.exists():
            print("Downloading checksum-pinned HotpotQA mirror", flush=True)
            with urllib.request.urlopen(url, timeout=180) as response:
                parquet.write_bytes(response.read())
        actual = hashlib.sha256(parquet.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("HotpotQA mirror checksum differs from pinned upstream LFS SHA256")
        rows = []
        for r in pq.read_table(parquet).to_pylist():
            rows.append({"_id": r["id"], "question": r["question"], "answer": r["answer"],
                         "type": r["type"], "level": r["level"],
                         "context": list(zip(r["context"]["title"], r["context"]["sentences"])),
                         "supporting_facts": list(zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]))})
        path = dest / "hotpot_dev_distractor_v1.json"
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        manifest[path.name] = {"url": url, "upstream_parquet_sha256": actual,
                               "conversion": "HF struct-of-lists to official HotpotQA JSON shape",
                               "license": "CC BY-SA 4.0", "attribution": "https://hotpotqa.github.io/",
                               "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
