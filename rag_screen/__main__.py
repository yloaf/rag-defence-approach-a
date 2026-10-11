import argparse
import json

from .data import prepare
from .experiment import run, train


def main():
    parser = argparse.ArgumentParser(description="Local, auditable RAG prompt-injection experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare")
    data.add_argument("--raw", default="data/raw/hotpot_dev_distractor_v1.json")
    data.add_argument("--output", default="data/prepared")
    data.add_argument("--limit", type=int, default=500)
    data.add_argument("--seed", type=int, default=42)
    fit = commands.add_parser("train")
    fit.add_argument("--data", default="data/prepared")
    fit.add_argument("--output", default="artifacts/detector_seed42")
    fit.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("run")
    evaluate.add_argument("--data", default="data/prepared")
    evaluate.add_argument("--checkpoint", default="artifacts/detector_seed42")
    evaluate.add_argument("--tokenizer", default="data/raw/qwen_tokenizer.json")
    evaluate.add_argument("--output", default="runs/pilot")
    evaluate.add_argument("--split", choices=["validation", "pilot", "test"], default="pilot")
    evaluate.add_argument("--groups", type=int, default=10, help="0 runs every source group")
    evaluate.add_argument("--arms", nargs="+", default=["none", "lexical_reject", "learned_reject", "learned_repair"])
    evaluate.add_argument("--model", default="qwen2.5:7b")
    evaluate.add_argument("--base-url", default="http://127.0.0.1:11434")
    evaluate.add_argument("--cache", default="artifacts/ollama.sqlite")
    evaluate.add_argument("--uncached", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if "limit" in args and args["limit"] < 20:
        parser.error("Use at least 20 questions for grouped partitions")
    if "groups" in args and args["groups"] < 0:
        parser.error("groups must be >= 0")
    if command == "prepare":
        args["raw_path"] = args.pop("raw")
    result = {"prepare": prepare, "train": train, "run": run}[command](**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
