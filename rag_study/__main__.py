import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Fresh, frozen RAG follow-up experiment")
    parser.add_argument("command", choices=["prepare", "train", "development", "test", "report", "contrast"])
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare()
    elif args.command == "train":
        from .training import train
        train()
    elif args.command in {"development", "test"}:
        from .study import run
        run(args.command, args.output)
    elif args.command == "report":
        from .analysis import report
        report(args.output or "runs/followup_test")
    else:
        from .analysis import contrast_audit
        contrast_audit(args.output or "runs/followup_test/context_contrast.json")


if __name__ == "__main__":
    main()
