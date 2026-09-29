"""Command-line interface for validation and end-to-end execution."""

from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Dict

from config import load_config, validate_config
from data import load_dataset, validate_dataset
from experiment import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ramhgcn")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data", help="Validate bundled multiplex inputs")
    validate.add_argument("--config", required=True)
    show = commands.add_parser("show-config", help="Print resolved configuration")
    show.add_argument("--config", required=True)
    run = commands.add_parser("run", help="Run validation, relation-wise CV, evaluation and ranking")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir")
    run.add_argument("--epochs", type=int)
    run.add_argument("--device", choices=["auto", "cpu", "cuda"])
    run.add_argument("--disable-ranking", action="store_true")
    return parser


def _overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    value = copy.deepcopy(config)
    if getattr(args, "output_dir", None):
        value["project"]["output_dir"] = args.output_dir
    if getattr(args, "epochs", None) is not None:
        value["training"]["epochs"] = args.epochs
    if getattr(args, "device", None):
        value["runtime"]["device"] = args.device
    if getattr(args, "disable_ranking", False):
        value["ranking"]["enabled"] = False
    validate_config(value)
    return value


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate-data":
        report = validate_dataset(load_dataset(config), config)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "show-config":
        printable = copy.deepcopy(config)
        print(json.dumps(printable, indent=2, ensure_ascii=False))
    elif args.command == "run":
        manifest = run_experiment(_overrides(config, args))
        print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
