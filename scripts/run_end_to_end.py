#!/usr/bin/env python3
"""Repository-local wrapper for the end-to-end CLI."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config, validate_config   
from experiment import run_experiment   


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--disable-ranking", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["project"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["runtime"]["device"] = args.device
    if args.disable_ranking:
        config["ranking"]["enabled"] = False
    validate_config(config)
    manifest = run_experiment(config)
    print("Completed: {}".format(manifest["output_dir"]))


if __name__ == "__main__":
    main()
