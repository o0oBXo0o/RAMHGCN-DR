#!/usr/bin/env python3
"""Run FULL and drop-one-relation experiments with a shared base configuration."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config, resolve_path, validate_config   
from experiment import run_experiment   
from runtime import write_json   


RELATIONS = ["DD", "DGD", "DPD", "DCD"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ramhgcn.yaml")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--only", choices=["FULL"] + RELATIONS)
    args = parser.parse_args()
    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=False)
    base = load_config(args.config)
    settings = [("FULL", [])] + [("DROP_{}".format(name), [name]) for name in RELATIONS]
    if args.only:
        target = "FULL" if args.only == "FULL" else "DROP_{}".format(args.only)
        settings = [value for value in settings if value[0] == target]
    rows = []
    manifests = {}
    for setting, disabled in settings:
        config = copy.deepcopy(base)
        config["project"]["output_dir"] = str(
            (output_root / setting.lower()).relative_to(ROOT)
        )
        config["project"]["name"] = "{}-{}".format(base["project"]["name"], setting.lower())
        config["model"]["disabled_relations"] = disabled
        config["ranking"]["enabled"] = False
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        validate_config(config)
        started = time.time()
        manifest = run_experiment(config)
        elapsed = time.time() - started
        metrics = json.loads(Path(manifest["metrics"]).read_text(encoding="utf-8"))
        row = {"setting": setting, "disabled_relation": ",".join(disabled), "seconds": elapsed}
        for metric, values in metrics["summary"].items():
            row["{}_mean".format(metric)] = values["mean"]
            row["{}_std".format(metric)] = values["std"]
        rows.append(row)
        manifests[setting] = manifest
    pd.DataFrame(rows).to_csv(output_root / "ablation_summary.csv", index=False)
    write_json(output_root / "ablation_manifest.json", manifests)
    print("Completed ablation suite: {}".format(output_root))


if __name__ == "__main__":
    main()
