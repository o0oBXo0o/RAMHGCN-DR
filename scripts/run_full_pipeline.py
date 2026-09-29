#!/usr/bin/env python3
"""Run raw preprocessing followed by training, evaluation and ranking."""

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config, validate_config   
from experiment import run_experiment   
from preprocessing.pipeline import (   
    load_preprocessing_config,
    prepare_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-config", default="configs/preprocessing.yaml")
    parser.add_argument("--experiment-config", default="configs/ramhgcn.yaml")
    parser.add_argument("--data-output-dir")
    parser.add_argument("--run-output-dir")
    args = parser.parse_args()
    preprocessing = prepare_data(
        load_preprocessing_config(args.preprocess_config),
        output_dir_override=args.data_output_dir or "",
    )
    data_dir = ROOT / preprocessing["output_dir"]
    experiment = copy.deepcopy(load_config(args.experiment_config))
    experiment["data"]["mat_file"] = str((data_dir / "multiplex_graph.mat").relative_to(ROOT))
    experiment["data"]["split_dir"] = str((data_dir / "splits").relative_to(ROOT))
    if args.run_output_dir:
        experiment["project"]["output_dir"] = args.run_output_dir
    validate_config(experiment)
    result = run_experiment(experiment)
    print(json.dumps({"preprocessing": preprocessing, "experiment": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
