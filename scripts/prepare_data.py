#!/usr/bin/env python3
"""Build all model inputs from the repository-local raw biomedical sources."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.pipeline import (   
    load_preprocessing_config,
    prepare_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/preprocessing.yaml")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    manifest = prepare_data(
        load_preprocessing_config(args.config),
        output_dir_override=args.output_dir or "",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
