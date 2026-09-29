#!/usr/bin/env python3
"""Verify the pinned reproducibility environment."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np   
import openpyxl   
import pandas as pd   
import rdkit   
import scipy   
import sklearn   
import torch   
import tqdm   
import yaml   


EXPECTED = {
    "python": "3.8.12",
    "numpy": "1.21.2",
    "scipy": "1.10.1",
    "pandas": "1.5.3",
    "rdkit": "2022.09.5",
    "scikit-learn": "1.2.2",
    "torch": "2.0.1",
    "PyYAML": "6.0",
    "openpyxl": "3.0.9",
    "tqdm": "4.67.1",
}
ACTUAL = {
    "python": ".".join(str(value) for value in sys.version_info[:3]),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "pandas": pd.__version__,
    "rdkit": rdkit.__version__,
    "scikit-learn": sklearn.__version__,
    "torch": torch.__version__.split("+")[0],
    "PyYAML": yaml.__version__,
    "openpyxl": openpyxl.__version__,
    "tqdm": tqdm.__version__,
}


def main() -> None:
    mismatches = []
    for name in EXPECTED:
        marker = "OK" if ACTUAL[name] == EXPECTED[name] else "DIFF"
        print("{:<14} expected={:<14} actual={:<14} {}".format(name, EXPECTED[name], ACTUAL[name], marker))
        if marker == "DIFF":
            mismatches.append(name)
    print("cuda_available={} gpu={}".format(
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    ))
    if mismatches:
        raise SystemExit("Environment mismatch: {}".format(", ".join(mismatches)))


if __name__ == "__main__":
    main()
