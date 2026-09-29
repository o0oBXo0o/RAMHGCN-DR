"""Randomness control, hashing and runtime provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if value not in ("cpu", "cuda"):
        raise ValueError("runtime.device must be auto, cpu or cuda")
    return torch.device(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_manifest(
    repository_root: Path,
    config_path: Path,
    source_files: Iterable[Path],
    data_files: Iterable[Path],
) -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_sha256": sha256_file(config_path),
        "source_sha256": {
            path.relative_to(repository_root).as_posix(): sha256_file(path)
            for path in sorted(source_files)
        },
        "data_sha256": {
            path.relative_to(repository_root).as_posix(): sha256_file(path)
            for path in sorted(data_files)
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
