"""Reproduce the source RDKit atom one-hot mean drug descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem


ELEMENTS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
    "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd",
    "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Sm", "Tc", "Gd", "Unknown",
]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.OTHER,
]


def _one_hot(value: object, choices: Sequence[object], unknown: bool = False) -> List[bool]:
    if value not in choices:
        if not unknown:
            raise ValueError("Value {} is outside {}".format(value, choices))
        value = choices[-1]
    return [value == choice for choice in choices]


def atom_features(atom: Chem.Atom) -> np.ndarray:
    values = (
        _one_hot(atom.GetSymbol(), ELEMENTS, unknown=True)
        + _one_hot(atom.GetDegree(), list(range(11)))
        + _one_hot(atom.GetTotalNumHs(), list(range(11)), unknown=True)
        + _one_hot(atom.GetExplicitValence(), list(range(11)), unknown=True)
        + _one_hot(atom.GetImplicitValence(), list(range(11)), unknown=True)
        + _one_hot(atom.GetFormalCharge(), list(range(-4, 5)))
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [atom.GetIsAromatic()]
    )
    return np.asarray(values, dtype=np.float32)


def smiles_descriptor(smiles: str, aggregation: str = "mean") -> np.ndarray:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("RDKit cannot parse SMILES: {}".format(smiles))
    matrix = np.asarray([atom_features(atom) for atom in molecule.GetAtoms()])
    if aggregation == "mean":
        return matrix.mean(axis=0)
    if aggregation == "sum":
        return matrix.sum(axis=0)
    if aggregation == "max":
        return matrix.max(axis=0)
    if aggregation == "concat":
        return np.concatenate([matrix.sum(axis=0), matrix.mean(axis=0), matrix.max(axis=0)])
    raise ValueError("Unsupported atom aggregation: {}".format(aggregation))


def generate_drug_features(
    path: Path,
    aggregation: str = "mean",
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int]]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    id_column = "ID" if "ID" in frame.columns else frame.columns[0]
    smiles_column = (
        "Connectivity_SMILES" if "Connectivity_SMILES" in frame.columns else frame.columns[2]
    )
    if frame[id_column].duplicated().any():
        raise ValueError("Drug identifiers must be unique")
    features = np.asarray(
        [smiles_descriptor(value, aggregation) for value in frame[smiles_column]],
        dtype=np.float32,
    )
    entities = pd.DataFrame(
        {
            "drug_id": frame[id_column].astype(str),
            "drug_name": frame["Name"].astype(str) if "Name" in frame.columns else frame[id_column].astype(str),
            "smiles": frame[smiles_column].astype(str),
        }
    )
    return entities, features, {
        "drugs": int(len(entities)),
        "drug_feature_dimensions": int(features.shape[1]),
    }
