"""Project heterogeneous biomedical paths into four drug--disease relations."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Set, Tuple

import pandas as pd


StringPair = Tuple[str, str]


def _digits(value: object) -> Iterable[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip().isdigit()]


def _map_add(mapping: DefaultDict[str, Set[str]], key: object, values: Iterable[str]) -> None:
    mapping[str(key).strip()].update(values)


def project_relations(
    direct_path: Path,
    drug_target_path: Path,
    phenotype_gene_path: Path,
    pathway_gene_path: Path,
    complex_gene_path: Path,
    allowed_drugs: Set[str],
    allowed_diseases: Set[str],
    distinct_pathway_genes: bool = True,
    distinct_complex_genes: bool = False,
) -> Tuple[Dict[str, Set[StringPair]], Dict[str, Any]]:
    direct = pd.read_csv(direct_path, encoding="utf-8-sig", dtype=str)
    direct.columns = ["Disease", "Drug"]
    drug_targets = pd.read_csv(drug_target_path, sep="\t", dtype=str)
    phenotypes = pd.read_csv(phenotype_gene_path, sep="\t", dtype=str)
    pathways = pd.read_csv(pathway_gene_path, sep="\t", dtype=str)
    complexes = pd.read_csv(complex_gene_path, sep="\t", dtype=str)

    drug_gene: DefaultDict[str, Set[str]] = defaultdict(set)
    for _, row in drug_targets[["KEGGID", "GeneList"]].dropna().iterrows():
        drug = str(row["KEGGID"]).strip()
        if drug in allowed_drugs:
            _map_add(drug_gene, drug, _digits(row["GeneList"]))

    disease_column = "MIMID"
    phenotype_gene_column = "GeneList" if "GeneList" in phenotypes.columns else phenotypes.columns[-1]
    gene_disease: DefaultDict[str, Set[str]] = defaultdict(set)
    for _, row in phenotypes.iterrows():
        disease = str(row[disease_column]).strip()
        if disease not in allowed_diseases:
            continue
        for gene in _digits(row[phenotype_gene_column]):
            gene_disease[gene].add(disease)

    pathway_id_column = "KEGID" if "KEGID" in pathways.columns else pathways.columns[0]
    pathway_gene_column = "GeneList" if "GeneList" in pathways.columns else pathways.columns[-1]
    gene_pathway: DefaultDict[str, Set[str]] = defaultdict(set)
    pathway_genes: DefaultDict[str, Set[str]] = defaultdict(set)
    for _, row in pathways.iterrows():
        pathway = str(row[pathway_id_column]).strip()
        for gene in _digits(row[pathway_gene_column]):
            gene_pathway[gene].add(pathway)
            pathway_genes[pathway].add(gene)

    complex_id_column = "ComplexID"
    complex_gene_column = "GeneList" if "GeneList" in complexes.columns else complexes.columns[-1]
    gene_complex: DefaultDict[str, Set[str]] = defaultdict(set)
    complex_genes: DefaultDict[str, Set[str]] = defaultdict(set)
    for _, row in complexes.iterrows():
        complex_id = str(row[complex_id_column]).strip()
        for gene in _digits(row[complex_gene_column]):
            gene_complex[gene].add(complex_id)
            complex_genes[complex_id].add(gene)

    relations: Dict[str, Set[StringPair]] = {
        "DD": {
            (str(row["Drug"]).strip(), str(row["Disease"]).strip())
            for _, row in direct.iterrows()
            if str(row["Drug"]).strip() in allowed_drugs
            and str(row["Disease"]).strip() in allowed_diseases
        },
        "DGD": set(),
        "DPD": set(),
        "DCD": set(),
    }
    for drug, first_genes in drug_gene.items():
        for first_gene in first_genes:
            for disease in gene_disease.get(first_gene, set()):
                relations["DGD"].add((drug, disease))
            for pathway in gene_pathway.get(first_gene, set()):
                for second_gene in pathway_genes[pathway]:
                    if distinct_pathway_genes and first_gene == second_gene:
                        continue
                    for disease in gene_disease.get(second_gene, set()):
                        relations["DPD"].add((drug, disease))
            for complex_id in gene_complex.get(first_gene, set()):
                for second_gene in complex_genes[complex_id]:
                    if distinct_complex_genes and first_gene == second_gene:
                        continue
                    for disease in gene_disease.get(second_gene, set()):
                        relations["DCD"].add((drug, disease))

    stats: Dict[str, Any] = {
        "drug_gene_pairs": int(sum(len(values) for values in drug_gene.values())),
        "gene_disease_pairs": int(sum(len(values) for values in gene_disease.values())),
        "pathway_gene_pairs": int(sum(len(values) for values in pathway_genes.values())),
        "complex_gene_pairs": int(sum(len(values) for values in complex_genes.values())),
        "projected_relation_pairs": {name: int(len(values)) for name, values in relations.items()},
        "distinct_pathway_genes": bool(distinct_pathway_genes),
        "distinct_complex_genes": bool(distinct_complex_genes),
    }
    return relations, stats
