"""Audit target-pair overlap, duplicate directions, and CV pair isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import grouped_stratified_folds, read_ddi


def canonical(a: int, b: int):
    return (a, b) if a <= b else (b, a)


def audit_dataset(root: Path):
    ddi = read_ddi(root / "ddi.txt")
    target_pairs = {canonical(int(a), int(b)) for a, b in ddi[:, :2]}
    directed = {(int(a), int(b)) for a, b in ddi[:, :2]}
    reverse_overlap = sum(
        1 for a, b in directed if a != b and (b, a) in directed
    ) // 2
    ddi_drugs = set(map(int, np.unique(ddi[:, :2])))
    auxiliary_drug_edges = set()
    target_overlap = set()
    with (root / "networks.txt").open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            fields = line.split()
            if len(fields) != 3:
                continue
            src, dst = map(int, fields[:2])
            pair = canonical(src, dst)
            if src in ddi_drugs and dst in ddi_drugs:
                auxiliary_drug_edges.add(pair)
            if pair in target_pairs:
                target_overlap.add(pair)

    folds = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(
        grouped_stratified_folds(ddi, 5, 2026), 1
    ):
        split_pairs = []
        for indices in (train_idx, val_idx, test_idx):
            split_pairs.append(
                {canonical(int(ddi[i, 0]), int(ddi[i, 1])) for i in indices}
            )
        folds.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "test_rows": int(len(test_idx)),
                "train_validation_pair_overlap": len(
                    split_pairs[0] & split_pairs[1]
                ),
                "train_test_pair_overlap": len(split_pairs[0] & split_pairs[2]),
                "validation_test_pair_overlap": len(
                    split_pairs[1] & split_pairs[2]
                ),
            }
        )

    return {
        "rows": int(len(ddi)),
        "unique_unordered_pairs": len(target_pairs),
        "reverse_pair_duplicates": reverse_overlap,
        "auxiliary_drug_drug_edges_raw": len(auxiliary_drug_edges),
        "target_pairs_in_auxiliary_kg_raw": len(target_overlap),
        "target_pairs_in_model_kg_after_filter": 0,
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("../dataset"))
    parser.add_argument(
        "--output", type=Path, default=Path("../results/leakage_audit.json")
    )
    args = parser.parse_args()
    report = {
        "protocol": "pair-disjoint outer CV and pair-disjoint inner validation",
        "auxiliary_kg_policy": "all edges between DDI drugs are excluded",
        "topology_policy": "training-positive graph with query-edge masking",
        "datasets": {
            name: audit_dataset(args.data_root / name) for name in args.datasets
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
