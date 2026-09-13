"""Run leakage-safe five-fold CV and save the requested metric table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configs import DATASETS, VARIANTS, get_config
from data import drug_disjoint_folds, grouped_stratified_folds, one_unseen_drug_folds
from metrics import save_cv_table
from train import train_fold


def parse_override(items):
    values = {}
    for item in items or []:
        key, raw = item.split("=", 1)
        if raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        values[key] = value
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--cache-root", type=Path, default=Path("../cache"))
    parser.add_argument("--output-root", type=Path, default=Path("../results"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--set", nargs="*", help="Overrides such as epochs=2 batch_size=64")
    args = parser.parse_args()

    overrides = parse_override(args.set)
    cfg = get_config(args.dataset, args.variant, overrides)
    payload = torch.load(args.cache_root / f"{args.dataset}.pt", map_location="cpu", weights_only=False)
    tag = args.tag or args.variant
    output_dir = args.output_root / args.dataset / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    rows = []
    if cfg["split_mode"] == "pair_grouped":
        folds = grouped_stratified_folds(payload["ddi"], cfg["folds"], cfg["seed"])
    elif cfg["split_mode"] == "drug_disjoint":
        if cfg.get("use_drug_id", True) or cfg.get("use_topology", True):
            raise ValueError(
                "drug_disjoint requires use_drug_id=False and use_topology=False"
            )
        folds = drug_disjoint_folds(payload["ddi"], cfg["folds"], cfg["seed"])
    elif cfg["split_mode"] == "one_unseen":
        if cfg.get("use_drug_id", True) or cfg.get("use_topology", True):
            raise ValueError("one_unseen requires use_drug_id=False and use_topology=False")
        folds = one_unseen_drug_folds(payload["ddi"], cfg["folds"], cfg["seed"])
    else:
        raise ValueError(f"Unsupported leakage-safe split mode: {cfg['split_mode']}")
    for fold, (train_idx, val_idx, test_idx) in enumerate(folds, 1):
        print(f"\n=== {args.dataset} / {tag} / fold {fold} ===", flush=True)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold_dir / "split_indices.npz",
            train=train_idx,
            validation=val_idx,
            test=test_idx,
        )
        ddi = payload["ddi"]
        split_audit = {}
        split_drugs = {}
        for name, indices in (
            ("train", train_idx), ("validation", val_idx), ("test", test_idx)
        ):
            drugs = sorted(map(int, np.unique(ddi[indices, :2])))
            split_drugs[name] = set(drugs)
            split_audit[name] = {
                "rows": int(len(indices)),
                "drugs": int(len(drugs)),
                "positive": int((ddi[indices, 3] == 1).sum()),
                "negative": int((ddi[indices, 3] == 0).sum()),
            }
        split_audit["drug_overlap"] = {
            "train_validation": len(split_drugs["train"] & split_drugs["validation"]),
            "train_test": len(split_drugs["train"] & split_drugs["test"]),
            "validation_test": len(split_drugs["validation"] & split_drugs["test"]),
        }
        with (fold_dir / "split_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(split_audit, handle, indent=2)
        row = train_fold(payload, train_idx, val_idx, test_idx, cfg, fold_dir, fold)
        rows.append(row)
        save_cv_table(rows, output_dir / "metrics.csv")
    print((output_dir / "metrics.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    main()
