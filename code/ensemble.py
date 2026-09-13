"""Create the fixed equal-weight two-expert final prediction table."""

import argparse
from pathlib import Path
import numpy as np
from metrics import binary_metrics, save_cv_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expert-a", default="molecular_expert")
    parser.add_argument("--expert-b", default="network_expert")
    parser.add_argument("--name", default="final_ensemble")
    parser.add_argument("--results", type=Path, default=Path("../results"))
    args = parser.parse_args()
    root = args.results / args.dataset
    target = root / args.name
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(1, 6):
        a = np.load(root / args.expert_a / f"fold_{fold}" / "predictions.npz")
        b = np.load(root / args.expert_b / f"fold_{fold}" / "predictions.npz")
        aligned = (
            np.array_equal(a["labels"], b["labels"])
            and np.array_equal(a["drug_pairs"], b["drug_pairs"])
            and np.array_equal(a["relations"], b["relations"])
            and np.array_equal(a["sample_indices"], b["sample_indices"])
        )
        if not aligned:
            raise ValueError(f"Fold {fold} predictions are not aligned")
        probability = 0.5 * a["probabilities"] + 0.5 * b["probabilities"]
        rows.append(binary_metrics(a["labels"], probability))
        fold_dir = target / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            fold_dir / "predictions.npz",
            labels=a["labels"],
            probabilities=probability,
            drug_pairs=a["drug_pairs"],
            relations=a["relations"],
            sample_indices=a["sample_indices"],
        )
    save_cv_table(rows, target / "metrics.csv")
    print((target / "metrics.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    main()
