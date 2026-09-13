"""Minimal command-line entry point for the CPST-DDI AAAI release."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def call(script: str, *arguments: str):
    command = [sys.executable, str(HERE / script), *arguments]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def available_datasets(data_root: Path):
    required = {"ddi.txt", "drug_smiles.txt", "networks.txt"}
    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and required.issubset({item.name for item in path.iterdir()})
    )


def main():
    parser = argparse.ArgumentParser(description="CPST-DDI reproducible pipeline")
    parser.add_argument(
        "stage", choices=["audit", "prepare", "main", "ablation", "all"]
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--data-root", type=Path, default=HERE.parent / "dataset")
    parser.add_argument("--cache-root", type=Path, default=HERE.parent / "cache")
    parser.add_argument("--output-root", type=Path, default=HERE.parent / "results")
    args = parser.parse_args()

    datasets = args.datasets or available_datasets(args.data_root)
    if not datasets:
        raise FileNotFoundError(f"No valid datasets found under {args.data_root}")

    if args.stage in {"audit", "all"}:
        call(
            "audit_leakage.py",
            "--datasets",
            *datasets,
            "--data-root",
            str(args.data_root),
            "--output",
            str(args.output_root / "leakage_audit.json"),
        )
    if args.stage in {"prepare", "all"}:
        call(
            "prepare_data.py",
            "--datasets",
            *datasets,
            "--data-root",
            str(args.data_root),
            "--cache-root",
            str(args.cache_root),
        )
    if args.stage in {"main", "all"}:
        for dataset in datasets:
            common = [
                "--dataset",
                dataset,
                "--cache-root",
                str(args.cache_root),
                "--output-root",
                str(args.output_root),
            ]
            call(
                "run_cv.py",
                *common,
                "--variant",
                "structure_expert",
                "--tag",
                "structure_expert",
            )
            call(
                "run_cv.py",
                *common,
                "--variant",
                "topology_expert",
                "--tag",
                "topology_expert",
            )
            call(
                "ensemble.py",
                "--dataset",
                dataset,
                "--expert-a",
                "structure_expert",
                "--expert-b",
                "topology_expert",
                "--results",
                str(args.output_root),
            )
    if args.stage in {"ablation", "all"}:
        for dataset in datasets:
            for variant in [
                "no_kg",
                "concat",
                "partial_ot",
                "no_counterfactual",
                "no_drug_id",
            ]:
                call(
                    "run_cv.py",
                    "--dataset",
                    dataset,
                    "--variant",
                    variant,
                    "--cache-root",
                    str(args.cache_root),
                    "--output-root",
                    str(args.output_root),
                )


if __name__ == "__main__":
    main()
