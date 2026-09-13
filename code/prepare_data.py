"""Build deterministic molecular/KG feature caches."""

import argparse
from pathlib import Path
from configs import BASE
from data import prepare_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["drugbank", "kegg", "ogbl-biokg"])
    parser.add_argument("--data-root", type=Path, default=Path("../dataset"))
    parser.add_argument("--cache-root", type=Path, default=Path("../cache"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in args.datasets:
        target = args.cache_root / f"{name}.pt"
        if target.exists() and not args.force:
            print(f"[skip] {target} already exists")
            continue
        payload = prepare_dataset(
            args.data_root / name, target,
            max_motifs=BASE["max_motifs"], max_kg_tokens=BASE["max_kg_tokens"], seed=BASE["seed"]
        )
        print(
            f"[done] {name}: {len(payload['ddi'])} rows, "
            f"{len(payload['records'])} DDI drugs, {payload['num_kg_relations']} KG relations, "
            f"{len(payload['missing_smiles'])} missing SMILES"
        )


if __name__ == "__main__":
    main()
