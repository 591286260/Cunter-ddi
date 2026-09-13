"""Central experiment configuration.

All paths are relative to the repository root unless overridden on the CLI.
The settings are deliberately conservative enough for a 12 GB GPU.
"""

from copy import deepcopy


BASE = {
    "seed": 2026,
    "folds": 5,
    # Canonical unordered drug pairs are kept in one fold.  This prevents
    # duplicate or reverse-pair leakage across train/validation/test sets.
    "split_mode": "pair_grouped",
    "epochs": 25,
    "patience": 5,
    "batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "hidden_dim": 96,
    "dropout": 0.20,
    "max_motifs": 12,
    "max_kg_tokens": 24,
    "transport_mass": 0.70,
    "transport_temperature": 0.15,
    "gw_weight": 0.30,
    "sinkhorn_iters": 8,
    "lambda_sufficiency": 0.10,
    "lambda_necessity": 0.10,
    "necessity_margin": 0.15,
    "keep_ratio": 0.30,
    "num_workers": 0,
    "use_drug_id": True,
}


DATASETS = {
    "drugbank": {},
    "kegg": {},
    "ogbl-biokg": {"batch_size": 1024, "epochs": 18, "patience": 4},
}


# Four submodels for the requested ablation study. Each removes one conceptual
# component while preserving the same encoders and data split.
VARIANTS = {
    "full": {},
    "structure_expert": {"use_topology": False},
    "topology_expert": {"use_topology": True},
    # Backward-compatible aliases used by the original experiment folders.
    "molecular_expert": {"use_topology": False},
    "network_expert": {"use_topology": True},
    "no_kg": {"use_kg": False, "use_transport": False, "use_counterfactual": False},
    "concat": {"use_transport": False, "use_counterfactual": False},
    "partial_ot": {"gw_weight": 0.0, "use_counterfactual": False},
    "no_counterfactual": {"use_counterfactual": False},
    "no_drug_id": {"use_drug_id": False},
    # Strict inductive setting: neither drug identity nor fold-local DDI
    # topology is available for drugs held out from model training.
    "cold_start": {"use_drug_id": False, "use_topology": False},
    # Transfer-oriented cold-start variant: raw neighbor identities are not
    # expected to generalize, whereas KG relation types and directions do.
    "cold_start_inductive": {
        "use_drug_id": False, "use_topology": False, "use_neighbor_id": False
    },
}


# Two architecture and two training hyperparameters. KEGG is selected because
# its five-fold results are stable, it is large enough to be representative,
# and its identifiers remain suitable for subsequent case inspection.
SENSITIVITY = {
    "hidden_dim": [64, 96, 128, 192],
    "transport_mass": [0.3, 0.5, 0.7, 0.9],
    "lr": [3e-4, 1e-3, 3e-3, 1e-2],
    "dropout": [0.0, 0.1, 0.2, 0.4],
}


def get_config(dataset: str, variant: str = "full", overrides=None):
    cfg = deepcopy(BASE)
    cfg.update(DATASETS[dataset])
    cfg.update({"use_kg": True, "use_transport": True, "use_counterfactual": True, "use_topology": True})
    cfg.update(VARIANTS[variant])
    if overrides:
        cfg.update(overrides)
    return cfg
