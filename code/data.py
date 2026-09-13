"""Dataset parsing, molecular tokenization, KG tokenization, and CV splits."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS, rdFingerprintGenerator
from sklearn.model_selection import KFold, StratifiedGroupKFold
from torch.utils.data import Dataset


ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
MOTIF_DIM = 32
FP_DIM = 1024
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_DIM)


def read_ddi(path: Path):
    """Read `drug1 drug2 relation label`; unlike networks.txt it has no header."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 4:
                continue
            rows.append(tuple(map(int, fields)))
    return np.asarray(rows, dtype=np.int64)


def read_smiles(path: Path):
    values = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t", 1)
            if len(fields) == 2:
                values[int(fields[0])] = fields[1]
    return values


def _atom_vector(atom):
    vec = np.zeros(MOTIF_DIM, dtype=np.float32)
    number = atom.GetAtomicNum()
    if number in ATOM_TYPES:
        vec[ATOM_TYPES.index(number)] = 1.0
    else:
        vec[10] = 1.0
    vec[11] = atom.GetDegree() / 4.0
    vec[12] = atom.GetTotalNumHs() / 4.0
    vec[13] = atom.GetFormalCharge() / 3.0
    vec[14] = float(atom.GetIsAromatic())
    vec[15] = float(atom.IsInRing())
    hyb = int(atom.GetHybridization())
    if 0 <= hyb < 8:
        vec[16 + hyb] = 1.0
    vec[24] = atom.GetMass() / 200.0
    vec[25] = float(atom.GetChiralTag()) / 4.0
    return vec


def molecule_tokens(smiles: str, max_tokens: int):
    """Return interpretable BRICS/ring tokens and their internal distances.

    BRICS bonds are cut conceptually and connected components become motifs.
    If BRICS finds no cut, ring systems plus remaining atoms are used. The
    representation is deterministic and cached by prepare_data.py.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return np.zeros((1, MOTIF_DIM), np.float32), np.zeros((1, 1), np.float32)

    cut_bonds = set()
    for (a, b), _ in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(int(a), int(b))
        if bond is not None:
            cut_bonds.add(bond.GetIdx())

    adjacency = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        if bond.GetIdx() not in cut_bonds:
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency[a].append(b)
            adjacency[b].append(a)

    seen, groups = set(), []
    for start in range(mol.GetNumAtoms()):
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            node = stack.pop()
            group.append(node)
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        groups.append(group)

    # Keep the largest chemically meaningful components under a fixed budget.
    groups = sorted(groups, key=lambda x: (-len(x), min(x)))[:max_tokens]
    atom_to_group = {}
    token_vectors = []
    for gid, atoms in enumerate(groups):
        for atom in atoms:
            atom_to_group[atom] = gid
        raw = np.stack([_atom_vector(mol.GetAtomWithIdx(a)) for a in atoms]).mean(0)
        raw[26] = min(len(atoms), 30) / 30.0
        raw[27] = np.mean([mol.GetAtomWithIdx(a).GetIsAromatic() for a in atoms])
        raw[28] = np.mean([mol.GetAtomWithIdx(a).IsInRing() for a in atoms])
        raw[29] = sum(mol.GetAtomWithIdx(a).GetFormalCharge() for a in atoms) / 5.0
        raw[30] = len(groups) / max_tokens
        raw[31] = 1.0
        token_vectors.append(raw)

    n = len(groups)
    graph = np.full((n, n), 99.0, dtype=np.float32)
    np.fill_diagonal(graph, 0.0)
    for bond in mol.GetBonds():
        ga = atom_to_group.get(bond.GetBeginAtomIdx())
        gb = atom_to_group.get(bond.GetEndAtomIdx())
        if ga is not None and gb is not None and ga != gb:
            graph[ga, gb] = graph[gb, ga] = 1.0
    for k in range(n):
        graph = np.minimum(graph, graph[:, k, None] + graph[None, k, :])
    graph[graph >= 99] = n
    if n > 1:
        graph /= graph.max()
    return np.stack(token_vectors), graph


def molecular_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(FP_DIM, dtype=np.float32)
    return np.asarray(MORGAN.GetFingerprintAsNumPy(mol), dtype=np.float32)


def build_kg_tokens(
    network_path: Path,
    drug_ids,
    max_tokens: int,
    seed: int,
    exclude_drug_drug_edges: bool = True,
):
    """Sample incident non-DDI KG edges for each relevant drug.

    A token is `(relation, direction, hashed-neighbor)`. Reservoir sampling
    prevents high-degree drugs from dominating memory and is deterministic.
    """
    drug_ids = set(map(int, drug_ids))
    reservoirs = {d: [] for d in drug_ids}
    counts = defaultdict(int)
    max_relation = 0
    max_entity = max(drug_ids) if drug_ids else 0
    rng = random.Random(seed)
    with network_path.open("r", encoding="utf-8") as handle:
        next(handle, None)  # networks.txt begins with the declared edge count.
        for line in handle:
            fields = line.split()
            if len(fields) != 3:
                continue
            src, dst, rel = map(int, fields)
            max_relation = max(max_relation, rel)
            max_entity = max(max_entity, src, dst)
            # A target drug pair must never be exposed as an auxiliary KG
            # neighbor.  Removing all edges between DDI drugs is conservative
            # and avoids dependence on relation-ID conventions.
            if exclude_drug_drug_edges and src in drug_ids and dst in drug_ids:
                continue
            for drug, neighbor, direction in ((src, dst, 0), (dst, src, 1)):
                if drug not in reservoirs:
                    continue
                counts[drug] += 1
                item = (rel, direction, neighbor)
                bucket = reservoirs[drug]
                if len(bucket) < max_tokens:
                    bucket.append(item)
                else:
                    pos = rng.randrange(counts[drug])
                    if pos < max_tokens:
                        bucket[pos] = item
    return reservoirs, counts, max_relation + 1, max_entity + 1


def prepare_dataset(dataset_dir: Path, cache_path: Path, max_motifs=12, max_kg_tokens=24, seed=2026):
    ddi = read_ddi(dataset_dir / "ddi.txt")
    smiles = read_smiles(dataset_dir / "drug_smiles.txt")
    drugs = np.unique(ddi[:, :2])
    kg, degrees, num_kg_relations, num_entities = build_kg_tokens(
        dataset_dir / "networks.txt",
        drugs,
        max_kg_tokens,
        seed,
        exclude_drug_drug_edges=True,
    )
    records = {}
    missing = []
    for drug in drugs:
        drug = int(drug)
        if drug not in smiles:
            missing.append(drug)
            mol_tokens = np.zeros((1, MOTIF_DIM), np.float32)
            mol_dist = np.zeros((1, 1), np.float32)
        else:
            mol_tokens, mol_dist = molecule_tokens(smiles[drug], max_motifs)
        records[drug] = {
            "mol_tokens": mol_tokens,
            "mol_dist": mol_dist,
            "fingerprint": molecular_fingerprint(smiles[drug]) if drug in smiles else np.zeros(FP_DIM, np.float32),
            "kg_tokens": kg.get(drug, []),
            "degree": degrees.get(drug, 0),
        }
    payload = {
        "ddi": ddi,
        "records": records,
        "num_kg_relations": num_kg_relations,
        "num_entities": num_entities,
        "num_ddi_relations": int(ddi[:, 2].max()) + 1,
        "missing_smiles": missing,
        "max_motifs": max_motifs,
        "max_kg_tokens": max_kg_tokens,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def grouped_stratified_folds(ddi, n_splits=5, seed=2026):
    """Pair-disjoint outer CV with a pair-disjoint inner validation split.

    Every row sharing the same unordered drug pair receives the same group ID,
    regardless of direction or candidate relation.  Consequently neither a
    duplicate nor a reverse pair can cross any split boundary.
    """
    labels = ddi[:, 3]
    canonical = np.sort(ddi[:, :2].astype(np.int64), axis=1)
    _, groups = np.unique(canonical, axis=0, return_inverse=True)
    outer = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    for fold, (train_val_idx, test_idx) in enumerate(
        outer.split(np.zeros(len(ddi)), labels, groups), 1
    ):
        inner = StratifiedGroupKFold(
            n_splits=10, shuffle=True, random_state=seed + fold
        )
        inner_train, inner_val = next(
            inner.split(
                np.zeros(len(train_val_idx)),
                labels[train_val_idx],
                groups[train_val_idx],
            )
        )
        tr_idx = train_val_idx[inner_train]
        val_idx = train_val_idx[inner_val]
        assert_pair_disjoint(ddi, tr_idx, val_idx, test_idx)
        yield np.asarray(tr_idx), np.asarray(val_idx), np.asarray(test_idx)


def _canonical_pair_set(ddi, indices):
    return {
        tuple(sorted((int(ddi[i, 0]), int(ddi[i, 1]))))
        for i in np.asarray(indices)
    }


def assert_pair_disjoint(ddi, train_idx, val_idx, test_idx):
    """Fail fast if an unordered drug pair occurs in more than one split."""
    train_pairs = _canonical_pair_set(ddi, train_idx)
    val_pairs = _canonical_pair_set(ddi, val_idx)
    test_pairs = _canonical_pair_set(ddi, test_idx)
    if train_pairs & val_pairs:
        raise ValueError("Canonical drug-pair leakage between train and validation")
    if train_pairs & test_pairs:
        raise ValueError("Canonical drug-pair leakage between train and test")
    if val_pairs & test_pairs:
        raise ValueError("Canonical drug-pair leakage between validation and test")


def drug_disjoint_folds(ddi, n_splits=5, seed=2026):
    """Strict two-unseen-drug cross-validation.

    Drugs, rather than rows or pairs, are partitioned.  A test row is retained
    only when both endpoints belong to the held-out test-drug fold.  Validation
    drugs are then held out from the remaining drugs in the same manner.  Rows
    crossing train/validation/test drug sets are intentionally discarded.
    Consequently, both drugs in every test instance are unseen during model
    fitting and checkpoint selection.
    """
    drugs = np.unique(ddi[:, :2].astype(np.int64))
    outer = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (remaining_pos, test_pos) in enumerate(outer.split(drugs), 1):
        remaining_drugs = drugs[remaining_pos]
        test_drugs = set(map(int, drugs[test_pos]))

        # A 20% drug-level validation subset gives enough validation pairs for
        # stable early stopping while preserving strict drug disjointness.
        inner = KFold(n_splits=5, shuffle=True, random_state=seed + fold)
        train_pos, val_pos = next(inner.split(remaining_drugs))
        train_drugs = set(map(int, remaining_drugs[train_pos]))
        val_drugs = set(map(int, remaining_drugs[val_pos]))

        a = ddi[:, 0].astype(np.int64)
        b = ddi[:, 1].astype(np.int64)
        train_idx = np.flatnonzero(
            np.isin(a, list(train_drugs)) & np.isin(b, list(train_drugs))
        )
        val_idx = np.flatnonzero(
            np.isin(a, list(val_drugs)) & np.isin(b, list(val_drugs))
        )
        test_idx = np.flatnonzero(
            np.isin(a, list(test_drugs)) & np.isin(b, list(test_drugs))
        )

        if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
            raise ValueError(f"Drug-disjoint fold {fold} contains an empty split")
        for name, indices in (
            ("train", train_idx), ("validation", val_idx), ("test", test_idx)
        ):
            if len(np.unique(ddi[indices, 3])) < 2:
                raise ValueError(
                    f"Drug-disjoint fold {fold} {name} split lacks one class"
                )

        observed_train = set(map(int, np.unique(ddi[train_idx, :2])))
        observed_val = set(map(int, np.unique(ddi[val_idx, :2])))
        observed_test = set(map(int, np.unique(ddi[test_idx, :2])))
        if observed_train & observed_val or observed_train & observed_test or observed_val & observed_test:
            raise ValueError(f"Drug identity leakage in drug-disjoint fold {fold}")

        yield np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def one_unseen_drug_folds(ddi, n_splits=5, seed=2026):
    """S2 cross-validation in which every evaluation pair has one unseen drug.

    The drug vocabulary is partitioned into anchor, validation-unseen and
    test-unseen sets. Training contains anchor--anchor pairs only. Validation
    and test contain, respectively, anchor--validation and anchor--test pairs;
    pairs with zero or two held-out endpoints are excluded from evaluation.
    Thus the held-out endpoint is genuinely unseen while the other endpoint is
    available as the clinically more realistic known-drug anchor.
    """
    drugs = np.unique(ddi[:, :2].astype(np.int64))
    outer = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    a = ddi[:, 0].astype(np.int64)
    b = ddi[:, 1].astype(np.int64)
    for fold, (remaining_pos, test_pos) in enumerate(outer.split(drugs), 1):
        remaining_drugs = drugs[remaining_pos]
        test_drugs = set(map(int, drugs[test_pos]))
        inner = KFold(n_splits=5, shuffle=True, random_state=seed + fold)
        anchor_pos, val_pos = next(inner.split(remaining_drugs))
        anchor_drugs = set(map(int, remaining_drugs[anchor_pos]))
        val_drugs = set(map(int, remaining_drugs[val_pos]))

        in_anchor_a = np.isin(a, list(anchor_drugs))
        in_anchor_b = np.isin(b, list(anchor_drugs))
        in_val_a = np.isin(a, list(val_drugs))
        in_val_b = np.isin(b, list(val_drugs))
        in_test_a = np.isin(a, list(test_drugs))
        in_test_b = np.isin(b, list(test_drugs))
        train_idx = np.flatnonzero(in_anchor_a & in_anchor_b)
        # An assigned anchor may have no anchor--anchor edge and therefore may
        # never actually occur in the training rows. Restrict evaluation to
        # anchors observed by the fitted model so every row has *exactly* one
        # unseen endpoint in operational, rather than merely assigned, terms.
        observed_anchor = set(map(int, np.unique(ddi[train_idx, :2])))
        seen_a = np.isin(a, list(observed_anchor))
        seen_b = np.isin(b, list(observed_anchor))
        val_idx = np.flatnonzero((seen_a & in_val_b) | (in_val_a & seen_b))
        test_idx = np.flatnonzero((seen_a & in_test_b) | (in_test_a & seen_b))

        if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
            raise ValueError(f"S2 fold {fold} contains an empty split")
        for name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
            if len(np.unique(ddi[indices, 3])) < 2:
                raise ValueError(f"S2 fold {fold} {name} split lacks one class")

        train_observed = set(map(int, np.unique(ddi[train_idx, :2])))
        unseen_per_test_row = np.asarray([
            int(int(x) not in train_observed) + int(int(y) not in train_observed)
            for x, y in ddi[test_idx, :2]
        ])
        if not np.all(unseen_per_test_row == 1):
            raise ValueError(f"S2 fold {fold} does not have exactly one unseen endpoint")
        test_unseen_observed = (
            set(map(int, np.unique(ddi[test_idx, :2]))) - anchor_drugs
        )
        if train_observed & test_unseen_observed:
            raise ValueError(f"Unseen-drug leakage in S2 fold {fold}")
        yield np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def training_graph_pair_features(ddi, train_indices):
    """Compute leakage-safe pair topology features from training positives only."""
    adjacency = defaultdict(set)
    for a, b, _, label in ddi[train_indices]:
        if label == 1:
            a, b = int(a), int(b)
            adjacency[a].add(b)
            adjacency[b].add(a)
    features = {}
    for a, b in np.unique(ddi[:, :2], axis=0):
        a, b = int(a), int(b)
        # Query-edge masking: when (a,b) is itself a positive training edge,
        # do not let its own existence alter the pair's degree-based features.
        na = adjacency[a] - {b}
        nb = adjacency[b] - {a}
        common = na & nb
        union = na | nb
        aa = sum(1.0 / max(np.log(len(adjacency[x]) + 1.0), 1e-6) for x in common)
        ra = sum(1.0 / max(len(adjacency[x]), 1) for x in common)
        values = np.asarray([
            np.log1p(len(na)), np.log1p(len(nb)), np.log1p(len(common)),
            len(common) / max(len(union), 1), aa, ra,
            np.log1p(len(na) * len(nb)), 0.0,
        ], dtype=np.float32)
        features[(a, b)] = values
        features[(b, a)] = values[[1, 0, 2, 3, 4, 5, 6, 7]]
    return features


class DDIDataset(Dataset):
    def __init__(self, payload, indices):
        self.payload = payload
        self.rows = payload["ddi"][indices]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return tuple(map(int, self.rows[index]))


_TENSOR_CACHE = {}


def _tensorize_records(payload, max_motifs, max_kg_tokens):
    """Convert per-drug Python objects to dense CPU tensors once.

    The previous straightforward collator rebuilt these arrays for every sample
    and every epoch, leaving the GPU idle. Dense indexing makes collation a few
    tensor gathers and lets CUDA remain the training bottleneck.
    """
    key = (id(payload["records"]), max_motifs, max_kg_tokens)
    if key in _TENSOR_CACHE:
        return _TENSOR_CACHE[key]
    records = payload["records"]
    drug_ids = sorted(records)
    lookup = {drug: i for i, drug in enumerate(drug_ids)}
    count = len(drug_ids)
    mol = torch.zeros(count, max_motifs, MOTIF_DIM)
    mol_mask = torch.zeros(count, max_motifs, dtype=torch.bool)
    mol_dist = torch.ones(count, max_motifs, max_motifs)
    fingerprint = torch.zeros(count, FP_DIM)
    kg_rel = torch.zeros(count, max_kg_tokens, dtype=torch.long)
    kg_dir = torch.zeros(count, max_kg_tokens, dtype=torch.long)
    kg_nei = torch.zeros(count, max_kg_tokens, dtype=torch.long)
    kg_mask = torch.zeros(count, max_kg_tokens, dtype=torch.bool)
    kg_dist = torch.ones(count, max_kg_tokens, max_kg_tokens)
    for i, drug in enumerate(drug_ids):
        rec = records[drug]
        mt = torch.as_tensor(rec["mol_tokens"][:max_motifs])
        n = len(mt)
        mol[i, :n] = mt
        mol_mask[i, :n] = True
        mol_dist[i, :n, :n] = torch.as_tensor(rec["mol_dist"][:n, :n])
        fingerprint[i] = torch.as_tensor(rec["fingerprint"])
        tokens = rec["kg_tokens"][:max_kg_tokens]
        n = len(tokens)
        if n:
            kg_rel[i, :n] = torch.tensor([x[0] for x in tokens])
            kg_dir[i, :n] = torch.tensor([x[1] for x in tokens])
            kg_nei[i, :n] = torch.tensor([x[2] for x in tokens])
            kg_mask[i, :n] = True
            r, d = kg_rel[i, :n], kg_dir[i, :n]
            local = 1.0 - 0.5 * (r[:, None] == r[None, :]).float() - 0.25 * (d[:, None] == d[None, :]).float()
            local.fill_diagonal_(0.0)
            kg_dist[i, :n, :n] = local
    packed = {"lookup": lookup, "mol": mol, "mol_mask": mol_mask, "mol_dist": mol_dist, "fingerprint": fingerprint,
              "kg_rel": kg_rel, "kg_dir": kg_dir, "kg_nei": kg_nei,
              "kg_mask": kg_mask, "kg_dist": kg_dist}
    _TENSOR_CACHE[key] = packed
    return packed


def make_collate(payload, max_motifs, max_kg_tokens):
    packed = _tensorize_records(payload, max_motifs, max_kg_tokens)

    def collate(rows):
        values = torch.tensor(rows, dtype=torch.long)
        a_idx = torch.tensor([packed["lookup"][int(x)] for x in values[:, 0]])
        b_idx = torch.tensor([packed["lookup"][int(x)] for x in values[:, 1]])
        mol = torch.cat([packed["mol"][a_idx], packed["mol"][b_idx]], 1)
        mol_mask = torch.cat([packed["mol_mask"][a_idx], packed["mol_mask"][b_idx]], 1)
        mol_dist = torch.ones(len(rows), max_motifs * 2, max_motifs * 2)
        mol_dist[:, :max_motifs, :max_motifs] = packed["mol_dist"][a_idx]
        mol_dist[:, max_motifs:, max_motifs:] = packed["mol_dist"][b_idx]
        kg_rel = torch.cat([packed["kg_rel"][a_idx], packed["kg_rel"][b_idx]], 1)
        kg_dir = torch.cat([packed["kg_dir"][a_idx], packed["kg_dir"][b_idx]], 1)
        kg_nei = torch.cat([packed["kg_nei"][a_idx], packed["kg_nei"][b_idx]], 1)
        kg_mask = torch.cat([packed["kg_mask"][a_idx], packed["kg_mask"][b_idx]], 1)
        kg_dist = torch.ones(len(rows), max_kg_tokens * 2, max_kg_tokens * 2)
        kg_dist[:, :max_kg_tokens, :max_kg_tokens] = packed["kg_dist"][a_idx]
        kg_dist[:, max_kg_tokens:, max_kg_tokens:] = packed["kg_dist"][b_idx]
        return {
            "mol": mol, "mol_mask": mol_mask, "mol_dist": mol_dist,
            "kg_rel": kg_rel, "kg_dir": kg_dir, "kg_nei": kg_nei,
            "kg_mask": kg_mask, "kg_dist": kg_dist,
            "relation": values[:, 2],
            "label": values[:, 3].float(),
            "drug_pair": values[:, :2],
            "drug_index": torch.stack([a_idx, b_idx], dim=1),
            "fingerprint": torch.stack([packed["fingerprint"][a_idx], packed["fingerprint"][b_idx]], dim=1),
            "pair_struct": torch.stack([
                torch.as_tensor(payload["pair_features"].get((int(a), int(b)), np.zeros(8, np.float32)))
                for a, b in values[:, :2]
            ]),
        }
    return collate
