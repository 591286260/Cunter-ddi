"""Training and evaluation utilities for one cross-validation fold."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DDIDataset, make_collate, training_graph_pair_features
from metrics import binary_metrics
from model import CPSTDDI


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def select_device():
    """Use CUDA only when this PyTorch build contains the GPU architecture."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    required = f"sm_{major}{minor}"
    compiled = set(torch.cuda.get_arch_list())
    if compiled and required not in compiled:
        print(
            f"[warning] GPU requires {required}, but this PyTorch build supports "
            f"{sorted(compiled)}; falling back to CPU. Install a matching CUDA "
            "PyTorch build to enable GPU training.",
            flush=True,
        )
        return torch.device("cpu")
    return torch.device("cuda")


def make_loader(payload, indices, cfg, shuffle):
    return DataLoader(
        DDIDataset(payload, indices), batch_size=cfg["batch_size"], shuffle=shuffle,
        num_workers=cfg["num_workers"], pin_memory=True,
        collate_fn=make_collate(payload, cfg["max_motifs"], cfg["max_kg_tokens"]),
    )


def training_loss(output, labels, cfg):
    loss = F.binary_cross_entropy_with_logits(output["logits"], labels)
    parts = {"bce": float(loss.detach())}
    if cfg["use_counterfactual"] and "keep_logits" in output:
        original = torch.sigmoid(output["logits"]).detach()
        keep = torch.sigmoid(output["keep_logits"])
        remove = torch.sigmoid(output["remove_logits"])
        sufficiency = F.mse_loss(keep, original)
        # Correct-label confidence must fall after removing the core plan.
        sign = labels * 2.0 - 1.0
        original_conf = torch.sigmoid(sign * output["logits"]).detach()
        remove_conf = torch.sigmoid(sign * output["remove_logits"])
        necessity = F.relu(cfg["necessity_margin"] - original_conf + remove_conf).mean()
        loss = loss + cfg["lambda_sufficiency"] * sufficiency + cfg["lambda_necessity"] * necessity
        parts.update(sufficiency=float(sufficiency.detach()), necessity=float(necessity.detach()))
    return loss, parts


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities, pairs, relations = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        labels.extend(batch["label"].cpu().numpy().tolist())
        probabilities.extend(torch.sigmoid(output["logits"]).cpu().numpy().tolist())
        pairs.extend(batch["drug_pair"].cpu().numpy().tolist())
        relations.extend(batch["relation"].cpu().numpy().tolist())
    return (
        binary_metrics(labels, probabilities),
        np.asarray(labels),
        np.asarray(probabilities),
        np.asarray(pairs),
        np.asarray(relations),
    )


def train_fold(payload, train_idx, val_idx, test_idx, cfg, fold_dir: Path, fold: int):
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"] + fold
    set_seed(seed)
    device = select_device()
    print(f"[device] {device}", flush=True)
    # Fold-local graph features are constructed exclusively from training
    # positive edges and then used for train/validation/test pair featurization.
    fold_payload = dict(payload)
    fold_payload["pair_features"] = training_graph_pair_features(payload["ddi"], train_idx)
    model = CPSTDDI(fold_payload, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = make_loader(fold_payload, train_idx, cfg, True)
    val_loader = make_loader(fold_payload, val_idx, cfg, False)
    test_loader = make_loader(fold_payload, test_idx, cfg, False)

    best_score, best_epoch, stale = -np.inf, 0, 0
    checkpoint = fold_dir / "best.pt"
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(batch, counterfactual=True)
                loss, _ = training_loss(output, batch["label"], cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * len(batch["label"])
            seen += len(batch["label"])

        val_metrics, _, _, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), **val_metrics})
        score = val_metrics["AUPR"]
        print(
            f"fold={fold} epoch={epoch:03d} loss={running/max(seen,1):.4f} "
            f"val_auc={val_metrics['AUC']:.4f} val_aupr={score:.4f}", flush=True
        )
        if score > best_score + 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, checkpoint)
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    metrics, labels, probs, pairs, relations = evaluate(model, test_loader, device)
    np.savez_compressed(
        fold_dir / "predictions.npz",
        labels=labels,
        probabilities=probs,
        drug_pairs=pairs,
        relations=relations,
        sample_indices=np.asarray(test_idx),
    )
    with (fold_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_epoch": best_epoch, **metrics}, handle, indent=2)
    return metrics
