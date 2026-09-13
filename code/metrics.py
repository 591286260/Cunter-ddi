"""Metric computation and exact TSV result formatting."""

from pathlib import Path
import csv
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score


METRIC_NAMES = ["Accuracy", "Precision", "F1", "AUC", "AUPR"]


def binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
        "AUPR": average_precision_score(y_true, y_prob),
    }


def save_cv_table(rows, path):
    """Save five folds plus Average and population Std in the requested form."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray([[r[k] for k in METRIC_NAMES] for r in rows], dtype=float)
    labels = [str(i + 1) for i in range(len(rows))] + ["Average", "Std"]
    output = np.vstack([values, values.mean(axis=0), values.std(axis=0, ddof=0)])
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow([""] + METRIC_NAMES)
        for label, row in zip(labels, output):
            writer.writerow([label] + [f"{x:.4f}" for x in row])
