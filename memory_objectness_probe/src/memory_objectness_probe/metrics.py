from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        mat[int(true), int(pred)] += 1
    return mat


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def class_recall(cm: np.ndarray) -> list[float]:
    recalls = []
    for i in range(cm.shape[0]):
        denom = cm[i, :].sum()
        recalls.append(float(cm[i, i] / denom) if denom else 0.0)
    return recalls


def binary_auc(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    # Mann-Whitney U, with average tie handling.
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[y == 1].sum()
    auc = (rank_sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def binary_auprc(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if np.sum(y == 1) == 0:
        return None
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.sum(y == 1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    *,
    scores_for_positive: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true_arr, y_pred_arr, num_classes)
    acc = float((y_true_arr == y_pred_arr).mean()) if y_true_arr.size else 0.0
    recalls = class_recall(cm)
    result: Dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": macro_f1_from_confusion(cm),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": recalls,
    }
    if num_classes == 2:
        result["balanced_accuracy"] = float(np.mean(recalls))
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        result["f1"] = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if scores_for_positive is not None:
            result["auroc"] = binary_auc(y_true_arr, scores_for_positive)
            result["auprc"] = binary_auprc(y_true_arr, scores_for_positive)
    return result
