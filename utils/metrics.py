import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix


def compute_uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return balanced_accuracy_score(y_true, y_pred) * 100


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro") * 100


def compute_class_recall(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> list:
    recalls = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append((y_pred[mask] == c).sum() / mask.sum() * 100)
    return recalls


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict:
    return {
        "uar": compute_uar(y_true, y_pred),
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "class_recall": compute_class_recall(y_true, y_pred, num_classes),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
    }
