from typing import List, Dict, Tuple
from collections import Counter


def compute_confusion_metrics(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    labels_set = sorted(set(labels) | set(predictions))
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for pred, true in zip(predictions, labels):
        if pred == true:
            if pred != "SALAH":
                tp += 1
            else:
                tn += 1
        else:
            if pred == "BENAR":
                fp += 1
            else:
                fn += 1

    accuracy = (tp + tn) / len(labels) if labels else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": len(labels),
        "classes": labels_set,
    }


def evaluate_sample_results(results: List[Dict[str, str]]) -> Dict[str, object]:
    predictions = [item["prediction"] for item in results]
    labels = [item["label"] for item in results]
    return compute_confusion_metrics(predictions, labels)
