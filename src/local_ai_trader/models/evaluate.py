"""Classification and uncalibrated probability metrics with explicit conventions."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score

from local_ai_trader.models.baseline import CLASS_NAMES, encode_labels
from local_ai_trader.settings import positive_integer


def evaluate_probabilities(labels, probabilities: np.ndarray, bins: int = 10) -> dict:
    encoded = encode_labels(labels)
    positive_integer(bins, "calibration_bins")
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(encoded), len(CLASS_NAMES)) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("Probabilities must be finite normalized up/down/flat distributions")
    predicted = values.argmax(axis=1)
    report = classification_report(encoded, predicted, labels=np.arange(3), target_names=list(CLASS_NAMES), output_dict=True, zero_division=0)
    one_hot = np.eye(3)[encoded]
    confidence = values.max(axis=1)
    correct = (predicted == encoded).astype(float)
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    calibration = []
    ece = 0.0
    for index in range(bins):
        selected = indices == index
        count = int(selected.sum())
        average = float(confidence[selected].mean()) if count else None
        accuracy = float(correct[selected].mean()) if count else None
        if count:
            ece += count / len(encoded) * abs(average - accuracy)
        calibration.append({
            "lower": index / bins, "upper": (index + 1) / bins, "count": count,
            "mean_max_probability": average, "accuracy": accuracy,
        })
    auc = {}
    classwise_bins = {}
    classwise_ece = {}
    for index, name in enumerate(CLASS_NAMES):
        binary = encoded == index
        auc[name] = float(roc_auc_score(binary, values[:, index])) if binary.any() and not binary.all() else None
        probability = values[:, index]
        assignments = np.minimum((probability * bins).astype(int), bins - 1)
        reliability = []
        error = 0.0
        for bucket in range(bins):
            selected = assignments == bucket
            count = int(selected.sum())
            average = float(probability[selected].mean()) if count else None
            frequency = float(binary[selected].mean()) if count else None
            if count:
                error += count / len(encoded) * abs(average - frequency)
            reliability.append({
                "lower": bucket / bins, "upper": (bucket + 1) / bins, "count": count,
                "mean_probability": average, "observed_frequency": frequency,
            })
        classwise_bins[name] = reliability
        classwise_ece[name] = float(error)
    return {
        "rows": len(encoded), "accuracy": float(accuracy_score(encoded, predicted)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]), "f1_macro": float(report["macro avg"]["f1-score"]),
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "log_loss": float(log_loss(encoded, values, labels=np.arange(3))),
        "brier_score": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
        "brier_convention": "mean sum of squared errors over all three classes; range 0 to 2",
        "ece": float(ece), "ece_convention": "top-label confidence, equal-width bins; final bin includes 1",
        "calibration_bins": calibration, "roc_auc_ovr_per_class": auc,
        "classwise_calibration_bins": classwise_bins, "classwise_ece": classwise_ece,
        "classwise_ece_macro": float(np.mean(list(classwise_ece.values()))),
        "roc_auc_ovr_macro": float(np.mean(list(auc.values()))) if all(value is not None for value in auc.values()) else None,
        "calibration_status": "uncalibrated",
    }
