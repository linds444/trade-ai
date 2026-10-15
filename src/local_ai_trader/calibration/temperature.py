"""Scalar temperature scaling of class probabilities without changing their rank."""

from dataclasses import asdict
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from local_ai_trader.models.baseline import CLASS_NAMES, encode_labels
from local_ai_trader.settings import CalibrationSettings, load_calibration_settings


def probability_logits(probabilities: np.ndarray, floor: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES) or not len(values) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("Temperature scaling needs finite normalized up/down/flat probabilities")
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or not 0 < floor <= 1e-6:
        raise ValueError("Probability floor must be positive and at most 1e-6")
    return np.log(np.maximum(values, floor))


def apply_temperature(probabilities: np.ndarray, temperature: float, floor: float = 1e-12) -> np.ndarray:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a positive finite number")
    with np.errstate(over="ignore", invalid="ignore"):
        logits = probability_logits(probabilities, floor) / temperature
    if not np.isfinite(logits).all():
        raise ValueError("Temperature scaling produced nonfinite logits")
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit_temperature(labels, probabilities: np.ndarray, config: CalibrationSettings) -> dict:
    """Minimize fit-period log loss only; always include identity as a candidate."""
    config = load_calibration_settings(asdict(config))
    targets = encode_labels(labels)
    logits = probability_logits(probabilities, config.probability_floor)
    if len(targets) != len(logits):
        raise ValueError("Calibration labels and probabilities have different lengths")

    def objective(temperature: float) -> float:
        scaled = logits / temperature
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]))

    result = minimize_scalar(objective, bounds=(config.min_temperature, config.max_temperature), method="bounded", options={"xatol": config.optimizer_tolerance, "maxiter": config.max_iterations})
    if not result.success or not math.isfinite(result.fun):
        raise ValueError("Temperature optimization did not converge; no calibration published")
    temperature = 1.0
    initial = objective(temperature)
    loss = initial
    for candidate in (float(result.x), config.min_temperature, config.max_temperature):
        candidate_loss = objective(candidate)
        if candidate_loss < loss - 1e-12:
            temperature, loss = candidate, candidate_loss
    return {
        "temperature": float(temperature), "fit_rows": len(targets),
        "objective_at_identity": initial, "fit_objective": loss,
        "objective": "mean negative log probability after flooring input scores",
        "optimizer": "scipy bounded scalar minimization", "optimizer_success": True,
        "optimizer_evaluations": int(result.nfev),
        "fit_class_counts": {name: int((targets == index).sum()) for index, name in enumerate(CLASS_NAMES)},
    }
