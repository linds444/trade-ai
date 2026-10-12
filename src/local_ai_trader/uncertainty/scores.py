"""Label-free diagnostic scores for aligned up/down/flat distributions."""

import numpy as np

from local_ai_trader.models.baseline import CLASS_NAMES


def probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Require nonempty, finite, normalized numeric distributions in class order."""
    values = np.asarray(probabilities)
    if values.dtype.kind not in "fiu" or values.ndim != 2 or values.shape[1] != len(CLASS_NAMES) or not len(values):
        raise ValueError("Probabilities must be a nonempty numeric row matrix in up/down/flat order")
    values = values.astype(np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("Probabilities must be finite normalized up/down/flat distributions")
    return values


def _entropy(values: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats, including the exact zero-probability limit."""
    logs = np.zeros_like(values)
    np.log(values, out=logs, where=values > 0)
    return -(values * logs).sum(axis=1)


def uncertainty_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Describe predictive ambiguity without labels or a calibrated correctness claim.

    Entropy is normalized by log(3): deterministic=0, uniform=1.
    The margin is largest minus second-largest probability. Neither entropy's
    complement nor max probability is reported as a probability of being correct.
    """
    values = probability_matrix(probabilities)
    ordered = np.sort(values, axis=1)
    return {
        "entropy_normalized": np.clip(_entropy(values) / np.log(len(CLASS_NAMES)), 0, 1),
        "top_class_probability": ordered[:, -1],
        "top_class_margin": ordered[:, -1] - ordered[:, -2],
    }


def model_disagreement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Equal-weight two-model Jensen-Shannon divergence divided by log(2).

    Identical distributions yield 0; disjoint supports yield 1. Inputs must refer
    to the same prediction times and class order; callers establish alignment.
    This is a comparison diagnostic, not an epistemic uncertainty guarantee.
    """
    left, right = probability_matrix(first), probability_matrix(second)
    if left.shape != right.shape:
        raise ValueError("Model probability matrices must have matching rows and classes")
    mixture = (left + right) / 2
    divergence = _entropy(mixture) - (_entropy(left) + _entropy(right)) / 2
    return np.clip(divergence / np.log(2), 0, 1)
