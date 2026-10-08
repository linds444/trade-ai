"""Train-only class-frequency and scaled logistic regression baselines."""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.settings import Settings

CLASS_NAMES = ("up", "down", "flat")


def encode_labels(labels) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or not len(values) or not all(value in CLASS_NAMES for value in values):
        raise ValueError("Labels must be a nonempty one-dimensional sequence of up/down/flat")
    return np.array([CLASS_NAMES.index(value) for value in values], dtype=np.int64)


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    if not set(FEATURE_COLUMNS).issubset(frame.columns) or frame.empty:
        raise ValueError("Missing model input features or empty dataset")
    for name in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
            raise ValueError(f"{name} must be numeric")
    values = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError("Model inputs must be finite")
    return values


@dataclass
class FittedBaselines:
    logistic: Pipeline
    naive_probabilities: np.ndarray

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        inputs = feature_matrix(frame)
        native = self.logistic.predict_proba(inputs)
        aligned = np.zeros((len(inputs), len(CLASS_NAMES)), dtype=np.float64)
        aligned[:, self.logistic.named_steps["classifier"].classes_] = native
        return {
            "naive": np.tile(self.naive_probabilities, (len(inputs), 1)),
            "logistic": aligned,
        }

    def checkpoint(self) -> dict:
        scaler = self.logistic.named_steps["scaler"]
        classifier = self.logistic.named_steps["classifier"]
        return {
            "schema_version": 1, "feature_names": list(FEATURE_COLUMNS),
            "class_names": list(CLASS_NAMES), "calibration": "none",
            "naive_probabilities": self.naive_probabilities.tolist(),
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "scaler_variance": scaler.var_.tolist(), "training_rows": int(scaler.n_samples_seen_),
            "logistic_classes": classifier.classes_.tolist(),
            "logistic_coefficients": classifier.coef_.tolist(),
            "logistic_intercept": classifier.intercept_.tolist(),
            "logistic_iterations": classifier.n_iter_.tolist(),
            "logistic_probability_link": "binary_sigmoid" if len(classifier.classes_) == 2 else "multinomial_softmax",
        }


def fit_baselines(train: pd.DataFrame, settings: Settings) -> FittedBaselines:
    inputs = feature_matrix(train)
    labels = encode_labels(train["target_class"])
    if len(np.unique(labels)) < 2:
        raise ValueError("Logistic regression needs at least two training classes; use more history")
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    naive = (counts + settings.naive_smoothing) / (len(labels) + len(CLASS_NAMES) * settings.naive_smoothing)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=settings.logistic_c, max_iter=settings.logistic_max_iter,
            solver="lbfgs", random_state=settings.random_seed,
        )),
    ])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(inputs, labels)
    except ConvergenceWarning as error:
        raise ValueError("Logistic regression did not converge; increase baseline.logistic_max_iter") from error
    return FittedBaselines(pipeline, naive)


def predict_checkpoint(checkpoint: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Repeat inference from JSON weights and the exact fitted preprocessing."""
    if checkpoint.get("schema_version") != 1 or checkpoint.get("feature_names") != list(FEATURE_COLUMNS) or checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Incompatible checkpoint schema, feature order or class order")
    mean = np.asarray(checkpoint["scaler_mean"], dtype=float)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=float)
    coefficients = np.asarray(checkpoint["logistic_coefficients"], dtype=float)
    intercept = np.asarray(checkpoint["logistic_intercept"], dtype=float)
    classes = np.asarray(checkpoint["logistic_classes"])
    naive = np.asarray(checkpoint["naive_probabilities"], dtype=float)
    if classes.ndim != 1 or len(classes) not in (2, 3) or classes.dtype.kind not in "iu" or len(np.unique(classes)) != len(classes) or not np.isin(classes, np.arange(3)).all():
        raise ValueError("Invalid checkpoint classes")
    width = len(FEATURE_COLUMNS)
    rows = 1 if len(classes) == 2 else len(classes)
    if mean.shape != (width,) or scale.shape != (width,) or coefficients.shape != (rows, width) or intercept.shape != (rows,) or naive.shape != (3,):
        raise ValueError("Invalid checkpoint parameter shapes")
    if not all(np.isfinite(values).all() for values in (mean, scale, coefficients, intercept, naive)) or not (scale > 0).all() or (naive < 0).any() or not np.isclose(naive.sum(), 1):
        raise ValueError("Invalid checkpoint parameters")
    inputs = (feature_matrix(frame) - mean) / scale
    scores = inputs @ coefficients.T + intercept
    if not np.isfinite(scores).all():
        raise ValueError("Inference produced nonfinite scores")
    if len(classes) == 2:
        values = scores[:, 0]
        positive = values >= 0
        p_one = np.empty(len(values))
        p_one[positive] = 1 / (1 + np.exp(-values[positive]))
        exponent = np.exp(values[~positive])
        p_one[~positive] = exponent / (1 + exponent)
        native = np.column_stack((1 - p_one, p_one))
    else:
        exponent = np.exp(scores - scores.max(axis=1, keepdims=True))
        native = exponent / exponent.sum(axis=1, keepdims=True)
    aligned = np.zeros((len(inputs), len(CLASS_NAMES)))
    aligned[:, classes] = native
    return {"naive": np.tile(naive, (len(inputs), 1)), "logistic": aligned}
