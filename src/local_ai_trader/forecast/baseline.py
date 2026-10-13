"""Training-mean and scaled Ridge forecasts of execution-aligned gross returns."""

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from local_ai_trader.data.validate import validate_candle_timeline
from local_ai_trader.decision.cost_aware import ReturnForecast
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.forecast.split import validate_execution_features
from local_ai_trader.models.baseline import feature_matrix
from local_ai_trader.settings import Settings, positive_integer, validate_symbol

MODEL_NAMES = ("mean", "ridge")


def return_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    if not frame.columns.is_unique:
        raise ValueError("Return model columns must be unique")
    for name in FEATURE_COLUMNS:
        if name in frame and pd.api.types.is_complex_dtype(frame[name]):
            raise ValueError("Return features must be real numeric values")
    return feature_matrix(frame)


def validate_ridge_alpha(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("return_forecast.ridge_alpha must be positive and finite")
    return float(value)


def load_return_ridge_alpha(path: Path) -> float:
    """Read a separate table without altering frozen classifier Settings."""
    with path.open("rb") as stream:
        values = tomllib.load(stream).get("return_forecast", {})
    if not isinstance(values, dict) or set(values) - {"ridge_alpha"}:
        raise ValueError("Unsupported return_forecast setting")
    return validate_ridge_alpha(values.get("ridge_alpha", 1.0))


def return_targets(frame: pd.DataFrame) -> np.ndarray:
    name = "execution_target_return"
    if name not in frame or frame.empty or not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]) or pd.api.types.is_complex_dtype(frame[name]):
        raise ValueError("Execution gross return targets must be nonempty and numeric")
    values = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values).all() or (values < -1).any():
        raise ValueError("Execution gross return targets must be finite and at least -1")
    return values


@dataclass
class FittedReturnBaselines:
    ridge: Pipeline
    training_mean: float
    metadata: dict

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        frame = validate_candle_timeline(frame, self.metadata["candle_seconds"])
        if frame.symbol.iloc[0] != self.metadata["symbol"] or frame.exchange.iloc[0] != self.metadata["exchange"]:
            raise ValueError("Return inference market disagrees with fitted model")
        inputs = return_feature_matrix(frame)
        prediction = self.ridge.predict(inputs)
        if not np.isfinite(prediction).all():
            raise ValueError("Return inference produced nonfinite values")
        return {"mean": np.full(len(inputs), self.training_mean), "ridge": prediction}

    def checkpoint(self) -> dict:
        scaler, regressor = self.ridge.named_steps["scaler"], self.ridge.named_steps["regressor"]
        return {
            **self.metadata, "schema_version": 1, "target_kind": "execution_open_to_open",
            "target_column": "execution_target_return", "return_unit": "fractional_reference_price_return",
            "feature_names": list(FEATURE_COLUMNS), "model_names": list(MODEL_NAMES),
            "training_mean_return": self.training_mean, "training_rows": int(scaler.n_samples_seen_),
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "scaler_variance": scaler.var_.tolist(), "ridge_coefficients": regressor.coef_.tolist(),
            "ridge_intercept": float(regressor.intercept_), "ridge_alpha": float(regressor.alpha),
            "ridge_solver": "svd", "normalization": "StandardScaler fitted on training rows only",
            "prediction_clipping": "none", "uncertainty_estimate": "not supplied by these point forecasts",
        }


def fit_return_baselines(train: pd.DataFrame, settings: Settings, ridge_alpha: float = 1.0) -> FittedReturnBaselines:
    """Fit both controls from training inputs/outcomes alone, using deterministic SVD."""
    alpha = validate_ridge_alpha(ridge_alpha)
    train = validate_execution_features(train, settings.candle_seconds, settings.horizon_steps, settings.backtest.latency_bars)
    inputs, outcomes = return_feature_matrix(train), return_targets(train)
    if len(train) < 2:
        raise ValueError("Return baselines require at least two training rows")
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=alpha, solver="svd"))])
    pipeline.fit(inputs, outcomes)
    fitted = FittedReturnBaselines(pipeline, float(outcomes.mean()), {
        "symbol": str(train.symbol.iloc[0]), "exchange": str(train.exchange.iloc[0]),
        "candle_seconds": settings.candle_seconds, "holding_bars": settings.horizon_steps,
        "latency_bars": settings.backtest.latency_bars,
        "training_last_label_known_at": train.execution_target_available_at.max().isoformat(),
    })
    # Validate all serialized parameters and reproduce native training inference.
    restored = predict_return_checkpoint(fitted.checkpoint(), train)
    for name, prediction in fitted.predict(train).items():
        np.testing.assert_allclose(restored[name], prediction, rtol=1e-10, atol=1e-12)
    return fitted


def _parameter(checkpoint: dict, name: str, shape: tuple) -> np.ndarray:
    values = np.asarray(checkpoint.get(name))
    if values.shape != shape or values.dtype.kind not in "fiu":
        raise ValueError(f"Invalid return checkpoint parameter {name}")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite return checkpoint parameter {name}")
    return values


def predict_return_checkpoint(checkpoint: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Inference from JSON weights and training scaling; labels are never inputs."""
    if (
        not isinstance(checkpoint, dict) or type(checkpoint.get("schema_version")) is not int or checkpoint["schema_version"] != 1
        or checkpoint.get("target_kind") != "execution_open_to_open" or checkpoint.get("target_column") != "execution_target_return"
        or checkpoint.get("return_unit") != "fractional_reference_price_return"
        or checkpoint.get("feature_names") != list(FEATURE_COLUMNS) or checkpoint.get("model_names") != list(MODEL_NAMES)
        or checkpoint.get("ridge_solver") != "svd"
    ):
        raise ValueError("Incompatible execution-return checkpoint schema or input order")
    for name in ("candle_seconds", "holding_bars", "latency_bars", "training_rows"):
        positive_integer(checkpoint.get(name), name)
    validate_ridge_alpha(checkpoint.get("ridge_alpha"))
    if not isinstance(checkpoint.get("symbol"), str):
        raise ValueError("Return checkpoint requires a market symbol")
    validate_symbol(checkpoint["symbol"])
    if checkpoint.get("exchange") != "coinbase":
        raise ValueError("Return checkpoints currently require Coinbase")
    candles = validate_candle_timeline(frame, checkpoint["candle_seconds"])
    if candles.symbol.iloc[0] != checkpoint["symbol"] or candles.exchange.iloc[0] != checkpoint["exchange"]:
        raise ValueError("Return inference market disagrees with checkpoint")
    width = len(FEATURE_COLUMNS)
    mean, scale = _parameter(checkpoint, "scaler_mean", (width,)), _parameter(checkpoint, "scaler_scale", (width,))
    variance = _parameter(checkpoint, "scaler_variance", (width,))
    coefficients = _parameter(checkpoint, "ridge_coefficients", (width,))
    training_mean = _parameter(checkpoint, "training_mean_return", ())
    intercept = _parameter(checkpoint, "ridge_intercept", ())
    if not (scale > 0).all() or (variance < 0).any() or training_mean < -1:
        raise ValueError("Invalid return checkpoint scaling or training mean")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        prediction = ((return_feature_matrix(candles) - mean) / scale) @ coefficients + intercept
    if not np.isfinite(prediction).all():
        raise ValueError("Return checkpoint inference produced nonfinite values")
    return {"mean": np.full(len(candles), float(training_mean)), "ridge": prediction}


def checkpoint_return_forecasts(checkpoint: dict, frame: pd.DataFrame, model: str = "ridge") -> list[ReturnForecast]:
    """Create aligned point forecasts only after all training labels were known.

    A forecast is not a profitability guarantee or a trade authorization.
    Freshness/account/risk checks still belong to the eventual paper coordinator.
    Invalid outputs are refused rather than clipped into plausible forecasts.
    """
    if model not in MODEL_NAMES:
        raise ValueError("Unknown return model")
    predictions = predict_return_checkpoint(checkpoint, frame)[model]
    if not isinstance(checkpoint.get("training_last_label_known_at"), str):
        raise ValueError("Return checkpoint requires training label availability")
    known_at = pd.Timestamp(checkpoint["training_last_label_known_at"])
    if pd.isna(known_at) or known_at.tzinfo is None:
        raise ValueError("Return checkpoint requires timezone-aware training label availability")
    times = frame.available_at.dt.tz_convert("UTC")
    if not (times > known_at).all():
        raise ValueError("Return forecasts cannot precede training label availability")
    if (predictions < -1).any():
        raise ValueError("Return forecast below -100% is invalid; no clipping is applied")
    return [ReturnForecast(float(value), checkpoint["symbol"], stamp.to_pydatetime(), checkpoint["candle_seconds"], checkpoint["latency_bars"], checkpoint["holding_bars"]) for value, stamp in zip(predictions, times, strict=True)]


def evaluate_return_predictions(outcomes, predictions) -> dict:
    """Fractional-return errors; constant-target R-squared is explicitly undefined."""
    actual, estimated = np.asarray(outcomes), np.asarray(predictions)
    if actual.ndim != 1 or actual.shape != estimated.shape or not len(actual) or actual.dtype.kind not in "fiu" or estimated.dtype.kind not in "fiu":
        raise ValueError("Return metrics require matching nonempty numeric vectors")
    actual, estimated = actual.astype(np.float64), estimated.astype(np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(estimated).all() or (actual < -1).any():
        raise ValueError("Return metrics require finite outcomes and predictions")
    with np.errstate(over="ignore", invalid="ignore"):
        errors = estimated - actual
        mse = float(np.mean(errors ** 2))
        target_variance = 0.0 if (actual == actual[0]).all() else float(np.mean((actual - actual.mean()) ** 2))
        result = {
            "rows": len(actual), "mae": float(np.mean(np.abs(errors))), "rmse": math.sqrt(mse),
            "r2": 1 - mse / target_variance if target_variance > 0 else None,
            "mean_error": float(errors.mean()), "mean_prediction": float(estimated.mean()), "mean_observed_return": float(actual.mean()),
            "invalid_forecast_rows": int((estimated < -1).sum()),
            "return_unit": "fractional_reference_price_return", "r2_convention": "undefined for constant outcomes",
            "interpretation": "point forecast errors; no profit, trade-confidence or independent-observation claim",
        }
    if any(not math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise ValueError("Return metric calculation produced nonfinite values")
    return result
