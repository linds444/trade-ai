"""Execute a frozen research protocol, retaining atomic bundles on interruption."""

from dataclasses import asdict, replace
import json
import logging
import os
from pathlib import Path

import pandas as pd

from local_ai_trader.calibration.run import create_calibration, validation_regions
from local_ai_trader.data.storage import write_json
from local_ai_trader.models.heldout import evaluate_heldout
from local_ai_trader.models.inference import load_frozen_partition
from local_ai_trader.models.train import file_digest, train_baseline_experiment
from local_ai_trader.settings import Settings
from local_ai_trader.walkforward.protocol import Protocol, load_protocol
from local_ai_trader.walkforward.results import publish_fold_backtest, publish_summary, print_summary

LOGGER = logging.getLogger(__name__)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def file_manifest(directory: Path, recursive: bool = True) -> dict[str, str]:
    paths = directory.rglob("*") if recursive else directory.iterdir()
    return {path.relative_to(directory).as_posix(): file_digest(path) for path in sorted(paths) if path.is_file()}


def record_bundle(state: dict, name: str, directory: Path, recursive: bool = True) -> None:
    actual = file_manifest(directory, recursive)
    if not actual:
        raise ValueError(f"Saved {name} bundle is empty or missing")
    if name in state and state[name] != actual:
        raise ValueError(f"Saved {name} bundle changed; refusing to replace it")
    state[name] = actual


def canonical(value: dict) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def native_hashes(checkpoint: dict) -> dict:
    return {checkpoint["model_file"]: checkpoint["model_sha256"]} if "model_file" in checkpoint else {}


def validate_model(protocol: Protocol, index: int, family: str, directory: Path) -> dict:
    checkpoint = read_json(directory / "checkpoint.json")
    metadata = checkpoint["training_metadata"]
    expected = {name: protocol.plan["fold_file_hashes"][f"fold_{index:02d}/{name}"] for name in ("split.json", "train.parquet", "validation.parquet")}
    frozen = asdict(protocol.settings)
    actual = metadata["settings"].copy()
    # Output directories may move; all numerical parameters remain bound.
    frozen.pop("models_dir")
    actual.pop("models_dir")
    if (
        metadata["input_hashes"] != expected or metadata["symbol"] != protocol.plan["symbol"]
        or metadata["exchange"] != protocol.plan["exchange"]
        or canonical(actual) != canonical(frozen)
        or checkpoint.get("model_type") != (None if family == "baseline" else protocol.plan["model_family"])
        or read_json(directory / "experiment.json") != metadata
    ):
        raise ValueError("Saved model does not match this fold's frozen training recipe")
    for filename, digest in native_hashes(checkpoint).items():
        if Path(filename).name != filename or file_digest(directory / filename) != digest:
            raise ValueError("Native model weights changed")
    # This validates input times, counts and class/feature schemas without test I/O.
    load_frozen_partition(directory, "validation", protocol.fold_directory(index))
    return checkpoint


def validate_calibration(protocol: Protocol, index: int, directory: Path, checkpoint: dict) -> None:
    calibration = read_json(directory / "calibration/calibration.json")
    snapshot = load_frozen_partition(directory, "validation", protocol.fold_directory(index))
    _, regions = validation_regions(snapshot.frame, protocol.settings.calibration.fit_fraction)
    names = {"naive", "logistic"} if checkpoint.get("model_type") is None else {checkpoint["model_type"]}
    if (
        calibration.get("schema_version") != 1
        or calibration.get("method") != "temperature_scaling_on_probabilities"
        or calibration["base_checkpoint_sha256"] != file_digest(directory / "checkpoint.json")
        or calibration["source_input_hashes"] != checkpoint["training_metadata"]["input_hashes"]
        or calibration["model_artifact_hashes"] != native_hashes(checkpoint)
        or calibration["parameters"] != asdict(protocol.settings.calibration)
        or calibration["regions"] != regions or set(calibration["models"]) != names
        or calibration["class_names"] != protocol.plan["class_names"]
        or calibration["feature_names"] != protocol.plan["feature_names"]
        or calibration.get("model_weights_refitted") is not False
        or calibration.get("test_data_opened") is not False
    ):
        raise ValueError("Saved calibration differs from the frozen fold recipe")
    parameters = protocol.settings.calibration
    for model in calibration["models"].values():
        temperature = model["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (float, int)) or not parameters.min_temperature <= temperature <= parameters.max_temperature:
            raise ValueError("Saved temperature is outside frozen bounds")


def validate_evaluation(protocol: Protocol, index: int, directory: Path, checkpoint: dict) -> None:
    report = read_json(directory / "test_evaluation/metrics.json")
    names = ("naive", "logistic") if checkpoint.get("model_type") is None else (checkpoint["model_type"],)
    expected_names = {name for model in names for name in (model, f"{model}_temperature")}
    if (
        report.get("schema_version") != 1 or report.get("partition") != "test"
        or report["checkpoint_sha256"] != file_digest(directory / "checkpoint.json")
        or report["split_manifest_sha256"] != protocol.plan["fold_file_hashes"][f"fold_{index:02d}/split.json"]
        or report["test_parquet_sha256"] != protocol.plan["fold_file_hashes"][f"fold_{index:02d}/test.parquet"]
        or report["calibration_checkpoint_sha256"] != file_digest(directory / "calibration/calibration.json")
        or report["model_artifact_hashes"] != native_hashes(checkpoint)
        or report["rows"] != protocol.plan["folds"][index - 1]["partitions"]["test"]["rows"]
        or report.get("refitted") is not False or report.get("calibration_fitted_during_evaluation") is not False
        or set(report["metrics"]) != expected_names
        or {p.name for p in (directory / "test_evaluation").iterdir()} != {"metrics.json", *(f"{name}.parquet" for name in expected_names)}
    ):
        raise ValueError("Saved evaluation does not match the frozen fold and model")


def fit_family(protocol: Protocol, index: int, family: str, root: Path) -> Path:
    settings: Settings = replace(protocol.settings, models_dir=root)
    if family == "baseline":
        return train_baseline_experiment(protocol.fold_directory(index), settings)[0]
    if protocol.plan["model_family"] == "xgboost":
        from local_ai_trader.models.boosting import train_boosting_experiment

        return train_boosting_experiment(protocol.fold_directory(index), settings)[0]
    from local_ai_trader.models.mlp import train_mlp_experiment

    return train_mlp_experiment(protocol.fold_directory(index), settings)[0]


def execute(protocol: Protocol, output: Path, state: dict) -> dict:
    def save():
        write_json(output / "state.json", state)

    runs: dict[int, dict[str, Path]] = {}
    # Phase one: freeze every model and temperature before opening any test file.
    for index in range(1, len(protocol.plan["folds"]) + 1):
        fold_output = output / f"fold_{index:02d}"
        fold_output.mkdir(exist_ok=True)
        fold_state = state["folds"].setdefault(str(index), {})
        runs[index] = {}
        for family in ("baseline", "selected"):
            protocol.verify()
            root = fold_output / f"{family}_models"
            root.mkdir(exist_ok=True)
            entry = fold_state.setdefault(family, {})
            if "run" in entry:
                directory = (output / entry["run"]).resolve()
            else:
                candidates = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
                if len(candidates) > 1:
                    raise ValueError("Multiple saved models exist for one fold; refusing ambiguous recovery")
                directory = candidates[0] if candidates else fit_family(protocol, index, family, root)
                entry["run"] = directory.relative_to(output).as_posix()
            if directory.resolve().parent != root.resolve():
                raise ValueError("Saved model path escapes its fold output directory")
            checkpoint = validate_model(protocol, index, family, directory)
            record_bundle(entry, "model", directory, recursive=False)
            protocol.verify()
            save()
            runs[index][family] = directory
            if not (directory / "calibration").exists():
                if "calibration" in entry:
                    raise ValueError("Previously saved calibration is missing")
                create_calibration(directory, protocol.settings.calibration, protocol.fold_directory(index))
            validate_calibration(protocol, index, directory, checkpoint)
            record_bundle(entry, "calibration", directory / "calibration")
            protocol.verify()
            save()
            LOGGER.info("Fold %s/%s %s model and calibration frozen", index, len(protocol.plan["folds"]), family)

    def verify_all():
        protocol.verify(include_test=True)
        for index, families in runs.items():
            fold_state = state["folds"][str(index)]
            for family, directory in families.items():
                entry = fold_state[family]
                record_bundle(entry, "model", directory, recursive=False)
                for bundle in ("calibration", "test_evaluation"):
                    if bundle in entry:
                        record_bundle(entry, bundle, directory / bundle)
            if "backtest" in fold_state:
                record_bundle(fold_state, "backtest", output / f"fold_{index:02d}/backtest")
        if "summary" in state:
            record_bundle(state, "summary", output / "summary")

    # Phase two: paired saved predictions, then costs on the same frozen opens.
    verify_all()
    for index, families in runs.items():
        fold_state = state["folds"][str(index)]
        for family, directory in families.items():
            entry = fold_state[family]
            verify_all()
            if not (directory / "test_evaluation").exists():
                if "test_evaluation" in entry:
                    raise ValueError("Previously saved evaluation is missing")
                evaluate_heldout(directory, protocol.fold_directory(index), calibrated=True)
            validate_evaluation(protocol, index, directory, read_json(directory / "checkpoint.json"))
            record_bundle(entry, "test_evaluation", directory / "test_evaluation")
            verify_all()
            save()
        fold_output = output / f"fold_{index:02d}"
        if not (fold_output / "backtest").exists():
            publish_fold_backtest(protocol, index, fold_output, families, verify_all)
        report = read_json(fold_output / "backtest/backtest.json")
        names = {f"{model}_{variant}" for model in ("naive", "logistic", protocol.plan["model_family"]) for variant in ("raw", "temperature")} | {"cash", "buy_and_hold"}
        if (
            report["plan_sha256"] != protocol.sha256 or report["fold"] != index
            or report["settings"] != asdict(protocol.settings.backtest)
            or report["test_parquet_sha256"] != protocol.plan["fold_file_hashes"][f"fold_{index:02d}/test.parquet"]
            or report.get("account_reset_each_fold") is not True
            or report.get("model_or_calibration_refitted") is not False
            or set(report["metrics"]) != names
        ):
            raise ValueError("Saved backtest differs from the frozen protocol")
        record_bundle(fold_state, "backtest", fold_output / "backtest")
        verify_all()
        save()
    verify_all()
    if not (output / "summary").exists():
        publish_summary(protocol, output, runs, verify_all)
    report = read_json(output / "summary/summary.json")
    if report["plan_sha256"] != protocol.sha256 or report["folds"] != len(runs):
        raise ValueError("Saved summary differs from the frozen protocol")
    record_bundle(state, "summary", output / "summary")
    verify_all()
    state["complete"] = True
    save()
    print_summary(report)
    LOGGER.info("Saved walk-forward research summary to %s", output / "summary")
    return report


def run_walkforward(directory: Path, resume: bool = False) -> tuple[Path, dict]:
    protocol = load_protocol(directory)
    output = protocol.directory / "execution"
    if output.exists() and not resume:
        raise ValueError("Walk-forward execution already exists; use --resume to verify and reuse saved results")
    output.mkdir(exist_ok=True)
    lock = output / "running.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError("Walk-forward running.lock exists; ensure its process has stopped before removing a stale lock and resuming") from error
    try:
        with stream:
            stream.write(json.dumps({"pid": os.getpid(), "started_at": pd.Timestamp.now(tz="UTC").isoformat()}))
        binding = {"schema_version": 1, "plan_sha256": protocol.sha256}
        path = output / "protocol.json"
        if path.exists():
            if read_json(path) != binding:
                raise ValueError("Execution belongs to a different frozen protocol")
        else:
            if any(p.name not in ("running.lock",) for p in output.iterdir()):
                raise ValueError("Execution directory has artifacts without a protocol binding")
            write_json(path, binding)
        state_path = output / "state.json"
        state = read_json(state_path) if state_path.exists() else {**binding, "folds": {}, "complete": False}
        if any(state.get(key) != value for key, value in binding.items()) or not isinstance(state.get("folds"), dict):
            raise ValueError("Execution journal belongs to a different protocol")
        return output, execute(protocol, output, state)
    finally:
        lock.unlink()
