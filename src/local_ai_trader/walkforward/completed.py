"""Read-only audit of complete research outputs before downstream reporting."""

from dataclasses import dataclass
from pathlib import Path

from local_ai_trader.models.train import file_digest
from local_ai_trader.walkforward.protocol import Protocol, load_protocol
from local_ai_trader.walkforward.run import (
    file_manifest, read_json, validate_model, validate_calibration, validate_evaluation,
)


@dataclass(frozen=True)
class CompletedExecution:
    protocol: Protocol
    output: Path
    state: dict
    state_hash: str
    binding_hash: str

    def verify(self) -> None:
        """Check every recorded bundle and source without fitting or overwriting."""
        if (self.output / "running.lock").exists():
            raise ValueError("Walk-forward execution is running; reporting requires completed frozen results")
        if file_digest(self.output / "state.json") != self.state_hash or file_digest(self.output / "protocol.json") != self.binding_hash:
            raise ValueError("Completed execution journal or binding changed during reporting")
        self.protocol.verify(include_test=True)

        def check(directory: Path, recorded: dict, recursive: bool = True):
            if not isinstance(recorded, dict) or not recorded or file_manifest(directory, recursive) != recorded:
                raise ValueError(f"Completed execution bundle changed or is missing: {directory.name}")

        for index in range(1, len(self.protocol.plan["folds"]) + 1):
            fold = self.state["folds"][str(index)]
            for family in ("baseline", "selected"):
                entry = fold[family]
                relative = Path(entry["run"])
                root = self.output / f"fold_{index:02d}/{family}_models"
                directory = self.output / relative
                if relative.is_absolute() or ".." in relative.parts or directory.resolve().parent != root.resolve():
                    raise ValueError("Completed model path escapes its recorded fold")
                check(directory, entry["model"], recursive=False)
                check(directory / "calibration", entry["calibration"])
                check(directory / "test_evaluation", entry["test_evaluation"])
            check(self.output / f"fold_{index:02d}/backtest", fold["backtest"])
        check(self.output / "summary", self.state["summary"])


def load_completed_execution(directory: Path) -> CompletedExecution:
    """Require complete plan-bound results; never invoke the execution runner."""
    protocol = load_protocol(directory)
    output = protocol.directory / "execution"
    state_path = output / "state.json"
    if not state_path.is_file():
        raise ValueError("Complete walk-forward execution is required before uncertainty reporting")
    state_hash = file_digest(state_path)
    state = read_json(state_path)
    binding_path = output / "protocol.json"
    binding_hash = file_digest(binding_path)
    binding = {"schema_version": 1, "plan_sha256": protocol.sha256}
    expected_folds = {str(index) for index in range(1, len(protocol.plan["folds"]) + 1)}
    if (
        not isinstance(state, dict) or state.get("complete") is not True or any(state.get(key) != value for key, value in binding.items())
        or not isinstance(state.get("folds"), dict)
        or read_json(binding_path) != binding or set(state.get("folds", {})) != expected_folds
    ):
        raise ValueError("Completed execution journal does not establish this frozen protocol")
    if "summary" not in state:
        raise ValueError("Completed execution journal has no summary bundle")
    for fold in state["folds"].values():
        if not isinstance(fold, dict) or not {"baseline", "selected", "backtest"}.issubset(fold):
            raise ValueError("Completed execution journal has incomplete fold bundles")
        for family in ("baseline", "selected"):
            if not isinstance(fold[family], dict) or not {"run", "model", "calibration", "test_evaluation"}.issubset(fold[family]):
                raise ValueError("Completed execution journal has incomplete model bundles")
            if not isinstance(fold[family]["run"], str):
                raise ValueError("Completed model path must be a relative string")
    snapshot = CompletedExecution(protocol, output, state, state_hash, binding_hash)
    snapshot.verify()
    for index in range(1, len(protocol.plan["folds"]) + 1):
        for family in ("baseline", "selected"):
            run = output / state["folds"][str(index)][family]["run"]
            checkpoint = validate_model(protocol, index, family, run)
            validate_calibration(protocol, index, run, checkpoint)
            validate_evaluation(protocol, index, run, checkpoint)
    summary = read_json(output / "summary/summary.json")
    if summary["plan_sha256"] != protocol.sha256 or summary["folds"] != len(expected_folds):
        raise ValueError("Completed summary differs from the frozen protocol")
    snapshot.verify()
    return snapshot
