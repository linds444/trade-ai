"""The independent signal gate; account authorization must still follow it.

Probability is a calibrated directional forecast. Confidence is a separate
validated reliability measure, not entropy or the largest model probability.
Missing evidence routes to local review and never grants execution permission.
"""

from dataclasses import dataclass
import math


def _unit(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


@dataclass(frozen=True)
class EntryPolicy:
    minimum_probability: float = 0.60
    minimum_confidence: float = 0.70
    maximum_positions: int = 5

    def __post_init__(self):
        if not _unit(self.minimum_probability) or not _unit(self.minimum_confidence):
            raise ValueError("Entry thresholds must be finite fractions in [0, 1]")
        if type(self.maximum_positions) is not int or self.maximum_positions < 1:
            raise ValueError("Maximum positions must be a positive integer")


@dataclass(frozen=True)
class SignalEvidence:
    action: str
    probability: float | None
    calibration_id: str | None
    confidence: float | None
    reliability_validation_id: str | None


@dataclass(frozen=True)
class GateResult:
    status: str
    reason: str
    eligible_for_account_risk_check: bool = False


def evaluate_entry_signal(evidence: SignalEvidence, occupied_positions: int,
                          policy: EntryPolicy = EntryPolicy()) -> GateResult:
    if type(occupied_positions) is not int or occupied_positions < 0:
        raise ValueError("Count holdings and pending entry reservations")
    if evidence.action not in ("BUY", "SELL", "HOLD", "AVOID"):
        return GateResult("BLOCK", "invalid_action")
    if evidence.action != "BUY":
        return GateResult("BLOCK", "action_not_entry")
    if not _unit(evidence.probability) or not isinstance(evidence.calibration_id, str) or not evidence.calibration_id.strip():
        return GateResult("REVIEW", "calibrated_probability_unavailable")
    if evidence.probability < policy.minimum_probability:
        return GateResult("BLOCK", "probability_below_minimum")
    if not _unit(evidence.confidence) or not isinstance(evidence.reliability_validation_id, str) or not evidence.reliability_validation_id.strip():
        return GateResult("REVIEW", "validated_confidence_unavailable")
    if evidence.confidence < policy.minimum_confidence:
        return GateResult("REVIEW", "confidence_below_minimum")
    if occupied_positions >= policy.maximum_positions:
        return GateResult("BLOCK", "position_count_limit")
    # This gate approves only the signal. Freshness, cost, cash, exposure,
    # drawdown and atomic persisted account authorization are separate checks.
    return GateResult("PASS", "signal_policy_passed", True)
