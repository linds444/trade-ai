import numpy as np
import pandas as pd
import pytest
from scipy.stats import entropy

from local_ai_trader.uncertainty.scores import uncertainty_scores, model_disagreement
from local_ai_trader.uncertainty.report import align_predictions, summarize_scores
from local_ai_trader.walkforward.results import TRUTH_COLUMNS


def test_entropy_margin_and_maximum_have_known_limits_and_do_not_mutate_inputs():
    values = np.array([[1, 0, 0], [1/3, 1/3, 1/3], [0.5, 0.5, 0], [0.1, 0.7, 0.2]])
    before = values.copy()
    scores = uncertainty_scores(values)
    np.testing.assert_allclose(scores["entropy_normalized"], entropy(values, axis=1) / np.log(3), atol=1e-15)
    np.testing.assert_allclose(scores["top_class_probability"], [1, 1/3, 0.5, 0.7])
    np.testing.assert_allclose(scores["top_class_margin"], [1, 0, 0, 0.5])
    assert scores["entropy_normalized"][0] == 0
    assert scores["entropy_normalized"][1] == pytest.approx(1)
    assert not any("confidence" in name for name in scores)
    np.testing.assert_array_equal(values, before)


def test_jensen_shannon_is_symmetric_bounded_and_distinguishes_distributions_with_same_top_class():
    first = np.array([[1, 0, 0], [0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
    second = np.array([[0, 1, 0], [0.6, 0.2, 0.2], [0.1, 0.7, 0.2]])
    expected = (entropy((first + second) / 2, axis=1) - (entropy(first, axis=1) + entropy(second, axis=1)) / 2) / np.log(2)
    values = model_disagreement(first, second)
    np.testing.assert_allclose(values, expected, atol=1e-15)
    np.testing.assert_allclose(values, model_disagreement(second, first), atol=1e-15)
    assert values[0] == pytest.approx(1)
    assert values[1] > 0
    assert values[2] == pytest.approx(0)
    assert ((values >= 0) & (values <= 1)).all()
    np.testing.assert_allclose(model_disagreement(first, first), 0, atol=1e-15)


@pytest.mark.parametrize("invalid", [
    [], [[0.5, 0.5]], [0.5, 0.25, 0.25], [[0.2, 0.2, 0.2]],
    [[float("nan"), 0, 1]], [[float("inf"), 0, 0]], [[-0.1, 0.5, 0.6]],
    [[1.1, 0, 0]], [[True, False, False]], [["0.5", "0.25", "0.25"]],
])
def test_invalid_distributions_are_rejected_by_both_scores(invalid):
    with pytest.raises(ValueError):
        uncertainty_scores(invalid)
    with pytest.raises(ValueError):
        model_disagreement(invalid, [[1, 0, 0]])


def test_disagreement_requires_same_number_of_rows():
    with pytest.raises(ValueError, match="matching"):
        model_disagreement([[1, 0, 0]], [[1, 0, 0], [0, 1, 0]])


def test_entropy_scores_do_not_use_outcomes_and_reporting_bands_cover_exact_one():
    frame = pd.DataFrame({"p_up": [1, 1/3], "p_down": [0, 1/3], "p_flat": [0, 1/3], "target_class": ["up", "down"]})
    scores = uncertainty_scores(frame[["p_up", "p_down", "p_flat"]].to_numpy())
    report = summarize_scores(frame, scores)
    changed = frame.copy()
    changed["target_class"] = ["down", "up"]
    changed["target_return"] = [0.9, -0.9]
    altered = uncertainty_scores(changed[["p_up", "p_down", "p_flat"]].to_numpy())
    for name in scores:
        np.testing.assert_array_equal(scores[name], altered[name])
    assert report["entropy_bands"][0]["rows"] == 1
    assert report["entropy_bands"][-1]["rows"] == 1
    assert report["entropy_bands"][-1]["upper_inclusive"] is True
    assert sum(row["rows"] for row in report["entropy_bands"]) == 2
    assert report["entropy_bands"][1]["observed_accuracy"] is None


@pytest.mark.parametrize("column", ["available_at", "symbol", "target_class", "fold"])
def test_alignment_includes_prediction_times_market_outcomes_and_fold(column):
    frame = pd.DataFrame({name: ["same"] for name in [*TRUTH_COLUMNS, "fold"]})
    changed = frame.copy()
    changed[column] = "different"
    with pytest.raises(ValueError, match="mismatched"):
        align_predictions(frame, changed)
