import numpy as np

from modelbrief.metrics.drift import population_stability_index, dataset_drift


def test_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(1)
    sample = rng.normal(size=2000)

    psi = population_stability_index(sample, sample)

    assert psi < 0.01


def test_psi_flags_a_clear_shift():
    rng = np.random.default_rng(1)
    expected = rng.normal(loc=0, scale=1, size=2000)
    actual = rng.normal(loc=3, scale=1, size=2000)  # shifted well away from `expected`

    psi = population_stability_index(expected, actual)

    assert psi >= 0.25


def test_dataset_drift_ranks_the_shifted_feature_first():
    rng = np.random.default_rng(1)
    n = 1500
    X_train = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    X_test = np.column_stack([rng.normal(size=n), rng.normal(loc=4, size=n)])  # 2nd col drifted

    result = dataset_drift(X_train, X_test, feature_names=["stable_feature", "drifted_feature"])

    assert result["available"] is True
    assert result["features"][0]["feature"] == "drifted_feature"
    assert result["features"][0]["status"] == "significant_shift"
    assert result["features"][-1]["status"] == "stable"
    assert result["significant_shift_count"] == 1


def test_dataset_drift_mismatched_feature_counts():
    result = dataset_drift(np.zeros((10, 3)), np.zeros((10, 2)))

    assert result["available"] is False
