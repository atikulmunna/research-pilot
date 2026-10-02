import math

from research_pilot.services import stats

A = [0.81, 0.83, 0.80, 0.84, 0.82]
B = [0.78, 0.79, 0.80, 0.77, 0.785]


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_welch_matches_scipy_reference():
    # Reference values from scipy.stats.ttest_ind(A, B, equal_var=False).
    out = stats.welch(A, B)
    assert close(out["t"], 4.041451884327389)
    assert close(out["p_value"], 0.004642733077761992, 1e-12)
    assert close(out["df"], 7.2)
    assert close(out["ci_low"], 0.014636505115723834)
    assert close(out["ci_high"], 0.05536349488427623)


def test_t_distribution_helpers():
    assert close(stats.t_sf_two_sided(2.0, 10), 0.07338803477074039, 1e-12)
    assert close(stats.t_ppf(0.975, 10), 2.2281388519649385, 1e-8)
    low, high = stats.mean_ci(A)
    assert close(low, 0.8003675683852245)
    assert close(high, 0.8396324316147756)


def test_holm_matches_statsmodels():
    assert stats.holm([0.01, 0.04, None, 0.03]) == [0.03, 0.06, None, 0.06]


def test_effect_size_and_degenerate_inputs():
    g = stats.hedges_g(A, B)
    assert g is not None and g > 2
    assert stats.welch([1.0], [2.0])["p_value"] is None
    assert stats.welch([1.0, 1.0], [1.0, 1.0])["p_value"] == 1.0
    assert stats.coefficient_of_variation([0.0, 0.0]) is None
    assert math.isnan(stats.mean([]))
