"""tests/test_confirmatory_d8.py — pytest gate for
run_experiment_B2B6_d8.py's mechanical functions: the power-calc formula,
the effect-size translation, and the degeneracy diagnosis. This is the first
time this repo has actually carried the S8-ii power procedure to a number
(every prior pass stopped at "REPORTED ONLY"), so it is pinned here rather
than only exercised once inside the runner's own print statements.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from run_experiment_B2B6_d8 import (D_CONFIRM, diagnose_degeneracy,  # noqa: E402
                                    effect_size_cells, power_n_trials)


def test_confirmatory_d_is_eight_and_unmoved():
    # SPEC S.11 blind draw, 2026-08-01. This constant must never be edited
    # to a different value for any reason (see the module docstring).
    assert D_CONFIRM == 8


def test_power_n_trials_matches_hand_computation():
    # z_(1-0.006/2) + z_0.80, from scipy.stats.norm.ppf, hand-checked against
    # the standard normal table: Phi^-1(0.997) ~= 2.7478, Phi^-1(0.80) ~= 0.8416.
    n, z = power_n_trials(sd=1.0, effect=1.0)
    assert z == pytest.approx(2.7478 + 0.8416, abs=2e-3)
    assert n == int(np.ceil(z ** 2))


def test_power_n_trials_scales_as_variance_over_effect_squared():
    n_small_effect, _ = power_n_trials(sd=1.0, effect=0.5)
    n_large_effect, _ = power_n_trials(sd=1.0, effect=2.0)
    # halving the effect should ~quadruple the required n (same z, (sd/effect)^2)
    assert n_small_effect > n_large_effect
    n_ref, z = power_n_trials(sd=1.0, effect=1.0)
    assert n_small_effect == pytest.approx((z / 0.5) ** 2, abs=1.0)


def test_effect_size_falls_back_to_one_cell_at_zero_baseline():
    effect, note = effect_size_cells(0.0)
    assert effect == 1.0
    assert "degenerate" in note


def test_effect_size_uses_smaller_of_twenty_percent_and_one_cell():
    # baseline large enough that 20% exceeds 1 cell -> use 1 cell (conservative)
    effect, note = effect_size_cells(10.0)
    assert effect == 1.0
    # baseline small enough that 20% is under 1 cell -> use the 20% value
    effect, note = effect_size_cells(2.0)
    assert effect == pytest.approx(0.4)


def _fake_pilot(gibf_vals, mmv_vals):
    return dict(
        delta_r_bar=dict(gibf=gibf_vals, mmv=mmv_vals),
        p_sep_rate=dict(gibf=float(np.mean([1.0] * len(gibf_vals))),
                        mmv=float(np.mean([1.0] * len(mmv_vals)))),
    )


def test_diagnose_degeneracy_flags_ceiling():
    pilot = _fake_pilot([0.0] * 50, [0.0] * 50)
    diag = diagnose_degeneracy(pilot)
    assert diag["at_ceiling"] == ["gibf", "mmv"]
    assert diag["adjudicable"] is False


def test_diagnose_degeneracy_allows_a_genuinely_variable_cell():
    rng = np.random.default_rng(0)
    gibf_vals = list(np.abs(rng.normal(1.0, 0.3, size=50)))
    mmv_vals = list(np.abs(rng.normal(1.2, 0.3, size=50)))
    pilot = _fake_pilot(gibf_vals, mmv_vals)
    diag = diagnose_degeneracy(pilot)
    assert diag["at_ceiling"] == []
    assert diag["adjudicable"] is True


def test_d8_minipilot_result_is_saturated_and_matches_b1s_own_grid():
    """Regression pin for THIS pass's actual finding (not a synthetic case):
    d=8 is at the ceiling in both the incoherent B1 grid (already committed,
    results/B_viability/b1_results.json key 'True|8|5.0') and the coherent
    mini-pilot this script runs. Re-running the mini-pilot end-to-end here
    would duplicate ~50 solver calls per test collection; instead this pins
    the already-recorded json artifact so a future code change that would
    silently alter that recorded finding is caught."""
    import json
    results = Path(__file__).resolve().parent.parent / "results" / "B_viability"
    manifest = json.loads((results / "confirmatory_d8_manifest.json").read_text())
    assert manifest["confirmatory_d"] == 8
    mp = manifest["minipilot_d8"]
    assert mp["gap_sd"] == 0.0
    assert mp["p_sep_rate"]["gibf"] == 1.0
    assert mp["p_sep_rate"]["mmv"] == 1.0
    assert manifest["degeneracy_diagnosis"]["adjudicable"] is False
    assert manifest["power_calc"]["n_trials"] is None
