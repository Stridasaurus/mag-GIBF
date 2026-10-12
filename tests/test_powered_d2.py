"""tests/test_powered_d2.py — pytest gate for run_experiment_B2B6_powered_d2.py.

Pins the mechanics that make the powered confirmatory run trustworthy BEFORE it
runs: the cell/arm plan, that every arm of a cell (oracle vs K-hat, the five
S.6(5) eps points, reduction ON vs OFF) is fed from ONE per-trial data draw,
the fragility rule, the WIN/LOSS/B6 rule logic, the K-hat breakdown, and the
smoke/confirmatory write guards. Every test uses throwaway seeds, never
MASTER_SEED, so no confirmatory trial is computed here.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

import run_experiment_B2B6_powered_d2 as R  # noqa: E402
from simulate import GRID_SHAPE, build_array_and_grid  # noqa: E402
from solvers import FAIRNESS_CONFIG, solve_all  # noqa: E402

TEST_SEED = 424242  # throwaway; never the master seed


@pytest.fixture(scope="module")
def tm():
    return build_array_and_grid(row_normalisation=R.ROW_NORMALISATION)


# ------------------------------------------------------------------ pins ---

def test_pins_unmoved():
    assert R.D_CONFIRM == 2                       # SPEC S.12
    assert R.N_TRIALS_POWERED == 10_000           # ratified 2026-09-23 ruling
    assert R.PHI_WIN_DEG == 90.0 and R.PHI_NULL_DEGS == (0.0, 180.0)
    assert R.SNR_PAIR == (5.0, 10.0) and R.N_SNAP == 64 and R.ABS_RHO == 0.85
    assert R.EPS_BAND == (0.1, 0.3, 1.0, 3.0, 10.0)
    assert R.EPS_BAND[R.HEADLINE_BAND_INDEX] == 1.0
    assert R.N_BOOT == 2000 and R.D_AXIS == (1, 2, 3, 5, 8)
    assert R.B6A_NSNAP_PINNED == 64
    assert R.parse_args([]).n_trials == 10_000 and R.parse_args([]).b6a_nsnap == 64


def test_master_seed_pinned_and_distinct_from_smoke():
    assert isinstance(R.MASTER_SEED, int) and R.MASTER_SEED > 0
    assert R.SMOKE_SEED != R.MASTER_SEED
    assert R.MASTER_SEED not in (20260706, 20260712, 20260721, 20260727,
                                 20260801, 20260828)


# ------------------------------------------------------------------ plan ---

def test_plan_structure():
    plan = R.build_plan()
    b2 = [c for c in plan if c.family == "B2"]
    assert len(b2) == 3 * 2 * 5 and len(plan) == 33 + 6   # + six off-grid cells
    ids = [c.cell_id for c in plan]
    assert len(set(ids)) == len(ids)              # the d=2 axis point is not duplicated
    conf = [c for c in plan if c.confirmatory]
    assert sorted(c.role for c in conf) == ["b6a"] + ["null"] * 4 + ["win"] * 2
    for c in conf:
        assert len(c.panel_arms) == 5
        assert c.panel_arms[R.HEADLINE_BAND_INDEX] == R.headline_arm(c)
        assert any(not a.reduction for a in c.arms)          # S.6(2) OFF pair
    assert [c.family for c in plan if c.role == "descriptive_b6a_underspecified"] == ["B6a128"]
    assert len([c for c in plan if c.role == "descriptive_forced_k"]) == 1
    assert all(any(a.no_early_exit for a in c.arms)
               for c in plan if c.family == "B2" and c.d == R.D_CONFIRM)
    assert all(not c.confirmatory for c in plan if c.family in ("B6a128", "B6a_forced10"))
    b6 = [c for c in plan if c.family == "B6a"][0]
    assert R.headline_arm(b6) == R.Arm("mdl", True, 1.0)
    assert R.Arm("oracle", True, 1.0) in b6.arms
    win5 = [c for c in b2 if c.role == "win" and c.snr_db == 5.0][0]
    assert b6.cell_id != win5.cell_id            # S.14 P6: independent seeds


def test_arm_uses():
    b6 = [c for c in R.build_plan() if c.family == "B6a"][0]
    uses = {a.key: R.arm_use(b6, a) for a in b6.arms}
    assert uses["mdl_on_x1"] == "headline"
    assert uses["oracle_on_x1"] == "b6a_oracle_reference"
    assert uses["mdl_off_x1"] == uses["oracle_off_x1"] == "descriptive_reduction_off"
    assert uses["mdl_on_x0.1"] == "sensitivity_panel"
    b6128 = next(c for c in R.build_plan() if c.family == "B6a128")
    assert R.arm_use(b6128, R.headline_arm(b6128)) == "descriptive_under_specified_b6a"
    assert all(R.arm_use(c, a) == "descriptive_forced_k"
               for c in R.build_plan() if c.family == "B6a_forced10" for a in c.arms)


def test_arm_config_only_changes_eps_and_reduction():
    for a in (R.Arm("oracle", True, 0.1), R.Arm("oracle", False, 1.0),
              R.Arm("mdl", True, 10.0)):
        cfg = R.arm_config(a)
        assert cfg.eps_frac == pytest.approx(FAIRNESS_CONFIG.eps_frac * a.eps_mult)
        assert cfg.grid_reduction is a.reduction
        for f in ("p_norm", "weight_floor", "beta", "max_iter", "tol", "min_active_factor"):
            assert getattr(cfg, f) == getattr(FAIRNESS_CONFIG, f)
    assert R.arm_config(R.Arm("oracle", True, 1.0)) == FAIRNESS_CONFIG


# ----------------------------------------------- paired arms share data ----

def _fake_solver(log):
    """Records exactly what each arm was fed; returns a well-formed output."""
    def solve(A, V, cfg):
        log.append(dict(V=hashlib.sha256(np.ascontiguousarray(V).tobytes()).hexdigest(),
                        V_first2=hashlib.sha256(np.ascontiguousarray(V[:, :2]).tobytes()).hexdigest(),
                        K=V.shape[1], eps_frac=cfg.eps_frac,
                        reduction=cfg.grid_reduction, cfg_id=id(cfg)))
        I = np.zeros(A.shape[1])
        I[0] = 1.0
        out = {"scale_s": 1.0,
               "l2": dict(I=I, realized_eps=0.1),
               "gibf": dict(I=I, realized_eps=[0.1] * V.shape[1], n_iter=[3] * V.shape[1]),
               "mmv": dict(I=I, realized_eps=0.2, n_iter=4)}
        return out
    return solve


def test_every_arm_of_a_trial_gets_the_same_data(tm):
    b6 = [c for c in R.build_plan() if c.family == "B6a"][0]
    log, datas = [], []
    arrays, _ = R.run_data_cell(tm, b6, 3, TEST_SEED, solve_fn=_fake_solver(log),
                                on_trial=lambda t, tr: datas.append(tr))
    n_arms = len(b6.arms)
    assert len(log) == 3 * n_arms
    for t in range(3):
        calls = log[t * n_arms:(t + 1) * n_arms]
        by_arm = dict(zip([a.key for a in b6.arms], calls))
        oracle_hashes = {c["V"] for k, c in by_arm.items() if k.startswith("oracle")}
        mdl_hashes = {c["V"] for k, c in by_arm.items() if k.startswith("mdl")}
        assert len(oracle_hashes) == 1 and len(mdl_hashes) == 1
        # the K-hat arm is the SAME eigendecomposition truncated differently
        khat = datas[t]["khat_mdl"]
        assert {c["K"] for k, c in by_arm.items() if k.startswith("mdl")} == {khat}
        if khat >= 2:
            assert by_arm["mdl_on_x1"]["V_first2"] == by_arm["oracle_on_x1"]["V"]
        # panel points: identical data, eps scaled, reduction ON
        for m in R.EPS_BAND:
            c = by_arm[R.Arm("mdl", True, m).key]
            assert c["eps_frac"] == pytest.approx(0.01 * m) and c["reduction"] is True
        assert by_arm["mdl_off_x1"]["reduction"] is False
    # recorded K-hat equals what the solver was fed
    assert list(arrays["_data"]["khat_mdl"]) == [d["khat_mdl"] for d in datas]
    assert list(arrays["mdl_on_x1"]["k_used"]) == [d["khat_mdl"] for d in datas]
    assert list(arrays["oracle_on_x1"]["k_used"]) == [2, 2, 2]


def test_b6a_fixed_k_and_aic_lookup_reuse_the_same_trials(tm):
    b6 = next(c for c in R.build_plan() if c.family == "B6a")
    arrays, _ = R.run_data_cell(tm, b6, 3, TEST_SEED,
                                solve_fn=_fake_solver([]))
    for label, counts in (("lookup_mdl", arrays["_data"]["khat_mdl"]),
                          ("lookup_aic", arrays["_data"]["khat_aic"])):
        for method in R.METHODS:
            expected = np.asarray([
                arrays[R.Arm(f"k{int(k)}", True, 1.0).key][f"dr_{method}"][i]
                for i, k in enumerate(counts)])
            assert np.array_equal(arrays[label][f"dr_{method}"], expected)
    summ = R.summarize_cell(b6, arrays)
    assert all(summ["direct_mdl_matches_forced_k_lookup"].values())
    assert set(summ["forced_k_sweep"]) == set(range(1, R.K_MAX + 1))


def test_generate_trial_is_deterministic_and_cell_keyed(tm):
    plan = R.build_plan()
    win5 = [c for c in plan if c.role == "win" and c.snr_db == 5.0][0]
    b6 = [c for c in plan if c.family == "B6a"][0]
    a = R.generate_trial(tm, win5, TEST_SEED, 7)
    b = R.generate_trial(tm, win5, TEST_SEED, 7)
    assert np.array_equal(a["X"], b["X"]) and np.array_equal(a["V"]["oracle"], b["V"]["oracle"])
    c = R.generate_trial(tm, win5, TEST_SEED, 8)
    assert not np.array_equal(a["X"], c["X"])
    d = R.generate_trial(tm, b6, TEST_SEED, 7)       # same physics, different cell id
    assert not np.array_equal(a["X"], d["X"])


def test_real_solver_record_shapes(tm):
    cell = [c for c in R.build_plan() if c.role == "d_axis_win"][0]
    arrays, timing = R.run_data_cell(tm, cell, 2, TEST_SEED, solve_fn=solve_all)
    rec = arrays["oracle_on_x1"]
    assert rec["niter_gibf"].shape == (2, R.K_MAX) and rec["eps_gibf"].shape == (2, R.K_MAX)
    assert np.all(rec["niter_gibf"][:, :2] >= 1) and np.all(rec["niter_gibf"][:, 2:] == -1)
    assert np.all(np.isnan(rec["eps_gibf"][:, 2:])) and np.all(rec["eps_gibf"][:, :2] > 0)
    assert np.all(rec["niter_mmv"] >= 1) and np.all(rec["eps_mmv"] > 0)
    assert set(timing) == {a.key for a in cell.arms} | {"_data"}


# ------------------------------------------------------------ fragility ----

def _pts(means_cis):
    return [dict(band_index=i, eps_mult=R.EPS_BAND[i], gap_mean=m, gap_ci=ci)
            for i, (m, ci) in enumerate(means_cis)]


def test_fragility_significant_reversal_is_fragile():
    pts = _pts([(0.5, (0.1, 0.9)), (0.4, (0.1, 0.7)), (0.6, (0.2, 1.0)),
                (0.3, (-0.1, 0.7)), (-0.4, (-0.7, -0.1))])
    f = R.fragility(pts)
    assert f["fragile"] is True and f["significant_reversal_at"] == [10.0]
    assert f["attenuated_at"] == [3.0]


def test_fragility_insignificant_reversal_is_not_fragile():
    pts = _pts([(0.5, (0.1, 0.9)), (0.4, (0.1, 0.7)), (0.6, (0.2, 1.0)),
                (0.3, (0.1, 0.7)), (-0.2, (-0.6, 0.1))])
    f = R.fragility(pts)
    assert f["fragile"] is False and f["significant_reversal_at"] == []
    assert f["attenuated_at"] == [10.0]


def test_fragility_same_sign_significant_is_robust():
    pts = _pts([(-0.5, (-0.9, -0.1))] * 5)
    f = R.fragility(pts)
    assert not f["fragile"] and f["attenuated_at"] == []


def test_fragility_zero_headline_has_no_sign():
    pts = _pts([(0.5, (0.1, 0.9)), (0.0, (-0.1, 0.1)), (0.0, (-0.1, 0.1)),
                (0.0, (-0.1, 0.1)), (-0.5, (-0.9, -0.1))])
    f = R.fragility(pts)
    assert f["fragile"] is False and f["headline_sign_undefined"] is True
    assert f["attenuated_at"] == []   # headline not significant: nothing to lose


def test_sensitivity_panel_uses_band_seeds_and_same_trials():
    cell = [c for c in R.build_plan() if c.role == "win"][0]
    rng = np.random.default_rng(1)
    n = 40
    arrays = {}
    for a in cell.panel_arms:
        arrays[a.key] = dict(dr_gibf=rng.normal(1, 1, n), dr_mmv=rng.normal(1, 1, n))
    p1 = R.sensitivity_panel(cell, arrays)
    p2 = R.sensitivity_panel(cell, arrays)
    assert p1 == p2                                   # deterministic (seeded)
    assert [p["eps_mult"] for p in p1["points"]] == list(R.EPS_BAND)
    # band seeds differ: identical data at two band points -> different CI draws
    same = {a.key: dict(dr_gibf=arrays[cell.panel_arms[0].key]["dr_gibf"],
                        dr_mmv=arrays[cell.panel_arms[0].key]["dr_mmv"])
            for a in cell.panel_arms}
    p = R.sensitivity_panel(cell, same)["points"]
    assert p[0]["gap_mean"] == p[1]["gap_mean"] and p[0]["gap_ci"] != p[1]["gap_ci"]


# -------------------------------------------------------------- rules ------

def _summary(g_mean, g_ci, m_mean, m_ci):
    return {"dr_gibf": (g_mean, g_ci), "dr_mmv": (m_mean, m_ci)}


def test_ci_non_overlap_is_strict():
    assert R.ci_non_overlap((0.0, 1.0), (1.1, 2.0))
    assert not R.ci_non_overlap((0.0, 1.0), (1.0, 2.0))
    assert not R.ci_non_overlap((0.0, 1.5), (1.0, 2.0))


def test_delta_r_arm():
    assert R.delta_r_arm(_summary(1.0, (0.9, 1.1), 2.0, (1.8, 2.2)), "gibf")["met"]
    # 20% effect but overlapping CIs
    r = R.delta_r_arm(_summary(1.0, (0.5, 1.9), 2.0, (1.8, 2.2)), "gibf")
    assert r["effect_met"] and not r["ci_non_overlap"] and not r["met"]
    # separated CIs but only 10% lower
    r = R.delta_r_arm(_summary(1.8, (1.75, 1.85), 2.0, (1.9, 2.1)), "gibf")
    assert r["ci_non_overlap"] and not r["effect_met"] and not r["met"]
    # exactly 20% lower counts ("at least 20%")
    assert R.delta_r_arm(_summary(1.6, (1.5, 1.7), 2.0, (1.9, 2.1)), "gibf")["effect_met"]
    # mirrored for the loss rule
    assert R.delta_r_arm(_summary(2.0, (1.8, 2.2), 1.0, (0.9, 1.1)), "mmv")["met"]
    assert not R.delta_r_arm(_summary(2.0, (1.8, 2.2), 1.0, (0.9, 1.1)), "gibf")["met"]


def test_smallest_d():
    assert R.smallest_d({1: 0.1, 2: 0.85, 3: 0.9, 5: 1.0, 8: 1.0}) == 2
    assert R.smallest_d({1: 0.1, 2: 0.2, 3: 0.3, 5: 0.4, 8: 0.5}) == math.inf
    # literal reading under non-monotonicity
    assert R.smallest_d({1: 0.9, 2: 0.1, 3: 0.1, 5: 1.0, 8: 1.0}) == 1
    assert R.smallest_d({1: 0.8, 2: 0.0, 3: 0.0, 5: 0.0, 8: 0.0}) == 1   # >= 0.8


def _psep_trials(first_d_gibf, first_d_mmv, n=200):
    out = {}
    for d in R.D_AXIS:
        out[d] = {"gibf": np.full(n, d >= first_d_gibf), "mmv": np.full(n, d >= first_d_mmv)}
    return out


def _ids(tag):
    return {d: f"test|{tag}|d={d}" for d in R.D_AXIS}


def test_d_axis_arm():
    r = R.d_axis_arm(_psep_trials(1, 3), "gibf", _ids("a"))
    assert r["met"] and r["d_min"] == {"gibf": 1.0, "mmv": 3.0}
    assert r["d_min_ci"]["gibf"] == (1.0, 1.0) and r["d_min_ci"]["mmv"] == (3.0, 3.0)
    assert not R.d_axis_arm(_psep_trials(2, 2), "gibf", _ids("b"))["met"]
    # never reaching 0.8 is +inf, which any finite d_min beats
    r = R.d_axis_arm(_psep_trials(5, 99), "gibf", _ids("c"))
    assert r["d_min"]["mmv"] == math.inf and r["met"]
    assert R.d_axis_arm(_psep_trials(5, 1), "mmv", _ids("d"))["met"]


def test_d_axis_arm_ci_overlap_blocks_a_noisy_edge():
    # GIBF sits exactly at P_sep = 0.8 at d=1 (24/30), so its point d_min is 1
    # (one cell below MMV's 2) but ~half the bootstrap draws fall below 0.8 and
    # push d_min to 2: the CIs overlap, so the arm is NOT met.
    pt = _psep_trials(2, 2, n=30)
    pt[1]["gibf"] = np.arange(30) < 24
    r = R.d_axis_arm(pt, "gibf", _ids("e"))
    assert r["d_min"] == {"gibf": 1.0, "mmv": 2.0} and r["effect_met"]
    assert r["d_min_ci"]["gibf"] == (1.0, 2.0) and r["d_min_ci"]["mmv"] == (2.0, 2.0)
    assert not r["ci_non_overlap"] and not r["met"]


def _view(win_gibf_better=True, null_mmv_better_at=(0.0,), fragile=()):
    good_g = _summary(1.0, (0.9, 1.1), 2.0, (1.8, 2.2))
    good_m = _summary(2.0, (1.8, 2.2), 1.0, (0.9, 1.1))
    tie = _summary(1.0, (0.8, 1.2), 1.0, (0.8, 1.2))
    v = {}
    for phi in (R.PHI_WIN_DEG,) + R.PHI_NULL_DEGS:
        for snr in R.SNR_PAIR:
            if phi == R.PHI_WIN_DEG:
                s = good_g if win_gibf_better else tie
            else:
                s = good_m if phi in null_mmv_better_at else tie
            v[(phi, snr)] = dict(summary=s, psep_trials=_psep_trials(2, 2),
                                 cell_ids=_ids(f"{phi}{snr}"),
                                 fragile=(phi, snr) in fragile)
    v["B6a"] = dict(summary=good_g, fragile=False, confirmatory=True)
    return v


def test_adjudicate_win_loss_b6():
    r = R.adjudicate(_view())
    assert r["win"]["met"] and r["loss"]["met"] and r["loss"]["met_at_phi"] == [0.0]
    assert r["b6"]["met"]
    assert not R.adjudicate(_view(win_gibf_better=False))["win"]["met"]
    assert not R.adjudicate(_view(null_mmv_better_at=()))["loss"]["met"]
    both = R.adjudicate(_view(null_mmv_better_at=(0.0, 180.0)))["loss"]
    assert both["met_at_phi"] == [0.0, 180.0]
    v = _view()
    v["B6a"]["summary"] = _summary(2.0, (1.8, 2.2), 1.0, (0.9, 1.1))
    b6 = R.adjudicate(v)["b6"]
    assert b6["met"] and b6["winner"] == "mmv" and b6["by_favored"]["mmv"]["met"]


def test_adjudicate_fragile_cell_blocks_its_rule():
    # win needs BOTH SNR cells; one fragile cell kills the win
    assert not R.adjudicate(_view(fragile=((90.0, 10.0),)))["win"]["met"]
    # loss needs both SNRs at >= 1 null; a fragile cell at the only winning null kills it
    assert not R.adjudicate(_view(fragile=((0.0, 5.0),)))["loss"]["met"]
    # ...but not if the other null also meets the standard
    r = R.adjudicate(_view(null_mmv_better_at=(0.0, 180.0), fragile=((0.0, 5.0),)))
    assert r["loss"]["met"] and r["loss"]["met_at_phi"] == [180.0]
    v = _view()
    v["B6a"]["fragile"] = True
    assert not R.adjudicate(v)["b6"]["met"]
    v = _view()
    v["B6a"]["confirmatory"] = False               # non-pinned n_snap -> descriptive
    assert not R.adjudicate(v)["b6"]["met"]


def test_win_can_be_met_by_the_d_axis_arm_alone():
    v = _view(win_gibf_better=False)
    for snr in R.SNR_PAIR:
        v[(R.PHI_WIN_DEG, snr)]["psep_trials"] = _psep_trials(1, 3)
    r = R.adjudicate(v)
    assert r["win"]["met"]
    assert not r["win"]["per_snr"][5.0]["delta_r_arm"]["met"]


def test_snr_may_use_different_win_arms_and_is_flagged_mixed():
    v = _view(win_gibf_better=False)
    v[(R.PHI_WIN_DEG, 5.0)]["summary"] = _summary(
        1.0, (0.9, 1.1), 2.0, (1.8, 2.2))
    v[(R.PHI_WIN_DEG, 10.0)]["psep_trials"] = _psep_trials(1, 3)
    result = R.adjudicate(v)["win"]
    assert result["met"] and result["mixed_arm"]
    assert result["per_snr"][5.0]["satisfying_arms"] == ["delta_r"]
    assert result["per_snr"][10.0]["satisfying_arms"] == ["d_axis"]


# ------------------------------------------------------------- B6a bits ----

def test_khat_breakdown_counts():
    kh = np.array([1, 2, 2, 6, 6, 6, 3])
    b = R.khat_breakdown(kh)
    assert b["counts"] == {"lt2": 1, "eq2": 2, "gt2": 4}
    assert b["hist"] == {1: 1, 2: 2, 3: 1, 6: 3}
    arm = {f"dr_{m}": np.arange(7, dtype=float) for m in R.METHODS}
    orc = {f"dr_{m}": np.zeros(7) for m in R.METHODS}
    b = R.khat_breakdown(np.array([2, 2, 2, 2, 2, 2, 2]), arm, orc)
    assert b["per_bin"]["lt2"] is None and b["per_bin"]["gt2"] is None
    assert b["per_bin"]["eq2"]["paired_diff_dr_gibf_mean"] == pytest.approx(3.0)


def test_b6a_paired_difference():
    n = 50
    rng = np.random.default_rng(5)
    kh = {f"dr_{m}": rng.normal(2, 1, n) for m in R.METHODS}
    kh.update({f"psep_{m}": rng.random(n) < 0.5 for m in R.METHODS})
    orc = {f"dr_{m}": rng.normal(1, 1, n) for m in R.METHODS}
    orc.update({f"psep_{m}": rng.random(n) < 0.7 for m in R.METHODS})
    idx = R.boot_indices(n, R.boot_rng("x"))
    per, summ = R.b6a_paired(kh, orc, idx)
    assert np.allclose(per["paired_diff_dr_gibf"], kh["dr_gibf"] - orc["dr_gibf"])
    assert summ["paired_diff_dr_mmv"][0] == pytest.approx(np.mean(kh["dr_mmv"] - orc["dr_mmv"]))


def test_b3_reading_gives_the_pinned_nsnap():
    rec = R.b3_reading()
    assert rec["procedure_result"] == 64
    assert rec["rows_5db"][64]["mdl_error_rate"] >= 0.20
    assert rec["rows_5db"][128]["mdl_error_rate"] < 0.20
    fake = {f"5.0|{n}": {"mdl_error_rate": 0.0} for n in R.B3_NSNAP_AXIS}
    assert R.b6a_nsnap_from_b3(fake) == 8                  # S8-vii fallback


# -------------------------------------------------------------- bootstrap --

def test_bootstrap_seeding_and_paired_indices():
    a = R.boot_indices(30, R.boot_rng("cell"))
    b = R.boot_indices(30, R.boot_rng("cell"))
    assert np.array_equal(a, b) and a.shape == (2000, 30)
    assert not np.array_equal(a, R.boot_indices(30, R.boot_rng("cell", 1)))
    x = np.arange(30, dtype=float)
    m, (lo, hi) = R.mean_ci(x, a)
    assert m == pytest.approx(14.5) and lo < m < hi
    m, ci = R.mean_ci(np.full(30, 2.0), a)                 # degenerate CI
    assert ci == (2.0, 2.0)


def test_summarize_arm_gap_is_signed_gibf_minus_mmv():
    n = 20
    rec = {f"dr_{m}": np.zeros(n) for m in R.METHODS}
    rec["dr_gibf"] = np.full(n, 3.0)
    rec["dr_mmv"] = np.full(n, 1.0)
    rec.update({f"psep_{m}": np.ones(n, dtype=bool) for m in R.METHODS})
    rec["niter_gibf"] = np.tile([13, 13, -1, -1, -1, -1], (n, 1)).astype(float)
    rec["niter_mmv"] = np.full(n, 13)
    s = R.summarize_arm(rec, R.boot_indices(n, R.boot_rng("s")))
    assert s["gap_gibf_minus_mmv"][0] == pytest.approx(2.0)
    assert s["niter_gibf_mean"] == pytest.approx(13.0)


# ---------------------------------------------------------------- guards ---

def test_smoke_refuses_results_dir():
    with pytest.raises(SystemExit):
        R.main(["--smoke", "--n-trials", "2", "--out-dir", str(R.DEFAULT_OUT)])


def test_non_confirmatory_n_refuses_results_dir():
    with pytest.raises(SystemExit):
        R.main(["--n-trials", "3", "--out-dir", str(R.DEFAULT_OUT)])


def test_only_is_smoke_only(tmp_path):
    with pytest.raises(SystemExit):
        R.main(["--n-trials", "3", "--only", "win", "--out-dir", str(tmp_path)])


def test_any_run_refuses_an_out_dir_with_a_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit):
        R.main(["--smoke", "--n-trials", "2", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--out-dir", str(tmp_path)]), clean=True)


def test_master_seed_run_must_write_under_results(tmp_path):
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--out-dir", str(tmp_path)]), clean=True)


def test_master_seed_run_requires_committed_clean_runner():
    out = R.RESULTS_ROOT / "B_viability" / "_never_created_test_dir"
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--out-dir", str(out)]), clean=False)
    assert not out.exists()


def test_nondefault_b6a_nsnap_cannot_touch_the_confirmatory_dir():
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--b6a-nsnap", "128"]), clean=True)


def test_nondefault_b6a_nsnap_is_a_descriptive_b6a_only_run():
    out = R.RESULTS_ROOT / "B_viability" / "_never_created_b6a_n128"
    run = R.resolve_run(R.parse_args(["--b6a-nsnap", "128", "--out-dir", str(out)]),
                        clean=True)
    assert run["confirmatory"] is False and run["roles"] == ("descriptive_b6a_underspecified",)
    assert run["seed"] == R.MASTER_SEED
    assert not out.exists()          # resolve_run never creates anything
    plan = R.build_plan(128)
    descriptive = [c for c in plan if c.role == "descriptive_b6a_underspecified"]
    assert len(descriptive) == 1 and descriptive[0].n_snap == 128
    assert descriptive[0].confirmatory is False and descriptive[0].panel_arms == ()
    assert len(descriptive[0].arms) == 1 and all(a.reduction for a in descriptive[0].arms)


def test_default_run_is_blocked_while_the_gates_are_unset(monkeypatch):
    if (R.DEFAULT_OUT / "manifest.json").exists():
        pytest.skip("the confirmatory run already exists")
    monkeypatch.setattr(R, "PRE_RUN_GATES_COMPLETE", False)
    with pytest.raises(SystemExit, match="airtightness review.*agent-chosen-values sweep"):
        R.resolve_run(R.parse_args([]), clean=True)


def test_smoke_run_is_never_confirmatory(tmp_path):
    run = R.resolve_run(R.parse_args(["--smoke", "--n-trials", "3",
                                      "--out-dir", str(tmp_path / "s")]))
    assert run["confirmatory"] is False and run["seed"] == R.SMOKE_SEED


def test_d_axis_arm_both_infinite_is_no_effect():
    r = R.d_axis_arm(_psep_trials(99, 99), "gibf", _ids("f"))
    assert r["d_min"] == {"gibf": math.inf, "mmv": math.inf}
    assert not r["effect_met"] and not r["met"]


def test_miss_and_matched_error_decomposes_delta_r_bar():
    from metrics import delta_r_bar, p_sep
    true_cells = [(16, 7), (16, 9)]
    I = np.zeros(GRID_SHAPE)
    I[16, 7] = 1.0
    I[17, 10] = 0.9                   # second peak 1 row + 1 col off
    miss, err = R.miss_and_matched_error(I, true_cells)
    assert miss is False
    assert err == pytest.approx(np.hypot(1, 1) / 2)
    assert err == pytest.approx(delta_r_bar(I, true_cells))
    I2 = np.zeros(GRID_SHAPE)
    I2[16, 7] = 1.0                   # only one detected peak -> miss
    miss, err = R.miss_and_matched_error(I2, true_cells)
    assert miss is True and err == pytest.approx(0.0)  # found-source error survives miss
    comp = R.matched_error_components(I2, true_cells)
    assert comp == dict(miss=True, found_sum=0.0, n_matched=1, found_mean=0.0)
    diag = float(np.hypot(GRID_SHAPE[0] - 1, GRID_SHAPE[1] - 1))
    assert delta_r_bar(I2, true_cells) == pytest.approx(diag / 2)   # penalty applied
    assert p_sep(I2, true_cells) is False


def test_peak_cap_is_two_primary_with_eight_peak_supplement():
    from metrics import delta_r_bar, find_peaks_2d, p_sep
    true_cells = [(10, 10), (10, 14)]
    image = np.zeros(GRID_SHAPE)
    image[10, 10] = 1.0
    image[10, 14] = 0.9
    image[2, 2] = 0.8
    image[28, 14] = 0.7
    assert len(find_peaks_2d(image)) == 2
    assert len(find_peaks_2d(image, max_peaks=8)) == 4
    assert p_sep(image, true_cells)
    assert p_sep(image, true_cells, max_peaks=8)
    assert delta_r_bar(image, true_cells) == delta_r_bar(image, true_cells, max_peaks=8)


def test_threshold_outcomes_and_conditional_ci_are_reported():
    n = 20
    rec = {f"dr_{m}": np.ones(n) for m in R.METHODS}
    rec.update({f"psep_{m}": np.ones(n, dtype=bool) for m in R.METHODS})
    rec.update({f"miss_{m}": np.zeros(n, dtype=bool) for m in R.METHODS})
    rec.update({f"outcome_{m}": np.array(["correct"] * 15 + ["mislocated"] * 5)
                for m in R.METHODS})
    rec["niter_gibf"] = np.ones((n, 1))
    rec["niter_mmv"] = np.ones(n)
    for tag in ("05", "20"):
        for m in R.METHODS:
            rec[f"dr_{tag}_{m}"] = np.ones(n)
            rec[f"psep_{tag}_{m}"] = np.array([True] * 12 + [False] * 8)
            rec[f"miss_{tag}_{m}"] = np.array([False] * 18 + [True] * 2)
            rec[f"outcome_{tag}_{m}"] = np.array(
                ["correct"] * 12 + ["mislocated"] * 6 + ["miss"] * 2)
            rec[f"matched_err_{tag}_{m}"] = np.arange(n, dtype=float)
    for m in R.METHODS:
        rec[f"matched_err_{m}"] = np.arange(n, dtype=float)
        rec[f"matched_err_sum_{m}"] = np.arange(n, dtype=float)
        rec[f"matched_n_{m}"] = np.ones(n, dtype=int)
        rec[f"dr_8peak_{m}"] = np.ones(n)
        rec[f"psep_8peak_{m}"] = np.ones(n, dtype=bool)
    idx = R.boot_indices(n, R.boot_rng("summary-test"))
    result = R.summarize_arm(rec, idx, "summary-test")
    assert result["threshold_20_outcome_counts_gibf"] == {
        "miss": 2, "correct": 12, "mislocated": 6}
    assert result["threshold_20_conditional_correct_error_gibf"]["n"] == 12
    assert result["supplementary_8peak_psep_gibf"][0] == 1.0


def test_no_early_exit_preserves_trial_arm_and_runs_full_iteration_budget(tm):
    cell = next(c for c in R.build_plan() if c.family == "B2" and c.d == R.D_CONFIRM)
    arm = next(a for a in cell.arms if a.no_early_exit)
    cfg = R.arm_config(arm)
    assert cfg.stop_early is False and cfg.grid_reduction is True
    arrays, _ = R.run_data_cell(tm, cell, 1, TEST_SEED, solve_fn=solve_all)
    rec = arrays[arm.key]
    assert np.all(rec["niter_gibf"][:, :2] == FAIRNESS_CONFIG.max_iter)
    assert np.all(rec["niter_mmv"] == FAIRNESS_CONFIG.max_iter)
    assert np.all(rec["stop_gibf"][:, :2] == "max_iter")
    assert np.all(rec["stop_mmv"] == "max_iter")


def test_real_solver_records_miss_fields(tm):
    cell = [c for c in R.build_plan() if c.role == "d_axis_null"][0]
    arrays, _ = R.run_data_cell(tm, cell, 2, TEST_SEED, solve_fn=solve_all)
    rec = arrays["oracle_on_x1"]
    for m in R.METHODS:
        miss = rec[f"miss_{m}"]
        assert miss.dtype == bool and miss.shape == (2,)
        assert np.all(np.isfinite(rec[f"matched_err_{m}"][miss]))
        assert np.allclose(rec[f"matched_err_sum_{m}"] / np.maximum(
            rec[f"matched_n_{m}"], 1), rec[f"matched_err_{m}"])
        assert np.allclose(rec[f"matched_err_{m}"][~miss], rec[f"dr_{m}"][~miss])
        assert np.all((rec[f"npeaks_{m}"] < 2) == miss)


# ------------------------------------------- off-grid row (A4) and d=1 panel --
# SPEC S.14 "Off-grid row (A4): specification" and P9 item 4.

# sha256 of (X, oracle V, K-hat, true cells) for on-grid trials 0..2 of 13 cells
# at TEST_SEED, computed from the runner BEFORE the off-grid change. The change
# must not disturb a single on-grid draw (the master-seed trial stream).
ONGRID_FINGERPRINTS = {
    "B2|phi=90|snr=5|d=1|nsnap=64#0":
        "5ef805171f35246f08e32ca434e5db02f6be952efe947ddd67d2cbba51d3cb82",
    "B2|phi=90|snr=5|d=1|nsnap=64#1":
        "dceb071dd7d7059a1cee7c6552e9a562341ba12b6b9d00727be1e2a99dfee8fe",
    "B2|phi=90|snr=5|d=1|nsnap=64#2":
        "91fc81d68d8c785e7f3d6c57e6318a38ac7651192ee84d5d117875b21fc79bb2",
    "B2|phi=90|snr=5|d=2|nsnap=64#0":
        "d662689ebd5ec396899cd552cc1e6f09644e047f4da21f8a9b9ee5bd5aa54c36",
    "B2|phi=90|snr=5|d=2|nsnap=64#1":
        "9975ae5fc7d7f7cb456f2bb30de7b17f3d2fb7f902efef6910180b2b25fbe0e3",
    "B2|phi=90|snr=5|d=2|nsnap=64#2":
        "48b6737a03cd7653c11adb26e2be6081aeee0755ae0f8b1bf5bcccf549e78928",
    "B2|phi=90|snr=5|d=8|nsnap=64#0":
        "cfa43e842729a98747fb3565680e0eb29107d697f6608aea7e378607a18613d0",
    "B2|phi=90|snr=5|d=8|nsnap=64#1":
        "e45416cc18d607cb2c743f8e9f0f12b7807aae1972339237c1e7947620ad73a4",
    "B2|phi=90|snr=5|d=8|nsnap=64#2":
        "b16c4b002211e38264b55cecca86ac9233f84816ed0c8aedc419c474d41e937f",
    "B2|phi=90|snr=10|d=1|nsnap=64#0":
        "8aa5345813f8ff5b319011cc9dc7e37b4f3bb5187a39f3bf099056a9701ac169",
    "B2|phi=90|snr=10|d=1|nsnap=64#1":
        "0d0f4a1c0ed4e03c9d5e0b9c4e7a9896dc93219ad766f6d96c8691fa684ae95d",
    "B2|phi=90|snr=10|d=1|nsnap=64#2":
        "3732be1ad545ced934eeabfa2d5a23a458b6d624c6105e05d71bacf4962edd07",
    "B2|phi=90|snr=10|d=2|nsnap=64#0":
        "03b9c2a449a2ed16b1389c9bce73b52225fdb67aa5be19539121015a0e7884ab",
    "B2|phi=90|snr=10|d=2|nsnap=64#1":
        "9fe359dabf483937e74e91eab8209d1604088d30e4ee79b6c68aa89f3c5e6d9c",
    "B2|phi=90|snr=10|d=2|nsnap=64#2":
        "d662da6d8531674abed1eafe8f5c2c8aac92e9949814976ae08c03a6bd7fb989",
    "B2|phi=90|snr=10|d=8|nsnap=64#0":
        "249389fc90e8a1a1cb24a7e82594140c0bd90c3c9dfc7b5111744eb985f46fa6",
    "B2|phi=90|snr=10|d=8|nsnap=64#1":
        "5aad143cb1b961bd0b752e5bce1fc440c630df357273566fa8be38b4c5074b87",
    "B2|phi=90|snr=10|d=8|nsnap=64#2":
        "2b2b8513aa3c60533c7597fb77cfe004698baf289bbb07ad1b04d2d2c05fa715",
    "B2|phi=0|snr=5|d=1|nsnap=64#0":
        "b07e1e1dd940af8c5fc7eb4976e3c61ed327a496d6a4dd6d4d1c7a5f4d950d94",
    "B2|phi=0|snr=5|d=1|nsnap=64#1":
        "fa7816c5e02f219a8a75337581d2fd6f08fc7c47bf7ec4482e0a23c56e90274d",
    "B2|phi=0|snr=5|d=1|nsnap=64#2":
        "bf2b2e2a357911f296ee9dea2258ed6791c50f39324f752c5e2696927c866de1",
    "B2|phi=0|snr=5|d=2|nsnap=64#0":
        "43c17b0b48a25c7fe0e90e22574bc530ca2ef4877d0d8845c309db2afa9ff790",
    "B2|phi=0|snr=5|d=2|nsnap=64#1":
        "5bc76cff5503c19e89b0423bcd9871945e756683e9cdf4a753c14a160b51f39b",
    "B2|phi=0|snr=5|d=2|nsnap=64#2":
        "bf55255297af46f219f52a8223caadd6f3199f8e38add2eb98beeb10895d6687",
    "B2|phi=0|snr=5|d=8|nsnap=64#0":
        "e4a7a66942f34bc96ddcca9dcb12510bd108bb67cd08b9fdadfeeec21b8c5bab",
    "B2|phi=0|snr=5|d=8|nsnap=64#1":
        "236360634a29e336d137860308734a0f6d506645e2ac6902af7104040c1c8eb1",
    "B2|phi=0|snr=5|d=8|nsnap=64#2":
        "252c0f7038aec5308890194728fa6c5ed572dfe81e82bdb4ccf73014804d4148",
    "B2|phi=0|snr=10|d=1|nsnap=64#0":
        "c200b62cf68275b2695fcdf05c2e9e112ea26a952437ae07555807b150ebe823",
    "B2|phi=0|snr=10|d=1|nsnap=64#1":
        "c656f36ac51c60f5678ae28cd73d751266e1f03d8c1ff12420af738b45a6a69f",
    "B2|phi=0|snr=10|d=1|nsnap=64#2":
        "6e13de163d9c9e5b0f6a244be87eaa9c523b72ad05cd48d00bc08364af99d14d",
    "B2|phi=0|snr=10|d=2|nsnap=64#0":
        "3daec706c270d18e0de3d68ade8c113097aa89074e3ecaa610ce87ae3e29582b",
    "B2|phi=0|snr=10|d=2|nsnap=64#1":
        "7d16daf66e5891ba036581899bbb0d4187439a9398653563f8ccdc2afe86225f",
    "B2|phi=0|snr=10|d=2|nsnap=64#2":
        "7e8664772fc9e4106c499bf0c93ce10346bcd87246f020e381d53f3b352bcf88",
    "B2|phi=0|snr=10|d=8|nsnap=64#0":
        "5557dab7fca3d96140e37851d32a3149d80443e7eaba67b6d66247db4d5a29d8",
    "B2|phi=0|snr=10|d=8|nsnap=64#1":
        "2ed8caf2a430f1479779cd605bbf299812f46a838a5d04aba65327aa50bee17e",
    "B2|phi=0|snr=10|d=8|nsnap=64#2":
        "90933cd6ca281666bff166e09fde4423e04581ca4bed92c92ece2c70a333f96a",
    "B6a|phi=90|snr=5|d=2|nsnap=64#0":
        "be17683627cda765025605dbf35d76918a588b48fa85b7043cc360064e193a9d",
    "B6a|phi=90|snr=5|d=2|nsnap=64#1":
        "09669a0d623ae80e8de54c81dad63293a2ff8308a0cf190b3376070e24f11882",
    "B6a|phi=90|snr=5|d=2|nsnap=64#2":
        "511731da785f22dd40fc41aa8953296071e2acf5ea80aeb2e94c78d77c73989c",
}


def _fingerprint(tr):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tr["X"]).tobytes())
    h.update(np.ascontiguousarray(tr["V"]["oracle"]).tobytes())
    h.update(str(tr["khat_mdl"]).encode())
    h.update(str([tuple(int(x) for x in tc) for tc in tr["true_cells"]]).encode())
    return h.hexdigest()


def _offgrid_cells():
    return [c for c in R.build_plan() if c.family == R.OFFGRID_FAMILY]


def test_on_grid_trials_are_byte_identical_to_before_the_offgrid_change(tm):
    plan = {c.cell_id: c for c in R.build_plan()}
    for key, want in ONGRID_FINGERPRINTS.items():
        cell_id, t = key.rsplit("#", 1)
        tr = R.generate_trial(tm, plan[cell_id], TEST_SEED, int(t))
        assert _fingerprint(tr) == want, key


def test_offgrid_plan_cells():
    plan = R.build_plan()
    off = _offgrid_cells()
    assert len(off) == 6
    assert {(c.phi_deg, c.snr_db) for c in off} == {
        (p, s) for p in (R.PHI_WIN_DEG,) + R.PHI_NULL_DEGS for s in R.SNR_PAIR}
    for c in off:
        assert c.d == R.D_CONFIRM and c.n_snap == R.N_SNAP
        assert c.role == "descriptive_offgrid" and not c.confirmatory
        assert c.panel_arms == () and c.arms == (R.Arm("oracle", True, 1.0),)
        assert R.arm_use(c, c.arms[0]) == "descriptive_offgrid"
        sib = [s for s in plan if s.cell_id == c.data_cell_id]
        assert len(sib) == 1 and sib[0].family == "B2" and sib[0].d == R.D_CONFIRM
        assert sib[0].role in ("win", "null") and c.cell_id != c.data_cell_id
    # adjudication selects its d axis by family == "B2": off-grid cells never join it
    for phi in (R.PHI_WIN_DEG,) + R.PHI_NULL_DEGS:
        for snr in R.SNR_PAIR:
            axis = [c for c in plan if c.family == "B2" and c.phi_deg == phi
                    and c.snr_db == snr and c.d == R.D_CONFIRM]
            assert len(axis) == 1
    assert R.OFFGRID_ROLE in R.ALL_ROLES
    # a descriptive B6a-only plan carries no off-grid cell
    assert not [c for c in R.build_plan(128) if c.family == R.OFFGRID_FAMILY]


def test_offgrid_zero_offset_reproduces_its_sibling_exactly(tm):
    plan = R.build_plan()
    off = _offgrid_cells()[0]
    sib = next(c for c in plan if c.cell_id == off.data_cell_id)
    a = R.generate_trial(tm, sib, TEST_SEED, 5)
    z = R.generate_trial(tm, off, TEST_SEED, 5, offset=(0.0, 0.0))
    assert np.array_equal(a["X"], z["X"])
    assert np.array_equal(a["V"]["oracle"], z["V"]["oracle"])
    assert [(float(r), float(c)) for r, c in a["true_cells"]] == list(z["true_cells"])


def test_offgrid_offsets_are_in_range_recorded_and_off_the_data_stream(tm):
    off = _offgrid_cells()[0]
    sib = next(c for c in R.build_plan() if c.cell_id == off.data_cell_id)
    before = R.generate_trial(tm, sib, TEST_SEED, 2)
    offs = []
    for t in range(30):
        tr = R.generate_trial(tm, off, TEST_SEED, t)
        u, v = tr["offset"]
        assert -0.5 <= u < 0.5 and -0.5 <= v < 0.5
        row, col_a = divmod(R.row_center_cols(R.D_CONFIRM)[0], R.GRID_SHAPE[1])
        assert tr["true_cells"][0] == (row + u, col_a + v)
        assert (tr["true_cells"][1][1] - tr["true_cells"][0][1]
                == pytest.approx(R.D_CONFIRM))                 # rigid pair
        offs.append((u, v))
    assert len(set(offs)) == 30                       # varies by trial
    again = R.generate_trial(tm, off, TEST_SEED, 7)
    assert again["offset"] == R.generate_trial(tm, off, TEST_SEED, 7)["offset"]
    after = R.generate_trial(tm, sib, TEST_SEED, 2)   # sibling untouched by the draws above
    assert np.array_equal(before["X"], after["X"])
    # the offset stream is not the data stream
    assert R.offgrid_rng(TEST_SEED, "k", 0).uniform() != R.trial_rng(TEST_SEED, "k", 0).uniform()


def test_offgrid_cell_runs_end_to_end_and_records_offsets(tm):
    off = _offgrid_cells()[0]
    arrays, _ = R.run_data_cell(tm, off, 3, TEST_SEED, solve_fn=solve_all)
    d = arrays["_data"]
    assert d["offset_u"].shape == (3,) and np.all(np.abs(d["offset_u"]) <= 0.5)
    rec = arrays["oracle_on_x1"]
    for m in R.METHODS:
        assert np.all(np.isfinite(rec[f"dr_{m}"]))
    summ = R.summarize_cell(off, arrays)
    assert all(a["descriptive"] and a["use"] == "descriptive_offgrid"
               for a in summ["arms"].values())
    assert "sensitivity_panel" not in summ


def test_offgrid_summary_bins_contrast_and_transitions():
    off = _offgrid_cells()[0]
    n = 400
    rng = np.random.default_rng(3)
    u, v = rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n)
    outcomes = np.array(["correct", "miss", "mislocated"])

    def rec(shift):
        out = {}
        for m in R.METHODS:
            out[f"dr_{m}"] = rng.normal(2.0, 1.0, n) + shift
            out[f"psep_{m}"] = rng.random(n) < 0.7
            out[f"miss_{m}"] = rng.random(n) < 0.2
            out[f"outcome_{m}"] = outcomes[rng.integers(0, 3, n)]
        return out

    on = rec(0.0)
    offr = {k: (np.asarray(x).copy()) for k, x in on.items()}
    offr["dr_gibf"] = on["dr_gibf"] + 1.0              # known paired shift
    arrays = {"_data": {"offset_u": u, "offset_v": v}, "oracle_on_x1": offr}
    s = R.offgrid_summary(off, arrays, on)
    assert sum(b["n"] for b in s["by_offset_magnitude"].values()) == n
    assert sum(b["n"] for b in s["by_axis_north"].values()) == n
    assert sum(b["n"] for b in s["by_axis_east"].values()) == n
    assert s["contrast_off_minus_on"]["dr_gibf"][0] == pytest.approx(1.0)
    assert s["contrast_off_minus_on"]["dr_mmv"][0] == pytest.approx(0.0)
    assert s["contrast_gap_gibf_minus_mmv"][0] == pytest.approx(1.0)
    for m in R.METHODS:
        assert sum(s["transitions_on_to_off"][m].values()) == n
        assert s["transitions_on_to_off"][m]["correct->correct"] == int(
            np.sum(on[f"outcome_{m}"] == "correct"))
    assert s["descriptive"] is True


def test_d1_cells_carry_the_descriptive_panel(tm):
    d1 = [c for c in R.build_plan() if c.family == "B2" and c.d == 1]
    assert len(d1) == 6
    for c in d1:
        assert not c.confirmatory
        assert len(c.panel_arms) == 5
        assert c.panel_arms[R.HEADLINE_BAND_INDEX] == R.headline_arm(c)
    arrays, _ = R.run_data_cell(tm, d1[0], 3, TEST_SEED, solve_fn=_fake_solver([]))
    summ = R.summarize_cell(d1[0], arrays)
    assert "sensitivity_panel" in summ
    panel = summ["sensitivity_panel"]
    assert panel["descriptive"] is True and len(panel["points"]) == 5
    assert "never ruled" in summ["d1_panel_role"]
    assert set(summ["d1_psep_by_eps"]) == {f"x{m:g}" for m in R.EPS_BAND}
    assert all(set(v) == set(R.SPARSE) for v in summ["d1_psep_by_eps"].values())
    assert all(isinstance(b, bool) for v in summ["d1_psep_meets_threshold_by_eps"].values()
               for b in v.values())
    # the descriptive panel arms are labelled descriptive; the headline arm is not
    flags = {k: a["descriptive"] for k, a in summ["arms"].items()}
    assert flags[R.headline_arm(d1[0]).key] is False
    assert flags[R.Arm("oracle", True, 0.1).key] is True


def test_master_seed_run_is_allowed_with_the_recorded_go_ahead():
    if (R.DEFAULT_OUT / "manifest.json").exists():
        pytest.skip("the confirmatory run already exists")
    assert R.PRE_RUN_GATES_COMPLETE is True           # recorded 2026-10-11
    run = R.resolve_run(R.parse_args([]), clean=True)
    assert run["confirmatory"] is True and run["seed"] == R.MASTER_SEED
    assert R.OFFGRID_ROLE in run["roles"]
    # an unclean runner still blocks it
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args([]), clean=False)


def test_smoke_may_run_the_offgrid_role_alone(tmp_path):
    run = R.resolve_run(R.parse_args(["--smoke", "--n-trials", "2", "--only",
                                      "descriptive_offgrid",
                                      "--out-dir", str(tmp_path / "s")]))
    assert run["roles"] == ("descriptive_offgrid",) and run["confirmatory"] is False
