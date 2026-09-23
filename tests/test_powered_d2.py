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
    assert R.N_TRIALS_POWERED == 1635             # SPEC S.12 power calc
    assert R.PHI_WIN_DEG == 90.0 and R.PHI_NULL_DEGS == (0.0, 180.0)
    assert R.SNR_PAIR == (5.0, 10.0) and R.N_SNAP == 64 and R.ABS_RHO == 0.85
    assert R.EPS_BAND == (0.1, 0.3, 1.0, 3.0, 10.0)
    assert R.EPS_BAND[R.HEADLINE_BAND_INDEX] == 1.0
    assert R.N_BOOT == 2000 and R.D_AXIS == (1, 2, 3, 5, 8)
    assert R.B6A_NSNAP_PINNED == 64
    assert R.parse_args([]).n_trials == 1635 and R.parse_args([]).b6a_nsnap == 64


def test_master_seed_pinned_and_distinct_from_smoke():
    assert isinstance(R.MASTER_SEED, int) and R.MASTER_SEED > 0
    assert R.SMOKE_SEED != R.MASTER_SEED
    assert R.MASTER_SEED not in (20260706, 20260712, 20260721, 20260727,
                                 20260801, 20260828)


# ------------------------------------------------------------------ plan ---

def test_plan_structure():
    plan = R.build_plan()
    b2 = [c for c in plan if c.family == "B2"]
    assert len(b2) == 3 * 2 * 5 and len(plan) == 31
    ids = [c.cell_id for c in plan]
    assert len(set(ids)) == len(ids)              # the d=2 axis point is not duplicated
    conf = [c for c in plan if c.confirmatory]
    assert sorted(c.role for c in conf) == ["b6a"] + ["null"] * 4 + ["win"] * 2
    for c in conf:
        assert len(c.panel_arms) == 5
        assert c.panel_arms[R.HEADLINE_BAND_INDEX] == R.headline_arm(c)
        assert any(not a.reduction for a in c.arms)          # S.6(2) OFF pair
    for c in plan:
        if not c.confirmatory:
            assert c.arms == (R.Arm("oracle", True, 1.0),)
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
    assert set(timing) == {"oracle_on_x1", "_data"}


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
