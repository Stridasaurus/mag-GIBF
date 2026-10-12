"""tests/test_b4_b6b.py — pytest gate for B4 (phantom phase) and B6b (support
jitter), SPEC S.15. Every test uses throwaway seeds, never B4B6B_SEED."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

import run_experiment_B4_B6b as R  # noqa: E402
import solvers  # noqa: E402
from phantom_phase import (C_KM_S, gibf_irls_complex, gibf_irls_scaled,  # noqa: E402
                           phantom_phase_matrix)
from simulate import (GRID_SHAPE, build_array_and_grid, build_csm,  # noqa: E402
                      eigendecompose, eigenmodes, row_center_cols,
                      simulate_snapshots, simulate_snapshots_general)
from solvers import FAIRNESS_CONFIG, solve_all, with_reduction  # noqa: E402

TEST_SEED = 777001


@pytest.fixture(scope="module")
def tm():
    return build_array_and_grid(row_normalisation="none")


def _trial_V(tm, d=2, snr=5.0, seed=11, coherence=None, A=None):
    idx_a, idx_b = row_center_cols(d)
    cols = np.array([[idx_a] * R.N_SNAP, [idx_b] * R.N_SNAP])
    rng = np.random.default_rng(seed)
    A = tm.A if A is None else A
    X, _ = simulate_snapshots_general(A, cols, [1.0, 1.0], rng, R.N_SNAP, snr,
                                      coherence=coherence)
    lam, U = eigendecompose(build_csm(X))
    return eigenmodes(lam, U, 2)


# ------------------------------------------------------------- seeds -------

def test_seeds_distinct_from_every_prior_seed():
    prior = {20260706, 20260712, 20260721, 20260727, 20260801, 20260828,
             2250821694}
    assert R.B4B6B_SEED not in prior and R.SMOKE_SEED != R.B4B6B_SEED
    assert isinstance(R.B4B6B_SEED, int) and R.B4B6B_SEED > 0


# ------------------------------------------------- general simulator -------

def test_general_simulator_matches_simulate_snapshots_for_constant_columns(tm):
    idx_a, idx_b = row_center_cols(3)
    rho = 0.85 * np.exp(1j * np.deg2rad(90.0))
    rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
    X1, C1 = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng1, 64, 5.0,
                                coherence=rho)
    cols = np.array([[idx_a] * 64, [idx_b] * 64])
    X2, C2 = simulate_snapshots_general(tm.A, cols, [1.0, 1.0], rng2, 64, 5.0,
                                        coherence=rho)
    assert np.allclose(X1, X2, rtol=1e-10, atol=0.0)
    assert np.allclose(C1, C2, rtol=1e-10, atol=0.0)
    # identical random stream consumption
    assert rng1.standard_normal() == rng2.standard_normal()


def test_general_simulator_shape_checks(tm):
    with pytest.raises(ValueError):
        simulate_snapshots_general(tm.A, np.zeros((2, 10), dtype=int), [1.0, 1.0],
                                   np.random.default_rng(1), 64, 5.0)


# ----------------------------------------------------- phantom kernel ------

def test_phantom_kernel_magnitudes_unchanged_and_complex(tm):
    # kernel entries are ~1e-13 T: use relative tolerances only (atol = 0)
    Ap = phantom_phase_matrix(tm, 800.0)
    scale = np.abs(tm.A).max()
    assert np.iscomplexobj(Ap) and Ap.shape == tm.A.shape
    assert np.allclose(np.abs(Ap), np.abs(tm.A), rtol=1e-12, atol=0.0)
    assert np.abs(Ap.imag).max() > 1e-3 * scale      # genuinely complex
    assert np.isrealobj(tm.A)                        # the real kernel is untouched


def test_phantom_kernel_tends_to_real_kernel_as_wavelength_grows(tm):
    Ap = phantom_phase_matrix(tm, 1.0e14)            # phase ~ 1e-11 rad
    assert np.abs(Ap - tm.A).max() / np.abs(tm.A).max() < 1e-6


def test_phantom_phase_varies_across_stations_not_just_poles(tm):
    # a column-constant phase would be a gauge; B4 needs a station-dependent ramp
    Ap = phantom_phase_matrix(tm, R.APERTURE_KM / 2.0)
    ph = np.angle(Ap / np.where(tm.A == 0, 1.0, tm.A + 0j))
    j = tm.A.shape[1] // 2
    assert np.ptp(np.unwrap(ph[:25, j])) > 0.5


def test_literal_wavelength_is_the_brief_definition_and_dose_axis_is_unaliased():
    k = R.kernel_wavelengths_km()
    assert k["real"] is None
    assert k["lit"] == pytest.approx(C_KM_S / 0.015)          # 15 mHz at light speed
    across = {n: R.APERTURE_KM / v for n, v in k.items() if v is not None}
    assert across["lit"] < 0.06                                 # about 0.04 wavelengths
    for w in R.B4_WAVELENGTHS_ACROSS:
        assert k[f"w{w:g}"] == pytest.approx(R.APERTURE_KM / w)
    # half-wavelength sampling on the pole grid for every arm
    assert R.POLE_SPACING_KM < 0.5 * min(v for v in k.values() if v is not None)


# --------------------------------------------- complex solver vs standard --

def test_complex_solver_equals_standard_gibf_on_a_real_kernel(tm):
    for red in (True, False):
        cfg = with_reduction(FAIRNESS_CONFIG, red)
        V = _trial_V(tm, d=2, snr=5.0, seed=21)
        std = solvers.gibf_irls(tm.A, V / np.linalg.norm(V[:, 0]), cfg)
        cpx = gibf_irls_complex(tm.A, V / np.linalg.norm(V[:, 0]), cfg)
        assert np.allclose(std["I"], cpx["I"], rtol=1e-8, atol=1e-14 * std["I"].max())
        assert std["n_iter"] == cpx["n_iter"]
        assert std["stop_reason"] == cpx["stop_reason"]
        scaled = gibf_irls_scaled(tm.A, V, cfg)
        full = solve_all(tm.A, V, cfg)["gibf"]
        assert np.allclose(scaled["I"], full["I"], rtol=1e-8,
                           atol=1e-14 * full["I"].max())


def test_standard_solvers_still_reject_a_complex_kernel(tm):
    Ap = phantom_phase_matrix(tm, 800.0)
    V = _trial_V(tm, d=2, snr=5.0, seed=3, A=Ap)
    with pytest.raises(AssertionError):
        solvers.gibf_irls(Ap, V, FAIRNESS_CONFIG)
    with pytest.raises(AssertionError):
        solvers.mmv_l1(Ap, V, FAIRNESS_CONFIG)


def test_complex_solver_runs_and_is_finite_on_phantom_kernel(tm):
    Ap = phantom_phase_matrix(tm, R.APERTURE_KM / 2.0)
    V = _trial_V(tm, d=3, snr=10.0, seed=9, A=Ap)
    out = gibf_irls_scaled(Ap, V, with_reduction(FAIRNESS_CONFIG, True))
    assert np.all(np.isfinite(out["I"])) and out["I"].shape == (tm.A.shape[1],)
    assert len(out["n_iter"]) == 2


# ----------------------------------------------------------- B6b ----------

def test_b6b_column_schedules_follow_the_spec():
    n_lon = GRID_SHAPE[1]
    idx_a, idx_b = row_center_cols(2)
    rng = np.random.default_rng(1)
    c, end, shift = R.b6b_columns("none", idx_a, idx_b, 64, n_lon, rng)
    assert np.all(c[0] == idx_a) and np.all(c[1] == idx_b) and end == idx_b and shift == 0
    c, end, shift = R.b6b_columns("step_away", idx_a, idx_b, 64, n_lon, rng)
    assert np.all(c[0] == idx_a)
    assert np.all(c[1][:32] == idx_b) and np.all(c[1][32:] == idx_b + 1)
    assert end == idx_b + 1 and shift == pytest.approx(0.5)
    c, end, shift = R.b6b_columns("step_toward", idx_a, idx_b, 64, n_lon, rng)
    assert np.all(c[1][:32] == idx_b) and np.all(c[1][32:] == idx_b - 1)
    assert end == idx_b - 1 and shift == pytest.approx(-0.5)
    c, end, shift = R.b6b_columns("iid", idx_a, idx_b, 64, n_lon, rng)
    assert np.all(c[0] == idx_a)
    offs = c[1] - idx_b
    allowed = {dr * n_lon + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    assert set(offs.tolist()) <= allowed and len(set(offs.tolist())) > 3
    assert end == idx_b and shift == 0
    with pytest.raises(ValueError):
        R.b6b_columns("sideways", idx_a, idx_b, 64, n_lon, rng)


def test_b6b_variants_share_draws_and_only_source_2_column_differs(tm):
    cells = {c.variant: c for c in R.build_plan() if c.family == "B6b"}
    assert {c.data_key for c in cells.values()} == {cells["none"].data_key}
    assert len({c.cell_id for c in cells.values()}) == 4
    rec = {v: R.run_b6b_cell(tm, c, 2, TEST_SEED, solve_fn=solve_all)
           for v, c in cells.items()}
    # metrics exist and are finite for every variant and method
    for v, r in rec.items():
        for m in R.METHODS:
            assert np.all(np.isfinite(r[f"dr_{m}"])) and r[f"psep_{m}"].dtype == bool


# ---------------------------------------------------------- B4 run --------

def test_b4_plan_and_pairing(tm):
    plan = R.build_plan()
    b4 = [c for c in plan if c.family == "B4"]
    b6 = [c for c in plan if c.family == "B6b"]
    assert len(b4) == len(R.B4_SNR_DB) * len(R.B4_D) == 25 and len(b6) == 4
    assert len({c.cell_id for c in plan}) == len(plan)
    assert all(c.role == "b4" for c in b4) and all(c.role == "b6b" for c in b6)
    kernels = R.build_kernels(tm)
    assert set(kernels) == {"real", "lit", "w1", "w2", "w5"}
    assert np.isrealobj(kernels["real"]) and all(
        np.iscomplexobj(kernels[k]) for k in ("lit", "w1", "w2", "w5"))
    cell = next(c for c in b4 if c.snr_db == 10.0 and c.d == 3)
    arrays = R.run_b4_cell(tm, {"real": kernels["real"], "lit": kernels["lit"]},
                           cell, 3, TEST_SEED)
    assert set(arrays) == {("real", True), ("real", False), ("lit", True), ("lit", False)}
    for rec in arrays.values():
        assert rec["dr"].shape == (3,) and np.all(np.isfinite(rec["dr"]))
        assert rec["niter"].shape == (3, 2) and rec["stop"].shape == (3, 2)
    # the literal phantom (0.04 wavelengths) is nearly real: same pairing, close maps
    assert np.allclose(arrays[("real", True)]["dr"], arrays[("lit", True)]["dr"], atol=3.0)
    summ = R.summarize_b4(cell, {**arrays})
    assert "lit_minus_real_on" in summ["paired_contrast"]
    assert set(summ["arms"]) == {"real_on", "real_off", "lit_on", "lit_off"}


def test_d_min_table_reads_literally():
    t = R.d_min_table({(5.0, "real", "on"): {1: 0.0, 2: 0.85, 3: 1.0, 5: 1.0, 8: 1.0},
                       (5.0, "w5", "on"): {1: 0.9, 2: 0.9, 3: 1.0, 5: 1.0, 8: 1.0},
                       (5.0, "lit", "off"): {1: 0.0, 2: 0.1, 3: 0.2, 5: 0.3, 8: 0.4}})
    assert t["5.0|real|on"] == 2 and t["5.0|w5|on"] == 1 and t["5.0|lit|off"] == "inf"


# ---------------------------------------------------------- guards --------

def test_smoke_refuses_results_dir_and_master_requires_clean(tmp_path):
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--smoke", "--out-dir", str(R.DEFAULT_OUT)]))
    run = R.resolve_run(R.parse_args(["--smoke", "--n-trials", "2",
                                      "--out-dir", str(tmp_path / "s")]))
    assert run["seed"] == R.SMOKE_SEED and run["master_seed_run"] is False
    out = R.RESULTS_ROOT / "B_viability" / "_never_created_b4_test"
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--out-dir", str(out)]), clean=False)
    assert not out.exists()
    if not (R.DEFAULT_OUT / "manifest.json").exists():
        ok = R.resolve_run(R.parse_args([]), clean=True)
        assert ok["master_seed_run"] is True and ok["seed"] == R.B4B6B_SEED
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--n-trials", "5", "--out-dir", str(out)]), clean=True)
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--only", "b4", "--out-dir", str(out)]), clean=True)


def test_any_run_refuses_an_out_dir_with_a_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--smoke", "--out-dir", str(tmp_path)]))


# ------------------------------------------------- parallel == serial -------

def test_parallel_and_serial_runs_are_identical(tmp_path):
    """Cells are independent and deterministic given the seed, so the worker
    count must not change a single saved number."""
    a, b = tmp_path / "serial", tmp_path / "parallel"
    R.main(["--smoke", "--n-trials", "3", "--only", "b6b", "--out-dir", str(a)])
    R.main(["--smoke", "--n-trials", "3", "--only", "b6b", "--workers", "2",
            "--out-dir", str(b)])
    files = sorted(p.name for p in a.glob("*.npz"))
    assert files and files == sorted(p.name for p in b.glob("*.npz"))
    for f in files:
        za, zb = np.load(a / f), np.load(b / f)
        assert sorted(za.files) == sorted(zb.files)
        for k in za.files:
            assert np.array_equal(za[k], zb[k]), (f, k)
