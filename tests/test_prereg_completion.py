"""tests/test_prereg_completion.py -- gate for the pre-registered completion
runner (SPEC S.17). Uses throwaway seeds except where a sealed cell's own seed
is the point (the fidelity checks)."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "viability-test"))

import run_experiment_B4_B6b as B4  # noqa: E402
import run_prereg_completion as R  # noqa: E402
import run_experiment_B2B6_powered_d2 as P  # noqa: E402

SEALED = R.SEALED_DIR


def test_seed_is_new_and_distinct_from_every_seed_in_the_seal():
    seal = (ROOT / "seal" / "SEAL.md").read_text(encoding="utf-8")
    sealed_seeds = {int(x) for x in re.findall(r"\b\d{9,10}\b", seal)}
    prior = {20260706, 20260712, 20260721, 20260727, 20260801, 20260828,
             P.MASTER_SEED, B4.B4B6B_SEED} | sealed_seeds
    assert R.COMPLETION_SEED not in prior and R.COMPLETION_SEED != R.SMOKE_SEED
    assert isinstance(R.COMPLETION_SEED, int) and R.COMPLETION_SEED > 0


def test_b2_plan_is_the_full_grid_minus_the_sealed_cells():
    sealed = R.sealed_b2_cell_ids()
    plan = R.b2_completion_plan()
    full = R.full_b2_grid()
    assert len(full) == 175 and len(sealed) == 30 and len(plan) == 145
    ids = [c.cell_id for c in plan]
    assert len(set(ids)) == 145 and not (set(ids) & sealed)
    assert set(ids) | sealed == {c.cell_id for c in full}
    assert all(c.role == R.ROLE_B2 and len(c.arms) == 1 for c in plan)
    assert all(c.arms[0] == P.headline_arm(c) for c in plan)


def test_rho_plan_pairs_with_the_sealed_cells_by_cell_id():
    sealed = R.sealed_b2_cell_ids()
    plan = R.rho_plan()
    assert len(plan) == 12
    assert {rho for _, rho in plan} == set(R.RHO_ENDPOINTS)
    for cell, _ in plan:
        assert cell.cell_id in sealed and cell.d == P.D_CONFIRM
    assert len({R.rho_stem(c, r) for c, r in plan}) == 12


def test_full_plan_sizes():
    jobs = R.plan_for(R.KINDS)
    counts = {k: sum(1 for j in jobs if j[0] == k) for k in R.KINDS}
    assert counts == {"b2": 145, "rho": 12, "b1": 25, "b3": 25}
    assert len({R.job_name(j) for j in jobs}) == len(jobs)
    assert len({R.job_stem(j) for j in jobs}) == len(jobs)


def test_fidelity_check_reproduces_a_sealed_prefix_bit_for_bit():
    ok, detail = R.fidelity_check(n=25)
    assert ok, detail
    ok2, detail2 = R.fidelity_check(n=25, rho=P.ABS_RHO)
    assert ok2, detail2


def test_rho_override_changes_the_data_and_is_restored():
    tm = R._tm()
    cell = R.rho_plan()[0][0]
    before = P.ABS_RHO
    a07, _ = R.run_b2_cell(tm, cell, 8, 4242, 0.70)
    assert P.ABS_RHO == before
    a85, _ = R.run_b2_cell(tm, cell, 8, 4242, None)
    key = P.headline_arm(cell).key
    assert not np.array_equal(a07[key]["dr_gibf"], a85[key]["dr_gibf"]) or \
        not np.array_equal(a07[key]["niter_mmv"], a85[key]["niter_mmv"])
    again, _ = R.run_b2_cell(tm, cell, 8, 4242, None)
    assert np.array_equal(again[key]["dr_gibf"], a85[key]["dr_gibf"])


def test_b1_draws_are_shared_across_reductions_and_not_the_pilot_seed():
    a = R.b1_rng(7, 2, 5.0, 3).standard_normal(4)
    b = R.b1_rng(7, 2, 5.0, 3).standard_normal(4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, R.b1_rng(7, 2, 5.0, 4).standard_normal(4))
    assert not np.array_equal(a, R.b1_rng(8, 2, 5.0, 3).standard_normal(4))


def test_b1_and_b3_cells_run_and_summarise():
    tm = R._tm()
    rec = R.run_b1_cell(tm, 3, 5.0, 4, 99)
    s = R.summarize_b1_cell(3, 5.0, rec)
    assert s["n_trials"] == 4 and set(s["reduction"]) == {"on", "off"}
    r3 = R.run_b3_cell(tm, 5.0, 64, 6, 99)
    s3 = R.summarize_b3_cell(5.0, 64, r3)
    assert sum(s3["mdl_khat_hist"].values()) == 6


def test_guards(tmp_path):
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--smoke", "--out-dir", str(R.DEFAULT_OUT)]))
    run = R.resolve_run(R.parse_args(["--smoke", "--n-trials", "2",
                                      "--out-dir", str(tmp_path / "s")]))
    assert run["master_seed"] == R.SMOKE_SEED and run["master_seed_run"] is False
    out = R.RESULTS_ROOT / "prereg_completion_never_created_test"
    with pytest.raises(SystemExit):                      # uncommitted runner
        R.resolve_run(R.parse_args(["--out-dir", str(out)]), clean=False)
    assert not out.exists()
    with pytest.raises(SystemExit):                      # wrong n
        R.resolve_run(R.parse_args(["--n-trials", "5", "--out-dir", str(out)]), clean=True)
    with pytest.raises(SystemExit):                      # subset is smoke only
        R.resolve_run(R.parse_args(["--kinds", "b1", "--out-dir", str(out)]), clean=True)
    with pytest.raises(SystemExit):                      # never into a sealed dir
        R.resolve_run(R.parse_args(["--out-dir",
                                    str(R.RESULTS_ROOT / "B_viability" / "x")]), clean=True)
    with pytest.raises(SystemExit):                      # outside results/
        R.resolve_run(R.parse_args(["--out-dir", str(tmp_path / "m")]), clean=True)
    ok = R.resolve_run(R.parse_args(["--out-dir", str(out)]), clean=True)
    assert ok["master_seed_run"] is True
    assert ok["master_seed"] == P.MASTER_SEED and ok["completion_seed"] == R.COMPLETION_SEED


def test_refuses_an_out_dir_with_a_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit):
        R.resolve_run(R.parse_args(["--smoke", "--out-dir", str(tmp_path)]))


def test_parallel_and_serial_runs_are_identical(tmp_path):
    a, b = tmp_path / "serial", tmp_path / "parallel"
    base = ["--smoke", "--n-trials", "3", "--kinds", "b2,rho", "--only-first", "2",
            "--skip-fidelity"]
    R.main(base + ["--out-dir", str(a)])
    R.main(base + ["--workers", "2", "--out-dir", str(b)])
    files = sorted(p.name for p in a.glob("*.npz"))
    assert files and files == sorted(p.name for p in b.glob("*.npz"))
    for f in files:
        za, zb = np.load(a / f), np.load(b / f)
        for k in za.files:
            x, y = za[k], zb[k]
            if k.startswith("_data__khat"):
                # K-hat is a diagnostic that no oracle-K arm reads; the MDL raw
                # argmin over a noise-only spectrum can flip on a last-ulp
                # difference between BLAS thread counts. Shapes must match; values are not compared.
                assert x.shape == y.shape, (f, k)
                continue
            if x.dtype.kind == "f" and not k.split("__")[-1].startswith(("dr_", "niter")):
                # BLAS thread count (serial parent vs one-thread workers) can
                # move continuous diagnostics in the last ulp; every scored
                # quantity (dr_*, psep_*, outcome_*, niter*) must be exact.
                assert np.allclose(x, y, rtol=1e-9, atol=0, equal_nan=True), (f, k)
            else:
                assert np.array_equal(x, y), (f, k)
    m = json.loads((a / "manifest.json").read_text())
    assert m["status"] == "pre_registered_completion" and m["smoke"] is True
