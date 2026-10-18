"""run_prereg_completion.py -- completion of the pre-registered work (SPEC S.17).

Pre-registered work that was never run (B2 grid cells, |rho| endpoints) or run
only at pilot size (B1, B3), executed AFTER the confirmatory adjudication.
Nothing here is adjudicated and nothing here can alter a sealed verdict. It is
kept apart by location: output goes to results/prereg_completion/, never into
a sealed directory.

New module only: the sealed powered runner is imported and reused; it is not
edited. Items (SPEC S.17):
  2. B2 completion grid   145 cells, headline arm, master seed 2250821694
  3. |rho| endpoints      12 cells,  same cell ids/draws as the sealed 0.85 cells
  4. B1 at full size      25 cells x 2 reductions, seed 2985373835
  5. B3 at full size      25 cells, seed 2985373835
and the fidelity self-check (item 10) that the completion path reproduces a
sealed cell's prefix bit-for-bit.

A3 workflow: committed clean before a master-seed run. Smoke runs use seed 1
and never write under results/.
"""

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

import run_experiment_B2B6_powered_d2 as P
from metrics import delta_r_bar, p_sep
from modeselect import estimate_n_sources
from simulate import (GRID_SHAPE, POLE_SPACING_KM, build_array_and_grid,
                      build_csm, eigendecompose, eigenmodes, geometry_record,
                      row_center_cols, simulate_snapshots)
from solvers import FAIRNESS_CONFIG, solve_all, with_reduction

COMPLETION_SEED = 2985373835
COMPLETION_SEED_PROVENANCE = (
    "secrets.randbits(32) in a fresh isolated interpreter (python -I) at "
    "2026-10-18T15:22:28Z; distinct from every prior seed (SPEC S.17 item 1)")
SMOKE_SEED = 1
N_TRIALS = 10_000

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS_ROOT = REPO / "results"
DEFAULT_OUT = RESULTS_ROOT / "prereg_completion"
SEALED_DIR = P.DEFAULT_OUT                       # results/B_viability/powered_d2

PHI_FULL = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)
SNR_FULL = (-5.0, 0.0, 5.0, 10.0, 20.0)
D_AXIS = P.D_AXIS
RHO_ENDPOINTS = (0.70, 0.95)
RHO_CELLS = tuple((phi, snr) for phi in (P.PHI_WIN_DEG,) + P.PHI_NULL_DEGS
                  for snr in P.SNR_PAIR)
SNR_B1 = (-5.0, 0.0, 5.0, 10.0, 20.0)
N_SNAP_B1 = 64
SNR_B3 = (-5.0, 0.0, 5.0, 10.0, 20.0)
NSNAP_B3 = (8, 16, 32, 64, 128)
D_B3 = 3

ROLE_B2 = "descriptive_completion"
ROLE_RHO = "descriptive_rho"
KINDS = ("b2", "rho", "b1", "b3")

EXPECTATIONS = {
    "E1": "at d=2, GIBF minus MMV in P_sep is not negative at phi=90 and not positive at phi=0",
    "E2": "P_sep at d=2 rises from phi=0 to phi=180 for every method",
    "E3": "at phi=90, P_sep is non-increasing in |rho| for every method",
    "E4": "B1 at full size agrees with the pilot's ordering of methods in >=80% of cells",
    "E5": "B3 MDL K-hat concentrates at 2 for SNR>=5 dB and degrades at -5 dB",
}


# ------------------------------------------------------------------- plans ---

def sealed_b2_cell_ids(manifest_path=SEALED_DIR / "manifest.json"):
    m = json.loads(Path(manifest_path).read_text())
    return {cid for cid in m["cells"] if cid.startswith("B2|")}


def full_b2_grid():
    return [P.DataCell("B2", phi, snr, d, P.N_SNAP, ROLE_B2,
                       (P.Arm("oracle", True, 1.0),))
            for phi in PHI_FULL for snr in SNR_FULL for d in D_AXIS]


def b2_completion_plan(sealed=None):
    """Full SPEC S.2-B2 grid minus the cells already sealed, in fixed order.
    The sealed set is read from the sealed manifest, never typed."""
    sealed = sealed_b2_cell_ids() if sealed is None else sealed
    return [c for c in full_b2_grid() if c.cell_id not in sealed]


def rho_plan():
    """(DataCell, |rho|) for each endpoint at the six confirmatory cells. The
    DataCell is the sealed d=2 cell's own id, so the draws are the same."""
    out = []
    for rho in RHO_ENDPOINTS:
        for phi, snr in RHO_CELLS:
            out.append((P.DataCell("B2", phi, snr, P.D_CONFIRM, P.N_SNAP, ROLE_RHO,
                                   (P.Arm("oracle", True, 1.0),)), rho))
    return out


def rho_stem(cell, rho):
    return f"rho{rho:g}_{cell.file_stem}"


# ------------------------------------------------------------ B2 / rho jobs ---

def run_b2_cell(tm, cell, n_trials, seed, abs_rho=None):
    """Sealed data path, optionally with the generator's |rho| overridden. The
    override is one module attribute of the sealed runner, restored after."""
    old = P.ABS_RHO
    try:
        if abs_rho is not None:
            P.ABS_RHO = float(abs_rho)
        arrays, timing = P.run_data_cell(tm, cell, n_trials, seed)
    finally:
        P.ABS_RHO = old
    return arrays, timing


def summarize_completion_cell(cell, arrays, abs_rho):
    summ = P.summarize_cell(cell, arrays)
    for s in summ["arms"].values():
        s["use"] = "descriptive_completion" if abs_rho is None else "descriptive_rho"
        s["descriptive"] = True
    summ["abs_rho"] = P.ABS_RHO if abs_rho is None else float(abs_rho)
    summ["phi_deg"], summ["snr_db"], summ["d"] = cell.phi_deg, cell.snr_db, cell.d
    return summ


# ------------------------------------------------------------------ B1 job ---

def b1_rng(seed, d, snr, trial):
    # reduction is NOT in the key: ON and OFF see the same draws (paired)
    return np.random.default_rng([int(seed), 2, zlib.crc32(repr("b1").encode()), int(d),
                                  zlib.crc32(repr(snr).encode()), int(trial)])


def run_b1_cell(tm, d, snr, n_trials, seed):
    idx_a, idx_b = row_center_cols(d)
    true_cells = [np.unravel_index(idx_a, GRID_SHAPE), np.unravel_index(idx_b, GRID_SHAPE)]
    cfgs = {True: with_reduction(FAIRNESS_CONFIG, True),
            False: with_reduction(FAIRNESS_CONFIG, False)}
    rec = {red: {f"{f}_{m}": np.empty(n_trials, dtype=bool if f in ("psep", "miss") else float)
                 for f in ("dr", "psep", "miss") for m in P.METHODS} for red in cfgs}
    for red in cfgs:
        rec[red]["niter_gibf"] = np.empty(n_trials)
        rec[red]["niter_mmv"] = np.empty(n_trials)
    for t in range(n_trials):
        rng = b1_rng(seed, d, snr, t)
        X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                                  n_snap=N_SNAP_B1, snr_db=snr)
        lam, U = eigendecompose(build_csm(X))
        V = eigenmodes(lam, U, P.K_ORACLE)
        for red, cfg in cfgs.items():
            res = solve_all(tm.A, V, cfg)
            for m in P.METHODS:
                I2d = res[m]["I"].reshape(GRID_SHAPE)
                rec[red][f"dr_{m}"][t] = delta_r_bar(I2d, true_cells)
                rec[red][f"psep_{m}"][t] = p_sep(I2d, true_cells)
                rec[red][f"miss_{m}"][t] = P.matched_error_components(I2d, true_cells)["miss"]
            rec[red]["niter_gibf"][t] = float(np.mean(res["gibf"]["n_iter"]))
            rec[red]["niter_mmv"][t] = float(res["mmv"]["n_iter"])
    return rec


def summarize_b1_cell(d, snr, rec):
    cid = f"B1full|snr={snr:g}|d={d}"
    n = len(rec[True]["dr_gibf"])
    idx = P.boot_indices(n, P.boot_rng(cid))
    out = dict(cell_id=cid, d=d, snr_db=snr, separation_km=round(d * POLE_SPACING_KM, 2),
               n_trials=n, reduction={})
    for red, r in rec.items():
        s = {}
        for m in P.METHODS:
            s[f"dr_{m}"] = P.mean_ci(r[f"dr_{m}"], idx)
            s[f"psep_{m}"] = P.mean_ci(r[f"psep_{m}"], idx)
            s[f"miss_rate_{m}"] = P.mean_ci(r[f"miss_{m}"], idx)
        s["gap_gibf_minus_mmv"] = P.mean_ci(r["dr_gibf"] - r["dr_mmv"], idx)
        s["niter_gibf_mean"] = float(np.mean(r["niter_gibf"]))
        s["niter_mmv_mean"] = float(np.mean(r["niter_mmv"]))
        out["reduction"]["on" if red else "off"] = s
    return out


# ------------------------------------------------------------------ B3 job ---

def b3_rng(seed, snr, n_snap, trial):
    return np.random.default_rng([int(seed), 3, zlib.crc32(repr("b3").encode()),
                                  zlib.crc32(repr(snr).encode()), int(n_snap), int(trial)])


def run_b3_cell(tm, snr, n_snap, n_trials, seed):
    idx_a, idx_b = row_center_cols(D_B3)
    k_mdl = np.empty(n_trials, dtype=int)
    k_aic = np.empty(n_trials, dtype=int)
    for t in range(n_trials):
        X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0],
                                  b3_rng(seed, snr, n_snap, t), n_snap=n_snap, snr_db=snr)
        lam, _ = eigendecompose(build_csm(X))
        k_mdl[t] = estimate_n_sources(lam, n_snap, criterion="mdl")[0]
        k_aic[t] = estimate_n_sources(lam, n_snap, criterion="aic")[0]
    return dict(khat_mdl=k_mdl, khat_aic=k_aic)


def summarize_b3_cell(snr, n_snap, rec):
    cid = f"B3full|snr={snr:g}|nsnap={n_snap}"
    n = len(rec["khat_mdl"])
    idx = P.boot_indices(n, P.boot_rng(cid))
    out = dict(cell_id=cid, snr_db=snr, n_snap=n_snap, n_trials=n)
    for crit in ("mdl", "aic"):
        k = rec[f"khat_{crit}"]
        out[f"{crit}_error_rate"] = P.mean_ci((k != 2).astype(float), idx)
        out[f"{crit}_mean"] = float(np.mean(k))
        out[f"{crit}_khat_hist"] = {int(v): int(np.sum(k == v)) for v in sorted(set(k.tolist()))}
    return out


# --------------------------------------------------------------------- jobs ---

_STATE = {}


def _tm():
    if "tm" not in _STATE:
        _STATE["tm"] = build_array_and_grid(row_normalisation=P.ROW_NORMALISATION)
    return _STATE["tm"]


def plan_for(kinds):
    jobs = []
    if "b2" in kinds:
        jobs += [("b2", c, None) for c in b2_completion_plan()]
    if "rho" in kinds:
        jobs += [("rho", c, rho) for c, rho in rho_plan()]
    if "b1" in kinds:
        jobs += [("b1", (d, snr), None) for d in D_AXIS for snr in SNR_B1]
    if "b3" in kinds:
        jobs += [("b3", (snr, ns), None) for snr in SNR_B3 for ns in NSNAP_B3]
    return jobs


def job_name(job):
    kind, spec, rho = job
    if kind == "b2":
        return spec.cell_id
    if kind == "rho":
        return f"{spec.cell_id}|rho={rho:g}"
    if kind == "b1":
        return f"B1full|snr={spec[1]:g}|d={spec[0]}"
    return f"B3full|snr={spec[0]:g}|nsnap={spec[1]}"


def job_stem(job):
    kind, spec, rho = job
    if kind == "b2":
        return spec.file_stem
    if kind == "rho":
        return rho_stem(spec, rho)
    if kind == "b1":
        return f"B1full_snr{spec[1]:g}_d{spec[0]}"
    return f"B3full_snr{spec[0]:g}_nsnap{spec[1]}"


def cell_job(job, n_trials, master_seed, completion_seed):
    """One job start to finish. Top level so worker processes can run it."""
    kind, spec, rho = job
    tm = _tm()
    t0 = time.time()
    if kind in ("b2", "rho"):
        arrays, _ = run_b2_cell(tm, spec, n_trials, master_seed, rho)
        summ = summarize_completion_cell(spec, arrays, rho)
        flat = {f"{ak}__{f}": np.asarray(a) for ak, rec in arrays.items()
                for f, a in rec.items()}
    elif kind == "b1":
        rec = run_b1_cell(tm, spec[0], spec[1], n_trials, completion_seed)
        summ = summarize_b1_cell(spec[0], spec[1], rec)
        flat = {f"{'on' if red else 'off'}__{f}": np.asarray(a)
                for red, r in rec.items() for f, a in r.items()}
    else:
        rec = run_b3_cell(tm, spec[0], spec[1], n_trials, completion_seed)
        summ = summarize_b3_cell(spec[0], spec[1], rec)
        flat = {k: np.asarray(v) for k, v in rec.items()}
    return dict(job=job, summ=summ, flat=flat, secs=time.time() - t0)


# ----------------------------------------------------------- fidelity check ---

FIDELITY_PREFIX = 200
FIDELITY_FIELDS = ("dr_gibf", "dr_mmv", "dr_l2", "psep_gibf", "psep_mmv", "psep_l2",
                   "niter_mmv", "eps_mmv")


def fidelity_check(n=FIDELITY_PREFIX, sealed_dir=SEALED_DIR, rho=None):
    """The completion path must reproduce the first n trials of a sealed cell
    bit-for-bit (SPEC S.17 item 10). Returns (ok, detail). `rho` exercises the
    |rho| override at the sealed 0.85 (identity)."""
    tm = _tm()
    cell = P.DataCell("B2", P.PHI_WIN_DEG, 5.0, P.D_CONFIRM, P.N_SNAP, "win",
                      (P.Arm("oracle", True, 1.0),))
    sealed = np.load(Path(sealed_dir) / f"{cell.file_stem}.npz")
    arrays, _ = run_b2_cell(tm, cell, n, P.MASTER_SEED, rho)
    key = P.headline_arm(cell).key
    bad = [f for f in FIDELITY_FIELDS
           if not np.array_equal(np.asarray(arrays[key][f]), sealed[f"{key}__{f}"][:n])]
    return (not bad), dict(cell_id=cell.cell_id, n=n, fields=list(FIDELITY_FIELDS),
                           mismatched=bad, abs_rho=(P.ABS_RHO if rho is None else rho))


# --------------------------------------------------------------- run guards ---

def _git(*args, cwd=HERE):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=False).stdout.strip()


def script_is_committed_clean():
    rel = Path(__file__).resolve().relative_to(REPO).as_posix()
    return not _git("status", "--porcelain", "--", rel, cwd=REPO) and bool(
        _git("ls-files", "--", rel, cwd=REPO))


def _is_under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="seed 1, any n, never under results/")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--kinds", default=",".join(KINDS), help="comma list of b2,rho,b1,b3")
    ap.add_argument("--only-first", type=int, default=None,
                    help="smoke: run only the first N jobs of the plan")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-fidelity", action="store_true", help="smoke only")
    return ap.parse_args(argv)


def resolve_run(args, clean=None):
    """Every write/seed guard, before any computation. Returns the seeds."""
    out_dir = Path(args.out_dir)
    kinds = tuple(k for k in args.kinds.split(",") if k)
    if bad := [k for k in kinds if k not in KINDS]:
        sys.exit(f"unknown kind(s): {bad}")
    if (out_dir / "manifest.json").exists():
        sys.exit(f"{out_dir} already holds a manifest.json; refusing to overwrite a run")
    if args.smoke:
        if _is_under(out_dir, RESULTS_ROOT):
            sys.exit("--smoke refuses to write under results/; pass --out-dir elsewhere")
        return dict(master_seed=SMOKE_SEED, completion_seed=SMOKE_SEED,
                    master_seed_run=False, kinds=kinds)
    if args.only_first is not None or args.skip_fidelity:
        sys.exit("--only-first and --skip-fidelity are smoke-test options")
    if set(kinds) != set(KINDS):
        sys.exit("a master-seed run covers every kind; subsets are smoke-test only")
    if args.n_trials != N_TRIALS:
        sys.exit(f"a master-seed run uses n_trials={N_TRIALS}; use --smoke for other n")
    if not _is_under(out_dir, RESULTS_ROOT):
        sys.exit("a master-seed run must write under results/ (use --smoke for scratch runs)")
    if _is_under(out_dir, RESULTS_ROOT / "B_viability") or _is_under(
            out_dir, RESULTS_ROOT / "A_flr_coherence") or _is_under(
            out_dir, RESULTS_ROOT / "V_kernel_validation"):
        sys.exit("a completion run never writes into a sealed directory")
    if clean is None:
        clean = script_is_committed_clean()
    if not clean:
        sys.exit("A3 workflow: commit this runner before a master-seed run")
    return dict(master_seed=P.MASTER_SEED, completion_seed=COMPLETION_SEED,
                master_seed_run=True, kinds=kinds)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None):
    args = parse_args(argv)
    run = resolve_run(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SPEC S.17 item 10: a mismatch stops the run before any new cell exists.
    fid = None
    if not args.skip_fidelity:
        ok, detail = fidelity_check()
        ok2, detail2 = fidelity_check(rho=P.ABS_RHO)
        fid = dict(plain=detail, rho_override_identity=detail2)
        if not (ok and ok2):
            sys.exit(f"fidelity self-check FAILED: {fid}")
        print("fidelity self-check passed (sealed prefix reproduced bit-for-bit)", flush=True)

    plan = plan_for(run["kinds"])
    if args.only_first is not None:
        plan = plan[:args.only_first]

    if args.workers > 1:
        for var in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ[var] = "1"

    t_start = time.time()
    results = []

    def _save(r):
        stem = job_stem(r["job"])
        np.savez_compressed(out_dir / f"{stem}.npz", **r["flat"])
        (out_dir / f"{stem}.json").write_text(json.dumps(r["summ"], indent=2,
                                                          default=_json_default))

    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(cell_job, j, args.n_trials, run["master_seed"],
                              run["completion_seed"]) for j in plan]
            for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                r = fut.result()
                _save(r)
                print(f"[{done}/{len(plan)}] {job_name(r['job'])} {r['secs']:.1f}s", flush=True)
                results.append(r)
    else:
        for i, j in enumerate(plan, 1):
            r = cell_job(j, args.n_trials, run["master_seed"], run["completion_seed"])
            _save(r)
            print(f"[{i}/{len(plan)}] {job_name(j)} {r['secs']:.1f}s", flush=True)
            results.append(r)

    order = {job_name(j): i for i, j in enumerate(plan)}
    results.sort(key=lambda r: order[job_name(r["job"])])
    summaries = {job_name(r["job"]): r["summ"] for r in results}
    timings = {job_name(r["job"]): round(r["secs"], 1) for r in results}

    rows = []
    for r in results:
        kind = r["job"][0]
        if kind in ("b2", "rho"):
            for arm, s in r["summ"]["arms"].items():
                row = dict(cell_id=job_name(r["job"]), kind=kind, arm=arm, n_trials=s["n_trials"],
                           abs_rho=r["summ"]["abs_rho"])
                for k in [f"dr_{m}" for m in P.METHODS] + [f"psep_{m}" for m in P.METHODS] + [
                        "gap_gibf_minus_mmv"]:
                    row[f"{k}_mean"], (row[f"{k}_ci_lo"], row[f"{k}_ci_hi"]) = s[k]
                rows.append(row)
    if rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    manifest = dict(
        experiment="Pre-registered completion, executed after adjudication (SPEC S.17)",
        status="pre_registered_completion",
        adjudication="none: nothing here is adjudicated and cannot alter a sealed verdict",
        master_seed_run=bool(run["master_seed_run"]), smoke=bool(args.smoke),
        master_seed=P.MASTER_SEED, completion_seed=COMPLETION_SEED,
        completion_seed_provenance=COMPLETION_SEED_PROVENANCE,
        seeds_used=dict(b2_rho=run["master_seed"], b1_b3=run["completion_seed"]),
        kinds=list(run["kinds"]), workers=int(args.workers),
        git_commit=_git("rev-parse", "HEAD"),
        script_committed_clean=script_is_committed_clean(),
        script_git_blob=_git("rev-parse",
                             f"HEAD:{Path(__file__).resolve().relative_to(REPO).as_posix()}",
                             cwd=REPO),
        script_sha256=hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        secsy_sha=_git("rev-parse", "HEAD", cwd=REPO / "secsy"),
        numpy=np.__version__, python=sys.version.split()[0],
        n_trials=args.n_trials, n_snap=P.N_SNAP, k_oracle=P.K_ORACLE,
        fairness_config=dataclasses.asdict(FAIRNESS_CONFIG),
        geometry=geometry_record(),
        fidelity_self_check=fid,
        expectations_stated_before_data=EXPECTATIONS,
        plan_sizes={k: sum(1 for j in plan if j[0] == k) for k in KINDS},
        cells=summaries, timing_s=timings,
        runtime_s=round(time.time() - t_start, 1),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                       default=_json_default))
    print(f"Wrote {len(plan)} jobs + manifest to {out_dir} in {manifest['runtime_s']}s")
    return manifest


if __name__ == "__main__":
    main()
