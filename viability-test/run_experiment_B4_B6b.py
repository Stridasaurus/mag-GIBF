"""run_experiment_B4_B6b.py — descriptive B4 (phantom-phase ablation) and B6b
(support jitter). SPEC S.15 is the specification; it was committed before this
file existed and before any trial of either experiment was run.

Neither experiment adjudicates anything. Neither enters D2's win/loss/B6a
rules, which are decided and recorded in ROADMAP S8 (2026-10-12). Both exist
because ROADMAP S4 exit condition 3 lists them and S.2 designed them.

  B4   GIBF only, B1's incoherent scenario, real kernel A against a phantom
       kernel A (.) exp(i Phi) at 0.04 (literal 15 mHz, light speed), 1, 2 and
       5 wavelengths across the station longitude span. Reduction ON headline,
       OFF twin. Trials of every kernel and reduction share random draws.
  B6b  Coherent phi = 90, |rho| = 0.85, 5 dB, d = 2, oracle K = 2: source 2's
       column varies by snapshot (none / step_away / step_toward / iid 3x3).
       L2, GIBF, MMV on the same trials.

MASTER SEED: drawn with `secrets.randbits(32)` in a fresh isolated interpreter
(`python -I`, scratch working directory, no repo import, no result file opened)
at 2026-10-12T13:11:47Z, before this file was written. Smoke runs use
SMOKE_SEED and never write under results/.
"""

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

import run_experiment_B2B6_powered_d2 as P
from metrics import delta_r_bar, p_sep
from phantom_phase import C_KM_S, gibf_irls_scaled, phantom_phase_matrix
from simulate import (GRID_SHAPE, N_STATION_LON, POLE_SPACING_KM,
                      STATION_LON_SPACING_KM, build_array_and_grid, build_csm,
                      eigendecompose, eigenmodes, geometry_record,
                      row_center_cols, simulate_snapshots_general)
from solvers import FAIRNESS_CONFIG, solve_all, with_reduction

# ----------------------------------------------------------------- seeds ---

B4B6B_SEED = 2411409774      # PINNED. Never regenerate, never change after any run.
B4B6B_SEED_PROVENANCE = (
    "secrets.randbits(32) in a fresh isolated interpreter (python -I, cwd = a "
    "scratch directory, no repo import, no result file opened), drawn "
    "2026-10-12T13:11:47Z, before this runner was written. Distinct from every "
    "prior seed in the repo (20260706, 20260712, 20260721, 20260727, 20260801, "
    "20260828, 2250821694).")
SMOKE_SEED = 1

# ---------------------------------------------------------------- pins -----

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS_ROOT = REPO / "results"
DEFAULT_OUT = RESULTS_ROOT / "B_viability" / "b4_b6b"

N_TRIALS = 10_000                          # SPEC S.15 item 6 / P14
N_SNAP = 64
K_ORACLE = 2
PSEP_THRESHOLD = 0.8
N_BOOT = 2000
CI_LEVEL = 0.95

B4_SNR_DB = (-5.0, 0.0, 5.0, 10.0, 20.0)  # B1's grid
B4_D = (1, 2, 3, 5, 8)
LITERAL_FREQUENCY_MHZ = 15.0               # millihertz, archived brief config
B4_WAVELENGTHS_ACROSS = (1.0, 2.0, 5.0)    # wavelengths across the station longitude span
APERTURE_KM = STATION_LON_SPACING_KM * (N_STATION_LON - 1)

B6B_PHI_DEG = 90.0
B6B_ABS_RHO = P.ABS_RHO                    # SLOT-1 midpoint, 0.85
B6B_SNR_DB = 5.0
B6B_D = 2
B6B_VARIANTS = ("none", "step_away", "step_toward", "iid")

METHODS = ("l2", "gibf", "mmv")
ALL_ROLES = ("b4", "b6b")

EXPECTATIONS = {
    "E1": "the literal-wavelength B4 arm is indistinguishable from the real arm within Monte-Carlo error",
    "E2": "B4 P_sep at small d is non-decreasing in wavelengths across the aperture",
    "E3": "any B4 gain is concentrated at d <= 2 (d >= 3 is at ceiling for the real kernel on the B1 grid)",
    "E4": "B6b jitter degrades MMV more than GIBF (the shared-support prior is false across the window)",
}


def kernel_wavelengths_km():
    """name -> wavelength in km (None = the real kernel)."""
    out = {"real": None,
           "lit": C_KM_S / (LITERAL_FREQUENCY_MHZ * 1.0e-3)}
    for w in B4_WAVELENGTHS_ACROSS:
        out[f"w{w:g}"] = APERTURE_KM / w
    return out


def build_kernels(tm):
    """name -> kernel matrix. Only B4 ever holds a complex kernel."""
    out = {}
    for name, lam in kernel_wavelengths_km().items():
        out[name] = tm.A if lam is None else phantom_phase_matrix(tm, lam)
    return out


# ------------------------------------------------------------- plan --------

@dataclasses.dataclass(frozen=True)
class Cell:
    family: str          # "B4" | "B6b"
    snr_db: float
    d: int
    variant: str         # B6b variant; "-" for B4
    role: str            # "b4" | "b6b"

    @property
    def cell_id(self):
        if self.family == "B4":
            return f"B4|snr={self.snr_db:g}|d={self.d}|nsnap={N_SNAP}"
        return (f"B6b|variant={self.variant}|phi={B6B_PHI_DEG:g}"
                f"|snr={self.snr_db:g}|d={self.d}|nsnap={N_SNAP}")

    @property
    def data_key(self):
        """Key of the random draws. Every kernel/reduction of a B4 cell, and
        every B6b variant, shares draws so contrasts are paired."""
        if self.family == "B4":
            return self.cell_id
        return (f"B6bdata|phi={B6B_PHI_DEG:g}|snr={self.snr_db:g}"
                f"|d={self.d}|nsnap={N_SNAP}")

    @property
    def file_stem(self):
        return (self.cell_id.replace("|", "_").replace("=", "")
                .replace("nsnap", "n"))


def build_plan():
    cells = [Cell("B4", snr, d, "-", "b4") for snr in B4_SNR_DB for d in B4_D]
    cells += [Cell("B6b", B6B_SNR_DB, B6B_D, v, "b6b") for v in B6B_VARIANTS]
    return cells


# ------------------------------------------------------------- B4 ----------

def _alloc_b4(n):
    return dict(dr=np.empty(n), psep=np.empty(n, dtype=bool),
                miss=np.empty(n, dtype=bool), npeaks=np.empty(n, dtype=int),
                niter=np.empty((n, K_ORACLE), dtype=int),
                stop=np.empty((n, K_ORACLE), dtype="U24"))


def run_b4_cell(tm, kernels, cell, n_trials, seed, solve_fn=gibf_irls_scaled):
    """GIBF on every kernel and both reductions, sharing the trial's draws."""
    idx_a, idx_b = row_center_cols(cell.d)
    cols = np.array([[idx_a] * N_SNAP, [idx_b] * N_SNAP])
    true_cells = [np.unravel_index(idx_a, GRID_SHAPE),
                  np.unravel_index(idx_b, GRID_SHAPE)]
    cfgs = {True: with_reduction(FAIRNESS_CONFIG, True),
            False: with_reduction(FAIRNESS_CONFIG, False)}
    out = {(k, red): _alloc_b4(n_trials) for k in kernels for red in (True, False)}
    for t in range(n_trials):
        for kname, A in kernels.items():
            rng = P.trial_rng(seed, cell.data_key, t)
            X, _ = simulate_snapshots_general(A, cols, [1.0, 1.0], rng, N_SNAP,
                                              cell.snr_db, coherence=None)
            lam, U = eigendecompose(build_csm(X))
            V = eigenmodes(lam, U, K_ORACLE)
            for red in (True, False):
                res = solve_fn(A, V, cfgs[red])
                I2d = res["I"].reshape(GRID_SHAPE)
                rec = out[(kname, red)]
                rec["dr"][t] = delta_r_bar(I2d, true_cells)
                rec["psep"][t] = p_sep(I2d, true_cells)
                comp = P.matched_error_components(I2d, true_cells)
                rec["miss"][t] = comp["miss"]
                rec["npeaks"][t] = comp["n_matched"]
                rec["niter"][t] = res["n_iter"]
                rec["stop"][t] = res["stop_reason"]
    return out


def summarize_b4(cell, arrays):
    n = len(arrays[("real", True)]["dr"])
    idx = P.boot_indices(n, P.boot_rng(cell.cell_id), N_BOOT)
    arms = {}
    for (k, red), rec in arrays.items():
        reasons = rec["stop"].ravel()
        arms[f"{k}_{'on' if red else 'off'}"] = dict(
            dr=P.mean_ci(rec["dr"], idx),
            psep=P.mean_ci(rec["psep"].astype(float), idx),
            miss_rate=P.mean_ci(rec["miss"].astype(float), idx),
            niter_mean=float(np.mean(rec["niter"])),
            stop_reason_counts={r: int(np.sum(reasons == r))
                                for r in ("cost_increase", "relative_tolerance",
                                          "active_set_floor", "max_iter")},
            n_trials=int(n))
    contrast = {}
    for red in (True, False):
        base = arrays[("real", red)]
        for (k, r2), rec in arrays.items():
            if r2 != red or k == "real":
                continue
            contrast[f"{k}_minus_real_{'on' if red else 'off'}"] = dict(
                psep=P.mean_ci(rec["psep"].astype(float) - base["psep"].astype(float), idx),
                dr=P.mean_ci(rec["dr"] - base["dr"], idx),
                miss=P.mean_ci(rec["miss"].astype(float) - base["miss"].astype(float), idx))
    return dict(cell_id=cell.cell_id, family="B4", role="descriptive_b4",
                separation_km=round(cell.d * POLE_SPACING_KM, 2),
                arms=arms, paired_contrast=contrast)


def d_min_table(psep_by_cell):
    """psep_by_cell[(snr, kernel, red)][d] -> P_sep. d_min = smallest d with
    P_sep >= 0.8, +inf if none (read literally, SPEC S.14 P3)."""
    out = {}
    for key, by_d in psep_by_cell.items():
        dmin = P.smallest_d(by_d, d_axis=tuple(sorted(by_d)), threshold=PSEP_THRESHOLD)
        out["|".join(str(x) for x in key)] = "inf" if math.isinf(dmin) else dmin
    return out


# ------------------------------------------------------------- B6b ---------

def b6b_columns(variant, idx_a, idx_b, n_snap, n_lon, rng):
    """(2, n_snap) column schedule. Source 1 is fixed; source 2 per SPEC S.15
    item 9. Returns (cols, end_col, mean_col) where end_col is source 2's
    displaced end-point column and mean_col its time-averaged column index
    (in column units, possibly fractional)."""
    c1 = np.full(n_snap, idx_a)
    half = n_snap // 2
    if variant == "none":
        c2 = np.full(n_snap, idx_b)
        end_col = idx_b
        mean_shift = 0.0
    elif variant in ("step_away", "step_toward"):
        sgn = 1 if variant == "step_away" else -1
        c2 = np.concatenate([np.full(half, idx_b), np.full(n_snap - half, idx_b + sgn)])
        end_col = idx_b + sgn
        mean_shift = sgn * (n_snap - half) / n_snap
    elif variant == "iid":
        drow = rng.integers(-1, 2, size=n_snap)
        dcol = rng.integers(-1, 2, size=n_snap)
        c2 = idx_b + drow * n_lon + dcol
        end_col = idx_b
        mean_shift = 0.0
    else:
        raise ValueError(f"unknown B6b variant {variant!r}")
    return np.stack([c1, c2]), end_col, mean_shift


def _alloc_b6b(n):
    rec = {}
    for m in METHODS:
        for tag in ("dr", "dr_end", "dr_avg"):
            rec[f"{tag}_{m}"] = np.empty(n)
        for tag in ("psep", "psep_end", "psep_avg", "miss"):
            rec[f"{tag}_{m}"] = np.empty(n, dtype=bool)
    return rec


def run_b6b_cell(tm, cell, n_trials, seed, solve_fn=solve_all):
    idx_a, idx_b = row_center_cols(cell.d)
    n_lon = GRID_SHAPE[1]
    row, col_a = divmod(idx_a, n_lon)
    _, col_b = divmod(idx_b, n_lon)
    rho = B6B_ABS_RHO * np.exp(1j * np.deg2rad(B6B_PHI_DEG))
    cfg = with_reduction(FAIRNESS_CONFIG, True)
    rec = _alloc_b6b(n_trials)
    for t in range(n_trials):
        rng = P.trial_rng(seed, cell.data_key, t)
        # jitter positions come from their own stream so the shared draws are
        # identical across variants
        jrng = np.random.default_rng([int(seed), zlib.crc32(b"b6b-iid"), int(t)])
        cols, end_col, mean_shift = b6b_columns(cell.variant, idx_a, idx_b, N_SNAP, n_lon, jrng)
        X, _ = simulate_snapshots_general(tm.A, cols, [1.0, 1.0], rng, N_SNAP,
                                          cell.snr_db, coherence=rho)
        lam, U = eigendecompose(build_csm(X))
        V = eigenmodes(lam, U, K_ORACLE)
        res = solve_fn(tm.A, V, cfg)
        end_row_col = divmod(end_col, n_lon)
        truths = {
            "": [(row, col_a), (row, col_b)],                           # nominal (primary)
            "_end": [(row, col_a), end_row_col],                         # displaced end point
            "_avg": [(row, col_a), (row, col_b + mean_shift)],           # time-averaged
        }
        for m in METHODS:
            I2d = res[m]["I"].reshape(GRID_SHAPE)
            for suffix, tc in truths.items():
                rec[f"dr{suffix}_{m}"][t] = delta_r_bar(I2d, tc)
                rec[f"psep{suffix}_{m}"][t] = p_sep(I2d, tc)
            rec[f"miss_{m}"][t] = P.matched_error_components(I2d, truths[""])["miss"]
    return rec


def summarize_b6b(cell, rec):
    n = len(rec["dr_gibf"])
    idx = P.boot_indices(n, P.boot_rng(cell.cell_id), N_BOOT)
    out = dict(cell_id=cell.cell_id, family="B6b", role="descriptive_b6b",
               variant=cell.variant, n_trials=int(n), arms={})
    for m in METHODS:
        out["arms"][m] = {
            "dr": P.mean_ci(rec[f"dr_{m}"], idx),
            "psep": P.mean_ci(rec[f"psep_{m}"].astype(float), idx),
            "miss_rate": P.mean_ci(rec[f"miss_{m}"].astype(float), idx),
            "dr_endpoint_ref": P.mean_ci(rec[f"dr_end_{m}"], idx),
            "psep_endpoint_ref": P.mean_ci(rec[f"psep_end_{m}"].astype(float), idx),
            "dr_time_avg_ref": P.mean_ci(rec[f"dr_avg_{m}"], idx),
            "psep_time_avg_ref": P.mean_ci(rec[f"psep_avg_{m}"].astype(float), idx),
        }
    out["gap_gibf_minus_mmv"] = P.mean_ci(rec["dr_gibf"] - rec["dr_mmv"], idx)
    return out


def b6b_contrast_vs_none(summaries, records):
    """Per variant: change in each method's delta r-bar and the GIBF-MMV gap
    relative to the `none` control, on the shared draws."""
    base = records["none"]
    out = {}
    for v, rec in records.items():
        if v == "none":
            continue
        n = len(rec["dr_gibf"])
        idx = P.boot_indices(n, P.boot_rng(f"B6b-contrast|{v}"), N_BOOT)
        out[v] = {m: P.mean_ci(rec[f"dr_{m}"] - base[f"dr_{m}"], idx) for m in METHODS}
        out[v]["gap_gibf_minus_mmv"] = P.mean_ci(
            (rec["dr_gibf"] - rec["dr_mmv"]) - (base["dr_gibf"] - base["dr_mmv"]), idx)
    return out


# --------------------------------------------------------- run control -----

def _git(*args, cwd=REPO):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=cwd).stdout.strip()
    except Exception:
        return "unknown"


def script_is_committed_clean():
    rel = Path(__file__).resolve().relative_to(REPO).as_posix()
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO)
    tracked = _git("ls-files", "--error-unmatch", rel)
    return r.returncode == 0 and bool(tracked)


def _is_under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smoke", action="store_true",
                   help="smoke test: SMOKE_SEED, never under results/, any n, --only allowed")
    p.add_argument("--only", default=None, help="comma-separated roles (smoke only): b4,b6b")
    p.add_argument("--workers", type=int, default=1,
                   help="worker processes (cells are independent and deterministic given\n                        the seed, so results do not depend on this)")
    return p.parse_args(argv)


def resolve_run(args, clean=None):
    """Every write/seed guard, evaluated before any computation. Returns
    dict(seed, master_seed_run, roles) or exits."""
    out_dir = Path(args.out_dir)
    if (out_dir / "manifest.json").exists():
        sys.exit(f"{out_dir} already holds a manifest.json; refusing to overwrite a run")
    if args.smoke:
        if _is_under(out_dir, RESULTS_ROOT):
            sys.exit("--smoke refuses to write under results/; pass --out-dir elsewhere")
        roles = tuple(args.only.split(",")) if args.only else ALL_ROLES
        return dict(seed=SMOKE_SEED, master_seed_run=False, roles=roles)
    if args.only is not None:
        sys.exit("--only is a smoke-test option")
    if args.n_trials != N_TRIALS:
        sys.exit(f"a master-seed run uses n_trials={N_TRIALS}; use --smoke for other n")
    if not _is_under(out_dir, RESULTS_ROOT):
        sys.exit("a master-seed run must write under results/ (use --smoke for scratch runs)")
    if clean is None:
        clean = script_is_committed_clean()
    if not clean:
        sys.exit("A3 workflow: commit this runner before a master-seed run "
                 "(working copy differs from HEAD or is untracked)")
    return dict(seed=B4B6B_SEED, master_seed_run=True, roles=ALL_ROLES)


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


def _save(out_dir, cell, arrays_flat, summ):
    np.savez_compressed(out_dir / f"{cell.file_stem}.npz", **arrays_flat)
    (out_dir / f"{cell.file_stem}.json").write_text(
        json.dumps(summ, indent=2, default=_json_default))


_STATE = {}


def _cell_job(cell, n_trials, seed):
    """One cell, start to finish. Top-level so worker processes can run it."""
    if "tm" not in _STATE:
        _STATE["tm"] = build_array_and_grid(row_normalisation=P.ROW_NORMALISATION)
    tm = _STATE["tm"]
    t0 = time.time()
    if cell.family == "B4":
        if "kernels" not in _STATE:
            _STATE["kernels"] = build_kernels(tm)
        arrays = run_b4_cell(tm, _STATE["kernels"], cell, n_trials, seed)
        summ = summarize_b4(cell, arrays)
        flat = {f"{k}_{'on' if red else 'off'}__{f}": a
                for (k, red), rec in arrays.items() for f, a in rec.items()}
        psep = {(cell.snr_db, k, "on" if red else "off", cell.d): float(np.mean(rec["psep"]))
                for (k, red), rec in arrays.items()}
        return dict(cell=cell, summ=summ, flat=flat, psep=psep, rec=None,
                    secs=time.time() - t0)
    rec = run_b6b_cell(tm, cell, n_trials, seed)
    return dict(cell=cell, summ=summarize_b6b(cell, rec), flat=dict(rec), psep={},
                rec=rec, secs=time.time() - t0)


def main(argv=None):
    args = parse_args(argv)
    run = resolve_run(args)
    seed = run["seed"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = [c for c in build_plan() if c.role in run["roles"]]

    # One BLAS thread per worker: the parallelism is across cells.
    if args.workers > 1:
        for var in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ[var] = "1"

    t_start = time.time()
    results = []
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_cell_job, c, args.n_trials, seed): c for c in plan}
            for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                r = fut.result()
                _save(out_dir, r["cell"], r["flat"], r["summ"])
                print(f"[{done}/{len(plan)}] {r['cell'].cell_id} done in {r['secs']:.1f}s",
                      flush=True)
                results.append(r)
    else:
        for i, c in enumerate(plan, 1):
            r = _cell_job(c, args.n_trials, seed)
            _save(out_dir, r["cell"], r["flat"], r["summ"])
            print(f"[{i}/{len(plan)}] {c.cell_id} done in {r['secs']:.1f}s", flush=True)
            results.append(r)

    order = {c.cell_id: i for i, c in enumerate(plan)}
    results.sort(key=lambda r: order[r["cell"].cell_id])
    cell_summaries = {r["cell"].cell_id: r["summ"] for r in results}
    timings = {r["cell"].cell_id: round(r["secs"], 1) for r in results}

    psep_by = {}
    for r in results:
        for (snr, k, red, d), v in r["psep"].items():
            psep_by.setdefault((snr, k, red), {})[d] = v
    b6b_records = {r["cell"].variant: r["rec"] for r in results if r["rec"] is not None}

    csv_rows = []
    for r in results:
        c, summ = r["cell"], r["summ"]
        if c.family == "B4":
            for arm, s_ in summ["arms"].items():
                csv_rows.append(dict(cell_id=c.cell_id, family="B4", arm=arm,
                                     dr_mean=s_["dr"][0], psep_mean=s_["psep"][0],
                                     miss_rate=s_["miss_rate"][0], niter_mean=s_["niter_mean"]))
        else:
            for m, s_ in summ["arms"].items():
                csv_rows.append(dict(cell_id=c.cell_id, family="B6b", arm=m,
                                     dr_mean=s_["dr"][0], psep_mean=s_["psep"][0],
                                     miss_rate=s_["miss_rate"][0], niter_mean=""))

    extras = {}
    complete = {k: v for k, v in psep_by.items() if len(v) == len(B4_D)}
    if complete:
        extras["d_min_psep_ge_0.8"] = d_min_table(complete)
    if "none" in b6b_records:
        extras["b6b_contrast_vs_none"] = b6b_contrast_vs_none(cell_summaries, b6b_records)

    if csv_rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    lit_lam = kernel_wavelengths_km()["lit"]
    manifest = dict(
        experiment="B4 phantom-phase ablation and B6b support jitter (descriptive; SPEC S.15)",
        master_seed_run=bool(run["master_seed_run"]), smoke=bool(args.smoke),
        master_seed=B4B6B_SEED, master_seed_provenance=B4B6B_SEED_PROVENANCE,
        seed_used=seed, workers=int(args.workers),
        git_commit=_git("rev-parse", "HEAD"),
        script_committed_clean=script_is_committed_clean(),
        script_sha256=hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        script_git_blob=_git("rev-parse", f"HEAD:{Path(__file__).resolve().relative_to(REPO).as_posix()}"),
        secsy_sha=_git("rev-parse", "HEAD", cwd=REPO / "secsy"),
        numpy=np.__version__, python=sys.version.split()[0],
        n_trials=args.n_trials, n_snap=N_SNAP, k_oracle=K_ORACLE,
        fairness_config=dataclasses.asdict(FAIRNESS_CONFIG),
        b4=dict(snr_db=list(B4_SNR_DB), d=list(B4_D),
                kernels={k: v for k, v in kernel_wavelengths_km().items()},
                wavelength_unit="km", aperture_km=APERTURE_KM,
                literal_frequency_mhz=LITERAL_FREQUENCY_MHZ,
                wavelengths_across_aperture=dict(
                    {"lit": APERTURE_KM / lit_lam},
                    **{f"w{w:g}": w for w in B4_WAVELENGTHS_ACROSS}),
                pole_spacing_km=POLE_SPACING_KM),
        b6b=dict(phi_deg=B6B_PHI_DEG, abs_rho=B6B_ABS_RHO, snr_db=B6B_SNR_DB, d=B6B_D,
                 variants=list(B6B_VARIANTS)),
        geometry=geometry_record(),
        expectations_stated_before_data=EXPECTATIONS,
        adjudication="none: both experiments are descriptive and never enter D2",
        cells=cell_summaries, extras=extras, timing_s=timings,
        runtime_s=round(time.time() - t_start, 1),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    print(f"Wrote {len(plan)} cells + manifest to {out_dir} in {manifest['runtime_s']}s")
    return manifest


if __name__ == "__main__":
    main()
