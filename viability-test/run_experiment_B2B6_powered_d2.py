"""run_experiment_B2B6_powered_d2.py — the POWERED confirmatory B2/B6a runner at
the accepted d=2 coordinate (127 km).

This script produces the pre-registered confirmatory result of the methods
paper. It is committed BEFORE its first real run (A3 workflow, repo CLAUDE.md);
the manifest records `script_sha256` and refuses a confirmatory run from a
dirty working copy of this file.

Pins implemented here (every value is cited; none is chosen in this file):

  * Confirmatory coordinate d = 2 (127 km) — SPEC S.12 (2026-08-28). SPEC S.3's
    table still reads d=3; superseded by S.11 (d=8), then by S.12 (d=2).
  * B2 WIN cells: phi = 90 deg x snr_db in {5, 10} — S.3 / ROADMAP S8-iii.
    NULL cells: phi in {0, 180} deg x the same SNR pair — S.3 / S8-vi.
    Coherent pair, |rho| = 0.85 (SLOT-1 midpoint, S.7), n_snap = 64, oracle
    K = 2, grid reduction ON (S.6(2)).
  * Reduction-OFF pair of every confirmatory cell at the same n_trials,
    DESCRIPTIVE only — S.6(2).
  * n_trials = 1635 — S.12 / ROADMAP S8 2026-08-28 power calc (provenance
    copied into the manifest from results/B_viability/confirmatory_d2_manifest
    .json). CLI-overridable for smoke tests only; any n != 1635 is stamped
    non-confirmatory and cannot be written under results/.
  * Win/loss rule second arm: "smallest d with P_sep >= 0.8 is >= 1 grid cell
    smaller along the full d axis {1,2,3,5,8}" — S.3 / ROADMAP S5-D2 / S8-iii.
    The loss rule uses the "same d reading" (S8-vi, S.3 loss row), so the d
    axis is run at phi = 90 AND at both nulls, x both SNRs, full n_trials,
    reduction ON. The d = 2 point of each axis IS the confirmatory cell
    (same data-cell id, same seeds) — not a duplicate.
  * S.6(5) regularization-sensitivity panel at every confirmatory cell
    (2 win + 4 null + B6a): both sparse solvers rerun at eps_frac x {0.1, 0.3,
    1, 3, 10} on the SAME trials (data generated once per trial, fed to every
    arm); x1 IS the headline arm (not recomputed). Fragility rule exactly as
    S.6(5): fragile iff some band point's mean gap is opposite-signed to the x1
    headline AND that point's 95% CI excludes zero. Panel CIs seeded from
    (cell id, band index).
  * B6a (S.2-B6, S.3 B6 row, S8-vii): phi = 90, snr = 5 dB, d = 2, n_snap =
    the largest of {8,16,32,64,128} with B3's MDL K-hat error rate >= 20% at
    5 dB (read from results/B_viability/b3_results.json -> 64; recorded in the
    manifest with the raw B3 row). K-hat = estimate_n_sources(lam, N, "mdl")
    fed identically to L2/GIBF/MMV, with the paired oracle-K = 2 arm on the SAME
    trials; per-trial paired differences and the K-hat <2 / =2 / >2 breakdown
    recorded. B6a's n_snap is a CLI parameter (default 64 = the pin); any other
    value stamps B6a DESCRIPTIVE. See SPEC S.14 item P1: at n_snap = 64 < 75
    channels the sample CSM is rank-deficient and MDL K-hat clips at K_MAX = 6
    -- flagged for Strider's ruling, NOT altered here.
  * Metrics per method per cell: Delta_r_bar and P_sep with the S8-v
    tolerance (1 grid cell, min-cost matched; metrics.py), signed GIBF-MMV
    gap, percentile-bootstrap 95% CIs, 2000 resamples, seeded from the cell id
    (S.1).
  * Logging: n_iter per solve (S.13 follow-up) and realized eps per solve
    (S.6(1)) into the per-cell .npz; full manifest per S.5 (config hash, git
    commit, secsy SHA, master seed, n_trials + power-calc provenance,
    geometry_record per S.11's last paragraph).
  * Adjudication (WIN / LOSS / B6 rules) is implemented in `adjudicate()` and
    reachable via `--adjudicate-only DIR`. `main()` NEVER calls it: the real
    verdict is applied as a separate, deliberate step after review.

Every decision the SPEC does not pin is listed in SPEC S.14 (PROPOSED — pending
Strider's ratification) and cross-referenced in comments as "S.14 Pn".

MASTER SEED: generated with Python `secrets` in a separate process that had no
access to any result (see MASTER_SEED_PROVENANCE), committed before any
execution of this script, smoke tests included. Smoke tests use SMOKE_SEED so
no confirmatory trial is ever computed before the real run.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from metrics import delta_r_bar, find_peaks_2d, p_sep
from modeselect import K_MAX, estimate_n_sources
from simulate import (GRID_SHAPE, POLE_SPACING_KM, build_array_and_grid,
                      build_csm, eigendecompose, eigenmodes, geometry_record,
                      row_center_cols, simulate_snapshots)
from solvers import FAIRNESS_CONFIG, solve_all, with_reduction

# ----------------------------------------------------------------- seeds ---

MASTER_SEED = 2250821694     # PINNED. Never regenerate, never change after any run.
MASTER_SEED_PROVENANCE = (
    "secrets.randbits(32) in a fresh isolated interpreter (python -I, cwd = a "
    "scratch directory, no repo import, no result file opened), drawn "
    "2026-09-23T07:38:11Z, committed in this runner before any execution of "
    "it (smoke tests use SMOKE_SEED, not this). Distinct from every prior "
    "seed in the repo (20260706, 20260712, 20260721, 20260727, 20260801, "
    "20260828).")
# Smoke-test seed: deliberately NOT the master seed, so a smoke run never
# computes any trial of the confirmatory run (S.14 P11).
SMOKE_SEED = 1

# ---------------------------------------------------------------- pins -----

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS_ROOT = REPO / "results"
DEFAULT_OUT = RESULTS_ROOT / "B_viability" / "powered_d2"
ROW_NORMALISATION = "none"          # identical to every prior B runner

D_CONFIRM = 2                       # SPEC S.12 — accepted 2026-08-16. DO NOT CHANGE.
D_AXIS = (1, 2, 3, 5, 8)            # SPEC S.2-B1 / S.10; the win rule's d axis (S8-iii)
PHI_WIN_DEG = 90.0                  # S8-iii
PHI_NULL_DEGS = (0.0, 180.0)        # S8-vi
SNR_PAIR = (5.0, 10.0)              # S8-iii
N_SNAP = 64                         # S8-iii
ABS_RHO = 0.85                      # SLOT-1 midpoint, SPEC S.7
K_ORACLE = 2

N_TRIALS_POWERED = 1635             # SPEC S.12 / ROADMAP S8 2026-08-28 power calc
EPS_BAND = (0.1, 0.3, 1.0, 3.0, 10.0)   # SPEC S.6(5); x1 is the headline
HEADLINE_BAND_INDEX = EPS_BAND.index(1.0)
N_BOOT = 2000                       # SPEC S.1 CI convention
CI_LEVEL = 0.95

B6A_SNR_DB = 5.0                    # S8-vii
B6A_NSNAP_PINNED = 64               # S8-vii procedure applied to B3 (verified at runtime)
B3_NSNAP_AXIS = (8, 16, 32, 64, 128)
B6A_ERR_THRESHOLD = 0.20            # S8-vii
B6A_NSNAP_FALLBACK = 8              # S8-vii

WIN_REL_THRESHOLD = 0.20            # "Delta_r_bar >= 20% lower" (S5-D2)
PSEP_THRESHOLD = 0.8                # "smallest d with P_sep >= 0.8" (S8-iii)
D_CELL_MARGIN = 1                   # ">= 1 grid cell smaller"

METHODS = ("l2", "gibf", "mmv")
SPARSE = ("gibf", "mmv")

B3_RESULTS = RESULTS_ROOT / "B_viability" / "b3_results.json"
B3_MANIFEST = RESULTS_ROOT / "B_viability" / "manifest.json"
POWER_MANIFEST = RESULTS_ROOT / "B_viability" / "confirmatory_d2_manifest.json"

# Hard-coded copy of the power calc so the provenance survives even if the
# JSON moves; the runtime copy is cross-checked against it.
POWER_CALC_EXPECTED = dict(n_trials=1635, alpha_ci=0.006, power_target=0.80,
                           variance_input=8.306556854463883,
                           effect_size_cells=0.7375417527999328,
                           source_commit="a42a873c61ba598d86f713a258a60cd6449c7c28",
                           source_script="viability-test/run_experiment_B2B6_d2.py",
                           source_seed=20260828)


# ------------------------------------------------------------ plan types ---

@dataclasses.dataclass(frozen=True)
class Arm:
    """One solver configuration applied to a data cell's trials."""
    k_source: str          # "oracle" (K=2) | "mdl" (K-hat, B6a)
    reduction: bool
    eps_mult: float = 1.0

    @property
    def key(self):
        return (f"{self.k_source}_{'on' if self.reduction else 'off'}"
                f"_x{self.eps_mult:g}")


@dataclasses.dataclass(frozen=True)
class DataCell:
    """A physical Monte-Carlo cell: one (family, phi, snr, d, n_snap). The data
    (X, S, lam, U, K-hat) depend only on this and the trial index; every arm is
    fed from the same per-trial computation."""
    family: str            # "B2" | "B6a"
    phi_deg: float
    snr_db: float
    d: int
    n_snap: int
    role: str              # "win" | "null" | "d_axis_win" | "d_axis_null" | "b6a"
    arms: tuple

    @property
    def cell_id(self):
        # Seed-bearing identity. Excludes reduction and eps (they are solver
        # arms, not data), so ON/OFF pairs and panel points share trials.
        return (f"{self.family}|phi={self.phi_deg:g}|snr={self.snr_db:g}"
                f"|d={self.d}|nsnap={self.n_snap}")

    @property
    def file_stem(self):
        return (f"{self.family}_phi{self.phi_deg:g}_snr{self.snr_db:g}"
                f"_d{self.d}_n{self.n_snap}")

    @property
    def confirmatory(self):
        return self.role in ("win", "null", "b6a")

    @property
    def panel_arms(self):
        """The S.6(5) band arms in band order, or () if not a panel cell."""
        k = "mdl" if self.family == "B6a" else "oracle"
        wanted = [Arm(k, True, m) for m in EPS_BAND]
        return tuple(wanted) if all(a in self.arms for a in wanted) else ()


def headline_arm(cell):
    """The adjudicating arm of a cell: reduction ON, x1 eps, oracle K for B2,
    MDL K-hat for B6a."""
    return Arm("mdl" if cell.family == "B6a" else "oracle", True, 1.0)


def arm_use(cell, arm):
    """What an arm is for (recorded per arm in every output):
      headline                   — the adjudicating arm (d=2 confirmatory
                                   cells, the d-axis points feeding the
                                   second rule arm, and B6a's K-hat arm)
      sensitivity_panel          — S.6(5) band point != x1 (fragility input)
      descriptive_reduction_off  — S.6(2) OFF pair, never adjudicated
      b6a_oracle_reference       — S.2-B6 paired oracle-K reference arm"""
    if arm == headline_arm(cell):
        return "headline"
    if not arm.reduction:
        return "descriptive_reduction_off"
    if arm.eps_mult != 1.0:
        return "sensitivity_panel"
    return "b6a_oracle_reference"


def build_plan(b6a_nsnap=B6A_NSNAP_PINNED):
    """The full cell/arm plan. Order is fixed so the run is reproducible."""
    cells = []
    for phi in (PHI_WIN_DEG,) + PHI_NULL_DEGS:
        is_win = phi == PHI_WIN_DEG
        for snr in SNR_PAIR:
            for d in D_AXIS:
                if d == D_CONFIRM:
                    arms = tuple(Arm("oracle", True, m) for m in EPS_BAND) + (
                        Arm("oracle", False, 1.0),)          # S.6(2) OFF pair
                    role = "win" if is_win else "null"
                else:
                    arms = (Arm("oracle", True, 1.0),)
                    role = "d_axis_win" if is_win else "d_axis_null"
                cells.append(DataCell("B2", phi, snr, d, N_SNAP, role, arms))
    b6_arms = tuple(Arm("mdl", True, m) for m in EPS_BAND) + (
        Arm("oracle", True, 1.0),                   # paired oracle reference (S.2-B6)
        Arm("mdl", False, 1.0), Arm("oracle", False, 1.0))   # S.6(2) OFF pair
    cells.append(DataCell("B6a", PHI_WIN_DEG, B6A_SNR_DB, D_CONFIRM, b6a_nsnap,
                          "b6a", b6_arms))
    return cells


# ----------------------------------------------------------- seeding -------

def _seed_component(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    return zlib.crc32(repr(x).encode())


def trial_rng(seed, cell_id, trial):
    """S.1 scheme: Generator seeded from (master_seed, cell_id, trial)."""
    return np.random.default_rng([int(seed), _seed_component(cell_id), int(trial)])


def boot_rng(cell_id, band_index=None):
    """S.1: bootstrap 'seeded from the cell id'; S.6(5): panel CIs 'seeded from
    (cell id, band index)'. Master seed deliberately NOT included (literal
    reading; S.14 P8)."""
    comps = [_seed_component(cell_id)]
    if band_index is not None:
        comps.append(int(band_index))
    return np.random.default_rng(comps)


def boot_indices(n, rng, n_boot=N_BOOT):
    return rng.integers(0, n, size=(n_boot, n))


# --------------------------------------------------------- data + solve ----

def generate_trial(tm, cell, seed, trial):
    """Everything that does not depend on the solver: X, S, eigendecomposition,
    MDL/AIC K-hat. Called ONCE per (cell, trial); every arm consumes it."""
    idx_a, idx_b = row_center_cols(cell.d)
    rho = ABS_RHO * np.exp(1j * np.deg2rad(cell.phi_deg))
    rng = trial_rng(seed, cell.cell_id, trial)
    X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                              n_snap=cell.n_snap, snr_db=cell.snr_db,
                              coherence=rho)
    S = build_csm(X)
    lam, U = eigendecompose(S)
    k_mdl, _, mdl_arr, clipped = estimate_n_sources(lam, cell.n_snap, "mdl")
    k_aic, _, _, _ = estimate_n_sources(lam, cell.n_snap, "aic")   # sensitivity record only (S8-viii)
    V = {"oracle": eigenmodes(lam, U, K_ORACLE), "mdl": eigenmodes(lam, U, k_mdl)}
    return dict(X=X, lam=lam, V=V, khat_mdl=int(k_mdl),
                khat_mdl_raw_argmin=int(np.argmin(mdl_arr)),
                khat_mdl_clipped=bool(clipped), khat_aic=int(k_aic),
                true_cells=[np.unravel_index(idx_a, GRID_SHAPE),
                            np.unravel_index(idx_b, GRID_SHAPE)])


def arm_config(arm):
    """ONE config object per arm, handed to solve_all so GIBF and MMV read the
    identical object (S.6(1)); only grid_reduction / eps_frac differ from
    FAIRNESS_CONFIG, and they differ identically for both solvers."""
    cfg = with_reduction(FAIRNESS_CONFIG, arm.reduction)
    if arm.eps_mult != 1.0:
        cfg = dataclasses.replace(cfg, eps_frac=FAIRNESS_CONFIG.eps_frac * arm.eps_mult)
    return cfg


def _pad(values, fill):
    out = np.full(K_MAX, fill, dtype=float)
    out[:len(values)] = values
    return out


def run_data_cell(tm, cell, n_trials, seed, solve_fn=solve_all, on_trial=None):
    """Run every arm of one data cell over n_trials. Returns (arrays, timing).

    arrays[arm.key][field] -> np.ndarray over trials, plus arrays["_data"] for
    the solver-independent per-trial record (K-hat etc.)."""
    arms = cell.arms
    data = {k: np.empty(n_trials, dtype=int) for k in
            ("khat_mdl", "khat_mdl_raw_argmin", "khat_aic")}
    data["khat_mdl_clipped"] = np.empty(n_trials, dtype=bool)
    data["lam_tracenorm_top8"] = np.empty((n_trials, 8))
    out = {}
    for a in arms:
        rec = {}
        for m in METHODS:
            rec[f"dr_{m}"] = np.empty(n_trials)
            rec[f"psep_{m}"] = np.empty(n_trials, dtype=bool)
            rec[f"npeaks_{m}"] = np.empty(n_trials, dtype=int)
        rec["k_used"] = np.empty(n_trials, dtype=int)
        rec["scale_s"] = np.empty(n_trials)
        rec["eps_l2"] = np.empty(n_trials)
        rec["eps_gibf"] = np.empty((n_trials, K_MAX))      # per mode, NaN-padded
        rec["niter_gibf"] = np.empty((n_trials, K_MAX))    # per mode, -1-padded
        rec["eps_mmv"] = np.empty(n_trials)
        rec["niter_mmv"] = np.empty(n_trials, dtype=int)
        out[a.key] = rec
    cfgs = {a.key: arm_config(a) for a in arms}
    timing = {a.key: 0.0 for a in arms}
    timing["_data"] = 0.0

    for t in range(n_trials):
        t0 = time.perf_counter()
        tr = generate_trial(tm, cell, seed, t)
        timing["_data"] += time.perf_counter() - t0
        for k in ("khat_mdl", "khat_mdl_raw_argmin", "khat_aic", "khat_mdl_clipped"):
            data[k][t] = tr[k]
        lam = tr["lam"]
        data["lam_tracenorm_top8"][t] = (lam * 2.0 / lam.sum())[:8]
        for a in arms:
            t1 = time.perf_counter()
            V = tr["V"][a.k_source]
            res = solve_fn(tm.A, V, cfgs[a.key])
            rec = out[a.key]
            for m in METHODS:
                I2d = res[m]["I"].reshape(GRID_SHAPE)
                rec[f"dr_{m}"][t] = delta_r_bar(I2d, tr["true_cells"])
                rec[f"psep_{m}"][t] = p_sep(I2d, tr["true_cells"])
                rec[f"npeaks_{m}"][t] = len(find_peaks_2d(I2d))
            rec["k_used"][t] = V.shape[1]
            rec["scale_s"][t] = res.get("scale_s", np.nan)
            rec["eps_l2"][t] = res["l2"]["realized_eps"]
            rec["eps_gibf"][t] = _pad(res["gibf"]["realized_eps"], np.nan)
            rec["niter_gibf"][t] = _pad(res["gibf"]["n_iter"], -1)
            rec["eps_mmv"][t] = res["mmv"]["realized_eps"]
            rec["niter_mmv"][t] = res["mmv"]["n_iter"]
            timing[a.key] += time.perf_counter() - t1
        if on_trial is not None:
            on_trial(t, tr)
    out["_data"] = data
    return out, timing


# ------------------------------------------------------------ statistics ---

def percentile_ci(boot_stats, level=CI_LEVEL, method="linear"):
    lo = (1.0 - level) / 2.0 * 100.0
    hi = 100.0 - lo
    return (float(np.percentile(boot_stats, lo, method=method)),
            float(np.percentile(boot_stats, hi, method=method)))


def mean_ci(x, idx):
    """Mean and percentile-bootstrap CI of x under precomputed resample
    indices (shared across methods within a cell, so CIs are paired)."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), percentile_ci(x[idx].mean(axis=1))


def summarize_arm(rec, idx):
    s = {}
    for m in METHODS:
        s[f"dr_{m}"] = mean_ci(rec[f"dr_{m}"], idx)
        s[f"psep_{m}"] = mean_ci(rec[f"psep_{m}"], idx)
    s["gap_gibf_minus_mmv"] = mean_ci(rec["dr_gibf"] - rec["dr_mmv"], idx)
    s["niter_gibf_mean"] = float(np.nanmean(np.where(rec["niter_gibf"] < 0, np.nan,
                                                      rec["niter_gibf"])))
    s["niter_mmv_mean"] = float(np.mean(rec["niter_mmv"]))
    s["niter_gibf_max"] = int(np.max(rec["niter_gibf"]))
    s["niter_mmv_max"] = int(np.max(rec["niter_mmv"]))
    s["n_trials"] = int(len(rec["dr_gibf"]))
    return s


def khat_breakdown(khat, arm_rec=None, oracle_rec=None):
    """S.2-B6: K-hat < 2 / = 2 / > 2 counts, and (descriptively) the per-bin
    mean Delta_r_bar per method in the K-hat arm and the paired K-hat - oracle
    difference, where records are given."""
    khat = np.asarray(khat)
    bins = {"lt2": khat < 2, "eq2": khat == 2, "gt2": khat > 2}
    out = {"counts": {b: int(np.sum(mask)) for b, mask in bins.items()},
           "hist": {int(k): int(np.sum(khat == k)) for k in sorted(set(khat.tolist()))}}
    if arm_rec is not None:
        per = {}
        for b, mask in bins.items():
            if not np.any(mask):
                per[b] = None
                continue
            per[b] = {f"dr_{m}_mean": float(np.mean(arm_rec[f"dr_{m}"][mask])) for m in METHODS}
            if oracle_rec is not None:
                for m in METHODS:
                    per[b][f"paired_diff_dr_{m}_mean"] = float(np.mean(
                        arm_rec[f"dr_{m}"][mask] - oracle_rec[f"dr_{m}"][mask]))
        out["per_bin"] = per
    return out


def b6a_paired(khat_rec, oracle_rec, idx):
    """S.2-B6: per-trial paired difference metric(K-hat) - metric(K=2) per
    method, on the same trials. Returns per-trial arrays and mean+CI."""
    per_trial, summ = {}, {}
    for m in METHODS:
        for metric in ("dr", "psep"):
            diff = (khat_rec[f"{metric}_{m}"].astype(float)
                    - oracle_rec[f"{metric}_{m}"].astype(float))
            per_trial[f"paired_diff_{metric}_{m}"] = diff
            summ[f"paired_diff_{metric}_{m}"] = mean_ci(diff, idx)
    return per_trial, summ


# ----------------------------------------------------------- the panel ------

def sensitivity_panel(cell, arrays, n_boot=N_BOOT):
    """S.6(5): signed mean gap + 95% CI per band point (bootstrap seeded from
    (cell id, band index)), plus the fragility / attenuation read."""
    rows = []
    for b, arm in enumerate(cell.panel_arms):
        rec = arrays[arm.key]
        gap = rec["dr_gibf"] - rec["dr_mmv"]
        idx = boot_indices(len(gap), boot_rng(cell.cell_id, b), n_boot)
        mean, ci = mean_ci(gap, idx)
        rows.append(dict(band_index=b, eps_mult=arm.eps_mult,
                         eps_frac=FAIRNESS_CONFIG.eps_frac * arm.eps_mult,
                         gap_mean=mean, gap_ci=ci))
    return dict(points=rows, **fragility(rows))


def _ci_excludes_zero(ci):
    return ci[0] > 0.0 or ci[1] < 0.0


def fragility(points, headline_index=HEADLINE_BAND_INDEX):
    """SPEC S.6(5) fragility rule, verbatim: a cell's verdict is
    regularization-FRAGILE iff any band point's mean gap is opposite-signed to
    the x1 headline AND that point's 95% CI excludes zero. Loss of
    significance without reversal = 'attenuated at band edge(s)', not fragile;
    both facets always reported.

    S.14 P9 edge cases: a headline mean of exactly 0 has no sign, so no point
    can be 'opposite-signed' (not fragile; flagged). 'Attenuated' is only
    defined when the headline's own CI excludes zero (there is no
    significance to lose otherwise)."""
    head = points[headline_index]
    hs = np.sign(head["gap_mean"])
    reversals, attenuated = [], []
    for p in points:
        if p["band_index"] == headline_index:
            continue
        opposite = hs != 0 and np.sign(p["gap_mean"]) == -hs
        if opposite and _ci_excludes_zero(p["gap_ci"]):
            reversals.append(p["eps_mult"])
        elif _ci_excludes_zero(head["gap_ci"]) and not _ci_excludes_zero(p["gap_ci"]):
            attenuated.append(p["eps_mult"])
    return dict(fragile=len(reversals) > 0, significant_reversal_at=reversals,
                attenuated_at=attenuated, headline_sign_undefined=bool(hs == 0))


# ------------------------------------------------------------ adjudication -

def ci_non_overlap(ci_better, ci_worse):
    """S.1: 'non-overlapping 95% CIs' = the two per-method CIs. Strict
    inequality (S.14 P7): touching CIs overlap; a degenerate [x, x] CI is an
    ordinary interval."""
    return ci_better[1] < ci_worse[0]


def delta_r_arm(summary, favored):
    """First arm: favored method's Delta_r_bar >= 20% lower than the other's
    AND non-overlapping per-method 95% CIs (S5-D2, S.1)."""
    other = "mmv" if favored == "gibf" else "gibf"
    fm, fci = summary[f"dr_{favored}"]
    om, oci = summary[f"dr_{other}"]
    effect = fm <= (1.0 - WIN_REL_THRESHOLD) * om
    sep = ci_non_overlap(fci, oci)
    return dict(met=bool(effect and sep), effect_met=bool(effect),
                ci_non_overlap=bool(sep), favored_mean=fm, other_mean=om,
                favored_ci=fci, other_ci=oci)


def smallest_d(psep_by_d, d_axis=D_AXIS, threshold=PSEP_THRESHOLD):
    """Smallest d on the axis with P_sep >= threshold; +inf if none reaches it.
    Read literally even if P_sep is non-monotone in d (S.14 P3)."""
    for d in sorted(d_axis):
        if psep_by_d[d] >= threshold:
            return float(d)
    return math.inf


def d_axis_arm(psep_trials, favored, cell_ids, n_boot=N_BOOT):
    """Second arm (S8-iii): smallest d with P_sep >= 0.8 is >= 1 grid cell
    smaller for the favored method, with non-overlapping 95% CIs.

    S.14 P3 (proposed): the CI is a percentile bootstrap of d_min per method,
    resampling each d-cell's trials independently (each d cell is its own
    Monte-Carlo cell with its own seeds), using each cell's own cell-id-seeded
    resample indices (so both methods share them); quantiles by the
    inverted-CDF rule because d_min is discrete and may be +inf.

    psep_trials[d][method] -> bool array over trials; cell_ids[d] -> cell id."""
    other = "mmv" if favored == "gibf" else "gibf"
    dmins = {}
    boots = {}
    idx_by_d = {d: boot_indices(len(psep_trials[d][favored]),
                                boot_rng(cell_ids[d]), n_boot) for d in D_AXIS}
    for m in (favored, other):
        point = smallest_d({d: float(np.mean(psep_trials[d][m])) for d in D_AXIS})
        rates = {d: psep_trials[d][m].astype(float)[idx_by_d[d]].mean(axis=1)
                 for d in D_AXIS}
        bd = np.full(n_boot, math.inf)
        for d in sorted(D_AXIS, reverse=True):
            bd = np.where(rates[d] >= PSEP_THRESHOLD, float(d), bd)
        dmins[m] = point
        boots[m] = percentile_ci(bd, method="inverted_cdf")
    effect = dmins[favored] <= dmins[other] - D_CELL_MARGIN
    sep = ci_non_overlap(boots[favored], boots[other])
    return dict(met=bool(effect and sep), effect_met=bool(effect),
                ci_non_overlap=bool(sep), d_min=dmins, d_min_ci=boots)


def _cell_rule(summary, psep_trials, cell_ids, favored, fragile):
    arm1 = delta_r_arm(summary, favored)
    arm2 = d_axis_arm(psep_trials, favored, cell_ids)
    # S.14 P4: per SNR level, the arms are OR'd; a regularization-fragile cell
    # cannot satisfy the rule (S.6(5): claim demoted to descriptive).
    return dict(met=bool((arm1["met"] or arm2["met"]) and not fragile),
                fragile=bool(fragile), delta_r_arm=arm1, d_axis_arm=arm2)


def adjudicate(cells_view):
    """Apply the pinned WIN, LOSS and B6 rules. NOT called by main().

    cells_view: {(phi, snr): {"summary": headline arm summary at d=2,
                              "psep_trials": {d: {m: bool array}},
                              "cell_ids": {d: cell id},
                              "fragile": bool}},
                plus key "B6a": {"summary": K-hat headline summary,
                                 "fragile": bool, "confirmatory": bool}.

    WIN  (S5-D2/S8-iii): at phi=90, BOTH SNRs meet the GIBF-favored rule.
    LOSS (S8-vi): at >= 1 of phi in {0,180}, BOTH SNRs meet the MMV-favored
                  rule; the other null always reported.
    B6   (S8-vii): win-rule standard at the single B6a cell — the Delta_r_bar
                  arm only, there being no d axis at that cell (S.14 P5)."""
    out = {"win": {}, "loss": {}, "b6": None}
    win_cells = {}
    for snr in SNR_PAIR:
        v = cells_view[(PHI_WIN_DEG, snr)]
        win_cells[snr] = _cell_rule(v["summary"], v["psep_trials"], v["cell_ids"],
                                    "gibf", v["fragile"])
    out["win"] = dict(met=all(c["met"] for c in win_cells.values()), per_snr=win_cells)
    per_null = {}
    for phi in PHI_NULL_DEGS:
        per_snr = {}
        for snr in SNR_PAIR:
            v = cells_view[(phi, snr)]
            per_snr[snr] = _cell_rule(v["summary"], v["psep_trials"], v["cell_ids"],
                                      "mmv", v["fragile"])
        per_null[phi] = dict(met=all(c["met"] for c in per_snr.values()), per_snr=per_snr)
    out["loss"] = dict(met=any(p["met"] for p in per_null.values()),
                       met_at_phi=[phi for phi, p in per_null.items() if p["met"]],
                       per_null=per_null)
    b6 = cells_view.get("B6a")
    if b6 is not None:
        arm = delta_r_arm(b6["summary"], "gibf")
        out["b6"] = dict(met=bool(arm["met"] and not b6["fragile"] and b6["confirmatory"]),
                         fragile=bool(b6["fragile"]), confirmatory=bool(b6["confirmatory"]),
                         delta_r_arm=arm)
    return out


def _load_cell_npz(path):
    z = np.load(path, allow_pickle=False)
    arrays = {}
    for key in z.files:
        arm, field = key.split("__", 1)
        arrays.setdefault(arm, {})[field] = z[key]
    return arrays


def adjudicate_from_dir(out_dir):
    """Rebuild the adjudication view from a finished run's per-cell .npz files
    and apply adjudicate(). A deliberate, separate step (--adjudicate-only)."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    plan = build_plan(manifest["b6a"]["n_snap"])
    loaded = {c.cell_id: _load_cell_npz(out_dir / f"{c.file_stem}.npz") for c in plan}
    view = {}
    for phi in (PHI_WIN_DEG,) + PHI_NULL_DEGS:
        for snr in SNR_PAIR:
            axis = {c.d: c for c in plan if c.family == "B2" and c.phi_deg == phi
                    and c.snr_db == snr}
            conf = axis[D_CONFIRM]
            head = headline_arm(conf).key
            arrays = loaded[conf.cell_id]
            idx = boot_indices(len(arrays[head]["dr_gibf"]), boot_rng(conf.cell_id))
            view[(phi, snr)] = dict(
                summary=summarize_arm(arrays[head], idx),
                psep_trials={d: {m: loaded[c.cell_id][headline_arm(c).key][f"psep_{m}"]
                                 for m in SPARSE} for d, c in axis.items()},
                cell_ids={d: c.cell_id for d, c in axis.items()},
                fragile=sensitivity_panel(conf, arrays)["fragile"])
    b6 = [c for c in plan if c.family == "B6a"][0]
    arrays = loaded[b6.cell_id]
    head = headline_arm(b6).key
    idx = boot_indices(len(arrays[head]["dr_gibf"]), boot_rng(b6.cell_id))
    view["B6a"] = dict(summary=summarize_arm(arrays[head], idx),
                       fragile=sensitivity_panel(b6, arrays)["fragile"],
                       confirmatory=bool(manifest["b6a"]["confirmatory"]))
    result = adjudicate(view)
    result["confirmatory_run"] = bool(manifest.get("confirmatory_run"))
    return result


# ------------------------------------------------------------- B3 reading --

def b6a_nsnap_from_b3(b3, snr_db=B6A_SNR_DB):
    """S8-vii procedure: largest n_snap in {8,..,128} with MDL K-hat error rate
    >= 20% at 5 dB; fallback 8. b3 is the parsed b3_results.json."""
    cands = [n for n in B3_NSNAP_AXIS
             if b3[f"{snr_db}|{n}"]["mdl_error_rate"] >= B6A_ERR_THRESHOLD]
    return max(cands) if cands else B6A_NSNAP_FALLBACK


def b3_reading():
    b3 = json.loads(B3_RESULTS.read_text())
    rows = {n: {k: b3[f"{B6A_SNR_DB}|{n}"][k] for k in
                ("mdl_error_rate", "mdl_khat_hist", "aic_error_rate", "aic_khat_hist")}
            for n in B3_NSNAP_AXIS}
    b3_manifest = json.loads(B3_MANIFEST.read_text()) if B3_MANIFEST.exists() else {}
    return dict(
        source=str(B3_RESULTS.relative_to(REPO)).replace("\\", "/"),
        source_run_commit=b3_manifest.get("git_commit"),
        source_seed=b3_manifest.get("seed"),
        source_d=3, source_note=("B3 was run at d=3 (run_experiment_B.py D_B3=3, "
                                 "incoherent pair); the S8-vii procedure reads it "
                                 "as-is. See SPEC S.14 P1."),
        rows_5db=rows, procedure_result=b6a_nsnap_from_b3(b3),
        khat_clip_note=("At 5 dB, MDL K-hat = 6 (= K_MAX clip) on 50/50 trials for "
                        "n_snap in {8,16,32,64}; K-hat = 2 on 50/50 at 128. n_snap=64 "
                        "< 75 channels makes the sample CSM rank-deficient; the raw "
                        "MDL argmin falls near the rank boundary and is clipped to 6. "
                        "Flagged for Strider's ruling (SPEC S.14 P1); not altered."))


# --------------------------------------------------------------- manifest --

def _git(*args, cwd=HERE):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=cwd).stdout.strip()
    except Exception:
        return "unknown"


def script_is_committed_clean():
    rel = Path(__file__).resolve().relative_to(REPO).as_posix()
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO)
    tracked = _git("ls-files", "--error-unmatch", rel, cwd=REPO)
    return r.returncode == 0 and bool(tracked)


def power_provenance():
    rec = dict(POWER_CALC_EXPECTED)
    if POWER_MANIFEST.exists():
        m = json.loads(POWER_MANIFEST.read_text())
        pc = m.get("power_calc", {})
        rec["runtime_copy"] = pc
        rec["runtime_copy_matches"] = (pc.get("n_trials") == POWER_CALC_EXPECTED["n_trials"]
                                       and m.get("git_commit") == POWER_CALC_EXPECTED["source_commit"])
    rec["citation"] = "SPEC_experiment_B.md S.12; ROADMAP.md S8 2026-08-28 entry"
    return rec


def run_config(n_trials, b6a_nsnap, seed, smoke):
    return dict(
        d_confirm=D_CONFIRM, d_axis=list(D_AXIS), phi_win_deg=PHI_WIN_DEG,
        phi_null_degs=list(PHI_NULL_DEGS), snr_pair=list(SNR_PAIR), n_snap=N_SNAP,
        abs_rho=ABS_RHO, k_oracle=K_ORACLE, n_trials=n_trials,
        eps_band=list(EPS_BAND), n_boot=N_BOOT, ci_level=CI_LEVEL,
        b6a=dict(phi_deg=PHI_WIN_DEG, snr_db=B6A_SNR_DB, d=D_CONFIRM, n_snap=b6a_nsnap,
                 k_estimator="mdl", k_max=K_MAX),
        fairness_config=dataclasses.asdict(FAIRNESS_CONFIG),
        row_normalisation=ROW_NORMALISATION, seed=seed, smoke=smoke,
        rules=dict(win_rel_threshold=WIN_REL_THRESHOLD, psep_threshold=PSEP_THRESHOLD,
                   d_cell_margin=D_CELL_MARGIN),
        plan=[dict(cell_id=c.cell_id, role=c.role, arms=[a.key for a in c.arms])
              for c in build_plan(b6a_nsnap)],
    )


# ------------------------------------------------------------------- main --

def _is_under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-trials", type=int, default=N_TRIALS_POWERED,
                   help="trials per cell (default 1635, the S.12 power calc)")
    p.add_argument("--b6a-nsnap", type=int, default=B6A_NSNAP_PINNED,
                   help="B6a n_snap (default 64 = the S8-vii pin; other values "
                        "stamp B6a DESCRIPTIVE)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smoke", action="store_true",
                   help="smoke test: SMOKE_SEED, never under results/, prints no statistics")
    p.add_argument("--only", default=None,
                   help="comma-separated cell roles to run (smoke only), e.g. win,b6a")
    p.add_argument("--adjudicate-only", type=Path, default=None,
                   help="apply the WIN/LOSS/B6 rules to a finished run directory and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.adjudicate_only is not None:
        res = adjudicate_from_dir(args.adjudicate_only)
        (Path(args.adjudicate_only) / "adjudication.json").write_text(
            json.dumps(res, indent=2, default=_json_default))
        print(f"Wrote {Path(args.adjudicate_only) / 'adjudication.json'}")
        return res

    confirmatory = (not args.smoke and args.n_trials == N_TRIALS_POWERED
                    and args.only is None)
    if args.smoke:
        seed = SMOKE_SEED
        if _is_under(args.out_dir, RESULTS_ROOT):
            sys.exit("--smoke refuses to write under results/; pass --out-dir elsewhere")
    else:
        if MASTER_SEED is None:
            sys.exit("MASTER_SEED is not pinned; refusing to run")
        seed = MASTER_SEED
        if args.only is not None:
            sys.exit("--only is a smoke-test option")
        if _is_under(args.out_dir, RESULTS_ROOT):
            if not confirmatory:
                sys.exit("only a full n_trials=1635 run may write under results/")
            if not script_is_committed_clean():
                sys.exit("A3 workflow: commit this runner before a confirmatory run "
                         "(working copy differs from HEAD or is untracked)")

    b3 = b3_reading()
    if args.b6a_nsnap == B6A_NSNAP_PINNED and b3["procedure_result"] != B6A_NSNAP_PINNED:
        sys.exit(f"S8-vii reading of B3 gives {b3['procedure_result']}, not the "
                 f"pinned {B6A_NSNAP_PINNED}; stop and report")
    b6a_confirmatory = args.b6a_nsnap == b3["procedure_result"]

    plan = build_plan(args.b6a_nsnap)
    if args.only:
        roles = set(args.only.split(","))
        plan = [c for c in plan if c.role in roles]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = run_config(args.n_trials, args.b6a_nsnap, seed, args.smoke)
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    tm = build_array_and_grid(row_normalisation=ROW_NORMALISATION)

    t_start = time.time()
    timings, cell_summaries, csv_rows = {}, {}, []
    for ci, cell in enumerate(plan):
        tc = time.time()
        arrays, timing = run_data_cell(tm, cell, args.n_trials, seed)
        timings[cell.cell_id] = timing
        summ = summarize_cell(cell, arrays)
        cell_summaries[cell.cell_id] = summ
        save_cell(out_dir, cell, arrays, summ)
        for arm_key, s in summ["arms"].items():
            csv_rows.append(_csv_row(cell, arm_key, s))
        print(f"[{ci + 1}/{len(plan)}] {cell.cell_id} ({cell.role}, "
              f"{len(cell.arms)} arms) done in {time.time() - tc:.1f}s")

    _write_csv(out_dir / "summary.csv", csv_rows)
    manifest = dict(
        experiment="Powered B2/B6a confirmatory run at d=2 (SPEC S.12)",
        confirmatory_run=bool(confirmatory),
        smoke=bool(args.smoke),
        master_seed=MASTER_SEED, master_seed_provenance=MASTER_SEED_PROVENANCE,
        seed_used=seed,
        git_commit=_git("rev-parse", "HEAD"),
        script_committed_clean=script_is_committed_clean(),
        script_sha256=hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        secsy_sha=_git("rev-parse", "HEAD", cwd=REPO / "secsy"),
        numpy=np.__version__, python=sys.version.split()[0],
        config_hash_sha256=config_hash, config=cfg,
        n_trials=args.n_trials, n_trials_provenance=power_provenance(),
        geometry=geometry_record(),
        confirmatory_d_km=round(D_CONFIRM * POLE_SPACING_KM, 3),
        b6a=dict(n_snap=args.b6a_nsnap, confirmatory=bool(b6a_confirmatory and confirmatory),
                 b3_reading=b3),
        preregistration=("SPEC_experiment_B.md S.1, S.2-B2, S.2-B6, S.3 (as amended by "
                         "S.11/S.12), S.5, S.6, S.7, S.12, S.13; ROADMAP.md S5-D2, "
                         "S8-ii..viii; proposed S.14 items P1..P12 pending ratification"),
        adjudication="NOT applied by this run; use --adjudicate-only after review",
        cells=cell_summaries,
        timing_s=timings,
        runtime_s=round(time.time() - t_start, 1),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    print(f"Wrote {len(plan)} cells + manifest to {out_dir} in "
          f"{manifest['runtime_s']}s")
    return manifest


def summarize_cell(cell, arrays):
    n = len(arrays["_data"]["khat_mdl"])
    idx = boot_indices(n, boot_rng(cell.cell_id))   # shared by every arm (paired CIs)
    summ = dict(cell_id=cell.cell_id, role=cell.role,
                confirmatory=cell.confirmatory,
                separation_km=round(cell.d * POLE_SPACING_KM, 2),
                arms={a.key: summarize_arm(arrays[a.key], idx) for a in cell.arms},
                khat_mdl=khat_breakdown(arrays["_data"]["khat_mdl"]),
                khat_mdl_clip_events=int(np.sum(arrays["_data"]["khat_mdl_clipped"])))
    for a in cell.arms:
        summ["arms"][a.key]["use"] = arm_use(cell, a)
        summ["arms"][a.key]["descriptive"] = arm_use(cell, a) in (
            "descriptive_reduction_off", "b6a_oracle_reference")
    if cell.panel_arms:
        summ["sensitivity_panel"] = sensitivity_panel(cell, arrays)
    if cell.family == "B6a":
        kh = arrays[Arm("mdl", True, 1.0).key]
        orc = arrays[Arm("oracle", True, 1.0).key]
        per_trial, paired = b6a_paired(kh, orc, idx)
        arrays["b6a_paired"] = per_trial
        summ["b6a_paired_diff_on"] = paired
        summ["khat_breakdown_on"] = khat_breakdown(arrays["_data"]["khat_mdl"], kh, orc)
    return summ


def save_cell(out_dir, cell, arrays, summ):
    flat = {}
    for arm_key, rec in arrays.items():
        for field, arr in rec.items():
            flat[f"{arm_key}__{field}"] = np.asarray(arr)
    np.savez_compressed(out_dir / f"{cell.file_stem}.npz", **flat)
    (out_dir / f"{cell.file_stem}.json").write_text(
        json.dumps(summ, indent=2, default=_json_default))


def _csv_row(cell, arm_key, s):
    row = dict(cell_id=cell.cell_id, role=cell.role, arm=arm_key,
               descriptive=s["descriptive"], n_trials=s["n_trials"])
    for k in [f"dr_{m}" for m in METHODS] + [f"psep_{m}" for m in METHODS] + ["gap_gibf_minus_mmv"]:
        mean, (lo, hi) = s[k]
        row[f"{k}_mean"], row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = mean, lo, hi
    for k in ("niter_gibf_mean", "niter_mmv_mean", "niter_gibf_max", "niter_mmv_max"):
        row[k] = s[k]
    return row


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
