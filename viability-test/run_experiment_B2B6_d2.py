"""run_experiment_B2B6_d2.py — powered B2/B6a confirmatory pass at the
blind-drawn d=2 coordinate.

Context (do not re-derive; see SPEC_experiment_B.md S.3 amendment / S.11 and
ROADMAP.md S8, and repo-root handoff.md for the d=8 pass this one mirrors):
the original confirmatory coordinate d=3 adjudicated as a ceiling/saturation
result (2026-07-27, both GIBF and MMV at Delta_r_bar=0.000, P_sep=1.00 on
50/50 trials) -- not a comparative result, so the pre-registered win AND
loss rules were both structurally unsatisfiable there at any n_trials.
Strider ruled (2026-07-30) that a NEW confirmatory d must be pre-registered
blind before B2/B6 run, and (2026-08-01) that the draw itself be a
documented random mechanism rather than his own pick. The blind draw (SPEC
S.11, secrets.choice() over {1,2,5,8}) landed on d=8 (508 km) -- which also
came back a ceiling result, even more decisively than d=3 (SD=0.0000 exact,
50/50 trials, all three solvers exact-recover; see run_experiment_B2B6_d8.py
and its PR, still open/unreviewed as of this script's writing).

The d=8 pass's own meta-finding is why THIS script exists: reading the
already-committed 2026-07-27 B1 incoherent-sweep grid (existing data, not a
new run) showed d=5 sits at the identical ceiling at every SNR from -5 to
20 dB, and d=1 sits at the opposite floor -- so of the four candidates in
the original blind-draw pool {1,2,5,8}, only d=2 was ever genuinely
uncertain; its outcome was not already knowable from data that existed
before either draw. Strider ruled (2026-08-16, reported at the team's
restart meeting, 2026-08-28) to accept d=2 as the confirmatory coordinate
on exactly that basis -- the cheapest path forward, and the one point in
the original pool that was never already determined. d=2 is NOT a second
blind draw; it is a direct acceptance of the pool's sole remaining
candidate, and per the same discipline as d=3 and d=8, it is NOT to be
re-picked, second-guessed, or adjusted for any reason once this script
starts.

Encouraging sign, reported here for context only and NOT used to bias
anything below: the already-committed B1 incoherent cell at d=2/5dB/
reduction-ON (results/B_viability/b1_results.json, key "True|2|5.0") is
NOT degenerate the way d=3 and d=8's matched B1 cells were -- gap_sd=8.1819
(vs 0.0000/0.0707 at the other two coordinates), P_sep spread across
methods (l2=0.00, gibf=0.94, mmv=0.84 -- l2 fails to separate at all here,
unlike the other two coordinates where every method saturated). This is
consistent with d=2 being the pool's one genuinely-uncertain point, but the
mini-pilot below is run and diagnosed exactly as mechanically as the d=3
and d=8 passes were -- a promising incoherent B1 reading does not exempt
the coherent mini-pilot from its own stop rule.

phi=90 deg, snr_db in {5,10}, n_snap=64, |rho|=0.85 (SLOT-1 midpoint) are
all UNCHANGED pins, identical to the d=3 and d=8 passes -- only the
confirmatory d differs.

This script runs, IN ORDER, STOPPING EARLY if the cell proves undecidable
(see the hard stop-rule note at the bottom of this docstring):

  1. B1's matched pilot cell re-measured fresh at d=2, snr=5dB, reduction ON,
     50 trials, incoherent equal-power pair -- mirrors run_experiment_B.py's
     run_b1() inner loop exactly, at this one cell, under this script's own
     seed (full self-containment; the 2026-07-27 pass already measured this
     cell as part of the full B1 grid -- see results/B_viability/b1_results
     .json key "True|2|5.0" -- and it is reported alongside for cross-check).
  2. The 50-trial coherent mini-pilot at phi=90, snr=5dB, d=2, n_snap=64,
     |rho|=0.85, reduction ON (SPEC S.6.2 pinned confirmatory setting) --
     identical procedure to run_experiment_B.py's run_minipilot(), d=2
     instead of d=3, fresh seed.
  3. The S8-ii power calculation: n_trials such that a true 20%/1-cell
     effect clears 95% CI non-overlap at >=80% power, variance = max(the two
     SDs above). Same formula as the d=8 pass's power_n_trials() below.

STOP RULE (hard constraint, matches the d=3/d=8 precedent exactly): if
either pilot cell shows a sparse solver pinned at Delta_r_bar=0.000 on
EVERY trial (ceiling) -- or at the grid diagonal on every trial (floor) --
then the pre-registered win rule and the mirrored loss rule are BOTH
structurally unsatisfiable at this cell at ANY n_trials. Per Strider's
explicit instruction (as applied identically at d=3 and d=8), this is NOT a
bug to route around: the script reports the finding plainly (numbers,
flagged for adjudication) and DOES NOT proceed to build/run powered B2/B6a.
If d=2 is ALSO degenerate, the fallback is Strider's call (pool redesign, or
reframing the paper around the floor->ceiling transition shape itself, per
the d=3 ceiling ruling's own precedent) -- this script does not invent one.

SEED: 20260828 (fresh; distinct from every prior seed in this repo --
20260706, 20260712, 20260721, 20260727, 20260801).
"""

import hashlib
import json
import math
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import norm

from metrics import delta_r_bar, find_peaks_2d, p_sep
from simulate import (POLE_SPACING_KM, build_array_and_grid, build_csm,
                      eigendecompose, eigenmodes, geometry_record,
                      row_center_cols, simulate_snapshots, GRID_SHAPE)
from solvers import FAIRNESS_CONFIG, SolverConfig, solve_all, with_reduction

SEED = 20260828
RESULTS = Path(__file__).resolve().parent.parent / "results" / "B_viability"
ROW_NORMALISATION = "none"

D_CONFIRM = 2              # Accepted 2026-08-16 (Strider). DO NOT CHANGE.
PHI_WIN_DEG = 90.0
PHI_NULL_DEGS = (0.0, 180.0)
SNR_AXIS = (5.0, 10.0)
N_SNAP = 64
ABS_RHO = 0.85             # SLOT-1 pinned midpoint

B1_PILOT_SNR_DB = 5.0
B1_PILOT_N_TRIALS = 50     # matches run_experiment_B.py's N_TRIALS_B1

MINIPILOT_N_TRIALS = 50    # pinned, handoff.md / SPEC S.3 power row

ALPHA_CI = 0.006           # ROADMAP S5-D2/S8-ii: "non-overlapping 95% CIs ~
                           # a difference test at alpha ~ 0.006 (conservative)"
POWER_TARGET = 0.80

REG_BAND_MULT = (0.1, 0.3, 1.0, 3.0, 10.0)   # SPEC S.6.5, x1 is the headline

# Historical cross-check, results/B_viability/b1_results.json key "True|2|5.0"
# (2026-07-27 pass, full B1 grid -- quoted here, not recomputed):
HISTORICAL_B1_D2_5DB = dict(
    gap_mean=-0.9588543819998319, gap_sd=8.181897444777546,
    p_sep_rate=dict(l2=0.0, gibf=0.94, mmv=0.84),
    delta_r_bar_mean=dict(l2=18.38854381999832, gibf=1.993312629199899,
                          mmv=2.952167011199731),
)


def _seed_component(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    return zlib.crc32(repr(x).encode())


def _rng(*components):
    return np.random.default_rng([SEED, *[_seed_component(c) for c in components]])


# ------------------------------------------------------- B1 matched cell ---

def b1_pilot_cell_d2():
    """Re-measure the B1 incoherent equal-power cell at d=2, snr=5dB,
    reduction ON, 50 trials -- mirrors run_experiment_B.py's run_b1() inner
    loop exactly, restricted to this one cell, under this script's own seed
    for full self-containment. Component tag scheme mirrors the d=8 script's
    (b1_pilot_d2 in place of b1_pilot_d8)."""
    tm = build_array_and_grid(row_normalisation=ROW_NORMALISATION)
    idx_a, idx_b = row_center_cols(D_CONFIRM)
    true_cells = [np.unravel_index(idx_a, GRID_SHAPE),
                 np.unravel_index(idx_b, GRID_SHAPE)]
    cfg = with_reduction(FAIRNESS_CONFIG, True)

    dr = {"l2": [], "gibf": [], "mmv": []}
    ps = {"l2": [], "gibf": [], "mmv": []}
    for trial in range(B1_PILOT_N_TRIALS):
        rng = _rng(1, "b1_pilot_d2", trial)
        X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                                  n_snap=N_SNAP, snr_db=B1_PILOT_SNR_DB)
        S = build_csm(X)
        lam, U = eigendecompose(S)
        V = eigenmodes(lam, U, 2)
        out = solve_all(tm.A, V, cfg)
        for name in ("l2", "gibf", "mmv"):
            I2d = out[name]["I"].reshape(GRID_SHAPE)
            dr[name].append(delta_r_bar(I2d, true_cells))
            ps[name].append(p_sep(I2d, true_cells))

    gap = [g - m for g, m in zip(dr["gibf"], dr["mmv"])]
    return dict(
        d=D_CONFIRM, snr_db=B1_PILOT_SNR_DB, reduction=True,
        separation_km=round(D_CONFIRM * POLE_SPACING_KM, 2),
        n_trials=B1_PILOT_N_TRIALS,
        delta_r_bar_mean={k: float(np.mean(v)) for k, v in dr.items()},
        delta_r_bar_sd={k: float(np.std(v, ddof=1)) for k, v in dr.items()},
        p_sep_rate={k: float(np.mean(v)) for k, v in ps.items()},
        gap_signed=gap, gap_mean=float(np.mean(gap)), gap_sd=float(np.std(gap, ddof=1)),
    )


# ------------------------------------------------------------ mini-pilot ---

def run_minipilot_d2():
    """50-trial coherent mini-pilot at phi=90 deg / 5 dB / d=2, |rho|=0.85
    (SLOT-1), both sparse solvers, grid reduction ON (pinned SPEC S.6.2 --
    the confirmatory setting). Identical procedure to
    run_experiment_B.py's run_minipilot(), d=2 in place of d=3."""
    tm = build_array_and_grid(row_normalisation=ROW_NORMALISATION)
    idx_a, idx_b = row_center_cols(D_CONFIRM)
    true_cells = [np.unravel_index(idx_a, GRID_SHAPE),
                 np.unravel_index(idx_b, GRID_SHAPE)]
    rho = ABS_RHO * np.exp(1j * np.deg2rad(PHI_WIN_DEG))
    cfg = with_reduction(FAIRNESS_CONFIG, True)

    dr = {"l2": [], "gibf": [], "mmv": []}
    ps = {"l2": [], "gibf": [], "mmv": []}
    npk = {"l2": [], "gibf": [], "mmv": []}
    for trial in range(MINIPILOT_N_TRIALS):
        rng = _rng(2, "minipilot_d2", trial)
        X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                                  n_snap=N_SNAP, snr_db=B1_PILOT_SNR_DB,
                                  coherence=rho)
        S = build_csm(X)
        lam, U = eigendecompose(S)
        V = eigenmodes(lam, U, 2)
        out = solve_all(tm.A, V, cfg)
        for name in ("l2", "gibf", "mmv"):
            I2d = out[name]["I"].reshape(GRID_SHAPE)
            dr[name].append(delta_r_bar(I2d, true_cells))
            ps[name].append(p_sep(I2d, true_cells))
            npk[name].append(len(find_peaks_2d(I2d)))

    gap = [g - m for g, m in zip(dr["gibf"], dr["mmv"])]
    return dict(
        phi_deg=PHI_WIN_DEG, snr_db=B1_PILOT_SNR_DB, d=D_CONFIRM,
        separation_km=round(D_CONFIRM * POLE_SPACING_KM, 2),
        abs_rho=ABS_RHO, n_snap=N_SNAP, n_trials=MINIPILOT_N_TRIALS, reduction=True,
        true_cells=[[int(c[0]), int(c[1])] for c in true_cells],
        delta_r_bar={k: v for k, v in dr.items()},
        p_sep_rate={k: float(np.mean(v)) for k, v in ps.items()},
        delta_r_bar_mean={k: float(np.mean(v)) for k, v in dr.items()},
        delta_r_bar_sd={k: float(np.std(v, ddof=1)) for k, v in dr.items()},
        n_peaks_hist={m: {int(k): int(np.sum(np.array(v) == k))
                          for k in sorted(set(v))} for m, v in npk.items()},
        gap_signed=gap, gap_mean=float(np.mean(gap)), gap_sd=float(np.std(gap, ddof=1)),
    )


def diagnose_degeneracy(pilot):
    """Same diagnostic run_experiment_B.py's main() and the d=8 pass apply:
    a sparse solver pinned at Delta_r_bar=0.000 (ceiling) or at the grid
    diagonal (floor) on EVERY trial makes the pre-registered win AND loss
    rules structurally unsatisfiable at this cell at any n_trials,
    regardless of whether the gap SD itself is nonzero (a single stray trial
    can make the SD nonzero without making the cell adjudicable -- exactly
    what happened at d=3)."""
    grid_diag = float(np.hypot(GRID_SHAPE[0] - 1, GRID_SHAPE[1] - 1))
    dr_all = pilot["delta_r_bar"]
    at_ceiling = [m for m in ("gibf", "mmv") if all(v == 0.0 for v in dr_all[m])]
    at_floor = [m for m in ("gibf", "mmv")
               if all(abs(v - grid_diag) < 1e-9 for v in dr_all[m])]
    p_sep_both_one = (pilot["p_sep_rate"]["gibf"] == 1.0
                      and pilot["p_sep_rate"]["mmv"] == 1.0)
    adjudicable = not (at_ceiling or at_floor)
    return dict(at_ceiling=at_ceiling, at_floor=at_floor,
               p_sep_both_one=p_sep_both_one, adjudicable=adjudicable,
               grid_diag=grid_diag)


# --------------------------------------------------------------- power calc

def power_n_trials(sd, effect):
    """n_trials for a paired (per-trial signed-gap) two-sided mean test to
    reach POWER_TARGET against a true mean gap of `effect`, at significance
    ALPHA_CI -- identical formula and rationale to the d=8 pass's
    power_n_trials(): the paired one-sample test on the signed difference
    gibf_dr[i] - mmv_dr[i], since GIBF and MMV run on the IDENTICAL trial
    (same seed, same X, same V, same A; SPEC S.1).

        n = ((z_{1-alpha/2} + z_{power}) * sd / effect) ** 2
    """
    z = norm.ppf(1.0 - ALPHA_CI / 2.0) + norm.ppf(POWER_TARGET)
    n = (z * sd / effect) ** 2
    return math.ceil(n), z


def effect_size_cells(mmv_baseline_dr):
    """Translate the win rule's OR'd '20% lower OR >=1 cell smaller' into a
    single numeric effect (grid cells) to power against -- identical logic
    to the d=8 pass's effect_size_cells()."""
    if mmv_baseline_dr <= 0.0:
        return 1.0, "baseline Delta_r_bar == 0 (degenerate 20% arm); falling back to the absolute 1-cell effect"
    twenty_pct = 0.20 * mmv_baseline_dr
    if twenty_pct < 1.0:
        return twenty_pct, f"20% of MMV baseline ({mmv_baseline_dr:.4f}) = {twenty_pct:.4f} cells, smaller than the 1-cell alternative -- using the smaller (conservative) effect"
    return 1.0, f"1-cell effect is smaller than 20% of MMV baseline ({twenty_pct:.4f} cells) -- using the smaller (conservative) effect"


# ------------------------------------------------------------------- main --

def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=== Confirmatory coordinate: d=2, accepted 2026-08-16 (Strider) "
         "as the sole untested pool candidate -- NOT re-picked ===")
    geom = geometry_record()
    print(f"  d axis in km: {geom['d_axis_km']}  |  d=2 = "
         f"{geom['d_axis_km'][2]} km")
    print(f"  Historical B1 d=2/5dB cell (2026-07-27, non-degenerate): "
         f"gap_mean={HISTORICAL_B1_D2_5DB['gap_mean']:.4f} "
         f"gap_sd={HISTORICAL_B1_D2_5DB['gap_sd']:.4f} "
         f"P_sep={HISTORICAL_B1_D2_5DB['p_sep_rate']}")

    print("\n=== Step 1: B1 matched pilot cell, re-measured (d=2, snr=5dB, "
         "reduction ON, incoherent, 50 trials) ===")
    b1_cell = b1_pilot_cell_d2()
    print(f"  gap mean={b1_cell['gap_mean']:.4f} sd={b1_cell['gap_sd']:.4f}")
    print(f"  per-method Delta_r_bar mean: {b1_cell['delta_r_bar_mean']}")
    print(f"  per-method P_sep: {b1_cell['p_sep_rate']}")

    print("\n=== Step 2: mini-pilot (50 trials, phi=90, 5dB, d=2, |rho|=0.85, "
         "reduction ON) ===")
    minipilot = run_minipilot_d2()
    (RESULTS / "minipilot_d2_results.json").write_text(json.dumps(minipilot, indent=2))
    print(f"  gap mean={minipilot['gap_mean']:.4f} sd={minipilot['gap_sd']:.4f}")
    print(f"  per-method Delta_r_bar mean (cells): "
         f"{ {k: round(v, 4) for k, v in minipilot['delta_r_bar_mean'].items()} }")
    print(f"  per-method P_sep: {minipilot['p_sep_rate']}")
    print(f"  detected-peak counts: {minipilot['n_peaks_hist']}")

    diag = diagnose_degeneracy(minipilot)
    print(f"\n  Degeneracy diagnosis: at_ceiling={diag['at_ceiling']} "
         f"at_floor={diag['at_floor']} "
         f"p_sep_both_one={diag['p_sep_both_one']} "
         f"adjudicable={diag['adjudicable']}")

    variance_input = max(b1_cell["gap_sd"], minipilot["gap_sd"])
    print(f"\n=== Step 3: S8-ii power calc ===")
    print(f"  variance input = max(B1 d=2/5dB SD, mini-pilot SD) = "
         f"max({b1_cell['gap_sd']:.4f}, {minipilot['gap_sd']:.4f}) = "
         f"{variance_input:.4f} cells")

    mmv_baseline = minipilot["delta_r_bar_mean"]["mmv"]
    effect, effect_note = effect_size_cells(mmv_baseline)
    print(f"  effect size: {effect_note} -> {effect:.4f} cells")

    n_trials_computed = None
    z_used = None
    if diag["adjudicable"] and variance_input > 0.0 and effect > 0.0:
        n_trials_computed, z_used = power_n_trials(variance_input, effect)
        print(f"  n = ((z_(1-alpha/2) + z_power) * sd / effect)^2, "
             f"alpha={ALPHA_CI}, power={POWER_TARGET}, "
             f"z_(1-alpha/2)+z_power={z_used:.4f}")
        print(f"  n = ({z_used:.4f} * {variance_input:.4f} / {effect:.4f})^2 "
             f"= {n_trials_computed} trials")
    else:
        print("  n_trials NOT computed: the cell is not adjudicable (see "
             "diagnosis above) and/or the variance input / effect size is "
             "degenerate (zero). A power calculation against a structurally "
             "unsatisfiable win/loss rule is meaningless -- see the "
             "STOP RULE in this script's module docstring.")

    manifest = dict(
        experiment="Powered B2/B6a confirmatory pass at the accepted d=2 coordinate",
        seed=SEED, numpy=np.__version__,
        git_commit=_git_commit(),
        script_sha256=hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        preregistration="SPEC_experiment_B.md S.1/S.2/S.3(amended)/S.6/S.10/S.11; "
                        "ROADMAP.md S5-D2, S8-ii...viii; d=2 acceptance 2026-08-16",
        confirmatory_d=D_CONFIRM,
        confirmatory_d_km=geom["d_axis_km"][2],
        historical_b1_d2_5db=HISTORICAL_B1_D2_5DB,
        b1_pilot_cell_d2=b1_cell,
        minipilot_d2=minipilot,
        degeneracy_diagnosis=diag,
        power_calc=dict(
            alpha_ci=ALPHA_CI, power_target=POWER_TARGET,
            variance_input=variance_input,
            effect_size_cells=effect, effect_size_note=effect_note,
            n_trials=n_trials_computed,
            z_sum=z_used,
            status=("COMPUTED" if n_trials_computed is not None
                    else "NOT COMPUTED -- cell not adjudicable (see degeneracy_diagnosis)"),
        ),
        runtime_s=round(time.time() - t0, 1),
    )
    (RESULTS / "confirmatory_d2_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote results to {RESULTS}")
    return manifest


if __name__ == "__main__":
    main()
