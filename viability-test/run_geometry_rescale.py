"""run_geometry_rescale.py — geometry-rescale pass: re-run the pinned
50-trial mini-pilot at a refined SECS pole grid.

BACKGROUND (ROADMAP §8, 2026-07-21 entry; ratified by Strider 2026-07-27,
vault `secs-gibf-viability-AT-DISPATCH.md` §8-ii sign-off): the 50-trial
mini-pilot at the exact §8-iii confirmatory coordinate (phi=90 deg, 5 dB,
d=3, |rho|=0.85, N=64, coherent, reduction ON) returned gap SD = 0.0 on
EVERY trial for all three solvers at the original 11x11 pole grid (§S.6.2's
confirmatory setting) -- a genuine ceiling effect (P_sep=1.00, delta_r_bar=0
always), not a solver bug (the single-source gate and B1's own d=1/d=2 rows
show the expected floor/cost structure at that same geometry). No n_trials
could ever power a comparison whose effect is structurally zero.

RULING (2026-07-27): the grid geometry moves, not the pre-registered
coordinates. d=3 was pinned data-independently (ROADMAP §8-iii, 2026-07-07);
the 11x11 pole grid was inherited from the archived brief's config/base.yaml
and flagged in the 2026-07-21 kickoff entry as "NOT pre-registered anywhere
current" -- so it is the geometry, not d=3/phi=90/snr/rho/reduction, that
gives way. phi, snr, d, |rho|, n_snap, n_trials, and grid-reduction=ON are
UNCHANGED below -- copied verbatim from run_experiment_B.py's mini-pilot.

GEOMETRY CHOICE (data-independent of THIS re-run's own result): the original
11x11 pole grid (18x28 deg span) puts one grid cell at ~102 km at this
array's latitude (70N). B1's own d=1/d=2 rows (already run, archived,
non-confirmatory, INCOHERENT regime) show d=1 (~102 km) is a total floor
(P_sep=0 always -- sources merge) and d=2 (~204 km) is the only grid point
with genuine cross-trial variance; d=3 (~306 km) and up are a ceiling
(P_sep=1 always). This script refines the pole grid to 17x17 poles over the
SAME 18x28 deg span (simulate.py's `build_array_and_grid` override
parameters, added this pass) -- purely a density increase, not a change of
footprint -- so that the pinned d=3 (3 grid cells) now falls at ~194 km,
inside the informative band B1 already located, rather than past it. This
choice is made from B1's EXISTING data, before this script's own mini-pilot
re-run produces a number, so it is not tuning to the confirmatory result.

If the re-run below still returns gap SD = 0.0 for every method, that is a
real finding about the design (a stop-and-ask trigger per this pass's
dispatch), not something to be fixed by trying another grid density here.

Run order (A3 workflow -- this script is committed before its first run):
  1. Single-source exact-recovery gate at the rescaled geometry (sanity
     check that the denser grid didn't introduce a solver artifact; not one
     of the three ruled steps, but a guard on their validity).
  2. The 50-trial mini-pilot re-run at phi=90, 5 dB, d=3, |rho|=0.85, N=64,
     reduction ON -- identical parameters to run_experiment_B.py's
     run_minipilot(), only the pole grid resolution differs.

SEED: 20260727 (fresh; distinct from 20260706/20260712/20260721).

CORRECTION (same-day, before the v1 result was reported to Strider --
caught by an advisor review, not by iterating on a disliked number): the
first cut of this script (git history: commit 26bbb78/753706f) had TWO
confounds relative to B1's d=1/d=2/d=3 calibration anchor, discovered by
comparing solver `n_iter` and the source pair's absolute position, not by
re-running the mini-pilot and disliking its result:

  (A) PLACEMENT DRIFT (fixed here). v1 used `RESCALED_COL_A = 7`, which
      centers the source pair near the grid's longitude midpoint. B1's own
      d=1/d=2/d=3 calibration was measured with the pair anchored near the
      station array's WESTERN EDGE (col_a=1 of 11, lon=-9.8 deg, essentially
      at the array's -10 deg boundary) -- sources near the array centroid
      are better-conditioned than edge sources, so "matched B1's informative
      d=2 regime" was only true of the SEPARATION, not the placement.
      RESCALED_COL_A is now 2 (lon=-9.625 deg on the 17-wide axis), matching
      the OLD grid's edge placement to within one grid cell.
      Re-running with ONLY this fixed (edge placement, still d=3, still
      17x17) reproduces the v1 ceiling exactly (gap SD still 0.0), so the
      "ceiling survives" finding is NOT an artifact of the placement drift.

  (B) SOLVER ITERATION-COUNT CONFOUND (flagged, NOT fixed -- outside this
      pass's authority). solvers.py's grid-reduction loop stops when
      `len(cols) < cfg.min_active_factor * n_channels` -- an ABSOLUTE
      threshold (`min_active_factor=2` is part of the PINNED
      `FAIRNESS_CONFIG`, SPEC S.6.1, 2026-07-19; n_channels=75 is fixed by
      the station array, untouched by this pass). At the original 11x11
      grid (n_grid=121), one reduction pass (121 -> 108 kept) already drops
      below `2*75=150`, so GIBF/MMV converge in a SINGLE IRLS iteration.
      At the rescaled 17x17 grid (n_grid=289), it takes SEVEN reduction
      passes (289->260->...->153->137) to cross the same absolute
      threshold -- i.e. refining the grid's PHYSICAL resolution, exactly as
      Ruling 1 authorized, has the SIDE EFFECT of also handing both sparse
      solvers substantially more IRLS refinement (7 passes vs 1) at the
      SAME pinned `FAIRNESS_CONFIG`. This means the rescaled mini-pilot's
      ceiling cannot be cleanly attributed to physical separation alone --
      it may partly (or largely) reflect the extra solver power. Resolving
      this would require either (i) changing the PINNED `min_active_factor`
      (forbidden -- SPEC S.6.1 fairness protocol), or (ii) choosing a grid
      density specifically to hold `n_iter` constant across geometries
      (which would make the geometry choice a function of solver mechanics
      rather than the physical calibration Ruling 1's justification rests
      on -- a different, undisciplined criterion this pass was not
      authorized to introduce). Per this pass's stop-and-ask triggers ("any
      genuine ambiguity in the pinned design"), this is reported to Strider
      unresolved, not patched. `n_iter` is now recorded per trial below so
      the archived data carries the evidence.
"""

import hashlib
import json
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np

from metrics import delta_r_bar, p_sep
from run_experiment_B import single_source_gate
from simulate import build_array_and_grid, build_csm, eigendecompose, eigenmodes, row_center_cols
from solvers import FAIRNESS_CONFIG, solve_all, with_reduction

SEED = 20260727
RESULTS = Path(__file__).resolve().parent.parent / "results" / "B_viability"
ROW_NORMALISATION = "none"  # unchanged (transfer.py default; matches the 07-21 run)

# --- Rescaled pole grid (density only; same span as the original 11x11) ---
RESCALED_N_POLE_LAT = 17
RESCALED_N_POLE_LON = 17
RESCALED_POLE_LAT_SPAN = 18.0   # unchanged from simulate.py's original
RESCALED_POLE_LON_SPAN = 28.0   # unchanged from simulate.py's original
RESCALED_GRID_SHAPE = (RESCALED_N_POLE_LAT, RESCALED_N_POLE_LON)
RESCALED_COL_A = 2   # CORRECTED (see module docstring "CORRECTION (A)"):
                     # anchors the d=3 pair near the array's western edge
                     # (lon=-9.625 deg), matching B1's own edge-placed
                     # d=1/d=2/d=3 calibration rows (col_a=1 of 11,
                     # lon=-9.8 deg) rather than the grid's longitude
                     # center. Free implementation choice (not a
                     # pre-registered coordinate), but must match the
                     # calibration anchor's placement, not just its
                     # separation, to be a valid comparison.

# --- Pinned mini-pilot coordinates (UNCHANGED from run_experiment_B.py) ---
MINIPILOT_PHI_DEG = 90.0
MINIPILOT_SNR_DB = 5.0
MINIPILOT_D = 3
MINIPILOT_ABS_RHO = 0.85   # SLOT-1 pinned midpoint
MINIPILOT_N_SNAP = 64
MINIPILOT_N_TRIALS = 50
MINIPILOT_REDUCTION = True   # §S.6.2 confirmatory setting


def _seed_component(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    return zlib.crc32(repr(x).encode())


def _rng(*components):
    return np.random.default_rng([SEED, *[_seed_component(c) for c in components]])


def build_rescaled_grid():
    """The refined SECS pole grid: same span (18x28 deg) as the original
    11x11 grid, denser (17x17) so 3 grid cells falls in the informative band
    B1 already located (see module docstring)."""
    return build_array_and_grid(
        row_normalisation=ROW_NORMALISATION,
        n_pole_lat=RESCALED_N_POLE_LAT, n_pole_lon=RESCALED_N_POLE_LON,
        pole_lat_span=RESCALED_POLE_LAT_SPAN, pole_lon_span=RESCALED_POLE_LON_SPAN,
    )


def run_minipilot_rescaled(tm):
    """Identical protocol to run_experiment_B.run_minipilot(), at the
    rescaled geometry. phi/snr/d/|rho|/n_snap/n_trials/reduction are the
    SAME pinned values; only tm (and hence the physical meaning of "d=3
    grid cells") differs."""
    idx_a, idx_b = row_center_cols(RESCALED_COL_A, MINIPILOT_D, grid_shape=RESCALED_GRID_SHAPE)
    true_cells = [np.unravel_index(idx_a, RESCALED_GRID_SHAPE),
                 np.unravel_index(idx_b, RESCALED_GRID_SHAPE)]
    rho = MINIPILOT_ABS_RHO * np.exp(1j * np.deg2rad(MINIPILOT_PHI_DEG))
    cfg = with_reduction(FAIRNESS_CONFIG, MINIPILOT_REDUCTION)

    # simulate_snapshots lives in simulate.py but is imported through
    # run_experiment_B's module in the original script; import directly here
    # to keep this script's dependency graph explicit.
    from simulate import simulate_snapshots

    dr = {"l2": [], "gibf": [], "mmv": []}
    ps = {"l2": [], "gibf": [], "mmv": []}
    gibf_n_iter = []   # per-trial list of per-mode iteration counts
    mmv_n_iter = []    # per-trial scalar iteration count
    for trial in range(MINIPILOT_N_TRIALS):
        rng = _rng(0, "minipilot_rescaled", trial)
        X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                                  n_snap=MINIPILOT_N_SNAP, snr_db=MINIPILOT_SNR_DB,
                                  coherence=rho)
        S = build_csm(X)
        lam, U = eigendecompose(S)
        V = eigenmodes(lam, U, 2)
        out = solve_all(tm.A, V, cfg)
        gibf_n_iter.append(list(out["gibf"]["n_iter"]))
        mmv_n_iter.append(int(out["mmv"]["n_iter"]))
        for name in ("l2", "gibf", "mmv"):
            I2d = out[name]["I"].reshape(RESCALED_GRID_SHAPE)
            dr[name].append(delta_r_bar(I2d, true_cells))
            ps[name].append(p_sep(I2d, true_cells))

    gap = [g - m for g, m in zip(dr["gibf"], dr["mmv"])]
    gibf_n_iter_flat = [it for trial_iters in gibf_n_iter for it in trial_iters]
    return dict(
        phi_deg=MINIPILOT_PHI_DEG, snr_db=MINIPILOT_SNR_DB, d=MINIPILOT_D,
        abs_rho=MINIPILOT_ABS_RHO, n_snap=MINIPILOT_N_SNAP,
        n_trials=MINIPILOT_N_TRIALS, reduction=MINIPILOT_REDUCTION,
        pole_grid=dict(n_pole_lat=RESCALED_N_POLE_LAT, n_pole_lon=RESCALED_N_POLE_LON,
                       pole_lat_span=RESCALED_POLE_LAT_SPAN,
                       pole_lon_span=RESCALED_POLE_LON_SPAN,
                       col_a=RESCALED_COL_A),
        delta_r_bar={k: v for k, v in dr.items()},
        p_sep_rate={k: float(np.mean(v)) for k, v in ps.items()},
        delta_r_bar_mean={k: float(np.mean(v)) for k, v in dr.items()},
        delta_r_bar_sd={k: float(np.std(v, ddof=1)) for k, v in dr.items()},
        gap_signed=gap, gap_mean=float(np.mean(gap)), gap_sd=float(np.std(gap, ddof=1)),
        # Confound-(B) evidence (see module docstring "CORRECTION (B)"):
        # solver iteration counts actually realized under the pinned
        # FAIRNESS_CONFIG at this grid density.
        gibf_n_iter_per_trial=gibf_n_iter,
        mmv_n_iter_per_trial=mmv_n_iter,
        gibf_n_iter_mean=float(np.mean(gibf_n_iter_flat)),
        gibf_n_iter_min=int(np.min(gibf_n_iter_flat)),
        gibf_n_iter_max=int(np.max(gibf_n_iter_flat)),
        mmv_n_iter_mean=float(np.mean(mmv_n_iter)),
        mmv_n_iter_min=int(np.min(mmv_n_iter)),
        mmv_n_iter_max=int(np.max(mmv_n_iter)),
    )


def original_grid_n_iter_reference(n_reference_trials=5):
    """Confound-(B) baseline: n_iter realized at the ORIGINAL 11x11 grid,
    same pinned coordinates, edge placement (col_a=1, matching B1's own
    calibration rows) -- so the manifest carries a direct side-by-side
    comparison to the rescaled grid's n_iter, not just an assertion."""
    tm_orig = build_array_and_grid(row_normalisation=ROW_NORMALISATION)
    grid_shape = (11, 11)
    idx_a, idx_b = row_center_cols(1, MINIPILOT_D, grid_shape=grid_shape)
    rho = MINIPILOT_ABS_RHO * np.exp(1j * np.deg2rad(MINIPILOT_PHI_DEG))
    cfg = with_reduction(FAIRNESS_CONFIG, MINIPILOT_REDUCTION)
    from simulate import simulate_snapshots
    gibf_iters, mmv_iters = [], []
    for trial in range(n_reference_trials):
        rng = _rng(1, "original_grid_reference", trial)
        X, _ = simulate_snapshots(tm_orig, [idx_a, idx_b], [1.0, 1.0], rng,
                                  n_snap=MINIPILOT_N_SNAP, snr_db=MINIPILOT_SNR_DB,
                                  coherence=rho)
        S = build_csm(X)
        lam, U = eigendecompose(S)
        V = eigenmodes(lam, U, 2)
        out = solve_all(tm_orig.A, V, cfg)
        gibf_iters.append(list(out["gibf"]["n_iter"]))
        mmv_iters.append(int(out["mmv"]["n_iter"]))
    return dict(n_grid=tm_orig.n_grid, n_channels=tm_orig.n_channels,
               min_active_factor_times_n_channels=int(cfg.min_active_factor * tm_orig.n_channels),
               gibf_n_iter_per_trial=gibf_iters, mmv_n_iter_per_trial=mmv_iters)


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=== Geometry-rescale pass (CORRECTED v2): build refined 17x17 "
         "SECS pole grid ===")
    tm = build_rescaled_grid()
    print(f"n_grid={tm.n_grid}, n_channels={tm.n_channels}")

    print("=== Single-source exact-recovery gate at the rescaled geometry ===")
    gate = single_source_gate(tm=tm)
    (RESULTS / "gate_result_rescaled_v2.json").write_text(json.dumps(gate, indent=2))
    if not gate["passed"]:
        print("GATE: FAIL —", gate["failures"])
        print("Geometry-rescale pass HALTED (solver bug at the rescaled "
             "geometry, not a science finding). See "
             f"{RESULTS / 'gate_result_rescaled_v2.json'}")
        raise SystemExit(1)
    print("GATE: PASS")

    print("=== Mini-pilot re-run (50 trials, phi=90, 5dB, d=3, "
         "reduction ON, 17x17 pole grid, edge-anchored placement) ===")
    minipilot = run_minipilot_rescaled(tm)
    (RESULTS / "minipilot_rescaled_v2_results.json").write_text(json.dumps(minipilot, indent=2))
    print(f"Mini-pilot (rescaled v2): GIBF-MMV gap mean={minipilot['gap_mean']:.4f} "
         f"sd={minipilot['gap_sd']:.4f}")
    for name in ("l2", "gibf", "mmv"):
        print(f"  {name}: delta_r_bar mean={minipilot['delta_r_bar_mean'][name]:.4f} "
             f"sd={minipilot['delta_r_bar_sd'][name]:.4f} "
             f"P_sep={minipilot['p_sep_rate'][name]:.2f}")
    print(f"  gibf n_iter: mean={minipilot['gibf_n_iter_mean']:.1f} "
         f"range=[{minipilot['gibf_n_iter_min']},{minipilot['gibf_n_iter_max']}]")
    print(f"  mmv  n_iter: mean={minipilot['mmv_n_iter_mean']:.1f} "
         f"range=[{minipilot['mmv_n_iter_min']},{minipilot['mmv_n_iter_max']}]")

    print("=== Confound-(B) reference: n_iter at the ORIGINAL 11x11 grid, "
         "same pinned coordinates ===")
    orig_ref = original_grid_n_iter_reference()
    (RESULTS / "original_grid_n_iter_reference.json").write_text(json.dumps(orig_ref, indent=2))
    print(f"  original grid: n_grid={orig_ref['n_grid']}, "
         f"min_active_factor*n_channels={orig_ref['min_active_factor_times_n_channels']}, "
         f"gibf n_iter samples={orig_ref['gibf_n_iter_per_trial'][:3]}")

    ceiling_survives = (minipilot["gap_sd"] == 0.0)
    if ceiling_survives:
        print("\nSTOP-AND-ASK TRIGGER: the rescaled mini-pilot STILL returns "
             "gap SD = 0.0 -- the ceiling survives the rescale AND the "
             "placement correction. Per this pass's dispatch, this is a "
             "real finding about the design, not something to fix by "
             "trying another grid density here. Do not iterate on grid "
             "density further; report to Strider.")
    print("\nSECOND, UNRESOLVED FINDING (confound B, flagged not fixed): "
         f"the rescaled grid gives GIBF/MMV a mean of "
         f"{minipilot['gibf_n_iter_mean']:.1f} IRLS iterations under the "
         "PINNED FAIRNESS_CONFIG, vs 1 iteration at the original 11x11 grid "
         "(see original_grid_n_iter_reference.json) -- a mechanical side "
         "effect of the pinned min_active_factor*n_channels stopping rule "
         "being an ABSOLUTE threshold that does not scale with grid "
         "density. The ceiling finding above cannot be cleanly attributed "
         "to physical separation alone until this is resolved; resolving "
         "it is outside this pass's authority (it would mean either "
         "changing the pinned FAIRNESS_CONFIG, or choosing grid density "
         "from solver mechanics rather than physical calibration). "
         "Reported to Strider unresolved.")

    script = Path(__file__).resolve()
    manifest = dict(
        pass_name="Geometry-rescale pass v2 (CORRECTED): edge-anchored placement "
                  "+ n_iter confound evidence",
        supersedes="manifest_geometry_rescale.json / minipilot_rescaled_results.json "
                  "(v1, git commits 26bbb78/753706f) -- v1 is left on disk, "
                  "unedited, per the append-only log-hygiene convention; v1's "
                  "own numbers are unchanged, but its placement (col_a=7, "
                  "grid-centered) diverges from B1's edge-anchored calibration "
                  "and it did not record n_iter. Re-running with only the "
                  "placement fixed reproduces v1's ceiling exactly.",
        seed=SEED, numpy=np.__version__,
        git_commit=_git_commit(),
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        preregistration="vault secs-gibf-viability-AT-DISPATCH.md, §8-ii sign-off "
                        "2026-07-27 (Rulings 1 and 2); ROADMAP.md §8 2026-07-27 entries",
        geometry=dict(
            original=dict(n_pole_lat=11, n_pole_lon=11, pole_lat_span=18.0, pole_lon_span=28.0,
                         cell_km_at_row_center=101.9, note="B1's own d=1/d=2/d=3 rows at this "
                         "geometry: ~102/~204/~306 km, edge-anchored placement (col_a=1)"),
            rescaled=dict(n_pole_lat=RESCALED_N_POLE_LAT, n_pole_lon=RESCALED_N_POLE_LON,
                         pole_lat_span=RESCALED_POLE_LAT_SPAN,
                         pole_lon_span=RESCALED_POLE_LON_SPAN,
                         col_a=RESCALED_COL_A,
                         d3_physical_km_approx=194.3,
                         justification="d=3 (3 grid cells) now falls at ~194 km, matching "
                         "B1's own already-archived d=2 informative cell (~204 km at the "
                         "original grid) rather than the original grid's d=3 ceiling "
                         "(~306 km) -- chosen from B1's existing data, before this run. "
                         "col_a=2 anchors the pair near the array's western edge "
                         "(lon=-9.625 deg), matching B1's own edge placement (lon=-9.8 "
                         "deg) -- corrected from v1's grid-centered col_a=7."),
            unchanged_pinned_coordinates=dict(phi_deg=MINIPILOT_PHI_DEG, snr_db=MINIPILOT_SNR_DB,
                                              d=MINIPILOT_D, abs_rho=MINIPILOT_ABS_RHO,
                                              n_snap=MINIPILOT_N_SNAP, n_trials=MINIPILOT_N_TRIALS,
                                              reduction=MINIPILOT_REDUCTION, tau_r=1.0),
        ),
        row_normalisation=ROW_NORMALISATION,
        fairness_config=dict(eps_frac=FAIRNESS_CONFIG.eps_frac,
                             p_norm=FAIRNESS_CONFIG.p_norm,
                             weight_floor=FAIRNESS_CONFIG.weight_floor,
                             beta=FAIRNESS_CONFIG.beta,
                             max_iter=FAIRNESS_CONFIG.max_iter,
                             tol=FAIRNESS_CONFIG.tol,
                             min_active_factor=FAIRNESS_CONFIG.min_active_factor),
        gate_passed=gate["passed"],
        minipilot_rescaled_v2=dict(gap_mean=minipilot["gap_mean"], gap_sd=minipilot["gap_sd"],
                                   n_trials=minipilot["n_trials"],
                                   gibf_n_iter_mean=minipilot["gibf_n_iter_mean"],
                                   mmv_n_iter_mean=minipilot["mmv_n_iter_mean"]),
        original_grid_n_iter_reference=dict(
            n_grid=orig_ref["n_grid"],
            min_active_factor_times_n_channels=orig_ref["min_active_factor_times_n_channels"],
            gibf_n_iter_samples=orig_ref["gibf_n_iter_per_trial"],
            mmv_n_iter_samples=orig_ref["mmv_n_iter_per_trial"],
        ),
        ceiling_survives=ceiling_survives,
        confound_b_niter_unresolved=True,
        status="STOP per dispatch: powered B2/B6 NOT run; n_trials NOT computed; "
              "ceiling-survives finding is confounded by the n_iter side effect "
              "(confound B) and returns to Strider unresolved, alongside this "
              "run's real (still-zero) SD",
        runtime_s=round(time.time() - t0, 1),
    )
    (RESULTS / "manifest_geometry_rescale_v2.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote v2 results + manifest to {RESULTS}")


if __name__ == "__main__":
    main()
