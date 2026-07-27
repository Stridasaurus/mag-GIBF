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
RESCALED_COL_A = 7   # centers the d=3 pair near the grid's longitude center
                     # ((17-1-3)//2 == 6.5, rounded to 7); free implementation
                     # choice, not a pre-registered coordinate.

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
    for trial in range(MINIPILOT_N_TRIALS):
        rng = _rng(0, "minipilot_rescaled", trial)
        X, _ = simulate_snapshots(tm, [idx_a, idx_b], [1.0, 1.0], rng,
                                  n_snap=MINIPILOT_N_SNAP, snr_db=MINIPILOT_SNR_DB,
                                  coherence=rho)
        S = build_csm(X)
        lam, U = eigendecompose(S)
        V = eigenmodes(lam, U, 2)
        out = solve_all(tm.A, V, cfg)
        for name in ("l2", "gibf", "mmv"):
            I2d = out[name]["I"].reshape(RESCALED_GRID_SHAPE)
            dr[name].append(delta_r_bar(I2d, true_cells))
            ps[name].append(p_sep(I2d, true_cells))

    gap = [g - m for g, m in zip(dr["gibf"], dr["mmv"])]
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
    )


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return "unknown"


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=== Geometry-rescale pass: build refined 17x17 SECS pole grid ===")
    tm = build_rescaled_grid()
    print(f"n_grid={tm.n_grid}, n_channels={tm.n_channels}")

    print("=== Single-source exact-recovery gate at the rescaled geometry ===")
    gate = single_source_gate(tm=tm)
    (RESULTS / "gate_result_rescaled.json").write_text(json.dumps(gate, indent=2))
    if not gate["passed"]:
        print("GATE: FAIL —", gate["failures"])
        print("Geometry-rescale pass HALTED (solver bug at the rescaled "
             "geometry, not a science finding). See "
             f"{RESULTS / 'gate_result_rescaled.json'}")
        raise SystemExit(1)
    print("GATE: PASS")

    print("=== Mini-pilot re-run (50 trials, phi=90, 5dB, d=3, "
         "reduction ON, 17x17 pole grid) ===")
    minipilot = run_minipilot_rescaled(tm)
    (RESULTS / "minipilot_rescaled_results.json").write_text(json.dumps(minipilot, indent=2))
    print(f"Mini-pilot (rescaled): GIBF-MMV gap mean={minipilot['gap_mean']:.4f} "
         f"sd={minipilot['gap_sd']:.4f}")
    for name in ("l2", "gibf", "mmv"):
        print(f"  {name}: delta_r_bar mean={minipilot['delta_r_bar_mean'][name]:.4f} "
             f"sd={minipilot['delta_r_bar_sd'][name]:.4f} "
             f"P_sep={minipilot['p_sep_rate'][name]:.2f}")

    ceiling_survives = (minipilot["gap_sd"] == 0.0)
    if ceiling_survives:
        print("\nSTOP-AND-ASK TRIGGER: the rescaled mini-pilot STILL returns "
             "gap SD = 0.0 -- the ceiling survives the rescale. Per this "
             "pass's dispatch, this is a real finding about the design, not "
             "something to fix by trying another grid density here. Do not "
             "iterate further; report to Strider.")
    else:
        print(f"\nReal gap SD obtained: {minipilot['gap_sd']:.4f} "
             f"(mean {minipilot['gap_mean']:.4f}, n_trials={minipilot['n_trials']}). "
             "This is the real-SD input for Strider's §8-ii power-calc "
             "sign-off. n_trials itself is NOT computed here (out of scope "
             "for this pass) and powered B2/B6 are NOT run.")

    script = Path(__file__).resolve()
    manifest = dict(
        pass_name="Geometry-rescale pass: refined SECS pole grid + rescaled mini-pilot re-run",
        seed=SEED, numpy=np.__version__,
        git_commit=_git_commit(),
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        preregistration="vault secs-gibf-viability-AT-DISPATCH.md, §8-ii sign-off "
                        "2026-07-27 (Rulings 1 and 2); ROADMAP.md §8 2026-07-27 entry",
        geometry=dict(
            original=dict(n_pole_lat=11, n_pole_lon=11, pole_lat_span=18.0, pole_lon_span=28.0,
                         cell_km_at_row_center=101.9, note="B1's own d=1/d=2/d=3 rows at this "
                         "geometry: ~102/~204/~306 km"),
            rescaled=dict(n_pole_lat=RESCALED_N_POLE_LAT, n_pole_lon=RESCALED_N_POLE_LON,
                         pole_lat_span=RESCALED_POLE_LAT_SPAN,
                         pole_lon_span=RESCALED_POLE_LON_SPAN,
                         col_a=RESCALED_COL_A,
                         d3_physical_km_approx=194.3,
                         justification="d=3 (3 grid cells) now falls at ~194 km, matching "
                         "B1's own already-archived d=2 informative cell (~204 km at the "
                         "original grid) rather than the original grid's d=3 ceiling "
                         "(~306 km) -- chosen from B1's existing data, before this run."),
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
        minipilot_rescaled=dict(gap_mean=minipilot["gap_mean"], gap_sd=minipilot["gap_sd"],
                                n_trials=minipilot["n_trials"]),
        ceiling_survives=ceiling_survives,
        status="STOP per dispatch: powered B2/B6 NOT run; n_trials NOT computed; "
              "this run's real SD returns to Strider's §8-ii power-calc sign-off",
        runtime_s=round(time.time() - t0, 1),
    )
    (RESULTS / "manifest_geometry_rescale.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote results + manifest to {RESULTS}")


if __name__ == "__main__":
    main()
