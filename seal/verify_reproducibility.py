"""seal/verify_reproducibility.py — regenerate a prefix of trials of the sealed
runs from their pinned seeds and compare them with the saved arrays.

This reads the sealed result files; it writes nothing and generates no trial
that is not already in them (it recomputes trials 0..N-1 of cells that exist).
Run from the repo root with the mhd-env interpreter:

    python seal/verify_reproducibility.py [N]

Numeric arrays are compared exactly and, separately, to rtol 1e-9 (a different
BLAS thread count can legitimately change the last bits); string arrays must be
equal. Exit code 0 only if every compared array matches to rtol 1e-9.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "viability-test"))

import run_experiment_B2B6_powered_d2 as P          # noqa: E402
import run_experiment_B4_B6b as Q                    # noqa: E402
from simulate import build_array_and_grid           # noqa: E402
from solvers import solve_all                        # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def compare(saved, fresh, label, stats):
    if saved.dtype.kind in "US" or fresh.dtype.kind in "US":
        ok = bool(np.array_equal(saved[:N], fresh[:N]))
        stats["exact"] += ok
        stats["close"] += ok
    else:
        a, b = np.asarray(saved[:N], dtype=float), np.asarray(fresh[:N], dtype=float)
        stats["exact"] += bool(np.array_equal(a, b, equal_nan=True))
        close = bool(np.allclose(a, b, rtol=1e-9, atol=0.0, equal_nan=True))
        stats["close"] += close
        if not close:
            print(f"    MISMATCH {label}: max abs diff {np.nanmax(np.abs(a - b)):.3g}")
    stats["n"] += 1


def main():
    tm = build_array_and_grid(row_normalisation=P.ROW_NORMALISATION)
    stats = dict(n=0, exact=0, close=0)

    print(f"== confirmatory run (seed {P.MASTER_SEED}), first {N} trials of selected cells")
    plan = {c.cell_id: c for c in P.build_plan()}
    for cid in ("B2|phi=90|snr=5|d=2|nsnap=64", "B2|phi=180|snr=10|d=2|nsnap=64",
                "B2|phi=0|snr=5|d=1|nsnap=64", "B6a|phi=90|snr=5|d=2|nsnap=64",
                "B2offgrid|phi=90|snr=5|d=2|nsnap=64"):
        cell = plan[cid]
        arrays, _ = P.run_data_cell(tm, cell, N, P.MASTER_SEED)
        z = np.load(REPO / "results" / "B_viability" / "powered_d2" / f"{cell.file_stem}.npz")
        before = stats["n"]
        for arm, rec in arrays.items():
            for field, arr in rec.items():
                key = f"{arm}__{field}"
                if key in z.files:
                    compare(z[key], np.asarray(arr), f"{cid} {key}", stats)
        print(f"  {cid}: {stats['n'] - before} arrays compared")

    print(f"== B4/B6b run (seed {Q.B4B6B_SEED}), first {N} trials of selected cells")
    kernels = Q.build_kernels(tm)
    plan = {c.cell_id: c for c in Q.build_plan()}
    for cid in ("B4|snr=5|d=2|nsnap=64", "B4|snr=-5|d=3|nsnap=64"):
        cell = plan[cid]
        arrays = Q.run_b4_cell(tm, kernels, cell, N, Q.B4B6B_SEED)
        z = np.load(REPO / "results" / "B_viability" / "b4_b6b" / f"{cell.file_stem}.npz")
        before = stats["n"]
        for (k, red), rec in arrays.items():
            for field, arr in rec.items():
                key = f"{k}_{'on' if red else 'off'}__{field}"
                compare(z[key], np.asarray(arr), f"{cid} {key}", stats)
        print(f"  {cid}: {stats['n'] - before} arrays compared")
    for v in ("step_toward", "iid"):
        cid = f"B6b|variant={v}|phi=90|snr=5|d=2|nsnap=64"
        cell = plan[cid]
        rec = Q.run_b6b_cell(tm, cell, N, Q.B4B6B_SEED, solve_fn=solve_all)
        z = np.load(REPO / "results" / "B_viability" / "b4_b6b" / f"{cell.file_stem}.npz")
        before = stats["n"]
        for field, arr in rec.items():
            compare(z[field], np.asarray(arr), f"{cid} {field}", stats)
        print(f"  {cid}: {stats['n'] - before} arrays compared")

    print(f"\n{stats['n']} arrays compared: {stats['exact']} bit-identical, "
          f"{stats['close']} equal to rtol 1e-9")
    return 0 if stats["close"] == stats["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
