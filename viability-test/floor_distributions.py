"""Audit A2 — archive per-trial floor-control distributions (bit-exact replay).

Runs AFTER the audit A1 sign-off (ROADMAP §8, 2026-07-10) and BEFORE Card A
Tier 2. Closes audit-D3: the §8-iv fail arm's "statistically indistinguishable
from the same-N floor controls" test (pinned by A1 as a one-sided two-sample
permutation test on the difference of means, 1e4 resamples, alpha=0.05) needs
per-trial distributions to run against; Tier 1 archived MC means only.

Method — replay, not re-draw: tier1_flr_coherence.py consumed a single seeded
RNG stream (seed 20260706) in a fixed order (steering QR, then per N: 3 rhos x
37 phis x n_mc trials x 2 normal draws each, then the two floor controls).
This script reconsumes that stream in the identical order, discarding the
calibration-curve draws and computing per-trial statistics only for the floor
controls. The saved distributions are therefore THE SAME TRIALS whose means
were archived in tier1_summary.csv — asserted below to the CSV's 4-decimal
precision before anything is written. Per the A1 sign-off, the per-trial
top-2 eigenvalue gap is archived too (the gap-conditioning criterion's test
consumes it).

Provenance (audit A3 workflow, first use): this script is committed BEFORE it
runs, the manifest records the HEAD that contains it plus this file's own
sha256, and the artifacts land in the following commit.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from tier1_flr_coherence import SEED, im_frac, kappa, steering

RESULTS = Path(__file__).resolve().parent.parent / "results" / "A_flr_coherence"
CSV_TOL = 5e-5  # tier1_summary.csv floors are rounded to 4 decimals


def consume_curve_draws(rng, n_snap, n_mc, n_phi=37, n_rho=3):
    """Reconsume the RNG draws tier1's calibration curves made, discarding them."""
    for _ in range(n_rho * n_phi * n_mc):
        rng.standard_normal((2, n_snap))
        rng.standard_normal((2, n_snap))


def per_trial_stats(a_s, rho, phi_rad, n_snap, n_mc, rng):
    """Identical simulation to tier1's simulate_stats (same draw order and
    shapes), but returning the per-trial arrays instead of their means."""
    r_s = np.array([[1.0, rho * np.exp(1j * phi_rad)],
                    [rho * np.exp(-1j * phi_rad), 1.0]])
    chol = np.linalg.cholesky(r_s + 1e-12 * np.eye(2))
    kaps = np.empty(n_mc)
    gaps = np.empty(n_mc)
    imfs = np.empty(n_mc)
    for t in range(n_mc):
        z = (rng.standard_normal((2, n_snap)) +
             1j * rng.standard_normal((2, n_snap))) / np.sqrt(2.0)
        x = a_s @ (chol @ z)
        csm = x @ x.conj().T / n_snap
        w, v = np.linalg.eigh(csm)
        kaps[t] = kappa(v[:, -1])
        gaps[t] = w[-1] - w[-2]
        imfs[t] = im_frac(csm)
    return kaps, gaps, imfs


def archived_floor_means():
    """Parse the floor rows out of the committed tier1_summary.csv."""
    means = {}
    for line in (RESULTS / "tier1_summary.csv").read_text().splitlines()[1:]:
        parts = line.split(",")
        if parts[0] in ("kappa_floor", "imfrac_floor"):
            stat = parts[0].split("_")[0]  # kappa | imfrac
            means[(stat, parts[1], int(parts[2]))] = float(parts[3])
    return means


def main():
    rng = np.random.default_rng(SEED)
    a_s = steering(rng)
    n_cfg = {64: 400, 1024: 100}
    controls = {"incoherent": (0.0, 0.0), "phi0_coherent": (0.95, 0.0)}

    archived = archived_floor_means()
    out = {}
    checks = []
    for n_snap, n_mc in n_cfg.items():
        consume_curve_draws(rng, n_snap, n_mc)
        for label, (rho, phi) in controls.items():
            kaps, gaps, imfs = per_trial_stats(a_s, rho, phi, n_snap, n_mc, rng)
            out[f"kappa_{label}_N{n_snap}"] = kaps
            out[f"gap_{label}_N{n_snap}"] = gaps
            out[f"imfrac_{label}_N{n_snap}"] = imfs
            for stat, arr in (("kappa", kaps), ("imfrac", imfs)):
                want = archived[(stat, label, n_snap)]
                got = float(arr.mean())
                ok = abs(got - want) < CSV_TOL
                checks.append(dict(stat=stat, control=label, N=n_snap,
                                   replayed_mean=got, archived_mean=want,
                                   match=bool(ok)))
                status = "OK " if ok else "FAIL"
                print(f"  [{status}] {stat:6s} {label:14s} N={n_snap:<5d} "
                      f"replayed {got:.6f} vs archived {want:.4f} "
                      f"(sd {arr.std(ddof=1):.4f}, n={len(arr)})")
    if not all(c["match"] for c in checks):
        raise SystemExit("replay does not reproduce the archived floor means "
                         "— NOT writing artifacts; investigate before Tier 2")

    np.savez(RESULTS / "floor_distributions.npz", **out)

    script = Path(__file__).resolve()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True,
                                cwd=script.parent).stdout.strip()
    except Exception:
        commit = "unknown"
    manifest = dict(
        seed=SEED, numpy=np.__version__, git_commit=commit,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        method="bit-exact RNG-stream replay of tier1_flr_coherence.py; "
               "per-trial floor-control stats; means asserted against "
               "tier1_summary.csv before writing",
        replay_checks=checks,
        n_mc=n_cfg,
        preregistration="ROADMAP.md §8 2026-07-10 (audit A1 sign-off; "
                        "permutation test alpha=0.05 consumes these arrays, "
                        "gap-conditioning consumes the gap arrays)",
    )
    (RESULTS / "manifest_floors.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote floor_distributions.npz ({len(out)} arrays) + "
          f"manifest_floors.json at commit {commit[:7]}")


if __name__ == "__main__":
    main()
