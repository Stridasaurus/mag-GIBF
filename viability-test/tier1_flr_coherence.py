"""Card A, Tier 1 — FLR source-coherence forward model (pure NumPy; no build deps).

Runs AFTER pre-registration §8-iv sign-off (2026-07-06, with the floor amendment)
and BEFORE any Tier-2 / ground-CSM work. See EXPERIMENT_CARD_A.md §A.3.1 and
ROADMAP.md §8 (2026-07-06 entry).

Produces, into results/A_flr_coherence/:
  1. kappa(phi) and ImFrac(phi) calibration curves on the pinned two-source
     scenario, MC-averaged, at N=64 (primary) and N=1024 (asymptotic secondary),
     for |rho| in {0.70, 0.85, 0.95}.
  2. Floor controls at both N, for both statistics: incoherent (rho=0) and
     phi=0 coherent (|rho|=0.95) — the fail arm of §A.4 is referenced to these.
  3. The co-primary threshold pinning (§8-iv procedure): ImFrac read off the
     |rho|=0.95, N=64 curve at the effective phi where kappa crosses 0.10 / 0.30.
  4. FLR geometry: pole-pair phase |phi_ij| vs spacing in resonance-width units
     (driven damped oscillator model, Southwood 1974 / Chen & Hasegawa 1974),
     max over pair placement, plus the physical-spacing table for Q in {5,10,20}.

Pinned scenario config (canonical calibration scenario going forward — the
2026-06-30 desk-check's exact array config was not committed; anchors are
checked for reproduction within tolerance and this file's config supersedes
it as the pinned reference):
  M=8 real orthonormal steering vectors (fixed seed), unit source powers,
  no sensor noise (finite-N leakage is the effect under calibration),
  n_mc=400 (N=64) / 100 (N=1024), phi grid 0..180 deg step 5.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

SEED = 20260706
RESULTS = Path(__file__).resolve().parent.parent / "results" / "A_flr_coherence"

# Okabe-Ito (colorblind-safe) hues, fixed assignment order
C_RHO = {0.95: "#0072B2", 0.85: "#E69F00", 0.70: "#009E73"}
C_IM = "#CC79A7"
C_GEOM = ["#0072B2", "#E69F00", "#009E73"]


# ---------------------------------------------------------------- statistics

def kappa(u):
    """Residual eigenvector complexity: min over global phase theta of
    ||Im(e^{i theta} u)|| / ||u||  ==  sqrt(lambda_min of the 2x2 Gram of
    [Re u, Im u]) / ||u||."""
    re, im = u.real, u.imag
    g = np.array([[re @ re, re @ im], [re @ im, im @ im]])
    lam_min = np.linalg.eigvalsh(g)[0]
    return float(np.sqrt(max(lam_min, 0.0)) / np.linalg.norm(u))


def im_frac(S):
    """Rotation-invariant CSM imaginary fraction ||Im S||_F / ||S||_F."""
    return float(np.linalg.norm(S.imag) / np.linalg.norm(S))


# ------------------------------------------------------- two-source scenario

def steering(rng, m=8):
    """Two fixed real orthonormal steering vectors (pinned by seed)."""
    a = rng.standard_normal((m, 2))
    q, _ = np.linalg.qr(a)
    return q  # m x 2, orthonormal columns


def simulate_stats(a_s, rho, phi_rad, n_snap, n_mc, rng):
    """MC-averaged (kappa of top eigenvector, eigenvalue gap, ImFrac) for the
    two-source scenario with source cross-spectrum [[1, rho e^{i phi}], [*, 1]]."""
    r_s = np.array([[1.0, rho * np.exp(1j * phi_rad)],
                    [rho * np.exp(-1j * phi_rad), 1.0]])
    # Cholesky with a hair of regularisation for rho -> 1
    chol = np.linalg.cholesky(r_s + 1e-12 * np.eye(2))
    kaps, gaps, imfs = [], [], []
    for _ in range(n_mc):
        z = (rng.standard_normal((2, n_snap)) +
             1j * rng.standard_normal((2, n_snap))) / np.sqrt(2.0)
        x = a_s @ (chol @ z)                       # m x N ground-free snapshots
        csm = x @ x.conj().T / n_snap
        w, v = np.linalg.eigh(csm)
        kaps.append(kappa(v[:, -1]))
        gaps.append(float(w[-1] - w[-2]))
        imfs.append(im_frac(csm))
    return float(np.mean(kaps)), float(np.mean(gaps)), float(np.mean(imfs))


# ----------------------------------------------------------- FLR phase model

def arg_r(u):
    """Phase of the driven-oscillator response R = 1/(gamma*omega*(u + i)),
    u = (omega_r^2 - omega^2)/(gamma*omega) the normalised detuning.
    Sweeps 0 -> -90 deg -> -180 deg as u goes +inf -> 0 -> -inf."""
    return -np.arctan2(1.0, u)


def pair_phase_max(sep):
    """|phi_ij| for a pole pair separated by `sep` resonance-width units,
    maximised over placement relative to the resonance (§A.4 'best case')."""
    centers = np.linspace(-6.0, 6.0, 481)
    u1, u2 = centers - sep / 2.0, centers + sep / 2.0
    dphi = np.abs(np.degrees(arg_r(u1) - arg_r(u2)))
    return float(dphi.max()), float(centers[dphi.argmax()])


# -------------------------------------------------------------------- main

def main():
    rng = np.random.default_rng(SEED)
    a_s = steering(rng)
    RESULTS.mkdir(parents=True, exist_ok=True)

    phi_deg = np.arange(0.0, 181.0, 5.0)
    rhos = [0.95, 0.85, 0.70]
    n_cfg = {64: 400, 1024: 100}

    curves = {}   # (rho, N) -> dict of arrays
    floors = {}   # (label, N) -> (kappa, imfrac)
    for n_snap, n_mc in n_cfg.items():
        for rho in rhos:
            k_arr, g_arr, i_arr = [], [], []
            for pd in phi_deg:
                k, g, imf = simulate_stats(a_s, rho, np.radians(pd),
                                           n_snap, n_mc, rng)
                k_arr.append(k); g_arr.append(g); i_arr.append(imf)
            curves[(rho, n_snap)] = dict(kappa=np.array(k_arr),
                                         gap=np.array(g_arr),
                                         imfrac=np.array(i_arr))
        # floor controls (amendment, §8-iv): incoherent and phi=0 coherent
        floors[("incoherent", n_snap)] = simulate_stats(
            a_s, 0.0, 0.0, n_snap, n_cfg[n_snap], rng)
        floors[("phi0_coherent", n_snap)] = simulate_stats(
            a_s, 0.95, 0.0, n_snap, n_cfg[n_snap], rng)

    # ---- co-primary threshold pinning (§8-iv procedure), rho=0.95, N=64
    cal = curves[(0.95, 64)]
    pins = {}
    for band in (0.10, 0.30):
        idx = int(np.argmax(cal["kappa"] >= band))
        # linear interpolation for the effective phi at the crossing
        if idx == 0:
            phi_eff = float(phi_deg[0])
        else:
            k0, k1 = cal["kappa"][idx - 1], cal["kappa"][idx]
            f = (band - k0) / (k1 - k0)
            phi_eff = float(phi_deg[idx - 1] + f * (phi_deg[idx] - phi_deg[idx - 1]))
        imf_at = float(np.interp(phi_eff, phi_deg, cal["imfrac"]))
        pins[band] = dict(phi_eff_deg=phi_eff, imfrac_threshold=imf_at)

    # ---- FLR geometry: |phi_ij| vs spacing (width units), max over placement
    seps = np.geomspace(0.05, 20.0, 120)
    phi_max = np.array([pair_phase_max(s)[0] for s in seps])
    # physical mapping: linear omega_r(x) = omega (1 + beta x); |u| = 2 Q beta x
    # => resonance half-width x_w = 1/(2 Q beta). Table for IMAGE-like spacings.
    phys = []
    for q in (5, 10, 20):
        for beta_per_100km in (0.05, 0.10, 0.20):
            x_w = 100.0 / (2 * q * beta_per_100km)       # km at |u| = 1
            for d_km in (100.0, 250.0):
                d_widths = d_km / x_w
                phys.append(dict(Q=q, beta_per_100km=beta_per_100km,
                                 halfwidth_km=x_w, spacing_km=d_km,
                                 spacing_widths=d_widths,
                                 phi_max_deg=pair_phase_max(d_widths)[0]))

    # ------------------------------------------------------------- outputs
    np.savez(RESULTS / "tier1_results.npz",
             phi_deg=phi_deg, seps_widths=seps, phi_max_deg=phi_max,
             **{f"kappa_rho{int(r*100)}_N{n}": curves[(r, n)]["kappa"]
                for r in rhos for n in n_cfg},
             **{f"imfrac_rho{int(r*100)}_N{n}": curves[(r, n)]["imfrac"]
                for r in rhos for n in n_cfg},
             **{f"gap_rho{int(r*100)}_N{n}": curves[(r, n)]["gap"]
                for r in rhos for n in n_cfg})

    lines = ["metric,condition,N,value"]
    for (lab, n), (k, _g, imf) in floors.items():
        lines += [f"kappa_floor,{lab},{n},{k:.4f}",
                  f"imfrac_floor,{lab},{n},{imf:.4f}"]
    for band, p in pins.items():
        lines += [f"phi_eff_at_kappa_{band},rho095,64,{p['phi_eff_deg']:.2f}",
                  f"imfrac_threshold_at_kappa_{band},rho095,64,"
                  f"{p['imfrac_threshold']:.4f}"]
    for row in phys:
        lines.append(
            f"phi_max_deg,Q{row['Q']}_beta{row['beta_per_100km']}_"
            f"d{int(row['spacing_km'])}km,-,{row['phi_max_deg']:.1f}")
    (RESULTS / "tier1_summary.csv").write_text("\n".join(lines) + "\n")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        commit = "unknown"
    manifest = dict(seed=SEED, numpy=np.__version__, git_commit=commit,
                    scenario=dict(M=8, steering="real orthonormal, seeded QR",
                                  sensor_noise=0.0, phi_grid_deg=[0, 180, 5],
                                  rhos=rhos, n_snapshots=list(n_cfg),
                                  n_mc=n_cfg),
                    preregistration="ROADMAP.md §8-iv (2026-07-06, floor amendment)",
                    pins={str(k): v for k, v in pins.items()})
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: calibration curves with decision bands
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, n_snap in ((ax1, 64), (ax2, 1024)):
        ax.axhspan(0.30, 1.0, color="0.92", zorder=0)
        ax.axhspan(0.0, 0.10, color="0.92", zorder=0)
        ax.text(178, 0.315, "H-A holds ≥ 0.30", ha="right", fontsize=8, color="0.35")
        ax.text(178, 0.015, "H-A fails < 0.10 (floor-referenced)", ha="right",
                fontsize=8, color="0.35")
        for rho in rhos:
            c = curves[(rho, n_snap)]
            ax.plot(phi_deg, c["kappa"], color=C_RHO[rho], lw=2,
                    label=f"κ, |ρ|={rho}")
        ax.plot(phi_deg, curves[(0.95, n_snap)]["imfrac"], color=C_IM, lw=2,
                ls="--", label="‖Im S‖/‖S‖, |ρ|=0.95")
        for lab, ls in (("incoherent", ":"), ("phi0_coherent", "-.")):
            ax.axhline(floors[(lab, n_snap)][0], color="0.25", lw=1, ls=ls)
        ax.axhline(floors[("incoherent", n_snap)][0], color="0.25", lw=0, )
        ax.text(2, floors[("incoherent", n_snap)][0] + 0.008,
                f"incoherent κ floor {floors[('incoherent', n_snap)][0]:.3f}",
                fontsize=7.5, color="0.25")
        ax.set_title(f"N = {n_snap}")
        ax.set_xlabel("inter-source phase φ (deg)")
        ax.set_xlim(0, 180); ax.set_ylim(0, 0.85)
        ax.grid(alpha=0.25, lw=0.5)
    ax1.set_ylabel("statistic value")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Card A Tier 1 — κ and ‖Im S‖/‖S‖ calibration vs φ "
                 "(pinned two-source scenario, MC-averaged)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_calibration_curves.png", dpi=160)

    # Fig 2: FLR pair phase vs spacing
    fig2, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhspan(70, 110, color="0.92", zorder=0)
    ax.text(0.06, 104, "exploitable band (~90°)", fontsize=8, color="0.35")
    ax.plot(seps, phi_max, color=C_GEOM[0], lw=2,
            label="max over placement |φ_ij|")
    ax.axhline(180, color="0.6", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("pole-pair separation (resonance half-width units, |u|=1)")
    ax.set_ylabel("|φ_ij| (deg)")
    ax.set_title("Card A Tier 1 — FLR inter-source phase vs pole spacing\n"
                 "(driven damped oscillator; best-case placement)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(RESULTS / "fig_phi_vs_spacing.png", dpi=160)

    # ----------------------------------------------------------- console
    print("=== anchors check (rho=0.95, N=64; desk-check 2026-06-30) ===")
    for target_phi, anchor in ((0, 0.0), (30, 0.22), (60, 0.44), (90, 0.65),
                               (180, 0.0)):
        got = float(np.interp(target_phi, phi_deg, cal["kappa"]))
        print(f"  kappa(phi={target_phi:>3}) = {got:.3f}   (desk-check {anchor})")
    for n_snap in n_cfg:
        k_i, _, imf_i = floors[("incoherent", n_snap)]
        k_0, _, imf_0 = floors[("phi0_coherent", n_snap)]
        print(f"  floors N={n_snap}: incoherent kappa={k_i:.3f} "
              f"imfrac={imf_i:.4f} | phi=0 coherent kappa={k_0:.3f} "
              f"imfrac={imf_0:.4f}")
    print("=== co-primary pinning (§8-iv) ===")
    for band, p in pins.items():
        print(f"  kappa={band} crossing at phi_eff={p['phi_eff_deg']:.1f} deg "
              f"-> ImFrac threshold {p['imfrac_threshold']:.4f}")
    print("=== physical spacing table (phi_max, deg) ===")
    for row in phys:
        print(f"  Q={row['Q']:>2} beta={row['beta_per_100km']:.2f}/100km "
              f"halfwidth={row['halfwidth_km']:>6.1f} km "
              f"d={int(row['spacing_km']):>3} km "
              f"({row['spacing_widths']:>5.2f} w) -> {row['phi_max_deg']:>6.1f}")


if __name__ == "__main__":
    main()
