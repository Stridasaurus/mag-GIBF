"""Gate V — kernel validation (ROADMAP.md §6 "Gate V"; build brief Appendix C).

Pass/halt build gate, not an experiment. Contents:

  V1  secsy API surface: accepted current_type keywords, return order, units,
      sign conventions, the theta-clip / under-pole NaN gotcha (adapter contract).
  V2  Realness / dtype / shape invariants of the B G-matrices.
  V3  CF-pair probe: the CF ground-field block at every latitude preset.
      Code-inspection provenance: secsy hard-codes Fukushima analytically
      (utils.py, curl_free branch: ``Ge_[below] *= 0``) — the library returns the
      field of the FULL pair (horizontal CF current + radial FACs) by construction,
      so the expected ground block is EXACTLY 0.0, not merely machine precision.
      The theorem's independent numerical confirmation is therefore V5, not V3.
  V4  DF independent validation: Biot-Savart quadrature of the DF SECS sheet
      current vs the library closed form (this is the block every downstream
      number rests on), plus the analytic under-pole radial-field spot check.
  V5  Fukushima pair independent confirmation: Biot-Savart of the horizontal CF
      sheet + the polar line FAC + the uniformly distributed return FACs sums to
      ~0 below the ionosphere, and matches the library CF field above it.

Deterministic (no RNG). Artifacts: results/V_kernel_validation/
    gateV_summary.json  manifest.json  gateV_fields.npz

Run:  conda run -n mhd-env python viability-test/gateV_kernel_validation.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np

import secsy
from secsy.utils import MU0, RE, get_SECS_B_G_matrices

RI = RE + 110e3  # SECS pole shell radius [m] (library default)
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "V_kernel_validation"

LATITUDE_PRESETS = {"high": 70.0, "mid": 55.0, "low": 35.0}  # brief §6.1

# quadrature resolution (midpoint rules; convergence checked at half resolution)
N_THETA, N_PHI = 2400, 4800

d2r = np.pi / 180.0


# ----------------------------------------------------------------- geometry --
def ecef(lat_deg, lon_deg, r):
    la, lo = np.asarray(lat_deg) * d2r, np.asarray(lon_deg) * d2r
    return np.stack(
        [r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)],
        axis=-1,
    )


def enu_basis(lat_deg, lon_deg):
    la, lo = lat_deg * d2r, lon_deg * d2r
    e = np.array([-np.sin(lo), np.cos(lo), 0.0])
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    u = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return e, n, u


def lib_B_enu(st_lat, st_lon, st_r, pole_lat, pole_lon, current_type, **kw):
    """Library field per unit amplitude at one station from one pole, ENU [T]."""
    Ge, Gn, Gr = get_SECS_B_G_matrices(
        [st_lat], [st_lon], st_r, [pole_lat], [pole_lon],
        current_type=current_type, RI=RI, **kw,
    )
    return np.array([Ge[0, 0], Gn[0, 0], Gr[0, 0]])


def enu_to_ecef_vec(v_enu, lat_deg, lon_deg):
    e, n, u = enu_basis(lat_deg, lon_deg)
    return v_enu[0] * e + v_enu[1] * n + v_enu[2] * u


# ------------------------------------------------- Biot-Savart line elements --
def semiinf_line_B(P, A, u_hat, I):
    """Field at P of a semi-infinite straight line from A to infinity along
    u_hat carrying current I [A] in the +u_hat direction. Vectorized over A
    ((..., 3)); P and u_hat broadcastable. On-axis points return 0 (the limit)."""
    d1 = P - A
    cross = np.cross(np.broadcast_to(u_hat, d1.shape), d1)
    rho2 = np.einsum("...i,...i", cross, cross)
    cos1 = np.einsum("...i,...i", np.broadcast_to(u_hat, d1.shape), d1) / np.linalg.norm(
        d1, axis=-1
    )
    factor = cos1 + 1.0  # (u.d2/|d2| -> -1 at the far end)
    with np.errstate(divide="ignore", invalid="ignore"):
        B = (MU0 * I / (4 * np.pi)) * cross * (factor / rho2)[..., None]
    return np.where(rho2[..., None] > 1e-12, B, 0.0)


def _selftest_line_formula():
    """Closed form vs brute-force quadrature along the line, random geometry."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=3) * 1e6
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    P = A + rng.normal(size=3) * 5e5
    closed = semiinf_line_B(P, A, u, 1.0)
    t = np.geomspace(1.0, 1e12, 400_000)
    pts = A + t[:, None] * u
    dl = np.diff(t)[:, None] * u
    mid = 0.5 * (pts[1:] + pts[:-1])
    R = P - mid
    num = (MU0 / (4 * np.pi)) * np.sum(
        np.cross(dl, R) / np.linalg.norm(R, axis=1)[:, None] ** 3, axis=0
    )
    return float(np.linalg.norm(closed - num) / np.linalg.norm(closed))


# ------------------------------------------------------ sheet-current fields --
def sheet_B(P_list, pole_lat, pole_lon, direction, n_theta=N_THETA, n_phi=N_PHI):
    """Biot-Savart field [T] at ECEF points P_list of the SECS surface current
    K = 1/(4*pi*RI) * cot(theta'/2) * e_dir on the shell r=RI, unit amplitude.

    direction: 'phi' (azimuthal, DF) or 'theta' (away from pole, CF horizontal).
    Midpoint quadrature, chunked over theta'.
    """
    p_hat = ecef(pole_lat, pole_lon, 1.0)
    e1 = np.cross([0.0, 0.0, 1.0], p_hat)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(p_hat, e1)

    th = (np.arange(n_theta) + 0.5) * (np.pi / n_theta)
    ph = (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    dth, dph = np.pi / n_theta, 2 * np.pi / n_phi
    cph, sph = np.cos(ph), np.sin(ph)

    P = np.asarray(P_list)
    B = np.zeros_like(P)
    c0 = 1.0 / (4 * np.pi * RI)  # K amplitude for I0 = 1 A

    chunk = 100
    for i0 in range(0, n_theta, chunk):
        t = th[i0 : i0 + chunk]
        st, ct = np.sin(t)[:, None], np.cos(t)[:, None]
        # shell points (nt, np, 3)
        radial = st[..., None] * (cph[None, :, None] * e1 + sph[None, :, None] * e2) \
            + ct[..., None] * p_hat
        xs = RI * radial
        e_phi = -sph[None, :, None] * e1 + cph[None, :, None] * e2
        e_phi = np.broadcast_to(e_phi, radial.shape)
        if direction == "phi":
            e_dir = e_phi
        else:  # 'theta': d(radial)/dtheta
            e_dir = ct[..., None] * (cph[None, :, None] * e1 + sph[None, :, None] * e2) \
                - st[..., None] * p_hat
        Kmag = c0 / np.tan(t / 2)[:, None]  # (nt, 1)
        dA = (RI**2) * st * dth * dph  # (nt, 1)
        w = (Kmag * dA)[..., None] * e_dir  # K * dA, (nt, np, 3)
        for k, Pk in enumerate(P):
            R = Pk[None, None, :] - xs
            r3 = np.linalg.norm(R, axis=-1, keepdims=True) ** 3
            B[k] += (MU0 / (4 * np.pi)) * np.sum(np.cross(w, R) / r3, axis=(0, 1))
    return B


def return_fac_B(P_list, pole_lat, pole_lon, n_theta=N_THETA // 2, n_phi=N_PHI // 2):
    """Field of the uniformly distributed OUTWARD return FACs of the CF system:
    semi-infinite radial lines from r=RI to infinity, surface current density
    1/(4*pi*RI^2) [A/m^2] for unit amplitude. (Smooth integrand; half resolution.)"""
    p_hat = ecef(pole_lat, pole_lon, 1.0)
    e1 = np.cross([0.0, 0.0, 1.0], p_hat)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(p_hat, e1)

    th = (np.arange(n_theta) + 0.5) * (np.pi / n_theta)
    ph = (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    dth, dph = np.pi / n_theta, 2 * np.pi / n_phi
    cph, sph = np.cos(ph), np.sin(ph)

    P = np.asarray(P_list)
    B = np.zeros_like(P)
    dens = 1.0 / (4 * np.pi * RI**2)

    chunk = 60
    for i0 in range(0, n_theta, chunk):
        t = th[i0 : i0 + chunk]
        st, ct = np.sin(t)[:, None], np.cos(t)[:, None]
        radial = st[..., None] * (cph[None, :, None] * e1 + sph[None, :, None] * e2) \
            + ct[..., None] * p_hat
        A = RI * radial  # line start points
        dI = dens * (RI**2) * st * dth * dph  # (nt, 1) current per element
        for k, Pk in enumerate(P):
            Bel = semiinf_line_B(Pk[None, None, :], A, radial, 1.0)
            B[k] += np.sum(dI[..., None] * Bel, axis=(0, 1))
    return B


# ------------------------------------------------------------------- checks --
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: dict = {"gate": "V", "checks": {}}
    ok = True

    # ---- V1: API surface -----------------------------------------------
    kw_accept = {}
    for kw in ["divergence_free", "curl_free", "divfree", "curlfree", "potential", "scalar"]:
        try:
            get_SECS_B_G_matrices([60.0], [0.0], RE, [62.0], [0.0], current_type=kw, RI=RI)
            kw_accept[kw] = "accepted"
        except Exception as exc:  # noqa: BLE001 - probing the API is the point
            kw_accept[kw] = f"rejected ({type(exc).__name__})"
    summary["checks"]["V1_keywords"] = kw_accept
    v1_ok = kw_accept["divergence_free"] == "accepted" and kw_accept["curl_free"] == "accepted"

    # under-pole gotcha: same lat/lon for station and pole
    B_up = lib_B_enu(60.0, 0.0, RE, 60.0, 0.0, "divergence_free")
    summary["checks"]["V1_under_pole_nan"] = {
        "Ge_Gn_nan": bool(np.isnan(B_up[0]) or np.isnan(B_up[1])),
        "Gr_finite": bool(np.isfinite(B_up[2])),
        "note": "theta clipped to ~0.026 deg by dpclip; horizontal components NaN "
        "when a pole shares a station's lat/lon (0/0 tangent normalization). "
        "The NaN also poisons the CF below-sheet zero (0 * NaN = NaN), so the "
        "hard-coded Fukushima zero is NOT immune. ADAPTER CONTRACT: transfer.py "
        "must guarantee no SECS pole shares a station's (lat, lon); assert "
        "np.isfinite(A).all() after every build.",
    }

    # return order: pole 0.5 deg from station -> DF field predominantly radial
    B_near = lib_B_enu(60.0, 0.0, RE, 60.5, 0.0, "divergence_free")
    v1_order = abs(B_near[2]) > 2 * max(abs(B_near[0]), abs(B_near[1]))
    summary["checks"]["V1_return_order"] = {
        "B_enu_near_pole": B_near.tolist(),
        "radial_dominates": bool(v1_order),
        "reading": "(Ge, Gn, Gr) per docstring; order and units independently "
        "pinned by V4's analytic under-pole radial match",
    }
    v1_ok = v1_ok and v1_order
    summary["checks"]["V1_pass"] = bool(v1_ok)

    # ---- V2: realness / dtype / shape / finiteness -----------------------
    # Station grid offset off the pole grid (audit W-A fix, 2026-07-10): the
    # original integer grids shared exact (lat, lon) pairs with the poles, so
    # per V1 these matrices contained NaN -- which isrealobj/dtype/shape all
    # pass. Offset like V3, and check the finiteness this gate itself made a
    # contract requirement.
    lats = np.array([68.0, 69.0, 70.0, 71.0, 72.0]) + 0.37
    lons = np.array([-8.0, -4.0, 0.0, 4.0, 8.0]) + 0.53
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    plat, plon = np.meshgrid(
        np.arange(66.0, 75.0, 1.0), np.arange(-12.0, 13.0, 4.0), indexing="ij"
    )
    mats = {}
    for ct in ["divergence_free", "curl_free"]:
        mats[ct] = get_SECS_B_G_matrices(
            glat.ravel(), glon.ravel(), RE, plat.ravel(), plon.ravel(),
            current_type=ct, RI=RI,
        )
    v2_real = all(np.isrealobj(G) and G.dtype == np.float64 for M in mats.values() for G in M)
    v2_shape = all(
        G.shape == (glat.size, plat.size) for M in mats.values() for G in M
    )
    v2_finite = all(np.isfinite(G).all() for M in mats.values() for G in M)
    summary["checks"]["V2_pass"] = bool(v2_real and v2_shape and v2_finite)
    summary["checks"]["V2_detail"] = {"real_float64": bool(v2_real), "shapes_ok": bool(v2_shape),
                                      "all_finite": bool(v2_finite)}
    ok = ok and v2_real and v2_shape and v1_ok

    # ---- V3: CF ground block at every preset ----------------------------
    v3 = {}
    v3_ok = True
    for name, lat0 in LATITUDE_PRESETS.items():
        # station grid deliberately offset from the pole grid: coincident
        # (lat, lon) columns are NaN by the V1 gotcha, tested there, not here.
        sl = np.arange(lat0 - 4, lat0 + 4.1, 2.0) + 0.7
        so = np.arange(-8.0, 8.1, 4.0) + 1.3
        gla, glo = np.meshgrid(sl, so, indexing="ij")
        pla, plo = np.meshgrid(
            np.arange(lat0 - 6, lat0 + 6.1, 2.0), np.arange(-12.0, 12.1, 4.0), indexing="ij"
        )
        entry = {}
        for tag, kw in [("sing0", 0), ("sing50km", 50e3)]:
            Ge, Gn, Gr = get_SECS_B_G_matrices(
                gla.ravel(), glo.ravel(), RE, pla.ravel(), plo.ravel(),
                current_type="curl_free", RI=RI, singularity_limit=kw,
            )
            m = max(np.abs(Ge).max(), np.abs(Gn).max(), np.abs(Gr).max())
            entry[tag] = {"max_abs_T_per_A": float(m), "exactly_zero": bool(m == 0.0)}
            v3_ok = v3_ok and (m == 0.0)
        # above the sheet the pair field must be nonzero:
        Ge, Gn, Gr = get_SECS_B_G_matrices(
            gla.ravel(), glo.ravel(), RE + 500e3, pla.ravel(), plo.ravel(),
            current_type="curl_free", RI=RI,
        )
        entry["above_sheet_max_abs"] = float(
            max(np.abs(Ge).max(), np.abs(Gn).max(), np.abs(Gr).max())
        )
        v3[name] = entry
    summary["checks"]["V3_cf_ground_block"] = v3
    summary["checks"]["V3_pass"] = bool(v3_ok)
    summary["checks"]["V3_provenance"] = (
        "analytic by construction: secsy utils.py curl_free branch zeroes the "
        "below-sheet field (Ge_[below] *= 0) per Amm & Viljanen 1999 / Vanhamaki & "
        "Juusola 2020 eqs 2.13-2.14 - the full Fukushima pair's field. The library "
        "asserts the theorem; V5 confirms it independently."
    )
    ok = ok and v3_ok

    # ---- V4: DF independent Biot-Savart ---------------------------------
    line_selftest = _selftest_line_formula()
    summary["checks"]["V4_line_formula_selftest_relerr"] = float(line_selftest)

    pole = (60.0, 0.0)
    stations = [(58.0, 0.0), (60.0, 3.0), (63.0, -2.0), (60.5, 0.5), (40.0, 10.0)]
    P = np.array([ecef(la, lo, RE) for la, lo in stations])

    B_lib = np.array(
        [
            enu_to_ecef_vec(lib_B_enu(la, lo, RE, *pole, "divergence_free"), la, lo)
            for la, lo in stations
        ]
    )
    B_q = sheet_B(P, *pole, direction="phi")
    # empirical sign convention of the library's positive DF amplitude:
    sign = 1.0 if np.linalg.norm(B_q - B_lib) < np.linalg.norm(B_q + B_lib) else -1.0
    rel = np.linalg.norm(sign * B_q - B_lib, axis=1) / np.linalg.norm(B_lib, axis=1)
    # convergence at half resolution, first station:
    B_half = sheet_B(P[:1], *pole, direction="phi", n_theta=N_THETA // 2, n_phi=N_PHI // 2)
    conv = float(np.linalg.norm(B_half[0] - B_q[0]) / np.linalg.norm(B_q[0]))

    # analytic under-pole radial spot check (station beneath the pole):
    s = RE / RI
    Br_analytic = MU0 * 1.0 / (4 * np.pi * RE) * (1.0 / (1.0 - s) - 1.0)
    Br_lib = lib_B_enu(*pole, RE, *pole, "divergence_free")[2]
    underpole_rel = float(abs(Br_lib - Br_analytic) / abs(Br_analytic))

    # self-test tolerance 1e-5: the numeric reference (400k log-spaced segments)
    # carries the error, not the closed form.
    v4_ok = bool(rel.max() < 1e-2 and underpole_rel < 1e-3 and line_selftest < 1e-5)
    summary["checks"]["V4_df_quadrature"] = {
        "stations": stations,
        "rel_err_per_station": rel.tolist(),
        "df_sign_convention": (
            "+1 amplitude = eastward (counterclockwise around pole seen from above)"
            if sign > 0
            else "+1 amplitude = clockwise around pole seen from above (K = -cot/(4piRI) e_phi)"
        ),
        "half_resolution_shift": conv,
        "underpole_Br_analytic_rel_err": underpole_rel,
        "Br_underpole_T_per_A": float(Br_lib),
    }
    summary["checks"]["V4_pass"] = v4_ok
    ok = ok and v4_ok

    # ---- V5: Fukushima pair, independent --------------------------------
    pole_shell = ecef(*pole, RI)
    p_hat = ecef(*pole, 1.0)

    B_sheet = sheet_B(P, *pole, direction="theta")           # horizontal CF sheet
    B_line = semiinf_line_B(P, pole_shell, p_hat, -1.0)      # I0 flowing INWARD at pole
    B_ret = return_fac_B(P, *pole)                           # distributed outward FACs
    B_tot = B_sheet + B_line + B_ret
    scale = np.maximum(
        np.linalg.norm(B_sheet, axis=1),
        np.maximum(np.linalg.norm(B_line, axis=1), np.linalg.norm(B_ret, axis=1)),
    )
    resid = np.linalg.norm(B_tot, axis=1) / scale

    # above the sheet: the pair's field vs the library CF row
    above = (60.5, 0.5, RE + 500e3)
    Pa = np.array([ecef(above[0], above[1], above[2])])
    B_above = (
        sheet_B(Pa, *pole, direction="theta")
        + semiinf_line_B(Pa, pole_shell, p_hat, -1.0)
        + return_fac_B(Pa, *pole)
    )[0]
    B_above_lib = enu_to_ecef_vec(
        lib_B_enu(above[0], above[1], above[2], *pole, "curl_free"), above[0], above[1]
    )
    sign_cf = 1.0 if np.linalg.norm(B_above - B_above_lib) < np.linalg.norm(
        B_above + B_above_lib
    ) else -1.0
    above_rel = float(
        np.linalg.norm(sign_cf * B_above - B_above_lib) / np.linalg.norm(B_above_lib)
    )

    v5_ok = bool(resid.max() < 1e-2 and above_rel < 2e-2)
    summary["checks"]["V5_fukushima_pair"] = {
        "ground_residual_over_largest_term": resid.tolist(),
        "term_norms_station0_T": {
            "sheet": float(np.linalg.norm(B_sheet[0])),
            "pole_line_fac": float(np.linalg.norm(B_line[0])),
            "return_facs": float(np.linalg.norm(B_ret[0])),
            "sum": float(np.linalg.norm(B_tot[0])),
        },
        "above_sheet_vs_library_rel_err": above_rel,
        "cf_sign_convention": (
            "+1 amplitude = sheet current AWAY from pole, line FAC INTO the shell at the pole"
            if sign_cf > 0
            else "+1 amplitude = sheet current TOWARD pole (opposite of the tested pair)"
        ),
    }
    summary["checks"]["V5_pass"] = v5_ok
    ok = ok and v5_ok

    # ---- verdict + artifacts --------------------------------------------
    summary["GATE_V"] = "PASS" if ok else "HALT"
    summary["runtime_s"] = round(time.time() - t0, 1)

    def sha(path):
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()

    manifest = {
        "gate": "V",
        "date": time.strftime("%Y-%m-%d"),
        "secsy_version": pkg_version("secsy"),
        "secsy_sha": sha(REPO / "secsy"),
        "repo_commit": sha(REPO),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "RI_m": RI,
        "quadrature": {"n_theta": N_THETA, "n_phi": N_PHI},
        "preregistration": "ROADMAP.md Gate V (2026-07-01 restructure)",
    }

    (OUT / "gateV_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    np.savez(
        OUT / "gateV_fields.npz",
        stations=np.array(stations),
        B_lib_df=B_lib, B_quad_df=B_q * sign,
        B_sheet_cf=B_sheet, B_line_fac=B_line, B_return_facs=B_ret, B_total_cf=B_tot,
    )

    print(json.dumps(summary, indent=2))
    print(f"\nGATE V: {summary['GATE_V']}  ({summary['runtime_s']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
