"""report_prereg.py -- the reporting owed by the pre-registration (SPEC S.17 items 6-8).

Reads the sealed arrays and the completion run; runs no trials and re-decides
nothing. Writes tables (CSV / Markdown), figures (PNG) and each figure's
underlying data (CSV) to results/reports/.

Colours: one fixed assignment per method (L2 green, GIBF blue, MMV vermilion;
Okabe-Ito members, validated for CVD separation) plus a distinct marker per
method, so identity never rests on colour alone.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import run_experiment_B2B6_powered_d2 as P
import run_prereg_completion as C
from simulate import GRID_SHAPE

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
SEALED_B2 = RESULTS / "B_viability" / "powered_d2"
SEALED_B4 = RESULTS / "B_viability" / "b4_b6b"
DEFAULT_COMPLETION = RESULTS / "prereg_completion"
DEFAULT_OUT = RESULTS / "reports"

COLOR = {"l2": "#009E73", "gibf": "#0072B2", "mmv": "#D55E00"}
MARK = {"l2": "s", "gibf": "o", "mmv": "^"}
LABEL = {"l2": "L2", "gibf": "GIBF", "mmv": "MMV"}
GRID_DIAG = math.hypot(GRID_SHAPE[0] - 1, GRID_SHAPE[1] - 1)      # 35.78 cells
HEAD = "oracle_on_x1"
CONF_CELLS = C.RHO_CELLS                                            # six (phi, snr) at d=2
POWER_SEED = 20261018                                               # analysis RNG for the power table only
POWER_REPS = 4000


def _read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------- data views ---

def b2_headline_rows(completion_dir):
    """Every B2 cell's headline-arm summary: the sealed 30 plus the completion
    145. Returns a list of dicts tagged with their source."""
    rows = []
    sealed = _read(SEALED_B2 / "manifest.json")["cells"]
    for cid, c in sealed.items():
        if not cid.startswith("B2|"):
            continue
        s = c["arms"][HEAD]
        phi, snr, d = (float(cid.split("|")[i].split("=")[1]) for i in (1, 2, 3))
        rows.append(_row(cid, "sealed", phi, snr, int(d), s))
    for p in sorted(Path(completion_dir).glob("B2_phi*.json")):
        c = _read(p)
        s = c["arms"][HEAD]
        rows.append(_row(c["cell_id"], "completion", c["phi_deg"], c["snr_db"], int(c["d"]), s))
    return rows


def _row(cid, source, phi, snr, d, s):
    r = dict(cell_id=cid, source=source, phi=phi, snr=snr, d=d)
    for m in P.METHODS:
        r[f"dr_{m}"], (r[f"dr_{m}_lo"], r[f"dr_{m}_hi"]) = s[f"dr_{m}"]
        r[f"psep_{m}"], (r[f"psep_{m}_lo"], r[f"psep_{m}_hi"]) = s[f"psep_{m}"]
    r["gap"], (r["gap_lo"], r["gap_hi"]) = s["gap_gibf_minus_mmv"]
    return r


def sealed_arrays(phi, snr):
    cell = P.DataCell("B2", phi, snr, P.D_CONFIRM, P.N_SNAP, "win", ())
    return np.load(SEALED_B2 / f"{cell.file_stem}.npz"), cell


# ---------------------------------------------- 6(a) the continuous bracket ---

def _rel_adv(favored, other, idx):
    """Relative reduction (other - favored) / other of the means, with a
    percentile-bootstrap CI under shared resample indices."""
    f, o = np.asarray(favored, float), np.asarray(other, float)
    est = (o.mean() - f.mean()) / o.mean()
    bf, bo = f[idx].mean(axis=1), o[idx].mean(axis=1)
    return est, P.percentile_ci((bo - bf) / bo)


def bracket_table():
    """Per confirmatory cell, both directions, at the three stored peak
    thresholds: the relative advantage with its CI, and whether the Delta r-bar
    arm of the rule (>=20% lower AND strictly non-overlapping per-method CIs)
    holds. The stored arrays carry three thresholds (5%, 10%, 20%), not a
    continuum, so the 'range' is those three points."""
    out = []
    for phi, snr in CONF_CELLS:
        z, cell = sealed_arrays(phi, snr)
        n = len(z[f"{HEAD}__dr_gibf"])
        idx = P.boot_indices(n, P.boot_rng(f"report-bracket|{cell.cell_id}"))
        for tag, thr in (("05", 0.05), ("", 0.10), ("20", 0.20)):
            sfx = f"_{tag}" if tag else ""
            g = z[f"{HEAD}__dr{sfx}_gibf"]
            m = z[f"{HEAD}__dr{sfx}_mmv"]
            for fav, oth, fv, ov in (("gibf", "mmv", g, m), ("mmv", "gibf", m, g)):
                est, ci = _rel_adv(fv, ov, idx)
                fm, fci = P.mean_ci(fv, idx)
                om, oci = P.mean_ci(ov, idx)
                out.append(dict(
                    phi=phi, snr=snr, threshold=thr, favored=fav,
                    rel_advantage=est, rel_ci_lo=ci[0], rel_ci_hi=ci[1],
                    effect_ge_20pct=bool(est >= P.WIN_REL_THRESHOLD),
                    ci_non_overlap=bool(P.ci_non_overlap(fci, oci)),
                    delta_r_arm_holds=bool(est >= P.WIN_REL_THRESHOLD
                                           and P.ci_non_overlap(fci, oci))))
    return out


# --------------------------------------------- 6(b) penalties and partition ---

def penalty_table():
    """Delta r-bar under other unmatched-source penalties, reconstructed from
    the stored found-source sums and counts, plus the pinned-penalty identity
    check and the miss / correct / mislocated partition."""
    out = []
    for phi, snr in CONF_CELLS:
        z, cell = sealed_arrays(phi, snr)
        for m in P.METHODS:
            s = z[f"{HEAD}__matched_err_sum_{m}"]
            nm = z[f"{HEAD}__matched_n_{m}"]
            dr = z[f"{HEAD}__dr_{m}"]
            outc = z[f"{HEAD}__outcome_{m}"]
            ident = float(np.max(np.abs((s + (2 - nm) * GRID_DIAG) / 2 - dr)))
            row = dict(phi=phi, snr=snr, method=m,
                       pinned_penalty_identity_max_abs_err=ident,
                       n_correct=int(np.sum(outc == "correct")),
                       n_mislocated=int(np.sum(outc == "mislocated")),
                       n_miss=int(np.sum(outc == "miss")))
            for lab, pen in (("0", 0.0), ("1x_diag", GRID_DIAG), ("2x_diag", 2 * GRID_DIAG)):
                row[f"dr_penalty_{lab}"] = float(np.mean((s + (2 - nm) * pen) / 2))
            corr = outc == "correct"
            e = z[f"{HEAD}__matched_err_{m}"]
            row["correct_trial_error_mean"] = float(np.mean(e[corr])) if corr.any() else None
            out.append(row)
    return out


# ----------------------------------------------------------- 6(c) K-hat ---

def khat_tables():
    man = _read(SEALED_B2 / "manifest.json")["cells"]
    rows = []
    for cid, c in man.items():
        if cid.startswith("B6a"):
            rows.append(dict(cell_id=cid, khat_mdl=json.dumps(c.get("khat_mdl")),
                             khat_breakdown_on=json.dumps(c.get("khat_breakdown_on")),
                             b6a_paired_diff_on=json.dumps(c.get("b6a_paired_diff_on")),
                             mdl_lookup=json.dumps(c.get("mdl_lookup")),
                             aic_lookup=json.dumps(c.get("aic_lookup")),
                             forced_k_sweep=json.dumps(c.get("forced_k_sweep"))))
    return rows


# --------------------------------------- 6(d) the rule's operating characteristic ---

def rule_power(fav, oth, n_reps=POWER_REPS, seed=POWER_SEED):
    """Smallest true relative advantage at which the Delta r-bar arm (>=20%
    point estimate AND strict per-method CI non-overlap) fires with probability
    >= 0.8, given this cell's observed per-trial sd, correlation and n. Normal
    approximation to the sample means; the CI half-width is 1.96 x the observed
    standard error. Returns (r80, power_at_20pct, grid)."""
    fav, oth = np.asarray(fav, float), np.asarray(oth, float)
    n = len(oth)
    se_f, se_o = fav.std(ddof=1) / math.sqrt(n), oth.std(ddof=1) / math.sqrt(n)
    rho = float(np.corrcoef(fav, oth)[0, 1])
    mu_o = float(oth.mean())
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_reps, 2))
    zf = z[:, 0]
    zo = rho * z[:, 0] + math.sqrt(max(0.0, 1 - rho * rho)) * z[:, 1]

    def power(r):
        mf = mu_o * (1 - r) + se_f * zf
        mo = mu_o + se_o * zo
        eff = (mo - mf) / mo >= P.WIN_REL_THRESHOLD
        sep = (mo - 1.96 * se_o) > (mf + 1.96 * se_f)
        return float(np.mean(eff & sep))

    grid = np.round(np.arange(0.0, 0.701, 0.01), 2)
    pw = np.array([power(r) for r in grid])
    hit = np.nonzero(pw >= 0.8)[0]
    return (float(grid[hit[0]]) if len(hit) else None), power(0.20), (grid, pw)


def power_table():
    out = []
    for phi, snr in CONF_CELLS:
        z, _ = sealed_arrays(phi, snr)
        g, m = z[f"{HEAD}__dr_gibf"], z[f"{HEAD}__dr_mmv"]
        for fav, oth, fv, ov in (("gibf", "mmv", g, m), ("mmv", "gibf", m, g)):
            r80, p20, _ = rule_power(fv, ov)
            out.append(dict(phi=phi, snr=snr, favored=fav, n=len(ov),
                            observed_other_mean=float(np.mean(ov)),
                            smallest_rel_effect_at_power_0p8=r80,
                            power_when_true_effect_is_exactly_20pct=p20))
    return out


# ---------------------------------------------------------------- figures ---

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.5})
    return plt


def fig_decision(rows, out):
    plt = _plt()
    d2 = [r for r in rows if r["d"] == P.D_CONFIRM]
    snrs = sorted({r["snr"] for r in d2})
    fig, ax = plt.subplots(2, len(snrs), figsize=(2.6 * len(snrs), 5.2), sharex=True, squeeze=False)
    data = []
    for j, snr in enumerate(snrs):
        cells = sorted([r for r in d2 if r["snr"] == snr], key=lambda r: r["phi"])
        phis = [r["phi"] for r in cells]
        a = ax[0][j]
        a.axhline(0, color="0.4", lw=0.8)
        for r in cells:
            conf = r["source"] == "sealed"
            a.errorbar(r["phi"], r["gap"], yerr=[[r["gap"] - r["gap_lo"]], [r["gap_hi"] - r["gap"]]],
                       fmt="o", ms=5, color="#444", mfc="#444" if conf else "white", capsize=2, lw=1)
        a.plot(phis, [r["gap"] for r in cells], color="#444", lw=0.8)
        a.set_title(f"{snr:g} dB")
        if j == 0:
            a.set_ylabel("GIBF minus MMV, mean Δr̄ (cells)\n(negative favours GIBF)")
        b = ax[1][j]
        for m in P.METHODS:
            b.errorbar(phis, [r[f"psep_{m}"] for r in cells],
                       yerr=[[r[f"psep_{m}"] - r[f"psep_{m}_lo"] for r in cells],
                             [r[f"psep_{m}_hi"] - r[f"psep_{m}"] for r in cells]],
                       color=COLOR[m], marker=MARK[m], ms=4, lw=1, capsize=2, label=LABEL[m])
        b.set_xlabel("inter-source phase φ (deg)")
        b.set_ylim(-0.02, 1.02)
        if j == 0:
            b.set_ylabel("P_sep at d = 2")
        for r in cells:
            data.append(dict(snr=snr, phi=r["phi"], source=r["source"], gap=r["gap"],
                             gap_lo=r["gap_lo"], gap_hi=r["gap_hi"],
                             **{f"psep_{m}": r[f"psep_{m}"] for m in P.METHODS}))
    ax[1][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Signed gap and P_sep against φ at d = 2 (filled: sealed confirmatory cells; open: completion)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_i_decision_phi.png", dpi=160)
    plt.close(fig)
    _write_csv(out / "fig_i_decision_phi.csv", data)


def fig_b1_floor(completion_dir, out):
    plt = _plt()
    cells = [_read(p) for p in sorted(Path(completion_dir).glob("B1full_*.json"))]
    if not cells:
        return
    snrs = sorted({c["snr_db"] for c in cells})
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    data = []
    for k, red in enumerate(("on", "off")):
        for snr in snrs:
            cs = sorted([c for c in cells if c["snr_db"] == snr], key=lambda c: c["d"])
            ds = [c["d"] for c in cs]
            g = [c["reduction"][red]["gap_gibf_minus_mmv"][0] for c in cs]
            ax[k].plot(ds, g, marker="o", ms=3, lw=1, label=f"{snr:g} dB")
            for c in cs:
                data.append(dict(reduction=red, snr=snr, d=c["d"],
                                 gap=c["reduction"][red]["gap_gibf_minus_mmv"][0],
                                 lo=c["reduction"][red]["gap_gibf_minus_mmv"][1][0],
                                 hi=c["reduction"][red]["gap_gibf_minus_mmv"][1][1]))
        ax[k].axhline(0, color="0.4", lw=0.8)
        ax[k].set_title(f"B1 incoherent floor, reduction {red.upper()}")
        ax[k].set_xlabel("separation d (grid cells)")
    ax[0].set_ylabel("GIBF minus MMV, mean Δr̄ (cells)")
    ax[1].legend(frameon=False, fontsize=7, title="SNR")
    fig.tight_layout()
    fig.savefig(out / "fig_i_b1_floor.png", dpi=160)
    plt.close(fig)
    _write_csv(out / "fig_i_b1_floor.csv", data)


def fig_b4(out):
    plt = _plt()
    man = _read(SEALED_B4 / "manifest.json")["cells"]
    kernels = ("real", "lit", "w1", "w2", "w5")
    shade = {"real": "#555", "lit": "#9ecae1", "w1": "#6baed6", "w2": "#3182bd", "w5": "#08306b"}
    snrs = sorted({float(c.split("|")[1].split("=")[1]) for c in man if c.startswith("B4|")})
    fig, ax = plt.subplots(1, len(snrs), figsize=(2.6 * len(snrs), 3.0), sharey=True)
    data = []
    for j, snr in enumerate(snrs):
        for k in kernels:
            ds, ps = [], []
            for cid, c in sorted(man.items()):
                if cid.startswith(f"B4|snr={snr:g}|"):
                    ds.append(int(cid.split("|")[2].split("=")[1]))
                    ps.append(c["arms"][f"{k}_on"]["psep"][0])
                    data.append(dict(snr=snr, kernel=k, d=ds[-1], psep=ps[-1]))
            o = np.argsort(ds)
            ax[j].plot(np.array(ds)[o], np.array(ps)[o], marker="o", ms=3, lw=1,
                       color=shade[k], label=k)
        ax[j].set_title(f"{snr:g} dB")
        ax[j].set_xlabel("d (cells)")
    ax[0].set_ylabel("GIBF P_sep, reduction ON")
    ax[-1].legend(frameon=False, fontsize=7, title="kernel")
    fig.tight_layout()
    fig.savefig(out / "fig_ii_b4_phantom_phase.png", dpi=160)
    plt.close(fig)
    _write_csv(out / "fig_ii_b4_phantom_phase.csv", data)


def fig_b6(out):
    plt = _plt()
    sealed = _read(SEALED_B2 / "manifest.json")["cells"]
    b6b = {k: v for k, v in _read(SEALED_B4 / "manifest.json")["cells"].items() if k.startswith("B6b|")}
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3))
    data = []
    variants = [(k.split("|")[1].split("=")[1], v) for k, v in b6b.items()]
    w = 0.26
    for i, m in enumerate(P.METHODS):
        xs = np.arange(len(variants)) + (i - 1) * w
        ys = [v["arms"][m]["dr"][0] for _, v in variants]
        lo = [v["arms"][m]["dr"][1][0] for _, v in variants]
        hi = [v["arms"][m]["dr"][1][1] for _, v in variants]
        ax[1].bar(xs, ys, w * 0.9, color=COLOR[m], label=LABEL[m],
                  yerr=[np.array(ys) - lo, np.array(hi) - ys], capsize=2)
        for (n_, _), y, l_, h_ in zip(variants, ys, lo, hi):
            data.append(dict(panel="b6b", variant=n_, method=m, dr=y, lo=l_, hi=h_))
    ax[1].set_xticks(range(len(variants)), [n for n, _ in variants], fontsize=8)
    ax[1].set_ylabel("mean Δr̄ (cells)")
    ax[1].set_title("B6b support jitter (phi 90, 5 dB, d 2)")
    ax[1].legend(frameon=False, fontsize=8)
    b6a = next((v for k, v in sealed.items() if k.startswith("B6a|")), None)
    if b6a and "khat_mdl" in b6a:
        kh = b6a["khat_mdl"]
        hist = kh.get("hist") or kh.get("counts") or kh
        if isinstance(hist, dict):
            ks = sorted(int(k) for k in hist if str(k).lstrip("-").isdigit())
            ax[0].bar(ks, [hist[str(k)] if str(k) in hist else hist[k] for k in ks], color="#777")
            for k in ks:
                data.append(dict(panel="b6a_khat", k=k, count=hist[str(k)] if str(k) in hist else hist[k]))
    ax[0].set_xticks(range(1, 7))
    ax[0].set_xlim(0.4, 6.6)
    ax[0].set_xlabel("MDL K-hat (true K = 2)")
    ax[0].set_ylabel("trials")
    ax[0].set_title("B6a K-hat breakdown (phi 90, 5 dB, n_snap 64)")
    fig.tight_layout()
    fig.savefig(out / "fig_iii_b6_panel.png", dpi=160)
    plt.close(fig)
    _write_csv(out / "fig_iii_b6_panel.csv", data)


def fig_regularization(out):
    plt = _plt()
    man = _read(SEALED_B2 / "manifest.json")["cells"]
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.2), sharey=True)
    data = []
    for j, phi in enumerate(sorted((P.PHI_WIN_DEG,) + P.PHI_NULL_DEGS)):
        for snr, ls in zip(P.SNR_PAIR, ("-", "--")):
            c = man[f"B2|phi={phi:g}|snr={snr:g}|d=2|nsnap=64"]
            pts = c["sensitivity_panel"]["points"]
            x = [p["eps_mult"] for p in pts]
            y = [p["gap_mean"] for p in pts]
            lo = [p["gap_ci"][0] for p in pts]
            hi = [p["gap_ci"][1] for p in pts]
            ax[j].errorbar(x, y, yerr=[np.array(y) - lo, np.array(hi) - y], marker="o", ms=4,
                           lw=1, ls=ls, color="#444", capsize=2, label=f"{snr:g} dB")
            for a_, b_, c_, d_ in zip(x, y, lo, hi):
                data.append(dict(phi=phi, snr=snr, eps_mult=a_, gap=b_, lo=c_, hi=d_))
        ax[j].axhline(0, color="0.4", lw=0.8)
        ax[j].set_xscale("log")
        ax[j].set_title(f"φ = {phi:g}°")
        ax[j].set_xlabel("ε multiple of the rule (x1 = Suzuki)")
    ax[0].set_ylabel("GIBF minus MMV, mean Δr̄ (cells)")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_iv_regularization.png", dpi=160)
    plt.close(fig)
    _write_csv(out / "fig_iv_regularization.csv", data)


def fig_b3(completion_dir, out):
    plt = _plt()
    cells = [_read(p) for p in sorted(Path(completion_dir).glob("B3full_*.json"))]
    if not cells:
        return
    snrs = sorted({c["snr_db"] for c in cells})
    nsn = sorted({c["n_snap"] for c in cells})
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
    data = []
    for k, crit in enumerate(("mdl", "aic")):
        z = np.full((len(snrs), len(nsn)), np.nan)
        for c in cells:
            z[snrs.index(c["snr_db"]), nsn.index(c["n_snap"])] = c[f"{crit}_error_rate"][0]
            data.append(dict(criterion=crit, snr=c["snr_db"], n_snap=c["n_snap"],
                             error_rate=c[f"{crit}_error_rate"][0],
                             lo=c[f"{crit}_error_rate"][1][0], hi=c[f"{crit}_error_rate"][1][1]))
        im = ax[k].imshow(z, origin="lower", vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax[k].set_xticks(range(len(nsn)), nsn)
        ax[k].set_yticks(range(len(snrs)), [f"{s:g}" for s in snrs])
        ax[k].set_xlabel("snapshots")
        ax[k].set_title(f"{crit.upper()} K-hat error rate (truth 2)")
        ax[k].grid(False)
    ax[0].set_ylabel("SNR (dB)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(out / "fig_v_b3_khat.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    _write_csv(out / "fig_v_b3_khat.csv", data)


# ------------------------------------------------- 8 kill-clause facts ---

def kill_table(rows, out):
    """Facts only. Does not say whether the clause is met."""
    adj = _read(SEALED_B2 / "adjudication.json")
    b4x = _read(SEALED_B4 / "manifest.json")["extras"].get("d_min_psep_ge_0.8", {})
    L = ["# Kill-clause evidence (facts only; the reading is Strider's)", "",
         "ROADMAP §5-D2: method-wide abandonment needs all four of (1) D2 = drop, (2) no advantage at "
         "nonzero φ in the coherent regime, (3) no B6 advantage under misspecification, (4) B4 attributing "
         "the acoustic-GIBF gap to absent kernel phase diversity. What the records say about each part:", "",
         "## Part 1: D2 outcome (sealed adjudication)", "",
         f"- WIN rule met: {adj['win']['met']}", f"- LOSS rule met: {adj['loss']['met']} "
         f"(at φ = {adj['loss'].get('met_at_phi')})", f"- B6 rule met: {adj['b6']['met']} "
         f"(winner {adj['b6'].get('winner')}, fragile {adj['b6'].get('fragile')})", "",
         "## Part 2: signed gap (GIBF minus MMV, mean Δr̄) at d = 2, nonzero φ, by SNR",
         "Negative favours GIBF. 'sealed' cells are confirmatory; 'completion' cells are descriptive.", "",
         "| φ | SNR | gap | 95% CI | source |", "|---|---|---|---|---|"]
    for r in sorted([r for r in rows if r["d"] == P.D_CONFIRM and r["phi"] > 0],
                    key=lambda r: (r["phi"], r["snr"])):
        L.append(f"| {r['phi']:g} | {r['snr']:g} | {r['gap']:.3f} | [{r['gap_lo']:.3f}, {r['gap_hi']:.3f}] "
                 f"| {r['source']} |")
    L += ["", "## Part 3: B6 (sealed)", "", f"- {json.dumps(adj['b6'], default=str)[:1500]}", "",
          "## Part 4: B4 d_min (smallest d with P_sep >= 0.8), GIBF, by SNR / kernel / reduction", "",
          f"```\n{json.dumps(b4x, indent=1, default=str)[:3000]}\n```", ""]
    (out / "kill_clause_evidence.md").write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------- main ---

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--completion-dir", default=str(DEFAULT_COMPLETION))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--only-sealed", action="store_true",
                    help="run the sealed-array parts only (6, kill table, B4/B6/regularization figures)")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    for sealed in ("B_viability", "A_flr_coherence", "V_kernel_validation"):
        if (RESULTS / sealed).resolve() in out.resolve().parents or (RESULTS / sealed).resolve() == out.resolve():
            sys.exit("reports never go into a sealed directory")
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(out / "table_bracket.csv", bracket_table())
    _write_csv(out / "table_penalties_partition.csv", penalty_table())
    _write_csv(out / "table_khat_b6a.csv", khat_tables())
    _write_csv(out / "table_rule_power.csv", power_table())
    fig_b4(out)
    fig_b6(out)
    fig_regularization(out)
    rows = b2_headline_rows(args.completion_dir) if not args.only_sealed else \
        b2_headline_rows(args.completion_dir)
    kill_table(rows, out)
    if not args.only_sealed:
        fig_decision(rows, out)
        fig_b1_floor(args.completion_dir, out)
        fig_b3(args.completion_dir, out)
    print(f"reports written to {out}")


if __name__ == "__main__":
    main()
