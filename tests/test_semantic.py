"""Semantic checks the 2026-07-07 repo audit had to do by hand (audit A4).

Three classes of drift these catch, all instances of "definition, code, and
archive disagreeing while every structural check passes":
  1. committed result artifacts no longer reproducing the numbers the
     documents quote (recompute from npz, compare to manifest/CSV);
  2. a gate summary claiming PASS while an individual check regressed;
  3. documents citing repo paths that do not exist.

Run: conda run -n mhd-env python -m pytest tests/ -q   (from the repo root)
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
TIER1 = ROOT / "results" / "A_flr_coherence"
GATEV = ROOT / "results" / "V_kernel_validation"

# Paths cited in docs that are allowed to not exist: historical quotes in the
# append-only §8 log (the audit entry quotes the broken pointer it found).
CITED_PATH_ALLOWLIST = {"viability-test/PLAN.md"}


def test_tier1_pins_recompute_from_npz():
    """The pinned co-primary thresholds must be recomputable from the
    committed npz via the §8-iv procedure (kappa-crossing interpolation on
    the |rho|=0.95, N=64 curves)."""
    d = np.load(TIER1 / "tier1_results.npz")
    manifest = json.loads((TIER1 / "manifest.json").read_text())
    phi = d["phi_deg"]
    kap = d["kappa_rho95_N64"]
    imf = d["imfrac_rho95_N64"]
    for band_str, pin in manifest["pins"].items():
        band = float(band_str)
        idx = int(np.argmax(kap >= band))
        assert idx > 0, "kappa curve should not start above the band"
        f = (band - kap[idx - 1]) / (kap[idx] - kap[idx - 1])
        phi_eff = phi[idx - 1] + f * (phi[idx] - phi[idx - 1])
        imf_at = float(np.interp(phi_eff, phi, imf))
        assert phi_eff == pytest.approx(pin["phi_eff_deg"], abs=1e-9)
        assert imf_at == pytest.approx(pin["imfrac_threshold"], abs=1e-9)


def _csv_floor_means():
    means = {}
    for line in (TIER1 / "tier1_summary.csv").read_text().splitlines()[1:]:
        parts = line.split(",")
        if parts[0] in ("kappa_floor", "imfrac_floor"):
            means[(parts[0].split("_")[0], parts[1], int(parts[2]))] = float(parts[3])
    return means


def test_floor_distributions_match_csv_means():
    """The A2 per-trial floor distributions must average to the archived
    tier1_summary.csv floor values (they are the same trials, replayed)."""
    d = np.load(TIER1 / "floor_distributions.npz")
    means = _csv_floor_means()
    assert len(means) == 8
    for (stat, label, n), want in means.items():
        key = f"{'kappa' if stat == 'kappa' else 'imfrac'}_{label}_N{n}"
        assert key in d.files
        assert float(d[key].mean()) == pytest.approx(want, abs=5e-5)
    # the A1 gap-conditioning test consumes per-trial gaps: they must exist
    for label in ("incoherent", "phi0_coherent"):
        for n in (64, 1024):
            assert f"gap_{label}_N{n}" in d.files


def test_gateV_summary_internally_consistent():
    """GATE_V must be PASS and every individual *_pass check true — a summary
    that says PASS over a failed sub-check is the audit's W-A in disguise."""
    s = json.loads((GATEV / "gateV_summary.json").read_text())
    assert s["GATE_V"] == "PASS"
    pass_keys = [k for k in s["checks"] if k.endswith("_pass")]
    assert len(pass_keys) >= 5  # V1..V5
    for k in pass_keys:
        assert s["checks"][k] is True, f"{k} failed while GATE_V says PASS"
    assert s["checks"]["V2_detail"]["all_finite"] is True


def test_doc_cited_paths_exist():
    """Every results/ or viability-test/ file path cited in the governing
    documents must exist on disk (audit W-E class)."""
    docs = [ROOT / "ROADMAP.md", ROOT / "EXPERIMENT_CARD_A.md",
            ROOT / "README.md", ROOT / "viability-test" / "SPEC_experiment_B.md"]
    pat = re.compile(r"(?:results|viability-test)/[A-Za-z0-9_./-]+")
    missing = []
    for doc in docs:
        for m in pat.findall(doc.read_text(encoding="utf-8")):
            path = m.rstrip(".,;:)")
            if path.endswith("/") or "." not in Path(path).name:
                continue  # directories / bare names: not file citations
            if path in CITED_PATH_ALLOWLIST:
                continue
            if not (ROOT / path).exists():
                missing.append(f"{doc.name}: {path}")
    assert not missing, f"cited paths not on disk: {missing}"
