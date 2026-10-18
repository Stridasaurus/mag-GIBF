"""tests/test_report_prereg.py -- the owed reporting (SPEC S.17 items 6-8) recomputes
from sealed arrays and reproduces the sealed numbers exactly where it can."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "viability-test"))

import report_prereg as RP  # noqa: E402
import run_experiment_B2B6_powered_d2 as P  # noqa: E402


def test_pinned_penalty_reconstruction_is_exact():
    rows = RP.penalty_table()
    assert len(rows) == 18
    assert max(r["pinned_penalty_identity_max_abs_err"] for r in rows) < 1e-9


def test_published_headline_means_are_recomputed_exactly():
    man = json.loads((RP.SEALED_B2 / "manifest.json").read_text(encoding="utf-8"))["cells"]
    for r in RP.penalty_table():
        cid = f"B2|phi={r['phi']:g}|snr={r['snr']:g}|d=2|nsnap=64"
        pub = man[cid]["arms"][RP.HEAD][f"dr_{r['method']}"][0]
        assert r["dr_penalty_1x_diag"] == pytest.approx(pub, abs=1e-9)


def test_bracket_table_shape_and_consistency():
    rows = RP.bracket_table()
    assert len(rows) == 6 * 3 * 2
    for r in rows:
        assert r["rel_ci_lo"] <= r["rel_advantage"] <= r["rel_ci_hi"] + 1e-12
        assert r["delta_r_arm_holds"] == (r["effect_ge_20pct"] and r["ci_non_overlap"])


def test_rule_power_behaves():
    rng = np.random.default_rng(5)
    oth = rng.normal(10, 3, 10_000)
    fav_same = oth + rng.normal(0, 1, 10_000)                 # no effect
    r80, p20, _ = RP.rule_power(fav_same, oth, n_reps=800)
    assert r80 is not None and 0.2 <= r80 <= 0.3 and 0.3 < p20 < 0.7
    r80b, p20b, (grid, pw) = RP.rule_power(oth * 0.5, oth, n_reps=800)
    assert pw[0] < 0.05 and pw[-1] > 0.95 and np.all(np.diff(pw) >= -0.05)


def test_reports_refuse_a_sealed_directory():
    with pytest.raises(SystemExit):
        RP.main(["--only-sealed", "--out-dir", str(RP.RESULTS / "B_viability" / "x")])
