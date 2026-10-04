"""Reproduce the 2026-09-28 A6 max-peaks=2 versus max-peaks=8 rescore.

This reruns only the already-seen B1/minipilot specimens with their original
seeds. It never reads or generates the powered runner's master seed and does
not alter the archived result files. Outputs are written beside the separate
rescore package under results/B_viability/peak-rule-rescoring/.
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VT = REPO / "viability-test"
RESULTS = REPO / "results" / "B_viability"
DEFAULT_OUT = RESULTS / "peak-rule-rescoring"
sys.path.insert(0, str(VT))

import metrics  # noqa: E402
import run_experiment_B as rb  # noqa: E402
import run_experiment_B2B6_d2 as r2  # noqa: E402
import run_experiment_B2B6_d8 as r8  # noqa: E402

RUNNERS = (rb, r2, r8)


def run_all():
    start = time.time()
    result = {}
    b1 = rb.run_b1()
    result["b1"] = {f"{r}|{d}|{s}": v for (r, d, s), v in b1.items()}
    result["minipilot_d3"] = rb.run_minipilot()
    result["b1_pilot_d2"] = r2.b1_pilot_cell_d2()
    result["minipilot_d2"] = r2.run_minipilot_d2()
    result["b1_pilot_d8"] = r8.b1_pilot_cell_d8()
    result["minipilot_d8"] = r8.run_minipilot_d8()
    print(f"specimen rescore completed in {time.time() - start:.0f}s", flush=True)
    return result


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda x: x.item()
                                 if hasattr(x, "item") else str(x)))


def _score_with_peak_cap(cap):
    original = {module: (module.delta_r_bar, module.p_sep) for module in RUNNERS}
    try:
        for module in RUNNERS:
            module.delta_r_bar = functools.partial(original[module][0], max_peaks=cap)
            module.p_sep = functools.partial(original[module][1], max_peaks=cap)
        return _jsonable(run_all())
    finally:
        for module, (dr, psep) in original.items():
            module.delta_r_bar, module.p_sep = dr, psep


def compare_cell(eight, two):
    # The two B1 pilot artifacts retain summary-level gap data only. The main
    # B1 and minipilot artifacts retain per-method trial scores.
    if "delta_r_bar" not in eight or "delta_r_bar" not in two:
        a = np.asarray(eight.get("gap_signed", []), dtype=float)
        b = np.asarray(two.get("gap_signed", []), dtype=float)
        return dict(
            comparison_level="summary_only",
            method_trial_arrays_available=False,
            gap_signed_mean_8=float(a.mean()) if a.size else None,
            gap_signed_mean_2=float(b.mean()) if b.size else None,
            gap_signed_changed_trials=(int(np.sum(~np.isclose(a, b)))
                                       if a.shape == b.shape and a.size else None),
            p_sep_rate_8=eight.get("p_sep_rate"),
            p_sep_rate_2=two.get("p_sep_rate"),
            gap_mean=[eight.get("gap_mean"), two.get("gap_mean")])
    result = {}
    for method in ("l2", "gibf", "mmv"):
        a = np.asarray(eight["delta_r_bar"][method], dtype=float)
        b = np.asarray(two["delta_r_bar"][method], dtype=float)
        result[method] = dict(
            dr8=float(a.mean()), dr2=float(b.mean()),
            changed_trials=int(np.sum(~np.isclose(a, b))),
            psep8=eight["p_sep_rate"][method], psep2=two["p_sep_rate"][method])
    result["gap_mean"] = [eight["gap_mean"], two["gap_mean"]]
    return result


def build_comparison(eight, two):
    comparison = {}
    for name, cells in eight.items():
        if name == "b1":
            for key, cell in cells.items():
                comparison[f"b1|{key}"] = compare_cell(cell, two[name][key])
        else:
            comparison[name] = compare_cell(cells, two[name])
    trial_comparable = {key: row for key, row in comparison.items()
                        if row.get("comparison_level") != "summary_only"}
    changed = {key: row for key, row in trial_comparable.items()
               if any(row[method]["changed_trials"]
                      for method in ("l2", "gibf", "mmv"))}
    summary_only = {key: row for key, row in comparison.items()
                    if row.get("comparison_level") == "summary_only"}
    return dict(
        comparison=comparison,
        changed_cells={"count": len(changed), "of_trial_comparable": len(trial_comparable),
                       "all_specimens": len(comparison), "keys": sorted(changed)},
        summary_only_cells={"count": len(summary_only), "keys": sorted(summary_only),
                            "method_level_change_count_available": False})


def verify_max8_reproduction(rescore):
    archived = {
        "b1": json.loads((RESULTS / "b1_results.json").read_text()),
        "minipilot_d3": json.loads((RESULTS / "minipilot_results.json").read_text()),
        "minipilot_d2": json.loads((RESULTS / "minipilot_d2_results.json").read_text()),
        "minipilot_d8": json.loads((RESULTS / "minipilot_d8_results.json").read_text()),
    }
    checks = {}
    for name, old in archived.items():
        new = rescore[name]
        if name == "b1":
            checks[name] = dict(
                cells=len(old),
                mismatched_cells=sorted(
                    key for key in old
                    if old[key]["delta_r_bar"] != new[key]["delta_r_bar"]))
        else:
            checks[name] = dict(match=(old["delta_r_bar"] == new["delta_r_bar"]))
    if any(check.get("mismatched_cells") or check.get("match") is False
           for check in checks.values()):
        raise RuntimeError(f"max-peaks=8 failed archived-result reproduction: {checks}")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--compare-existing", action="store_true",
                        help="build comparison.json from max_peaks_2/8.json without rerunning")
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    if out == RESULTS.resolve() or RESULTS.resolve() not in out.parents:
        parser.error("output must be a new directory below results/B_viability/")
    out.mkdir(parents=True, exist_ok=True)
    if args.compare_existing:
        peak8 = json.loads((out / "max_peaks_8.json").read_text())
        peak2 = json.loads((out / "max_peaks_2.json").read_text())
        checks = verify_max8_reproduction(peak8)
        comparison = build_comparison(peak8, peak2)
        comparison["max8_reproduction"] = checks
        (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps({"reproduction": checks,
                          "changed_cells": comparison["changed_cells"]}, indent=2))
        return
    results = {}
    for cap in (8, 2):
        print(f"scoring with max_peaks={cap}", flush=True)
        results[cap] = _score_with_peak_cap(cap)
        (out / f"max_peaks_{cap}.json").write_text(
            json.dumps(results[cap], indent=2) + "\n")
    checks = verify_max8_reproduction(results[8])
    comparison = build_comparison(results[8], results[2])
    comparison["max8_reproduction"] = checks
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({"reproduction": checks,
                      "changed_cells": comparison["changed_cells"]}, indent=2))


if __name__ == "__main__":
    main()
