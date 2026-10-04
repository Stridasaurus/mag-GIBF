# mag-GIBF

Viability test for **SECS-GIBF**: applying Generalized Inverse Beamforming (GIBF), an
eigenmode-per-mode sparse imaging method originally developed for aeroacoustics
(Suzuki 2011), to magnetometer arrays imaging ionospheric current systems via the
SECS (Spherical Elementary Current Systems) source basis.

**Start here, in this order:**

1. This README (repo orientation, environment setup)
2. [`BUILD_BRIEF.md`](BUILD_BRIEF.md) — **onboarding brief**: current build state, the two
   non-negotiable invariants, the Gate-V/`transfer.py` adapter contract, the audit-A1 pins,
   and what not to do. Reconciles the sources below into one place. Replaces the archived
   `docs/archive/GIBF_viability_BUILD_BRIEF.md`.
3. [`ROADMAP.md`](ROADMAP.md) — **canonical, live source of truth** for the research
   state, decision tree, and build order. If anything below and the roadmap disagree,
   the roadmap wins.
4. [`EXPERIMENT_CARD_A.md`](EXPERIMENT_CARD_A.md) and
   [`viability-test/SPEC_experiment_B.md`](viability-test/SPEC_experiment_B.md) —
   the frozen experiment designs, once you need that level of detail.

## Read this before opening any older doc or notes

The project terminology was **renamed 2026-07-01** after peer review. Anything dated
before that — old shared notes, screenshots, prior conversations — uses the *old*
names. `ROADMAP.md` §0 has the full rename map; the short version:

| Old name | Current name |
|---|---|
| Card C / Experiment C (FLR source-coherence) | **Card A / Experiment A** |
| Card A / Experiment A (CF/DF identifiability) | **Validation Gate V** |
| Decision node D1 | retired |
| Card B5 | retired |
| — | Card B6 added |

## Where the project stands right now

*(Refreshed 2026-10-03. `ROADMAP.md` §9 is the current frontier; `ROADMAP.md` §8 and
`viability-test/SPEC_experiment_B.md` §S.14 preserve the decision record.)*

- **Validation Gate V, Card A Tier 1 and Tier 2, `transfer.py`, and the Experiment-B
  pilot stack** remain complete as recorded in the historical entries below.
- **Powered Experiment B at `d=2`:** Strider selected 10,000 trials per cell, replacing
  the historical 1,635 estimate. The candidate runner implements the recovered §S.14
  decisions: symmetric win/null d-axes, peak/threshold scoring, stop-reason and
  no-early-exit comparisons, B6a `n_snap=64` plus descriptive `n_snap=128`, fixed-K/AIC
  lookup, and miss/error decomposition. The 64-snapshot B6a row is intentionally
  overspecified; the 128-snapshot row is descriptive and under-specified.
- **Current implementation state:** the powered runner and its SPEC/status repairs are
  on the `b2b6-powered-runner-d2` candidate branch. The confirmatory/master-seed run
  has **not** been executed and is blocked by two unresolved design details in SPEC
  §S.14: the exact off-grid offset/cells and whether the descriptive `d=1` regularization
  panel binds a d-min verdict. Do not infer either choice.
- **Next valid action:** finish review and land the candidate, then resolve those two
  pre-run details. Only after the design is complete and the runner is committed clean
  may the master-seed experiment run. The old Shane-Gilbertie Card-A assignment in
  `handoff.md` is historical.

Full experiment-by-experiment detail: `ROADMAP.md` §8 (append-only log) and §9
(status). Do not re-derive from this summary — it is intentionally short.

## Repo layout

| Path | What it is |
|---|---|
| `ROADMAP.md` | Canonical research state: decision tree, invariants, glossary, build order, experiment log |
| `handoff.md` | Historical: the Experiment-B-kickoff task handoff (2026-07-12). Superseded as a live assignment doc — see README "Where the project stands" above for the current frontier. |
| `EXPERIMENT_CARD_A.md` | Card A (FLR source-coherence) experiment design |
| `viability-test/` | The actual build — validation/experiment scripts and their frozen specs |
| `viability-test/gateV_kernel_validation.py` | Gate V: runs the `secsy` CF-pair probe + realness/DF checks |
| `viability-test/tier1_flr_coherence.py` | Card A Tier 1: κ/φ calibration curves, threshold pinning |
| `viability-test/floor_distributions.py` | Audit A2: per-trial floor-control distributions (bit-exact Tier-1 replay) |
| `viability-test/transfer.py` | DF-only real transfer matrix A (Gate-V-pinned secsy adapter; coincidence guard, finiteness assert, row-norm, cache) |
| `tests/` | Semantic pytest gate (audit A4): artifacts recompute, gate consistency, doc-path existence |
| `pytest.ini` | Scopes `pytest` collection to `tests/` — without it a bare `pytest -q` also collects the `secsy` submodule's own `test_scripts/`, which import `lompe` and abort collection |
| `viability-test/SPEC_experiment_B.md` | Frozen Experiment B design (B1–B4, B6) |
| `results/` | Committed output artifacts (figures, summaries) from the scripts above — reproducible, not hand-edited |
| `secsy/`, `mhd-ibf-reconstruction/`, `magnetometer-time-series-simulator/` | Git submodules. Only `secsy` is currently load-bearing (editable-installed into the conda env, pinned to a specific commit); the other two are still scaffolding |
| `BUILD_BRIEF.md` | **Current onboarding brief** (2026-07-12): reconciles ROADMAP + SPEC §S.9 + Gate-V pins + audit-A1 + the `transfer.py` contract into one self-consistent read. Replaces the archived brief below |
| `docs/archive/GIBF_viability_BUILD_BRIEF.md` | **Archived 2026-07-11.** Claude's original build spec (June 2026), superseded by `BUILD_BRIEF.md` + `ROADMAP.md` + `SPEC_experiment_B.md`; kept for history only. Uses pre-rename terminology — its own header explains the mapping |
| `legacy_mhd_notebook.ipynb` | Legacy MHD notebook (pre-restructure) — the prior-practice MDL-13 cutoff cited by ROADMAP §8-viii lives here; renamed 2026-07-10 from `mhd_notebook (1).ipynb` |

## Environment setup

```bash
# 1. Clone with submodules (or run the update command below if already cloned)
git clone --recurse-submodules https://github.com/Stridasaurus/mag-GIBF.git
cd mag-GIBF
git submodule update --init --recursive   # if you cloned without --recurse-submodules

# 2. Create the conda environment (name must be exactly mhd-env — .envrc expects it)
conda create -n mhd-env python=3.11 numpy scipy matplotlib pytest -y
conda activate mhd-env

# 3. Install the pinned secsy submodule editable
pip install -e ./secsy
```

If you use `direnv`, the repo's `.envrc` auto-activates `mhd-env` on `cd` (run `direnv
allow` once). It's OS-conditional (macOS vs Windows paths), so it works unmodified on
either.

## Smoke-test your setup

Both scripts below are deterministic/seeded — your output should match what's already
committed in `results/`:

```bash
conda run -n mhd-env python -m pytest -q
#  -> 12 passed   (<1s; pytest.ini scopes collection to tests/ only —
#     a bare `pytest -q` without it will also try to collect the secsy
#     submodule's own test_scripts/, which import lompe and are not
#     part of this project)

conda run -n mhd-env python viability-test/gateV_kernel_validation.py
#  -> GATE V: PASS   (~8s)

conda run -n mhd-env python viability-test/tier1_flr_coherence.py
#  -> kappa/phi calibration curves + threshold pinning   (~16s)
```

If either fails, check first that `secsy`'s installed version matches the SHA pinned
in `results/V_kernel_validation/manifest.json` — the `secsy` API has drifted across
versions before and Gate V exists specifically to catch that.
