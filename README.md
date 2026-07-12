# mag-GIBF

Viability test for **SECS-GIBF**: applying Generalized Inverse Beamforming (GIBF), an
eigenmode-per-mode sparse imaging method originally developed for aeroacoustics
(Suzuki 2011), to magnetometer arrays imaging ionospheric current systems via the
SECS (Spherical Elementary Current Systems) source basis.

**Start here, in this order:**

1. This README (repo orientation, environment setup)
2. [`ROADMAP.md`](ROADMAP.md) — **canonical, live source of truth** for the research
   state, decision tree, and build order. If anything below and the roadmap disagree,
   the roadmap wins.
3. [`EXPERIMENT_CARD_A.md`](EXPERIMENT_CARD_A.md) and
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

- **Validation Gate V: PASSED** (2026-07-07). Confirms the `secsy` library implements
  the Fukushima/Amm theorem correctly — curl-free (CF) current produces exactly zero
  ground field for radial FACs — and pins the exact adapter contract (`secsy` keyword
  map, `(Ge, Gn, Gr)` return order, a coincidence/NaN guard) that `transfer.py` must
  satisfy.
- **Card A Tier 1: complete** (2026-07-06). Forward-modeled realistic FLR source pairs;
  pinned the co-primary thresholds (κ 0.394 holds / 0.137 fails) for whether
  inter-source phase survives ionospheric integration to the ground.
- **Experiment B design: frozen** (2026-07-07), pending the build below.
- **`transfer.py`: built and pytest-gated** (2026-07-10) — the real, DF-only transfer
  matrix adapter, built against the contract Gate V pinned. Semantic-drift audit items
  A1-A3 landed the same sitting.
- **Frontier / next thing to run: Card A Tier 2** — the H-A adjudication hinge, design
  fully pinned 2026-07-11 (`ROADMAP.md` §8). **Currently assigned to Shane Gilbertie —
  see [`handoff.md`](handoff.md) for the complete, self-contained task.**

**Before picking up new work, check `ROADMAP.md` and ask in the team channel** —
several build-order items (`transfer.py`, the eigendecomposition/mode-selection step)
were assigned informally before this repo existed; confirm what's already started
before duplicating it.

## Repo layout

| Path | What it is |
|---|---|
| `ROADMAP.md` | Canonical research state: decision tree, invariants, glossary, build order, experiment log |
| `handoff.md` | Self-contained task handoff for the current assignee of the active frontier node (Card A Tier 2, 2026-07-11: Shane Gilbertie) |
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
| `docs/archive/GIBF_viability_BUILD_BRIEF.md` | **Archived 2026-07-11.** Claude's original build spec (June 2026), superseded by `ROADMAP.md` + `SPEC_experiment_B.md`; kept for history only. Uses pre-rename terminology — its own header explains the mapping |
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
