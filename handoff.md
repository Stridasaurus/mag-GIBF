# Handoff — Card A Tier 2 (the H-A hinge)

> **Assigned to:** Shane Gilbertie, 2026-07-11.
> **This file is written to be run from cold** — point Claude Code at this repo after
> cloning and it should be able to complete the task below without needing anything
> that isn't already in this repository. If you (human or Claude) hit a real decision
> this file and its links don't resolve, stop and ask Strider — do not improvise a
> pre-registered value. See "If you get stuck" at the bottom.

## The one-line task

Run **Card A Tier 2**: push the FLR source-coherence model through the real DF-only
transfer matrix `A`, build the ground CSM, and adjudicate whether inter-source phase
survives ionospheric integration to the ground (**H-A holds / fails / marginal**) —
using the design pinned below. Land the result as a new `ROADMAP.md` §8 log entry,
fill the two Card-A slots in `viability-test/SPEC_experiment_B.md` §S.7, and open a
PR. **Stop there** — do not start building Experiment B (B1–B4/B6). That is
deliberately out of scope for this handoff; Strider reviews the Tier 2 PR first.

## Read in this order

1. This file (task + pinned design + workflow).
2. [`README.md`](README.md) — repo orientation, environment setup, terminology
   rename map (old docs use pre-2026-07-01 names).
3. [`ROADMAP.md`](ROADMAP.md) §5 (node A, in the Decision Tree), §8's
   **2026-07-11 entry** ("Card A Tier 2 design PINNED..." — the complete pinned
   design, canonical; this handoff summarizes it but that entry is authoritative if
   they ever disagree), and §9 (current status).
4. [`EXPERIMENT_CARD_A.md`](EXPERIMENT_CARD_A.md) — the experiment card. §A.3.3 is
   Tier 2's method description, §A.4 is the decision rule, §A.8 is required outputs.
5. [`viability-test/tier1_flr_coherence.py`](viability-test/tier1_flr_coherence.py) —
   Tier 1's implementation. Tier 2 reuses its statistics (`kappa`, `im_frac`), its FLR
   phase model (`arg_r`), and its coding conventions (manifest format, seed handling,
   figure style). Read it before writing Tier 2's script — don't reinvent these.
6. [`viability-test/transfer.py`](viability-test/transfer.py) — the real transfer
   matrix builder Tier 2 calls. Its docstring is the Gate-V-pinned adapter contract
   (keyword names, return order, the coincidence-guard hazard). Read the docstring in
   full before calling `build_transfer_matrix`.
7. [`viability-test/floor_distributions.py`](viability-test/floor_distributions.py) —
   the reference implementation of the permutation test you'll reuse for the fail arm
   (see below).

## Environment (from README, repeated here for completeness)

```bash
git clone --recurse-submodules https://github.com/Stridasaurus/mag-GIBF.git
cd mag-GIBF
git submodule update --init --recursive   # if cloned without --recurse-submodules

conda create -n mhd-env python=3.11 numpy scipy matplotlib pytest -y
conda activate mhd-env
pip install -e ./secsy
```

**Verify before touching Tier 2** (should both pass unmodified — if either doesn't,
stop and diagnose before writing new code; something about your environment or clone
diverges from what's pinned):

```bash
python -m pytest -q
#  -> 12 passed
python viability-test/gateV_kernel_validation.py
#  -> GATE V: PASS
```

## The pinned design (full text: `ROADMAP.md` §8, 2026-07-11 entry)

Every parameter below was locked by Strider before any Tier 2 number exists — this is
the project's pre-registration discipline (see `ROADMAP.md` §2's log-hygiene
invariant and the append-only §8 Experiment Log). **Do not deviate from these without
appending a dated ROADMAP §8 entry explaining why**, the same way every prior
amendment in this project is logged (e.g. the 2026-07-06 floor amendment, the
2026-07-10 audit A1 sign-off). If a value below looks locally suboptimal once you're
in the code, that is not sufficient reason to change it — pre-registration only works
if it precedes the result.

### 1. Station geometry

- **Primary:** synthetic meridional chain, 10 stations, fixed longitude 20.0°E,
  centered 68.0°N, uniform meridional spacing swept `d ∈ {100, 150, 250}` km
  (`Δlat = d / 111.19` km-per-degree), ground radius `r = RE` (the `transfer.py`
  default — don't pass an explicit `r`).
- **Descriptive secondary (one run, reported not adjudicated):** the real IMAGE
  Finnish meridional chain — SOR, KEV, KIL, MUO, PEL, OUJ, HAN, NUR, TAR. Look up
  current published coordinates for these IMAGE stations at run time (they are public
  station metadata) and record exactly what you used in the run's `manifest.json` —
  this is a descriptive check, so pinning the *procedure* (use the real IMAGE Finnish
  chain) is sufficient; the exact coordinate values don't need pre-registration.
- **SECS pole grid:** latitude spans the station span plus 2 station-spacings margin
  on each side, sampled at half the station spacing. Longitude spans ±6° about the
  station meridian, sampled at whatever Δlon gives roughly half-station-spacing
  azimuthal resolution at 68°N (station spacing in degrees longitude ≈ station
  spacing in km / (111.19 × cos(68°))). **Offset the pole grid by half a pole spacing
  in longitude from the station meridian** — this is required, not optional: it's the
  Gate V V1 coincidence guard (`transfer.py` raises `ValueError` on any exact
  station–pole lat/lon match, and near-coincidence degrades conditioning even when it
  doesn't hit the guard exactly).
- **Descriptive row:** longitude extent doubled to ±12°, to measure how sensitive
  `κ_ground` is to the azimuthal truncation width (relevant because the source
  profile is flat in longitude — see below — so the grid's longitude extent is
  effectively a source-width parameter).

### 2. Sensor noise

- **Primary (adjudicating):** sensor-noiseless. This matches the Tier 1 calibration
  that pinned the 0.394/0.137 thresholds and the archived floor distributions
  (`results/A_flr_coherence/floor_distributions.npz`) — both were computed with zero
  sensor noise so that finite-N leakage was the only effect under calibration. Adding
  sensor noise to the primary run would break the permutation-test comparison against
  those floors.
- **Descriptive sensitivity rows:** `snr_db ∈ {10, 20}` (additive complex Gaussian
  noise per station-channel, calibrated to the stated SNR against the mean ground
  signal power). Noise robustness as its own axis belongs to Experiment B, not here.

### 3. Longitude source profile

- **Primary:** flat — the FLR source amplitude `s_t(x)` is azimuthally uniform across
  the pole grid's longitude extent. Isolates the latitudinal washout mechanism, which
  is the physics under test.
- **Descriptive row:** one run with a Gaussian azimuthal envelope, `σ_lon = 2°`, to
  bound how much finite azimuthal width would change `κ_ground`.

### 4. Monte Carlo and sweep parameters

- `n_mc = 400` trials at the pre-registered primary snapshot count `N = 64` (matches
  Tier 1's `n_mc`, so the permutation test against the archived floor distributions
  compares equal-`n` samples — do not use a different `n_mc` at N=64).
- `n_mc = 100` at `N = 1024` (asymptotic secondary, matches Tier 1).
- Sweeps: `Q ∈ {5, 10, 20}` (FLR quality factor), `|ρ| ∈ {0.70, 0.85, 0.95}` (source
  coherence magnitude), `β ∈ {0.05, 0.10, 0.20}` per 100 km (resonance latitudinal
  gradient) — all three sets are exactly Tier 1's values (see
  `tier1_flr_coherence.py`'s `phys` table), reused rather than re-chosen.
- **Placement:** for each `(Q, β, d, |ρ|)` cell, sweep the resonant latitude `x_r`
  across the station latitude span in 41 uniform steps and take the max over
  placement (Card A §A.4's "best case a real array could sample" — same principle
  Tier 1's `pair_phase_max` already implements for the abstract geometry curve;
  Tier 2 does the analogous sweep through the real kernel).
- **Seed:** `20260712` (fresh — do not reuse Tier 1's `20260706`).

### 5. `|ρ|` on the continuum grid

Tier 1's two-source model has an explicit `R_s = [[1, ρe^{iφ}],[ρe^{-iφ}, 1]]`
between exactly two poles. Tier 2 has a continuum of poles along the resonance
profile, so `|ρ|` needs an operational definition: **calibrate `σ_bg` per cell so
that the complex coherence magnitude between the two poles nearest `x_r ± 1`
resonance half-width equals the target `|ρ|`** (this is the continuum analogue of
Tier 1's two-pole `R_s`, evaluated at the pair that sits closest to Tier 1's "one pole
near resonance, one off by ~one width" exploitable geometry).

### 6. `κ_source` computation

Compute `κ_source` from the **sampled** source snapshots' CSM at the same `N` (not
from the analytic `R_s`/continuum covariance in closed form). This keeps the
source-vs-ground gap (`κ_source − κ_ground`) isolating the kernel's effect, not a
mismatch between an analytic source statistic and a finite-sample ground statistic.

### 7. Verdict aggregation over the realistic grid

The experiment card pins max-over-placement within one `(Q, β, d, |ρ|)` cell, but not
how the full realistic grid of cells aggregates into one H-A verdict. Pinned now,
before any Tier 2 number exists:

- **H-A holds** if **at least one** realistic-grid cell has MC-averaged
  max-over-placement `κ_ground ≥ 0.30` **with the co-primary concurring**
  (`‖Im S_ground‖/‖S_ground‖ ≥ 0.394`) at that same cell. This is an existence claim —
  it matches Card A §A.4's framing ("the *best case* a real array could sample") and
  Tier 1's own finding that band occupancy is parameter-dependent, not universal.
- **H-A fails** if **every** realistic-grid cell fails its arm: `κ_ground < 0.10`, or
  statistically indistinguishable from the same-N floor controls per the gap-conditioned,
  permutation-tested fail arm (`ROADMAP.md` §8, 2026-07-10 audit A1 entry — reuse
  `floor_distributions.py`'s per-trial archived distributions and test procedure
  exactly; 10⁴ resamples, α=0.05, seeded), with the co-primary concurring in the same
  floor-referenced sense.
- **Marginal** otherwise. Report the full `κ_ground(Q, |ρ|, d)` surface — this becomes
  B2's weighting per `ROADMAP.md` node A, regardless of which verdict lands.
- `κ` is always the **top-1** definition (audit A1, 2026-07-10) — do not use a top-2
  combination anywhere.
- Archive **per-trial** arrays (not just MC means) at whichever cell(s) end up
  adjudicating the verdict — the permutation test needs them, exactly as
  `floor_distributions.py` archived Tier 1's per-trial floor arrays.

## Invariants — do not violate (from `ROADMAP.md` §2, restated for this task)

- `A` is real. `assert np.isrealobj(A)` — `transfer.py` already asserts finiteness and
  realness internally; don't work around it.
- DF-only. Never pass `current_types=("curlfree",)` or include CF columns in any
  ground inversion — CF is excluded by theorem (Gate V), not by threshold.
- Build the CSM from complex frequency-bin phasor snapshots, never raw real time
  samples.
- Always pre-register before looking at results — you already have every threshold
  and rule above; do not adjust any of them after seeing a `κ_ground` number.
- Log hygiene: append to `ROADMAP.md` §8, never edit a past entry.

## Workflow (repo-standard, see the audit A3 entry in `ROADMAP.md` §8, 2026-07-10)

1. Work on a branch, e.g. `tier2-flr-coherence`.
2. Write `viability-test/tier2_flr_coherence.py` (naming mirrors
   `tier1_flr_coherence.py`). **Commit the runner before its first real run**
   (provenance workflow — a manifest that cites a commit must actually contain the
   script that produced it; see the audit-D2 defect this project hit and fixed).
3. Run it. Outputs go to `results/A_flr_coherence/` alongside the Tier 1 artifacts
   (new files, e.g. `tier2_results.npz`, `tier2_summary.csv`, updated or new
   `manifest.json` / figures — don't overwrite Tier 1's files). Include a
   `script_sha256` field in the manifest (the A3 workflow standard).
4. Add tests to `tests/` covering at minimum: the verdict-aggregation logic given a
   synthetic `κ_ground` surface, and that the permutation test correctly reproduces
   against a known archived floor distribution. `pytest -q` must stay green.
5. Append the result to `ROADMAP.md` §8 as a new dated entry (append-only — see
   existing entries for the expected level of detail: the exact numbers, the verdict,
   and which cell(s) adjudicated it).
6. Fill `viability-test/SPEC_experiment_B.md` §S.7 — replace the SLOT-1 placeholder
   (`|ρ|` = 0.85 mid / {0.70, 0.95} endpoints) with Tier 2's actual realistic range,
   and write SLOT-2's φ-weighting/framing per the verdict.
7. Update `ROADMAP.md` §9's status block (the "Active node" / "Most load-bearing next
   move" / "Sequence" lines) to reflect the landed verdict.
8. Open a PR against `main`. In the PR description, state the verdict plainly
   (holds/fails/marginal) and link the `ROADMAP.md` §8 entry.
9. **Stop.** Do not start Experiment B (B1–B4/B6) — Strider reviews this PR first;
   the review doubles as the team's ratification checkpoint for the audit A1 pins
   (κ=top-1, the permutation test, the |ρ|=0.95 calibration curve) that this task
   depends on.

## If you get stuck

Every design parameter needed to run Tier 2 is pinned above or in `ROADMAP.md` §8's
2026-07-11 entry. If something genuinely isn't resolvable from this repo (a real
ambiguity, not just an implementation-detail choice you could reasonably make either
way), do not guess and do not silently pick a default — that's exactly the
semantic-drift failure mode the 2026-07-07 repository audit found and fixed (see
`ROADMAP.md` §8's audit entries). Stop, write down the specific question, and ask
Strider directly rather than the "team channel" — this task was scoped to be
answerable from the repo alone, so a real gap here is itself useful information for
him.
