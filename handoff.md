# Handoff — Experiment B kickoff: solver stack + B1 (cost/pilot/floor) + B3 (mode-selection surfaces)

> **Assigned to:** a standard-tier Claude Code session (Strider's call, confirmed 2026-07-20).
> This file is the assignment — run it from cold per the note below. Strider's only role in
> Experiment B is the **§8-ii power-calc sign-off** that gates powered B2/B6; he is not the
> executor. (A 2026-07-19 record briefly listed him as the assignee, which turned that gate into
> an assignment and stalled the build for a day; corrected 2026-07-20.)
> **Written:** 2026-07-12, by the frontier session that verified the Tier 2 PR (verification memo
> in Strider's research vault). The previous handoff (Card A Tier 2) completed 2026-07-12 —
> H-A HOLDS, `ROADMAP.md` §8 — and lives in git history (`git show 00b426a:handoff.md`).
>
> **This file is written to be run from cold** — point Claude Code at this repo after cloning
> and it should complete the task below from the repository alone. If you hit a real decision
> this file and its links don't resolve, stop and ask Strider — do not improvise a
> pre-registered value. See "If you get stuck" at the bottom.

## GATES — do not start until all three are true

1. **PR #1 (`tier2-flr-coherence`) is merged.** The Tier-2 PR review is the team's ratification
   checkpoint for the audit-A1 pins; Experiment B consumes those pins.
2. **The ratification §8 entry has been appended to `ROADMAP.md`** (Strider; drafted in the
   vault review memo). It resolves the gap-conditioning scale reading — the fail-arm
   statistic your permutation code must implement (expected resolution: **trace-normalized**,
   gap × 2/trace(S); confirm against the actual appended entry, which is authoritative).
3. ~~**Strider has named the assignee**~~ — **CLEARED 2026-07-20** (standard-tier session; header updated).

## The one-line task

Build the Experiment-B module stack (solvers, metrics, mode selection, simulation extensions,
cell runner) per `viability-test/SPEC_experiment_B.md`, pass the single-source exact-recovery
gate, then run **B1** (full grid, both grid-reduction settings) and **B3** (both criteria),
plus the **50-trial mini-pilot** at the φ=90°/5 dB/`d=3` cell. Land results + a §8 entry +
open a PR. **Stop before powered B2/B6** — the power calculation (§8-ii: n_trials from
max(B1 pilot SD, mini-pilot SD), **reduction-ON rows** per pinned §S.6.2) is reviewed by
Strider before any confirmatory cell runs. The fairness protocol is **PINNED as of
2026-07-19** (SPEC §S.6 + ROADMAP §8 amendment) — the power-calc sign-off is the only
remaining gate.

## Why this scope line (so you don't re-derive it)

B1 and B3 **never adjudicate D2** (SPEC §S.2: B1 is cost/pilot/floor; B3 is descriptive,
feeding B6a's `n_snap` choice). They were therefore safe to run before the fairness protocol
pinned — and it is now pinned (2026-07-19). The confirmatory machinery (B2's φ-resolved
signed gap at the §8-iii cells, B6a) waits for the power-calc sign-off — that boundary is
exactly the pre-registration discipline that made Tier 2's verdict trustworthy.

## Read in this order

1. This file.
2. `README.md` — repo orientation, environment, rename map.
3. `BUILD_BRIEF.md` (repo root, the **current** onboarding brief, 2026-07-12) — the two
   invariants, Gate-V/`transfer.py` adapter pins, audit-A1 amendments, §S.9 supersessions,
   the AIC→MDL code-check verdict. The archived `GIBF_viability_BUILD_BRIEF.md` is stale;
   where they conflict, `SPEC_experiment_B.md` and `ROADMAP.md` win.
4. `viability-test/SPEC_experiment_B.md` — **your primary design document.** §S.1 (shared MC
   skeleton), §S.2 (B1/B3 designs), §S.4 (modeselect), §S.5 (outputs), §S.6 (fairness protocol
   — **PINNED 2026-07-19**; implement exactly as pinned, incl. the shared config object,
   the V/√λ₁ scale convention, and the three-solver two-tier gate), §S.7
   (both Card-A slots now FILLED), §S.8 (module deltas — your build checklist).
5. `ROADMAP.md` §2 (invariants), §5 node D2, §6 Card B, §8's 2026-07-12 entry (Tier 2) and
   the ratification entry above it once appended, §9.
6. `viability-test/transfer.py` docstring (adapter contract) and
   `viability-test/tier2_flr_coherence.py` (conventions to mirror: manifest format,
   seed-derivation scheme, per-trial archiving, figure style).

## Build checklist (SPEC §S.8, expanded)

- `viability-test/solvers.py` (or a package — mirror existing layout conventions): L2
  baseline, per-mode GIBF (L1-IRLS, per brief/SPEC), joint MMV-L1 (L2,1 row-sparse).
  **Fairness invariants (SPEC §S.1/§S.6):** identical `V`, `A`, `eps_frac`, `weight_floor`,
  `p_norm`, `max_iter`, `tol`, grid-reduction setting per run; MMV's μ and GIBF's per-mode
  penalty tied through the same `eps_frac` scaling rule; `assert np.isrealobj(A)` in every
  solver (B4's phantom arm is not in this handoff's scope).
- `viability-test/modeselect.py`: Wax–Kailath in the log domain, `criterion="mdl"` default,
  returns `K̂` + full AIC(k)/MDL(k) arrays, eigenvalue floor 1e-18, `K̂` clipped to [1, 6]
  with clip events counted (SPEC §S.4).
- `viability-test/metrics.py`: min-cost matched assignment, `τ_r = 1` grid cell (§8-v — not
  the archived brief's 1.5), `P_sep`, `Δr̄`.
- `simulate.py` additions per §S.8 (coherent-pair model per SPEC §S.2-B2 formula; B6b truth
  variants can be stubbed behind `truth.jitter` but are not exercised in this handoff).
- `viability-test/run_experiment_B.py`: the §S.1 skeleton; per-cell npz with **per-trial,
  per-method** metrics; CI convention = percentile bootstrap, 2000 resamples, seeded from
  cell id (§S.1).
- Tests in `tests/`: solver fairness (identical shared params actually shared), the
  single-source gate as a pytest, metrics pins (τ_r=1, assignment), modeselect MDL/AIC on
  synthetic spectra with known K. `pytest -q` stays green — 22 passing now; do not break them.

## Run order (each with its own commit before first run — A3 workflow)

1. **Single-source exact-recovery gate (SPEC §S.6.4, pinned):** one isolated DF source at
   the grid cell nearest the array centroid, oracle K=1, `n_snap=64` — **all three solvers**,
   two tiers: (i) noise-free deterministic, (ii) 20 dB MC × 10 trials, all pass; exact cell
   = global argmax (τ=0); sparse solvers at both reduction settings. **Failure HALTS
   Experiment B** (solver bug, not science). Archive the gate result like Gate V's artifacts.
2. **B1:** `d ∈ {1,2,3,5,8}` cells × `snr_db ∈ {−5,0,5,10,20}`, `n_snap=64`, oracle K=2,
   incoherent equal-power pair, grid reduction **on and off** (SPEC §S.2-B1). Deliverables:
   signed-gap cost surfaces, the §8-ii pilot SD, the floor panel arrays.
3. **B3:** `snr_db ∈ {−5,0,5,10,20}` × `n_snap ∈ {8,16,32,64,128}`, record BOTH AIC and MDL
   `K̂` distributions vs truth K=2 (SPEC §S.2-B3). Deliverable: the MDL error-rate table over
   (snr, n_snap) — B6a's `n_snap` is read from it per §8-vii (largest n_snap where MDL error
   ≥ 20% at 5 dB, fallback 8). Compute and REPORT that value; do not run B6a.
4. **Mini-pilot:** 50 trials at φ=90°, snr 5 dB, `d=3`, `n_snap=64`, `|ρ|=0.85` (SLOT-1),
   both solvers, **grid reduction ON** (pinned §S.6.2 — the confirmatory setting; the §8-ii
   SD inputs read from reduction-ON rows) — its SD is the other input to the §8-ii power
   calc. Report both SDs and a candidate n_trials; **do not run the powered cells.**

Outputs to `results/B_viability/` per SPEC §S.5 (manifest with config hash, git commit,
`script_sha256`, master seed). Seed: derive fresh (date-based, like Tier 2's 20260712 —
record it; do not reuse 20260706/20260712).

## Invariants (ROADMAP §2, restated)

- `A` real, DF-only, from `transfer.py` — never rebuild it ad hoc, never pass CF types.
- CSM from complex frequency-bin phasor snapshots; carry √λ (eigenmode = √λᵢ·uᵢ).
- GIBF and MMV receive the identical `V` and `A` in every cell.
- Report signed GIBF−MMV gaps, never magnitudes.
- Pre-registration: every threshold you need is already pinned (§8-ii…viii, SPEC §S.3);
  nothing is tuned after seeing a result. Log hygiene: §8 is append-only.

## Stop rule

After B1 + B3 + mini-pilot land (results, §8 entry, §9 status update, PR): **stop.** The
powered B2/B6 confirmatory runs start only after Strider reviews the PR and signs off the
power calc (the fairness protocol is already pinned, 2026-07-19). Do not draft manifesto text.

## If you get stuck

Same rule as the Tier-2 handoff: if something genuinely isn't resolvable from this repo (a
real ambiguity, not an implementation choice), do not guess — stop, write down the specific
question, and ask Strider. A real gap here is itself useful information.
