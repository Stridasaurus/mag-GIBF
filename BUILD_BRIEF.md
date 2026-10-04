# SECS-GIBF Viability Test — Build Brief (onboarding, current as of 2026-10-03)

**This is the current, self-consistent onboarding brief for the `viability-test/` build.**
It **replaces** the archived `docs/archive/GIBF_viability_BUILD_BRIEF.md` (June 2026,
pre-rename, doubly stale) as the doc a fresh session or collaborator reads to get oriented.
It does **not** re-derive the full method — it reconciles the four live sources
(`ROADMAP.md`, `viability-test/SPEC_experiment_B.md`, `EXPERIMENT_CARD_A.md`, and the built
`viability-test/transfer.py`) into one place and points to them for depth.

Originally written 2026-07-12 and refreshed 2026-10-03. Current project state is reconciled
against ROADMAP §9 and SPEC §S.14; the Gate-V pins, Audit-A1 amendments, and `transfer.py`
DF-only adapter contract remain as recorded below.

## Reading order (authority)

1. `README.md` — repo orientation, environment, the 2026-07-01 rename map.
2. **`ROADMAP.md` — canonical, live source of truth** (decision tree, invariants §2, glossary
   §3, build order §6, experiment log §8, status §9). If anything disagrees, the roadmap wins.
3. **`viability-test/SPEC_experiment_B.md` — the frozen Experiment-B design** (B1–B4, B6). Wins
   over any older brief; its §S.9 lists what it supersedes.
4. `EXPERIMENT_CARD_A.md` — the frozen Card A (FLR source-coherence) design.
5. This brief — the reconciled current state + the two invariants + what not to do.

The archived brief is **history only** (methods-paper backfill). Do not tick its checklist or
trust its terminology; it predates the rename, the φ-sweep, Gate V, and `transfer.py`.

---

## The two non-negotiable invariants (honour everywhere)

These override every convenience and every acoustic-beamforming habit. Violating either
silently invalidates the study (ROADMAP §2; archived brief §0.2).

1. **The transfer matrix `A` is REAL. There is no propagation phase.** No `exp(i k r)`
   anywhere except the deliberate phantom-phase ablation (Experiment B4). Outside B4,
   `assert np.isrealobj(A)` in every solver.
2. **Snapshots are COMPLEX frequency-bin phasors, not raw time samples.** Real `X` ⇒ real
   CSM ⇒ real eigenvectors ⇒ the premise collapses. All phase lives in the complex source
   phasors → complex CSM → complex eigenvectors.

Plus, settled since the archived brief was written:

3. **DF-only by theorem — never include CF columns in a ground-inversion `A`.** Fukushima
   (1976) / Amm (1997): radial FACs (which the SECS CF basis has by construction) produce
   exactly zero ground field. CF is excluded by theorem, not threshold; the only CF
   computation anywhere is Gate V's probe. `transfer.py` is DF-only by default.
4. **Carry the eigenvalue magnitude:** the eigenmode is `vᵢ = √λᵢ · uᵢ` (Suzuki Eq. 3). Never
   invert the unit eigenvector.
5. **Fair comparison:** GIBF and MMV-L1 receive the identical `V` and real `A` and the same
   solver params, under the matched-regularization / symmetric-grid-reduction protocol with
   the regularization-sensitivity panel (SPEC §S.6). Report the **signed** `GIBF−MMV` gap.

---

## Current state (refreshed 2026-10-03)

| Item | State | Evidence |
|---|---|---|
| Card A Tier 1 / Tier 2; Gate V; `transfer.py` | complete | `results/`; ROADMAP §8 |
| Experiment-B pilot stack (B1, B3, d=3/d=8/d=2 pilots) | completed; old results preserved | ROADMAP §8; SPEC §§S.11–S.12 |
| Powered d=2 design | settled except two pre-run details | SPEC §S.14; direct Claude decision sessions cited there |
| Selected trial count | 10,000 per cell; historical 1,635 superseded | SPEC §S.14; `run_experiment_B2B6_powered_d2.py` |
| Powered runner / decision-state repair | implemented on `b2b6-powered-runner-d2`; confirmatory run not executed | candidate runner, tests, and current branch |
| Remaining scientific choices | exact off-grid offset/cells; whether the d=1 panel binds a d-min verdict | SPEC §S.14 |

The master-seed run is blocked until those two details are resolved. No confirmatory
result exists. The fresh next action is to resolve the two details, then review/land the
runner before any master-seed execution. The old Shane-Gilbertie Card-A assignment in
`handoff.md` is historical.

---

## Gate-V pins / `transfer.py` adapter contract (reconciled — the pinned facts)

Everything `transfer.py` relies on was pinned by Gate V (ROADMAP §8 2026-07-07;
`results/V_kernel_validation/`), not assumed. The built adapter's contract (`transfer.py`
docstring):

- **function:** `secsy.utils.get_SECS_B_G_matrices(glat, glon, r, plat, plon, current_type=…, RI=…)`
- **secsy version:** `1.0.1.dev38+g6f699cbd` (editable install of the `./secsy` submodule pin;
  recorded in the Gate V manifest). `secsy`'s API has drifted across versions — Gate V exists to
  catch that; if the pinned SHA changes, re-run Gate V.
- **keyword map:** config `'divfree' → 'divergence_free'`, `'curlfree' → 'curl_free'`. Invalid
  strings raise `ValueError` (they do **not** default). All `secsy` contact goes through the one
  thin adapter `_secsy_G`; nothing else in the repo names a `secsy` keyword.
- **return order:** `(Ge, Gn, Gr)` — east, north, **radial (up)**, each `(n_stations, n_poles)`;
  order and units independently pinned by V4's analytic under-pole radial match.
- **units:** tesla per unit SECS amplitude [A]; agrees with independent Biot-Savart quadrature to
  ≤ 1.6e-4 (V4/V5).
- **coincidence hazard (V1):** a station–pole `(lat, lon)` coincidence produces `NaN` in the
  horizontal columns that poisons even the analytic CF zero (`0 × NaN`). `transfer.py` **forbids**
  coincidence (`ValueError`) and asserts `np.isfinite(A).all()` after every build.
- **theorem (V3):** the CF ground block is analytically **exactly 0** below the shell (`secsy`
  hard-codes the full Fukushima pair). DF-only is the default and the Experiment-B invariant; the
  CF block is exposed only for explicit probes.
- **`A` assembly:** `vstack([Ge, Gn, Gr])` per current type → `(3·n_stations, n_types·n_poles)`,
  real float64. Row normalisation `'none'` or `'rms'` (each row to unit mean `|row|`); `A` stays
  real; the applied `row_scale` is recorded. sha256-keyed npz cache (derived data, gitignored).
- **provenance (audit A3):** the runner is committed **before** first use; manifests carry
  `script_sha256`. This is the standing workflow from `transfer.py` onward.

`tests/test_transfer.py` (8 tests; 12 with the semantic suite) gates: shape/realness, the
coincidence guard, row-norm + raw recovery, cache round-trip + key discrimination, unit source
vectors, keyword rejection, and the codified Fukushima expectation `‖CF block‖ ≤ 1e-12·‖DF block‖`
at high-latitude geometry (a documented constant, not a silent pass).

---

## Audit A1 amendments — ratified design and completed Tier-2 work (do not re-open)

Signed off 2026-07-10 (ROADMAP §8), resolving audit-D1/D3/D4. Carry these verbatim into Tier 2:

1. **κ operational definition = top-1** (the top CSM eigenvector only), matching the Tier-1
   calibration that pinned the co-primary thresholds. All "top-2" wording in older docs is
   amended to top-1. The eigenvalue gap is still reported alongside κ.
2. **Fail-arm floor-indistinguishability test:** one-sided two-sample **permutation test on the
   difference of means** (10⁴ resamples, seeded), H₀: mean(statistic) ≤ mean(same-N floor-control
   distribution), **α = 0.05**. "Indistinguishable from floor" ≡ H₀ not rejected. Applies to
   κ_ground vs its κ floor and to the co-primary `‖Im S‖/‖S‖` vs its floor.
3. **Gap-conditioning criterion:** near-degenerate ≡ the Tier-2 per-trial top-2 eigenvalue-gap
   distribution is **not** significantly above the incoherent floor's gap distribution (same
   permutation test, same α); rank-1-like otherwise. Rank-1-like → κ_ground referenced to the
   φ=0 coherent floor; near-degenerate → κ uninformative, the co-primary decides alone.
4. **Co-primary calibration curve pinned at `|ρ|=0.95`** (the curve the 0.394/0.137 thresholds
   were read from). Sensitivity recorded: `|ρ|=0.85 → 0.370/0.128`, `|ρ|=0.70 → 0.322/0.114`
   (conservative *against* H-A if realized `|ρ|` lands low — accepted).

Pinned Card A bands (ROADMAP §8-iv, node A): **κ_ground ≥ 0.30 → H-A holds**; **< 0.10 → H-A
fails** (report which failure mode: source-phase vs ground-washout); 0.10–0.30 → marginal. The
fail arm is floor-referenced and gap-conditioned per (2)–(3). Floors (canonical, N=64): incoherent
κ floor **0.283** (degeneracy-dominated); φ=0 coherent κ floor **0.012**; co-primary floor
**0.068**. These bracket the fail band from opposite sides — hence the gap-conditioning.

The pre-registered Experiment-B adjudication cells are likewise pinned (SPEC §S.3): win rule at
B2 **φ=90°, snr∈{5,10} dB, d=3, N=64**; loss rule mirrored at **φ∈{0°,180°}**; B6a at
**φ=90°, d=3, 5 dB**, `n_snap` read from B3's MDL error curve. `P_sep`: correctly placed = within
**1 grid cell**; distinct = peaks match different true sources.

---

## SPEC §S.9 — supersessions of the archived brief (explicit; do not trust the old brief here)

1. Archived brief §9-B1 "headline test / the figure *is* the decision" → **B1 is cost/pilot/floor**;
   the decision figure is B2's φ-resolved signed gap (07-01 restructure).
2. Archived brief §9-B2 fixed `rho: 0.95` → **SLOT-1** (`|ρ|` inherited from Card A's realistic
   range; midpoint at confirmatory cells).
3. Archived brief §6.6 `τ_r = 1.5` cells → **1 cell** (§8-v).
4. Archived brief §6.5 "AIC **or** MDL" → **MDL primary** (§8-viii) — see the verdict below.
5. Archived brief §9-B5 and its "Experiment A" (CF/DF identifiability) → **retired / replaced by
   Gate V** (07-01; theorem). The latitude sweep, subspace statistic, and crossing-latitude
   deliverable no longer exist. B5 deleted.

---

## AIC→MDL implementation status

MDL remains the solver-fed mode-count estimator, clipped to `[1, 6]`; B3 records both MDL and
AIC. The current powered runner records AIC per trial and derives its descriptive AIC-fed
comparison by lookup over the fixed-K=1…6 runs. It does not add an AIC-fed solver arm. See SPEC
§S.14 P13 and `viability-test/run_experiment_B2B6_powered_d2.py`.

---

## Current run guardrails

- Do not run the powered/master-seed experiment while either SPEC §S.14 pre-run detail remains
  unresolved: the exact off-grid offset/cells, and whether the descriptive d=1 regularization
  panel is binding when d-min decides a verdict.
- Do not alter original result artifacts when applying the separately recorded peak rescoring.
- Preserve the two core invariants above: real DF-only `A` outside B4, complex frequency-bin
  phasor snapshots, eigenmodes weighted by `√λ`, and identical GIBF/MMV inputs.
- Do not reopen the ratified Card-A/Audit-A1 rulings or treat old 1,635-trial text as current.

---

## Pointers

- Canonical state / decisions: `ROADMAP.md` (§8 log, §9 status).
- Frozen Experiment-B design: `viability-test/SPEC_experiment_B.md`.
- Frozen Card A: `EXPERIMENT_CARD_A.md`.
- Built adapter + tests: `viability-test/transfer.py`, `tests/test_transfer.py`.
- Gate V artifacts + paper paragraph: `results/V_kernel_validation/` (`PARAGRAPH.md`).
- Card A Tier 1 artifacts: `results/A_flr_coherence/`.
- Vault: `[[secs-gibf-viability]]` (project note), `[[wax-kailath-1985-notes]]` (the AIC/MDL basis).
- Archived (history only): `docs/archive/GIBF_viability_BUILD_BRIEF.md`.
