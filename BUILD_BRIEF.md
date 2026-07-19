# SECS-GIBF Viability Test — Build Brief (onboarding, current)

**This is the current, self-consistent onboarding brief for the `viability-test/` build.**
It **replaces** the archived `docs/archive/GIBF_viability_BUILD_BRIEF.md` (June 2026,
pre-rename, doubly stale) as the doc a fresh session or collaborator reads to get oriented.
It does **not** re-derive the full method — it reconciles the four live sources
(`ROADMAP.md`, `viability-test/SPEC_experiment_B.md`, `EXPERIMENT_CARD_A.md`, and the built
`viability-test/transfer.py`) into one place and points to them for depth.

Written 2026-07-12 (Fable session), reconciling: SPEC §S.9 supersessions · the Gate-V pins ·
the audit A1 amendments · the `transfer.py` DF-only adapter contract.

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

## Current state (what is built / passed / frozen / next)

| Item | State | Evidence |
|---|---|---|
| Rename (Card C→A; old Card A→Gate V; B5 retired; B6 added) | done 2026-07-01 | README rename map; ROADMAP §0 |
| Card A **Tier 1** (FLR κ/φ calibration, threshold pinning) | **complete** 2026-07-06 | `results/A_flr_coherence/`; ROADMAP §8 |
| **Gate V** (secsy CF-pair probe + DF validation) | **PASSED** 2026-07-07 | `results/V_kernel_validation/`; ROADMAP §8 |
| Repository audit (14/14 numeric checks; 4 semantic-drift defects D1–D4) | done 2026-07-07 | ROADMAP §8 |
| Audit **A1** (κ=top-1; permutation floor test; gap-conditioning; \|ρ\|=0.95 co-primary pin) | **signed off** 2026-07-10 | ROADMAP §8 |
| Audit **A2** (per-trial floor distributions, bit-exact replay) + **A3** (provenance workflow) | **complete** 2026-07-10 | `floor_distributions.py`; ROADMAP §8 |
| **`transfer.py`** (DF-only real-`A` adapter, pytest-gated) | **built** 2026-07-10 | `viability-test/transfer.py`; `tests/test_transfer.py` |
| Experiment-B design SPEC (B1–B4, B6) | **frozen** except 2 Card-A slots | `SPEC_experiment_B.md` |
| **Card A Tier 2** (the H-A hinge adjudication) | **NEXT — handed to Shane Gilbertie** (see below) | ROADMAP §9 |
| Experiment B (B1 pilot → powered B2/B6, B3/B4 alongside) | blocked behind Tier 2 | SPEC §S.7 sequencing gate |

**Sequencing gate:** ~~Gate V~~ → ~~`transfer.py`~~ → **Card A Tier 2** → fill SPEC Card-A slots
→ Experiment B → `DECISION_MEMO.md` → paper. Timeline: 11 weeks, build + paper in parallel.

> **Do NOT execute Card A Tier 2 from this brief.** Its design is pinned and its execution is
> **handed off to Shane Gilbertie** (branch+PR workflow; four design params + verdict-aggregation
> rule pre-registered). Strider reviews the PR when it lands (verdict + SPEC §S.7 slot fill).
> Experiment B does not start before that review. This brief is onboarding, not the run.

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

## Audit A1 amendments — the pinned pre-registration for Card A Tier 2 (do not re-open)

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

## AIC→MDL switch — code-check verdict (2026-07-12)

**Verdict: decided at the design layer, not yet implemented; no code conflict.** The switch
flagged in `[[wax-kailath-1985-notes]]` (MDL is consistent, AIC over-estimates source count even
asymptotically — Wax & Kailath 1985) is resolved to **MDL-primary** across ROADMAP §8-viii and
SPEC §S.4/§S.8/§S.9-4: `estimate_n_sources(eigvals, n_snapshots, criterion="mdl")` defaults to
MDL; `criterion` stays an argument so B3 scores **both** AIC and MDL; `K̂` clipped to `[1, 6]`.

Checked against the actual code: **there is no `modeselect.py` and no `estimate_n_sources`
implementation anywhere in the repo** — Experiment B is unbuilt (blocked behind Card A Tier 2), so
there is nothing to contradict the decision. (A grep for `AIC|MDL|estimate_n_sources|criterion`
over the codebase returns a single incidental hit — the word "criterion" in a comment in
`floor_distributions.py` about the gap-conditioning criterion, unrelated to mode selection.) The
only extant mode-selection artifact is `legacy_mhd_notebook.ipynb`'s MDL-13 cutoff, which is
consistent prior practice. **Action:** when `modeselect.py` is written, it must default
`criterion="mdl"` per SPEC §S.4; no change is needed now.

---

## What NOT to do (traps)

- **Do NOT run Card A Tier 2** — handed to Shane Gilbertie (above). This brief onboards; it is not
  the run.
- **Do NOT over-build.** Only Gate V, Card A, and Experiment B1–B4/B6. No robustness sweeps beyond
  B6, no real-data demo (synthetic-only paper confirmed), no CF claims of any kind (theorem cited;
  Stub X′ parked).
- **Do NOT trust the archived brief's stale sections** (the §S.9 list above; its pre-rename
  terminology throughout).
- **Do NOT introduce a complex `A`** outside Experiment B4. **Do NOT build the CSM from real time
  samples.** **Do NOT drop the `√λ`.** **Do NOT feed GIBF and MMV different inputs.**
- **Do NOT re-open the audit A1 pre-registrations** (κ=top-1, the permutation test + α, the
  gap-conditioning rule, the `|ρ|=0.95` pin) — they are signed off before any Tier-2 number exists;
  reopening them post-hoc breaks the pre-registration discipline the whole project rests on.

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
