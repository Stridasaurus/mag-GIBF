# SECS‑GIBF — Research Roadmap

> **Repository:** `https://github.com/Stridasaurus/GIBF`
> **Layer:** Research Roadmap (Decide phase). This document sequences the *decisions*; it never feeds the build cascade directly. Only a manifesto does.
> **Status:** active — frontier is **Experiment A, Tier 1** (FLR source‑coherence forward model), runnable **today** in two steps: sign off §8‑iv, then run. Manifesto #1 (Viability Test & Methods Paper) is being expanded now from **Stub 0**.
> **Restructured 2026‑07‑01** after peer review. All §8 log entries dated before 2026‑07‑01 use the *old* names; read them through the map in §0.

---

## 0. Rename map (2026‑07‑01 restructure)

| Old name | New name | Why |
|---|---|---|
| Card C / Experiment C (FLR source‑coherence) | **Card A / Experiment A** | It runs first (highest information value, zero build dependency); the alphabet now matches the run order. |
| Card A / Experiment A (CF/DF identifiability) | **Validation Gate V** | Retired as an *experiment*: its outcome is fixed by theorem (Fukushima 1976; Amm 1997). What survives is its keystone — the `secsy` CF‑pair probe — which is a *build validation*, not a decision. |
| Card B / Experiment B | **unchanged** | B1–B4 keep their names. **B5 retired** (CF/DF attribution — moot by theorem). **B6 added** (prior‑misspecification stress cells). |
| Decision node D1 (CF identifiability fork) | **RETIRED** (tombstone in §5) | The CF‑marginal / CF‑observable / CF‑collinear branches are unreachable with a correct adapter. |
| Node C (Card C's design‑setting node) | **Node A** | Follows the card rename. |
| Decision nodes D2, D3 | **unchanged** | Kept for log continuity; D2 remains the sole stub‑routing methods decision. |
| Q1 / Q2 (research questions) | **Q1 = phase survival (Experiment A); Q2 = per‑mode value (Experiment B)** | Old Q1 (CF identifiability) is answered by theorem and absorbed into Gate V. |
| "the eigen‑decomposition layer under test" | **"per‑mode independence under test"** | Both GIBF and MMV‑L1 consume the *same* eigendecomposition; D2 adjudicates per‑mode vs shared‑support, not eigendecomposition vs none. |

---

## 1. Purpose

Sequence the experiments that retire the two critical unknowns about SECS‑GIBF — **whether realistic FLR inter‑source phase survives to the ground CSM** (Experiment A) and **whether per‑mode independence earns its keep against a joint shared‑support inversion** (Experiment B → D2) — so that the full‑pipeline direction is chosen from evidence, not from the method's name.

CF identifiability is **no longer an unknown**. In this model class it is settled negatively by theorem: for radial FACs — which the SECS CF basis has *by construction* — the CF current plus its FAC produces exactly zero ground magnetic field, independent of the conductance distribution (Fukushima 1976; Amm 1997). It is carried as **Validation Gate V** (does the code implement the theorem?), cited in the paper as theory, and never re‑litigated as an experiment.

---

## 2. Invariant vision & boundaries

True across *every* branch below. Anything that varies by branch lives in a stub, not here.

**Vision.** Physically consistent imaging of ionospheric current systems from magnetometer arrays, using a sparse SECS source basis, with every methodological claim (does FLR phase survive to the ground CSM? does per‑mode independence earn its keep, and in which φ / misspecification regime?) decided by a pre‑registered, executable rule rather than asserted — and with settled theory (Fukushima/Amm) cited as theory, not re‑run as experiment.

**Domain of validity (carried into every manifesto).** The quasi‑static / real‑kernel treatment holds when array aperture `L` and signal period `T` satisfy `L/c ≪ T` (equivalently `L ≪ λ`): ground arrays of 10²–10³ km, frequencies 10⁻³–1 Hz (Pc3–5, Pi2, substorm current wedge, FAC structures; IMAGE / SuperMAG‑class arrays). Full‑wave regimes (e.g. VLF > 3 kHz) are **out of scope** and would require a dyadic EM Green's function.

**Invariants — never violate, in any branch:**

- **Never** apply a propagation phase to the transfer matrix. `A` is **real**; there is no `exp(i k r)`. The *only* exception anywhere in the program is the deliberate phantom‑phase ablation (Experiment B4). Outside it, `assert np.isrealobj(A)`.
- **Never** include CF columns in `A` for any ground inversion — excluded by theorem, not by threshold. The only CF computation anywhere in the program is Gate V's probe.
- **Always** build the CSM from **complex frequency‑bin phasor snapshots**, never raw real time samples. Real `X` ⇒ real `S` ⇒ real eigenvectors ⇒ the premise collapses. All phase lives in the complex source phasors → complex CSM → complex eigenvectors.
- **Always** carry the eigenvalue magnitude: the eigenmode is `vᵢ = √λᵢ · uᵢ` (Suzuki Eq. 3). Never invert the unit eigenvector.
- **Always** feed any head‑to‑head method comparison the **identical** `V` and the **identical** real `A`. Identical inputs are *necessary but not sufficient*: GIBF (per‑mode L1‑IRLS + grid reduction) and MMV‑L1 (joint L2,1) cannot share a literal parameter set, so the comparison is only fair under an explicit **matched‑regularization / matched‑effective‑sparsity protocol**, with grid reduction applied **symmetrically** (both methods or neither), and with the **sign of the gap reported across a band of regularization strengths** at the primary cells (a verdict robust to the tuning knob is a result; a verdict at one knob setting is an anecdote). The protocol is pinned in the manifesto; the *requirement* is invariant here.
- **Always** report the **signed** `GIBF−MMV` gap, never a magnitude alone. The thesis is sign‑neutral (§7 Stub 0); either sign is the finding.
- **Always** pre‑register each experiment's decision rule and threshold *before* looking at results, and record them in `DECISION_MEMO.md`.
- **Log hygiene:** corrections to §8 are appended as new dated entries referencing the old entry — never inline edits or annotations. (The 2026‑07‑01 annotation inside the 2026‑06‑30 pre‑registration entry is grandfathered as the last exception.)

---

## 3. Glossary (cross‑branch)

One name per concept, used identically in this roadmap, in every manifesto, and in code.

| Term | Meaning |
|---|---|
| **SECS** | Spherical Elementary Current Systems; the source basis. |
| **CF** | Curl‑free SECS part — divergence of the ionospheric current, closing via FACs. Excluded from all ground inversions by theorem (see **Fukushima theorem**, **Gate V**). |
| **DF** | Divergence‑free SECS part — rotational/solenoidal current closed within the sheet. The sole basis for ground `A`. |
| **FAC** | Field‑aligned current. |
| **Fukushima theorem** | Radial FACs + their CF current produce **exactly zero** ground magnetic field below the ionosphere, for **any** conductance distribution (Fukushima 1976 proved the uniform case; Amm 1997 removed the uniformity assumption). The crucial assumption is *radial* FAC — which the SECS CF elementary system satisfies by construction, at every pole latitude. Hence CF **is** (not "may be") unrecoverable from ground data in this model class; the question is a theorem, not an experiment. Lifting it requires non‑radial FACs (see **Stub X′**). |
| **`A` (transfer matrix)** | Real geometric coupling matrix `B(ω) = A · s(ω)`, mapping unit SECS amplitudes → ground field. Real in every branch except the B4 ablation. **DF columns only**; CF columns exist only inside Gate V's probe. |
| **`secsy`** | Library providing the SECS field kernels `(Ge, Gn, Gu)`. API has drifted across versions; pin and verify per build. |
| **CSM** | Cross‑spectral matrix `S = (1/N) X Xᴴ`; complex Hermitian PSD. |
| **eigenmode `vᵢ`** | `√λᵢ · uᵢ` (Suzuki Eq. 3). The shared input to every inversion under comparison. |
| **GIBF** | Generalized Inverse Beamforming: invert each eigenmode **independently** with a per‑mode sparse (L1‑IRLS) prior, then accumulate intensity across modes. The **per‑mode independence** under test (colloquially "the eigen layer" — but note both GIBF and MMV‑L1 consume the same eigendecomposition; D2 adjudicates per‑mode vs shared‑support, not eigendecomposition vs none). |
| **MMV‑L1** | The competitor: a single **joint** inversion of the same `K` eigenmodes with a **shared‑support** (L2,1 row‑sparse) prior. Same `V`, same `A`; differs from GIBF only in shared‑support vs independent‑per‑mode. |
| **IRLS** | Iteratively reweighted least squares; the L1 solver core. |
| **grid reduction (β)** | Irreversible pruning of the active source set by factor β = 0.9 per iteration; switches the solve underdetermined → overdetermined. Applied symmetrically (both methods or neither); report the method × reduction 2×2 where feasible. |
| **mode selection** | Choosing the number of signal eigenmodes via complex AIC / MDL (Wax–Kailath 1985). In **B6a** the *estimated* `K̂` is fed to the solvers (not oracle `K`). |
| **observability ratio `ρ_obs(j)`** | `‖A·e_CF(j)‖ / ‖A·e_DF(j)‖` at grid point `j`. **Repurposed 2026‑07‑01:** now Gate V's validation statistic — expected ≈ machine precision when the adapter implements the full Fukushima pair; a materially nonzero value indicates an **adapter bug** (most likely: horizontal CF current returned without its FAC), never physics. |
| **`‖Im S‖/‖S‖`** | CSM imaginary fraction (rotation‑invariant). Card A's **co‑primary** statistic; the observable online gate for D2's liability branch. Finite‑N floor ~`1/√N` (measured 0.060 at N=64, 0.0013 at N=10⁵). |
| **`P_sep`** | Separation success rate: fraction of trials in which both true sources resolve as distinct, correctly‑placed peaks. **Pinned (§8‑v):** *correctly placed* = within **1 grid cell** of a true source under minimum‑cost matched assignment; *distinct* = the two reported peaks match **different** true sources. |
| **inter‑source relative phase `φ`** | The phase of the complex coherence `ρ = |ρ|·e^{iφ}` between two coherent sources. **Load‑bearing:** with a real `A`, the CSM acquires exploitable imaginary structure *only* when `φ ≠ 0, 180°`. Verified: residual eigenvector complexity ≈ 0.00 at φ=0 and 180°, 0.22/0.44/0.65 at 30/60/90°, ≈ 0.14 incoherent at N=64 (finite‑sample). |
| **residual eigenvector complexity `κ`** | Global‑phase‑removed imaginary content of the top‑2 CSM eigenmodes (`min_θ ‖Im(e^{iθ}u)‖/‖u‖`). Reported at the **source** (`κ_source`, from `R_s`) and at the **ground** (`κ_ground`, from the ground CSM after the real kernel); the gap is the ionospheric‑integration "washout." **Caveat:** eigenvector rotation is ill‑conditioned when `λ₁ ≈ λ₂` (equal‑power sources) — always report the eigenvalue gap alongside, Monte‑Carlo‑average in Tier 2, and cross‑check with the rotation‑invariant `‖Im S‖/‖S‖`. |
| **FLR quality factor `Q`** | `Q ≡ ω/γ` for the driven‑damped‑oscillator field‑line‑resonance model (Card A); realistic `Q ≈ 5–20` sets how narrow (in latitude) the ~180° source cross‑phase transition is. |
| **`Δr̄`** | Mean two‑source localisation error (minimum‑cost matched assignment; error averaged over matched pairs). |
| **phantom‑phase ablation** | Experiment B4 only: a *fake* complex `A_phase = A ⊙ exp(i·Φ)` emulating a propagating‑wave phase ramp, run GIBF‑only, to quantify what the absent acoustic phase diversity *would* buy. |
| **B6 (stress cells)** | Prior‑misspecification cells: (a) solver‑fed `K̂` from AIC/MDL, (b) ≤1‑cell source‑position jitter across the snapshot window. The only cells where the shared‑support prior is *false* — hence the only cells with a principled mechanism for a per‑mode win. Bridge to D3 / real data. |
| **Gate V** | The kernel validation gate (was Experiment A): the `secsy` CF‑pair probe + realness assertions + DF spot checks. Not a decision; a pass/halt. |
| **snapshot** | One complex frequency‑bin phasor column of `X`. |
| **DECISION_MEMO.md** | The headline deliverable: pre‑registered thresholds, the decision figures, and the go/no‑go decisions. The paper's proto‑results section. |

---

## 4. Research‑phase exit conditions

These are *not* the project's success criteria — they are the criteria for **stopping deciding and locking the next manifesto**. The viability test (Manifesto #1) is the effort that reaches them.

Research is "done deciding" when **all** of the following hold:

1. **Correctness gates green** so the decisions are trustworthy: single‑source exact recovery for all three solvers in the noise‑free well‑separated case; CSM provably *not* near‑real for a multi‑phase scenario; √λ carried (Suzuki Eq. 3); the `secsy` adapter's verified version + conventions recorded; **Gate V passed** — CF + FAC ground field ≡ 0 at machine precision at every latitude preset, documented as the theorem's numerical confirmation (headline calibration, not a finding). `pytest -q` green in < 1 minute.
2. **Q1 (Experiment A) resolved:** the H‑A verdict landed per the §5 node‑A bands (holds / marginal / fails), with the κ_source‑vs‑κ_ground washout gap reported; B2's φ range, weighting, and `|ρ|` range set from it; the paper's framing fork chosen.
3. **Q2 (Experiment B) resolved:** B1 complete with mean ± 95 % CI per method per cell; the §[D2] rule applied — **both the win rule and the loss rule (§8‑vi)** — to land the D2 outcome; B2 (coherent, φ‑swept), B3 (mode selection), B4 (phantom‑phase), and **B6 (misspecification stress cells, §8‑vii)** complete. **B2 may not be designed until Experiment A has resolved** — its result sets B2's realistic φ range/weighting and `|ρ|`, and the primary confirmatory cells must be pre‑registered before B2 runs (§8‑iii).
4. **`DECISION_MEMO.md` filled** with thresholds pre‑registered before results, all decisions recorded, and a recommendation written.
5. **Methods paper drafted** to submission‑ready state for the target venue. *Confirmed synthetic‑only*: no real‑data example is required, so the real‑data demo stays downstream (Stub F/S) and is **out of Manifesto #1's scope**. Much of the paper — intro, physics, domain of validity, the Fukushima/Gate‑V paragraph, algorithm, the pre‑registered hypotheses and decision rules — drafts from the manifesto and planning docs *in parallel with the build*; only the results/decision sections wait on the experiments.

**No‑kill floor (quadrant map).** Every cell of (Experiment A outcome) × (D2 outcome) carries a headline:

- *H‑A holds × GIBF wins:* per‑mode imaging earns its keep in the physically dominant ULF regime, with B4 as mechanism.
- *H‑A holds × GIBF loses:* exploitable phase exists, survives to the ground — and per‑mode inversion *still* cannot beat a joint inversion of the same modes; B4 shows why (acoustic GIBF's power came from kernel phase diversity, which no quasi‑static real‑kernel problem has). A transferable negative for the whole real‑kernel inverse‑problem class.
- *H‑A fails (either mode):* Experiment A itself is the headline — FLR cross‑phase, robustly measured *between stations*, does not survive into the ground CSM's eigenstructure (or is near‑real at the source). A ground‑observability result the ULF community can use independent of GIBF.
- *B6 win with clean‑cell loss:* the method's actual value proposition is robustness to prior misspecification, not resolution — arguably the most useful positive available.

The research phase cannot terminate in project death; it terminates by *selecting which downstream manifesto to write*.

When met: select the downstream branch by the `(Q1 × Q2)` outcome (§7) and expand that stub into Manifesto #2.

---

## 5. Decision Tree (assumption‑anchored — load‑bearing)

Thin navigation; the cards in §6 hold the method detail. The full build context lives in the repo's `viability-test/PLAN.md` (the master build brief — **flagged stale**, see §9), which a manifesto will hand off to.

```
V — VALIDATION GATE (not a decision; the outcome is fixed by theorem).
    Fukushima 1976 / Amm 1997: for radial FACs — the SECS CF construction, at every pole
    latitude — the CF current + its FAC produce EXACTLY ZERO ground field, independent of
    the conductance distribution. DF-only for ground arrays is therefore a THEOREM in this
    model class, not a finding. The old D1 fork (CF-marginal / CF-observable / CF-collinear)
    is retired: those branches are unreachable with a correct adapter, and with
    ‖A·e_CF‖ ≈ 1e-15 the old subspace-separation statistic is 0/0 numerical noise.
    Probe: the secsy CF-pair check (PLAN Appendix C) — does the library return the FULL
    Fukushima pair (horizontal CF current + its radial FAC)? Expected: ground field at
    machine precision at EVERY latitude preset (the theorem is latitude-independent;
    there is no crossing latitude to report).
      pass -> A's realness + physics asserted; DF-only A built; Tier 2 / Experiment B
              unblocked. The paper cites the theorem, with the probe as its numerical
              confirmation (one paragraph; headline CALIBRATION, not a result).
      fail -> HALT everything downstream of A: the kernel reflects the LIBRARY, not the
              physics (most likely: horizontal CF current without its FAC). Fix the
              adapter; nothing consuming A is trustworthy until green.
    Status: FIRST BUILD ACTION (needs secsy). Does not block Experiment A Tier 1.

A — Assumption H-A: a realistic field-line resonance induces inter-source phase phi
    substantially away from 0/180 deg between SECS poles, AND that phase survives through
    the real kernel A to the ground CSM (not washed out by ionospheric spatial integration).
    Experiment: A (Card A, §6; was Card C).
    NOT a stub-selecting decision (unlike D2/D3) — it sets Experiment B's DESIGN (B2's phi
    range, weighting, |rho| range) and the paper's FRAMING. D2's rule alone governs the
    Stub F/S routing; node A never overrides it.
    Rule: kappa_ground (top-2 ground-CSM residual eigenvector complexity) at realistic
    Q = 5-20, |rho| = 0.7-0.95, IMAGE-like spacing, maximised over pole-pair placement.
    Co-primary: ‖Im S‖/‖S‖ (rotation-invariant), threshold pinned from Tier 1's
    calibration curve BEFORE Tier 2 runs (§8-iv). Report the eigenvalue gap with kappa.
      H-A holds   (kappa_ground >= 0.30)        -> B2 centers its sweep on the realistic
                                                    phase-offset band; framing: exploitable
                                                    structure exists in the physically
                                                    dominant regime.                 -> D2
      marginal    (0.10 <= kappa_ground < 0.30) -> B2 sweep unchanged (full 0-180 arc) but
                                                    reweighted by the kappa(Q,|rho|) curve;
                                                    report as a graded regime.       -> D2
      H-A fails   (kappa_ground < 0.10)         -> B2 still runs the full arc (the abstract
                                                    cost/benefit result stands), but the
                                                    framing pivots to "phase structure does
                                                    not survive to the ground CSM"; report
                                                    WHICH failure mode (source-phase vs
                                                    ground washout) and the source-vs-ground
                                                    gap as the mechanistic figure — itself
                                                    a headline.                      -> D2
    Kill: none — every outcome is publishable. Only halt: do not trust kappa_ground (Tier 2)
    until Gate V is green.
    Status: ACTIVE. Tier 1 (pure NumPy, no build dependency) runs TODAY after §8-iv
    sign-off. Tier 2 slots after transfer.py + Gate V, before B2 is designed.

D1 — RETIRED 2026-07-01 (tombstone; kept for log continuity).
     Was: the CF-identifiability fork on median rho_obs thresholds (1e-2 / 1e-1) with a
     subspace-separation gate. Retired because the outcome is analytic (see V): in the
     radial-FAC SECS model class rho_obs is identically ~0, the thresholds can never be in
     play, and the DF-only "branch" is a restatement of Fukushima, not a finding. B5
     retired with it. CF/FAC science routes to Stub X (satellite) or Stub X' (dipolar-FAC,
     Paper 2) — never through this pipeline's ground A.

D2 — Assumption: a real (phase-less) kernel makes per-mode GIBF NO better than joint MMV-L1.
     Experiment: B (Card B; B2 phi-swept headline, B1 cost/pilot/floor, B3/B4 supporting,
     B6 stress cells).
     PRE-REQUISITE: B2's phi range, weighting, and |rho| are set by Experiment A's result
     (node A above). Do not finalize B2's design before Experiment A completes.
     STRUCTURAL NOTE (desk-checks 2026-06-30; framing CORRECTED 2026-07-01): with a real A,
     exploitable complex structure exists only for coherent sources with phi != 0, 180,
     peaking near phi = 90 (verified: 0.00 / 0.22 / 0.44 / 0.65 at 0/30/60/90 deg; null
     again at 180). CORRECTION: the shared-support prior is correct in EVERY clean
     synthetic cell — B2 as well as B1 — because the two true sources occupy the same two
     grid cells across all eigenmodes in BOTH regimes. B1 and B2 differ in the DATA's
     structure, not in the PRIOR's correctness. Both arms receive the identical complex V,
     so "structure exists at phi=90" does not by itself supply a mechanism for a per-mode
     advantage. Therefore the structurally favored outcome is MMV >= GIBF everywhere
     ("drop" or "free option"); a GIBF win at phi~90, if it occurs, is an algorithmic
     (optimization-path) effect — treat H-B's alternative as the alternative, not the
     expectation. B6 is the ONLY cell class where the shared-support prior is FALSE, and
     hence the only place with a principled reason to expect a per-mode win.
     Rule (WIN — unchanged): GIBF Δr̄ >= 20% lower OR resolves >= 1 grid cell smaller, with
           NON-OVERLAPPING 95% CIs, at >= 2 SNR levels, at the PRE-REGISTERED primary
           phi cells (§8-iii) in the phi-swept coherent regime.
     Rule (LOSS — new, mirrored, §8-vi): MMV Δr̄ >= 20% lower OR >= 1 cell smaller, same
           CI and power standard, at the pre-registered NULL cells (phi in {0, 180}).
           Required to call "liability" as a finding rather than an eyeball; without it,
           liability-vs-free-option is post hoc.
     POWER: non-overlapping 95% CIs ~ a difference test at alpha ~ 0.006 (conservative).
            n_trials set by the pre-registered PROCEDURE (§8-ii): variance input =
            max(B1 pilot SD, a 50-trial mini-pilot SD at the B2 primary cell) — B1's
            incoherent variance need not transfer to the near-rank-1 coherent regime.
            Then set n_trials so a TRUE 20% / 1-cell effect clears non-overlap at >= 80%
            power. Without this, "inconclusive" and "drop" are confounded.
     SIGN, NOT JUST MAGNITUDE: report the SIGNED GIBF-MMV gap at each phi. phi is LATENT
            (imaged, not observed); the observable pre-inversion gate is ‖Im S‖/‖S‖
            (high exactly when phi is exploitable, floor ~1/sqrt(N)). So:
       GIBF worth it     (win rule met in the exploitable band)   -> per-mode retained
                                                                     -> Stub F
       free option       (GIBF >= MMV for ALL phi incl. nulls AND B6; strictly better
                          somewhere)                              -> run per-mode ALWAYS;
                                                                     no gate -> Stub F
       liability         (win near 90 AND loss rule met at nulls) -> gate on the OBSERVABLE
                                                                     ‖Im S‖/‖S‖, or drop
                                                                     unless domain says
                                                                     phase-offset
                                                                     -> Stub F (gated) / S
       robustness pivot  (<= MMV on clean cells BUT win rule met in B6)
                                                                  -> per-mode retained as a
                                                                     ROBUSTNESS option;
                                                                     thesis pivots from
                                                                     resolution to
                                                                     robustness-to-
                                                                     misspecification
                                                                     -> Stub F (B6-framed) / S
       drop              (no win anywhere, incl. phi~90 AND B6)   -> per-mode dropped
                                                                     -> Stub S
       inconclusive      (CIs overlap everywhere AT ADEQUATE POWER)
                                                                  -> refine SNR/phi grid;
                                                                     re-run.
     Kill (defeats the project's own namesake): if neither the win rule nor the B6 rule is
           met, do NOT carry per-mode inversion into the full pipeline despite "GIBF" being
           the project name. Method-wide abandonment is warranted when D2 = drop AND the
           coherent regime shows no advantage AT NONZERO phi (an in-phase-only B2 is a null
           by construction and CANNOT satisfy this) AND B6 shows no advantage under
           misspecification AND B4 attributes the acoustic-GIBF gap to the absent kernel
           phase diversity.
     Status: pending (blocked on Experiment A + Gate V + the Monte-Carlo machinery).

D3 — (future, one-line until reached, inside Stub F or S)
     Assumption: a real event affords ENOUGH independent stationary snapshots for the CSM
     spectrum to reveal the planted source count (AIC/MDL stable). B3 and B6a only partially
     de-risk this on synthetic data; real data is the true test. Reached only on first
     contact with IMAGE/SuperMAG data downstream.
```

---

## 6. Experiment Cards

Written to run without this conversation. Full method, configs, and tests are in `viability-test/PLAN.md` (the master build brief — **stale**, see §9); the cards below carry only what the *decisions* need.

### Card A — FLR source‑coherence forward model *(was Card C; full standalone card: `EXPERIMENT_CARD_A.md`, repo root)*

- **Assumption (H‑A):** a realistic field‑line resonance induces inter‑source phase `φ` substantially away from 0/180° between SECS poles, **and** that phase survives to the ground CSM through the real kernel `A` (i.e. is not washed out by ionospheric spatial integration). The null has two failure modes to separate: source‑phase failure (the FLR maps to near‑real `ρ`) vs ground‑washout failure (φ≠0 at the source but the ground CSM is near‑real).
- **Method:** Tier 1 (pure NumPy, no `secsy`) — driven‑damped‑oscillator FLR model (Southwood 1974; Chen & Hasegawa 1974): `R(x) = 1/(ω_r(x)² − ω² + iγω)`; `φ_ij = arg R(x_i) − arg R(x_j)` vs pole spacing relative to resonance width (`Q ≈ 5–20`); source CSM `R_s = R Rᴴ + σ_bg²·I` with `σ_bg` tuning `|ρ|` (0.7–0.95). **Tier 1 also produces the calibration curves of *both* κ and `‖Im S‖/‖S‖` against φ on the desk‑check scenario**, from which the co‑primary threshold is pinned (§8‑iv). Tier 2 (needs the real DF‑only `A` from `transfer.py`, **after Gate V**) — push `R_s` through `A`, build the ground CSM, measure `κ_ground` vs `κ_source`; the gap is the washout.
- **Runnable check / metric:** `κ_ground` at realistic `Q`, `|ρ|`, IMAGE‑like spacing (~100–250 km), maximised over pole‑pair placement. Bands: `≥ 0.30` → H‑A holds; `< 0.10` → H‑A fails (report which failure mode); `0.10–0.30` → marginal, feed the curve into B2 as a weighting. Co‑primary `‖Im S‖/‖S‖` at its Tier‑1‑pinned threshold; report the top‑2 eigenvalue gap alongside κ (rotation sensitivity at `λ₁≈λ₂`); Monte‑Carlo‑average Tier 2 over driver/background realisations.
- **Timebox / cost:** small — 1–2 day spike; Tier 1 deterministic, Tier 2 a light MC.
- **Kill:** none — both outcomes are publishable. Only halt: do not trust `κ_ground` until Gate V is green.
- **Where it runs:** Tier 1 — anywhere, **today**, pure NumPy. Tier 2 — `viability-test/` (e.g. `scratch/flr_coherence.py`), after `transfer.py` + Gate V, before B2 is designed.
- **Outputs that prove the result:** `results/A_flr_coherence/` `.npz` (`R(x)` profile, `φ_ij` distribution, `κ_source`, `κ_ground`, sweeps over `Q`/`|ρ|`/spacing, **both φ‑calibration curves**) + `.csv` + `manifest.json`; two figures (`φ_ij` vs pole spacing; `κ_ground`/`κ_source` vs `Q`/`|ρ|` with the 0.10/0.30 bands); the H‑A verdict + **the realistic `|ρ|` range handed to B2** + the resulting B2 φ range and thesis framing.

### Gate V — kernel validation *(was Card A / Experiment A; retired as an experiment)*

- **What it is:** a pass/halt build gate, not a decision. Content: (i) the `secsy` CF‑pair probe (PLAN Appendix C) — verify the library returns the *full* Fukushima pair (horizontal CF current **+** its radial FAC); (ii) `assert np.isrealobj(A)` and convention checks; (iii) DF single‑source ground‑field spot checks against the analytic expressions.
- **Expected result (known in advance):** `ρ_obs ≈ 10⁻¹⁵` (machine precision) at **every** latitude preset — the theorem is latitude‑independent, exact, and conductance‑independent (Fukushima 1976; Amm 1997). Anything materially above machine precision is an **adapter bug** (most likely the horizontal CF piece without its FAC), never physics. There is no latitude crossing, no threshold fork, no subspace statistic — those deliverables are deleted (they belonged to the retired D1; with `‖A·e_CF‖ ≈ 10⁻¹⁵` the direction of the CF ground vector is numerical noise and the subspace fraction is 0/0).
- **Deliverable:** probe outputs archived under `results/V_kernel_validation/` + `manifest.json` (`secsy` version pinned) + **one paper paragraph**: the theorem cited as theory, the probe reported as its numerical confirmation and as the headline calibration on which every downstream number rests.
- **Sequencing:** the **first build action**; gates `transfer.py`, Card A Tier 2, and everything in Experiment B. Does **not** block Card A Tier 1.

### Card B — Does per‑mode independence earn its keep? *(sub‑experiments B1–B4 unchanged in name; B5 retired; B6 new)*

- **Assumption (H‑B):** for a real (phase‑less) kernel, per‑mode GIBF does **not** separate two sources better than a joint MMV‑L1 inversion of the *same* eigenmodes. *(Structural note, §5‑D2: on clean cells the shared‑support prior is correct everywhere, so H‑B is the structurally favored outcome; the alternative has an articulated mechanism only in B6.)*
- **Method (spike, Monte‑Carlo):** shared skeleton — simulate complex phasor snapshots → CSM → `eigh` → eigenmodes `√λ·u` → feed the **identical** `V` and real DF‑only `A` to three solvers (L2 min‑norm baseline, GIBF per‑mode IRLS, MMV‑L1 joint) **under the matched‑regularization / symmetric‑grid‑reduction protocol with the regularization‑sensitivity panel** (root invariant §2). Constituent runs:
  - **B1 — incoherent separation (cost side + pilot + floor):** two equal‑power DF sources, incoherent; `d ∈ {1,2,3,5,8}` cells × `snr_db ∈ {−5,0,5,10,20}`, `K=2` known, `n_snap=64`; grid reduction on and off. Triple duty: (a) **cost measurement** — incoherent eigenmodes are 2‑source mixtures (verified), so MMV is expected `≥` GIBF, penalty largest at small `d`; report the signed gap. (b) **pilot** for the power calc. (c) finite‑sample **floor**. Do **not** decide D2 on B1 alone.
  - **B2 — coherent‑with‑phase (the decision figure):** `|ρ|` **inherited from Card A's realistic range** (no longer fixed at 0.95), sweeping `φ ∈ {0,30,60,90,120,150,180}°` × the B1 SNR/separation grid, with φ range/weighting set by Card A. Report the **signed** `GIBF−MMV` gap per φ. Primary confirmatory cells pre‑registered (§8‑iii); the rest of the grid is descriptive. **The φ‑resolved signed‑gap figure is the Experiment‑B decision figure.** `n_trials` per the amended power procedure (§8‑ii).
  - **B3 — mode‑selection validity:** `snr_db × n_snap ∈ {8,16,32,64,128}`; does AIC/MDL recover `K=2`, and how many snapshots are needed? Feeds B6a its realistic `K̂` error modes; partial de‑risk of D3.
  - **B4 — phantom‑phase ablation (the "why"):** GIBF‑only, real `A` vs `A_phase = A ⊙ exp(i·Φ)`. Isolates how much of acoustic‑GIBF's power came from kernel phase diversity this problem lacks. The key explanatory figure. *(B4 puts phase in the* kernel*; B2 puts phase in the* sources*.)*
  - **B5 — RETIRED** *(CF/DF attribution; moot — CF is excluded by theorem, see Gate V).*
  - **B6 — prior‑misspecification stress cells (new; the only principled per‑mode regime):** **(a)** feed both solvers the *estimated* `K̂` from AIC/MDL (not oracle `K`) at the low‑snapshot / low‑SNR corners where B3 shows `K̂` errs; **(b)** ≤1‑cell source‑position jitter across the snapshot window (mild nonstationarity — the shared‑support assumption is then false). Same metrics, same CI/power standard, own pre‑registered primary cell + rule (§8‑vii). Rationale: on clean cells MMV's prior is exactly correct and it cannot structurally lose, so a GIBF loss there proves less than it appears; B6 closes that reviewer escape route if GIBF loses, and detects a robustness value proposition if GIBF wins. Bridge to D3 / real data.
- **Runnable check / metric:** mean `Δr̄` and `P_sep` (pinned tolerance, §8‑v) with bootstrap/t 95 % CIs per method per cell; apply the §[D2] **win** rule at the primary φ cells and the **loss** rule (§8‑vi) at the null cells; smallest resolved separation at `P_sep ≥ 0.8` per method; the B6 rule (§8‑vii).
- **Timebox / cost:** the bulk of the build (simulate, csm, modeselect, three solvers, metrics, plotting, MC runner). Gate each module on its `pytest` test before running cells. *(Net scope vs the pre‑restructure plan is smaller: old Experiment A's sweep machinery and B5 are gone; B6 + the mini‑pilot are cheaper than what they replace.)*
- **Kill:** see D2. If neither the win rule nor the B6 rule is met, do **not** carry per‑mode inversion downstream.
- **Where it runs:** `viability-test/`, `run_experiment_B.py`. Pure NumPy/SciPy after `transfer.py`.
- **Outputs that prove the result:** `results/B_viability/` `.npz` + `.csv` per cell + the **φ‑resolved signed‑gap decision figure** (B1 floor as reference panel) + the B4 phantom‑phase figure + the **B6 misspecification panel** + the regularization‑sensitivity panel; `manifest.json`; the Question‑2 table in `DECISION_MEMO.md`.

---

## 7. Manifesto Stubs

One paragraph per terminal branch, fields aligned to the manifesto skeleton so expansion is mechanical.

**Stub 0 — Viability Test & Methods Paper** *(being expanded into Manifesto #1 now).*
*Scope:* build the self‑contained `viability-test/` codebase, pass Gate V, run Experiments A and B (B1–B4, B6), and draft a methods paper. *Thesis (sign‑neutral — rewritten 2026‑07‑01):* a **phase‑ and misspecification‑resolved characterization of the *sign* of the per‑mode‑vs‑shared‑support gap on a real kernel** — the φ‑resolved signed gap (B2) over the physically grounded operating map (Experiment A), the cost floor (B1), the misspecification regime (B6), and the mechanism (B4) — leaving the domain scientist an evidence‑based operating rule per phenomenon. Either sign of the gap is the finding; the paper's framing forks on Experiment A's verdict (exploitable‑regime map vs ground‑washout headline), not on GIBF winning. Gate V appears as one theorem‑plus‑confirmation paragraph, **not** as a finding. *Core users:* the research team and the paper's reviewers; secondarily, whoever builds the downstream pipeline and inherits the decisions. *Defining constraint:* correctness, reproducibility, and a *clean decisive result* over generality or performance — pre‑registered thresholds (win **and** loss side), fair GIBF‑vs‑MMV comparison with the sensitivity panel, real DF‑only `A`, complex snapshots — inside an **11‑week** window. *Rough capability list:* config + geometry + real‑`A` transfer (Gate V); FLR coherence model (Card A); simulate + CSM + mode‑selection; three solvers; metrics + plotting; experiment runners A and B + figure regeneration; `DECISION_MEMO.md` + paper draft. *Non‑goals:* the full pipeline; no robustness sweeps beyond B6; **no real‑data demo** (synthetic‑only confirmed); **no CF claims of any kind** (theorem cited, Stub X′ parked).

**Stub F — Full SECS‑GIBF Pipeline (per‑mode layer retained).** *Selected when* D2 = worth it, free option, liability‑but‑gated (gated on observable `‖Im S‖/‖S‖`), or **robustness pivot** (retained for misspecification robustness per B6, framed accordingly). *Scope:* productionise SECS‑GIBF as a reusable imaging pipeline and apply it to real events. *Core users:* space‑physics analysts using IMAGE/SuperMAG data. *Defining constraint:* fidelity on real, non‑stationary data — where D3 becomes load‑bearing and where B6's misspecification result is the leading indicator. *Capabilities (rough):* eigendecomposition + per‑mode sparse inversion (gated per D2's verdict); robustness suite (sensor reduction, altitude mismatch, grid offset); L2/L1‑SECS comparison; real IMAGE/SuperMAG event demonstration; possibly Earth‑induction corrections. DF‑only by theorem.

**Stub S — Sparse‑SECS Pipeline (per‑mode layer dropped).** *Selected when* D2 = drop. *Scope:* the same imaging goal; the pipeline reduces to a single joint sparse (MMV‑L1) SECS inversion (the top‑K subspace may remain as an implementation‑level compression — dropping "per‑mode independence" is the decision, and the paper says so precisely). The paper reframes around *when* per‑mode inversion helps and why it doesn't here (carried by B2 + B4 + B6). *Core users:* same analysts. *Defining constraint:* a simpler, defensible sparse inversion that still resolves superimposed sources; D3 still applies on real data. *Capabilities (rough):* joint sparse SECS solver, robustness suite, real‑data demo. *Non‑goal:* per‑mode inversion.

**Stub X — CF from Satellite / Ground–Satellite Fusion (someday option).** *A dormant, separately‑chosen thread*, not auto‑armed by any outcome: DF‑only ground imaging is the present goal, so X is pursued only if CF/FAC science is later wanted. Orthogonal to F/S — attaches to either. *Scope:* recover CF/FACs where they *are* observable — satellite magnetometers (e.g. Swarm, AMPERE) — possibly fusing ground (DF) with satellite (CF). *Core users:* FAC/electrodynamics researchers. *Defining constraint:* a fundamentally different observation geometry and data source; an exploratory thread, not a refinement of the ground pipeline. *Capabilities (rough):* satellite data ingest, a satellite‑appropriate transfer kernel, ground+satellite joint inversion. *Non‑goal:* claiming CF from ground data alone.

**Stub X′ — Dipolar‑FAC CF Ground Observability (Paper 2 — parked, 2026‑07‑01).** *Deliberately deferred; do not let it creep into the 11 weeks.* *Scope:* replace the radial‑FAC CF SECS with **dipolar elementary current systems** (FACs along dipole field lines; Vanhamäki et al. 2020, Earth Planets Space). Fukushima's cancellation *requires* radial FACs, so with dipolar FACs the mid/low‑latitude CF ground‑observability question becomes genuinely open and live — and the machinery retired with D1 (the `ρ_obs` latitude sweep, the subspace‑separation identifiability statistic, the crossing‑latitude deliverable) is exactly the right instrument *there*. *Prerequisites:* this paper's validated DF pipeline, Gate V's discipline extended to the dipolar kernels (new basis functions ⇒ new validation burden — the reason it cannot fit inside Manifesto #1). *Core users:* mid/low‑latitude electrodynamics researchers. *Defining constraint:* a novel observability claim deserving its own paper's full attention, on the foundation Paper 1 builds either way.

---

## 8. Experiment Log (append‑only)

Immutable history of results. Append corrections; never edit a past entry.

- **2026‑06‑30 — Roadmap created.** Frontier set at the viability test (D1 → D2). No experiments run yet; codebase not yet built. Manifesto #1 (Viability Test & Methods Paper) opened from Stub 0. Pre‑registered thresholds for D1 (`ρ_obs`: `1e‑2`, `1e‑1`) and D2 (≥20 % `Δr̄` / ≥1 cell, non‑overlapping 95 % CIs, ≥2 SNR levels) are fixed and recorded here before any result exists.
- **2026‑06‑30 — Scoping decisions (pre‑experiment).** Venue accepts a synthetic‑only methods paper → real‑data demo stays downstream, out of Manifesto #1. CF/FAC science deferred → Stub X is a someday option, not auto‑armed. Paper drafting to run in parallel with the build (no fixed phase split). None of these touch a result; the D1/D2 thresholds above are unchanged.
- **2026‑06‑30 — Structural desk‑check (analytical, moved the frontier).** Verified in NumPy that with a real `A`, the CSM is real‑up‑to‑global‑phase for incoherent and for in‑phase‑coherent (`φ=0`) sources, so the eigenmodes carry no exploitable complex structure there; exploitable structure appears only for coherent sources with `φ≠0`. Measured residual eigenvector complexity (global phase removed): incoherent ≈ 0.14 (N=64), `φ=0` ≈ 0.01, `φ=90°` ≈ 0.62; `‖Im S‖/‖S‖` for incoherent falls 0.060 (N=64) → 0.0013 (N=10⁵). **Consequences:** (1) B1 (incoherent) reclassified from headline to *floor* — GIBF ≈ MMV there is expected, not informative. (2) Added **inter‑source phase `φ` as an explicit swept axis** to B2, now the discriminating headline; the φ‑resolved `P_sep` is the Experiment‑B decision figure. (3) Hardened the D2 kill so an in‑phase‑only B2 (a null by construction) cannot satisfy the method‑wide‑abandonment AND‑condition. (4) D1 rule gated on a subspace‑separation statistic (`ρ_obs` is observability, not identifiability). (5) Fairness invariant operationalised (matched regularization, symmetric grid reduction). (6) D2 requires a pre‑hoc power analysis. *No experiment has run; this is a desk result that reshaped the forward tree, not a result entry to be corrected later.*
- **2026‑06‑30 — Second desk‑check + cost/benefit reframe + FLR grounding.** (a) Verified incoherent signal eigenmodes are **2‑source mixtures** (top mode projects 0.74–0.99 onto *both* steering vectors, equal power), so B1 is the **cost side**: MMV's shared‑support prior is correct‑and‑stronger, GIBF expected `≤ MMV` and plausibly worse (penalty largest at small `d`). B1 reclassified floor → cost/pilot/floor (triple duty). (b) Verified the full φ arc: residual complexity is **null at φ=0 AND φ=180**, peak ~0.65 at φ=90 (0/0.22/0.44/0.65 at 0/30/60/90). B2 sweep extended to `φ ∈ [0,180]`. (c) Because φ is **latent** (imaged, not observed), replaced "gate on φ" with: report the **signed** `GIBF−MMV` gap; if GIBF ≥ MMV everywhere → run always (free option); if it wins near 90 but loses at the nulls → gate on the **observable** `‖Im S‖/‖S‖` or drop. (d) `n_trials` procedure pinned: powered off the **B1 pilot** SD (not a guessed number). (e) FLR/ULF literature (Waters 1991; Chi & Russell 1998; Samson; Vellante 2004) confirms a systematic ~180° cross‑phase reversal across the resonance; read as source‑current phase structure inherited through the real kernel — consistent with the "phase in `s`" premise and supporting the φ‑sweep as spanning the *realistic* regime.
- **2026‑06‑30 — Pre‑registrations (pin BEFORE running; provisional values to confirm).** (i) **D1 subspace‑separation threshold:** CF counts as *identifiable* (not merely observable) only if the median fraction of CF ground‑signal energy lying *outside* the DF column span exceeds **0.5**. *Corrected 2026‑07‑01: the original draft additionally stated this as "equiv. smallest principal angle > 30°," but `sin²(30°) = 0.25`, not `0.5` — the angle that actually corresponds to a 0.5 outside‑energy fraction is `45°` (`sin²(45°) = 0.5`), not 30°. Separately, "median fraction across grid points" and "smallest principal angle" (an extremal, single worst‑case direction) are not the same statistic in the first place, so the two were never a precise equivalence — treat the energy‑fraction rule as primary and drop the angle framing, or restate it as a genuinely independent secondary check.* Below the 0.5 threshold, downgrade CF‑observable → CF‑marginal. *Provisional — confirm or adjust before Experiment A runs; do not tune after seeing results.* (ii) **B2 `n_trials`:** set by the pre‑registered procedure — target ≥80% power for a true 20% `Δr̄` / 1‑cell effect to clear non‑overlapping 95% CIs, using the B1 pilot SD as the variance input. The *procedure* is fixed now; the *number* is computed once the pilot lands.
(iii) **B2 multiplicity (gap identified 2026‑07‑01, needs sign‑off before Experiment B runs):** B2 sweeps 7 `φ` values × the B1 SNR/separation grid — on the order of 100+ cells, each individually eligible for a non‑overlapping‑CI call. The §[D2] rule as written ("non‑overlapping CIs at ≥2 SNR levels in the φ‑swept regime") does not say *which* cells are confirmatory vs. exploratory, so a reviewer can reasonably ask whether the reported win was cherry‑picked from the grid. Fix before running: pre‑register the primary confirmatory cell(s) (the obvious candidate is `φ≈90°` at the SNR level(s) used for the power calculation in (ii)) as the cells the go/no‑go rule is actually adjudicated on; report the full `φ`‑resolved curve as descriptive/exploratory alongside it, not as additional shots at the same threshold.
- **2026‑07‑01 — Roadmap reconciliation pass (structural, no result).** Card C (FLR source‑coherence forward model) existed as a standalone experiment card but was not wired into this roadmap: absent from §5's decision tree, §6's card list, and §9's status, despite being written specifically to resolve §9's own open question #1. Fixed: added node **C** to §5 (explicitly *not* a stub‑selecting decision like D1/D2/D3 — it sets Card B's design and the paper's framing only; corrected Card C §C.9's overreaching claim that a failing result should "route toward Stub S"); added Card C to §6; updated §9's active node, running experiment, and open‑question‑1 status accordingly; flagged that Card C Tier 1 has no build dependency and should run *before* Card A by information value (cheaper, resolves the most load‑bearing open question). Also caught and fixed a numeric inconsistency in the D1 provisional threshold (§8‑i: the stated "0.5 energy fraction ⇔ 30° principal angle" equivalence is wrong — `sin²(30°)=0.25`, not `0.5`; the two statistics also aren't the same quantity in the first place). Added §8‑iii flagging an unaddressed multiple‑comparisons risk in the B2 φ×SNR sweep (~100+ cells, no pre‑registered primary cell). Flagged (not fixed) that the referenced master build brief (`viability-test/PLAN.md` / repo's untracked `GIBF_viability_BUILD_BRIEF.md`) predates the φ‑sweep/Card‑C revisions and will need its own pass before Manifesto #1 hands off to it.
- **2026‑07‑01 — Peer‑review restructure (structural, no result; supersedes parts of earlier entries per the map in §0).** External peer review of the roadmap; changes verified against the SECS literature before adoption. **(1) D1 retired by theorem.** In the radial‑FAC SECS model class, the CF current + its FAC produce *exactly zero* ground field, independent of the conductance distribution (Fukushima 1976; Amm 1997; the crucial assumption is radial FAC, which the SECS CF basis satisfies by construction at every latitude). The D1 fork's CF‑observable/CF‑collinear branches were therefore unreachable with a correct adapter; the DF‑only "branch" was a restatement, not a finding; with `‖A·e_CF‖ ≈ 10⁻¹⁵` the subspace‑separation statistic is 0/0 numerical noise; and the "latitude at which `ρ_obs` crosses 1e‑1" deliverable does not exist (the theorem is latitude‑independent). Old Experiment A demoted to **Validation Gate V**, whose entire content is the already‑flagged keystone (the `secsy` CF‑pair probe) plus realness/spot checks. Pre‑registration (i) retired; **B5 retired**. **(2) Renames per §0** — old Card C → **Card A** (first by information value; Tier 1 runnable today); D2/D3 keep their names for log continuity; "eigen layer under test" → "per‑mode independence under test" (both arms consume the same eigendecomposition; by the L2,1 norm's right‑unitary invariance, MMV‑on‑eigenmodes ≈ MMV‑on‑snapshots + subspace denoising, so what D2 isolates is per‑mode vs shared‑support). **(3) Structural‑note correction.** The 2026‑06‑30 framing implied MMV's shared‑support prior is correct *in B1 specifically*; in fact it is correct in **every clean synthetic cell, B2 included** (the sources occupy the same two grid cells across all eigenmodes in both regimes) — B1 and B2 differ in the *data's* structure, not the *prior's* correctness, and both arms receive the identical complex `V`. Consequently MMV ≥ GIBF everywhere is the structurally favored D2 outcome, and H‑B's alternative has no articulated mechanism on clean cells. **Stub 0's thesis rewritten sign‑neutral** so no outcome requires reframing after the fact. **(4) B6 added** (prior‑misspecification stress cells: solver‑fed `K̂` from AIC/MDL; ≤1‑cell position jitter) — the only cell class where the shared‑support prior is false and a per‑mode win has a principled mechanism; new **robustness‑pivot** outcome added to D2; method‑wide‑abandonment condition extended to require B6 failure too. **(5) Loss‑side rule added** (mirrored 20 %/1‑cell standard at the φ∈{0,180} null cells, §8‑vi) so liability‑vs‑free‑option is adjudicated, not eyeballed. **(6) Power procedure amended:** variance input = max(B1 pilot SD, 50‑trial mini‑pilot SD at the B2 primary cell) — incoherent‑regime variance need not transfer to the near‑rank‑1 coherent regime. **(7) `P_sep` tolerance pinned** (§8‑v). **(8) B2's `|ρ|`** now inherited from Card A's realistic range (was fixed 0.95). **(9) Card A co‑primary statistic:** `‖Im S‖/‖S‖` (rotation‑invariant), threshold pinned from Tier 1's φ‑calibration curve *before* Tier 2 runs; κ always reported with the top‑2 eigenvalue gap (eigenvector rotation is ill‑conditioned at `λ₁≈λ₂`) and MC‑averaged in Tier 2. **(10) Stub X′ added** (dipolar‑FAC CF ground observability, Vanhamäki et al. 2020) as Paper 2, parked — the retired D1 machinery is the right instrument there, where Fukushima's radial‑FAC assumption genuinely breaks. **(11) Fairness protocol extended** with the regularization‑sensitivity panel (sign of the gap across a band of regularization strengths at the primary cells). **(12) Log‑hygiene rule adopted:** corrections are appended entries referencing the old; no inline edits (the 07‑01 annotation inside the 06‑30 entry is grandfathered as the last exception). *Net scope change: removes old Experiment A's sweep machinery and B5; adds B6 and a mini‑pilot — schedule risk down.*
- **2026‑07‑01 — Pre‑registrations (updated set; pin BEFORE running).** **(i) RETIRED** with D1 (see restructure entry). **(ii) B2 `n_trials` (amended):** procedure unchanged — ≥80 % power for a true 20 % `Δr̄` / 1‑cell effect to clear non‑overlapping 95 % CIs — with the variance input now `max(B1 pilot SD, 50‑trial mini‑pilot SD at the B2 primary cell)`. The *procedure* is fixed now; the *number* is computed once the pilots land. **(iii) B2 primary confirmatory cells (unchanged, sign‑off still required before B runs):** candidate `φ≈90°` at the two SNR levels used in the power calculation; the rest of the φ×SNR grid is descriptive/exploratory. **(iv) Card A thresholds — SIGN OFF TODAY, BEFORE Tier 1 RUNS:** `κ_ground` bands **0.30** (H‑A holds) / **0.10** (H‑A fails) confirmed as final; the `‖Im S‖/‖S‖` co‑primary threshold is pinned by the fixed *procedure*: Tier 1 maps both statistics against φ on the desk‑check scenario, and the co‑primary threshold is read off at the same effective φ values as the κ bands, *before* Tier 2 runs. **(v) `P_sep` tolerance:** *correctly placed* = within **1 grid cell** of a true source under minimum‑cost matched assignment; *distinct* = the two reported peaks match different true sources. **(vi) Loss‑side rule:** MMV `Δr̄` ≥20 % lower OR ≥1 cell smaller, non‑overlapping 95 % CIs, at ≥2 SNR levels, at the pre‑registered null cells `φ ∈ {0°, 180°}`. **(vii) B6 rule:** same effect‑size / CI / power standard as the win rule; primary cell = the solver‑fed‑`K̂` variant (B6a) at the mid‑SNR, mid‑separation cell — confirm the exact cell before Experiment B runs.

---

## 9. Status / current frontier

- **Active node: A (Card A, Tier 1 — FLR source‑coherence forward model) — run TODAY, in two steps.** **Step 1:** sign off pre‑registration §8‑iv (the κ bands and the co‑primary pinning procedure) — two minutes, and it is the pre‑registration discipline actually working: thresholds confirmed *before* the first number exists. **Step 2:** run the Tier‑1 spike (pure NumPy, zero build dependency): the `R(x)` profile, `φ_ij` vs pole spacing and `Q`, and the κ / `‖Im S‖/‖S‖` calibration curves vs φ. Do **not** run Tier 2 until Gate V is green.
- **First build action:** **Gate V** (the `secsy` CF‑pair probe + realness/spot checks) — gates `transfer.py`, Card A Tier 2, and everything in Experiment B. Its outcome is known in advance (theorem); its job is proving the *code* implements the physics. Report as the headline calibration.
- **Sequence:** Card A Tier 1 (today) → Gate V → `transfer.py` → Card A Tier 2 → design B2/B6 (requires §8‑iii/vi/vii sign‑off + Card A's φ/`|ρ|` result) → Experiment B (B1 pilot → mini‑pilot → powered B2/B6, with B3/B4 alongside) → `DECISION_MEMO.md` → paper close.
- **Manifesto being written:** **Manifesto #1 — Viability Test & Methods Paper**, expanding Stub 0 (thesis now sign‑neutral). The master build brief (`viability-test/PLAN.md` / untracked `GIBF_viability_BUILD_BRIEF.md`) is **flagged stale, now doubly**: it predates the φ‑sweep/Card‑C revisions *and* the 2026‑07‑01 restructure (D1 retirement, renames, B6, loss rule). It needs a revision pass before Manifesto #1 hands off to it.
- **Timeline:** 11 weeks, no fixed internal split; build and paper in parallel. The restructure **removes** scope (old Experiment A's latitude‑sweep machinery, the subspace statistic, B5) and **adds** less than it removes (B6, one mini‑pilot, the sensitivity panel) — net schedule risk down. Ordering gate unchanged: Gate V before anything consuming `A`; Card A Tier 2 before B2 is designed.
- **Venue question — resolved:** synthetic‑only acceptable; no real‑data risk in the 11 weeks.
- **Headline framing:** the Experiment‑B decision rests on the **φ‑resolved signed gap** over Card A's physically grounded operating map, with B1 as the paired cost measurement, B6 as the misspecification regime, and B4 as the mechanism. Gate V remains the highest‑leverage build risk and is reported as the headline calibration.
- **Open questions that gate airtightness (resolve before/at manifesto lock):**
  1. **Fairness protocol (publication‑critical).** Matched regularization + symmetric grid reduction + the L2 baseline + the single‑source exact‑recovery gate + the regularization‑sensitivity panel, concrete enough that a reviewer can say neither "MMV was under‑tuned" nor "that isn't Suzuki's GIBF." Pin in the manifesto.
  2. **Threshold sign‑offs:** §8‑iv **today** (before Tier 1); §8‑iii, ‑vi, ‑vii before Experiment B runs.
  3. **Stale build brief** (see above) — revise before Manifesto #1 handoff.
- **Most load‑bearing next move:** sign off §8‑iv, then run **Card A, Tier 1 — today**. No build dependency; it resolves the hinge question (does realistic FLR phase land in the exploitable band?), sets B2's design, and is the first half of the paper's contribution landing.

**Done‑check:** the next experiment is **Card A, Tier 1** (source‑coherence forward model; §8‑iv signed off first); then **Gate V** (first build action); research ends when Card A has fixed B2's design, D2 is landed per its pre‑registered **win and loss** rules including B6, the memo is filled, and the paper is drafted (§4).
