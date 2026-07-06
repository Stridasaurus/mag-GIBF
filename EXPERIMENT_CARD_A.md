# Experiment Card A — FLR source‑coherence forward model

> **Renamed 2026‑07‑01** (was Experiment Card C; see `ROADMAP.md` §0 rename map). The alphabet now matches the run order: this is the **first experiment**, and Tier 1 runs **today**.
>
> **Layer:** Research Roadmap (Decide phase) — an **Experiment Card**, not a SPEC. It slots into `ROADMAP.md` §6 alongside Card B and Gate V. It is written to be **runnable without this conversation's context**. The technical SPEC (module contracts, `secsy` adapter, paths) is written later in Claude Code with repo context; do not write it here.
>
> **Resolves:** `ROADMAP.md` §9 open question — the load‑bearing "hinge." Its result is a **pre‑registered input to D2's design and to the paper thesis**, and (per Stub 0's sign‑neutral thesis) one of the paper's two co‑equal contributions: D2 is the methods result, this card is the physics result, and B4 is the mechanism that welds them together. It is **not** a branch to a different manifesto stub — D2 alone routes stubs.
>
> **Provenance:** motivated by the 2026‑06‑30 desk‑checks (residual eigenvector complexity is null at φ=0/180, peaks ~0.65 at φ=90) and the FLR cross‑phase literature (Waters 1991; Chi & Russell 1998; Vellante 2004). Uses the roadmap glossary verbatim.
>
> **Before running (pre‑registration §8‑iv):** confirm the κ_ground bands (0.30 / 0.10) and the co‑primary pinning procedure below. Do not look at a single number first.
>
> **✔ SIGNED OFF 2026‑07‑06, with the floor amendment** (see `ROADMAP.md` §8, 2026‑07‑06 entry): snapshot count pre‑registered at **N=64** (N=1024 asymptotic secondary); Tier‑1 calibration must include incoherent and φ=0 **floor controls at N=64 for both statistics**; the fail arm of §A.4 is **floor‑referenced**. Tier 1 is now unblocked.

---

## A.0 The question, in one line

For a **physically realistic field‑line resonance (FLR)**, does the inter‑source phase `φ` seen at the **ground CSM** — after the real kernel `A` and ionospheric spatial integration — land in the exploitable band (near 90°) or collapse to a null (0°/180°)? This sets B2's realistic operating point (φ range, weighting, and `|ρ|`) and decides the paper's framing fork: "exploitable structure exists in the physically dominant regime" versus "phase structure does not survive to the ground CSM."

## A.1 Assumption under test

> **H‑A:** A realistic FLR induces, between distinct SECS source elements, a coherence `ρ = |ρ|·e^{iφ}` with `φ` substantially different from 0 and 180°, and that phase **survives to the ground CSM** through the real kernel; so the physically realistic ULF regime is one in which the CSM eigenstructure carries exploitable complex structure.

The null of H‑A has two distinct failure modes, and the experiment must separate them:
- **Source‑phase failure:** the FLR itself maps to near‑real `ρ` between SECS poles (φ≈0 or ≈180 for the geometries that occur). Then B2's realistic point is a null on physics grounds.
- **Ground‑washout failure:** the source phase is φ≠0, but ionospheric spatial integration + the real kernel smear it, so the **ground** CSM is near‑real even though the source `R_s` is not. This is the Anderson (1989) / Poulter & Allan (1985) effect — ground observations show reduced latitudinal variation relative to the source. If this dominates, the complex eigenstructure is inert on ground arrays *regardless* of source physics — which is itself a headline result about ground observability of source‑phase structure.

## A.2 Background sufficient to run standalone

The SECS‑GIBF premise is `B(ω) = A·s(ω)` with **`A` real**; all phase lives in the complex source phasors `s`. The FLR cross‑phase measured on the ground (~180° reversal across the resonance) is therefore *not* a propagation phase across the array — it is the spatial phase structure of the standing Alfvén wave, carried in the **source currents** and inherited through the real `A`. This card tests whether, once that source‑phase structure is pushed through `A` (which mixes nearby poles = spatial integration), the ground CSM retains exploitable imaginary structure.

Two phases must never be conflated: **inter‑observer** phase (across ground stations — the classic FLR cross‑phase signature) versus **inter‑source** phase (the off‑diagonal phase of the source cross‑spectral matrix `R_s` between SECS poles). B2 is parameterised by the latter. This card is the bridge that establishes the latter from a model of the former's physical cause.

`A` is **DF‑only**: CF columns are excluded by theorem (Fukushima 1976; Amm 1997 — see `ROADMAP.md` Gate V), not by threshold.

## A.3 Model (throwaway spike; deterministic core + light Monte Carlo)

### A.3.1 FLR source‑phase model (Tier 1 — needs no `secsy`; runs today)

Model each field line (indexed by latitude `x`, mapping to a resonant frequency `ω_r(x)`) as a **driven damped harmonic oscillator** (Southwood 1974; Chen & Hasegawa 1974). The current‑phasor response at a monochromatic driver frequency `ω` is

```
R(x) = 1 / ( ω_r(x)² − ω²  +  i·γ·ω )
```

- `|R(x)|` peaks at the resonant latitude `x_r` where `ω_r(x_r)=ω`.
- `arg R(x)` sweeps **0 → −90° → −180°** as `ω_r(x)` decreases through `ω` (verify sign against your convention; the *range* is the physics, ~180° total).
- Linearise `ω_r²−ω² ≈ slope·(x−x_r)` across the array; then the phase transition width is `~ γω/slope`. Define quality factor `Q ≡ ω/γ`; realistic FLR `Q ≈ 5–20` ⇒ a **narrow** resonance ⇒ the phase reversal is concentrated over a small latitude band.

The inter‑source phase between poles at `x_i, x_j` is `φ_ij = arg R(x_i) − arg R(x_j)`. **Geometric prediction to confirm numerically:** a pair straddling the *full* resonance sees `φ≈180°` (a null); a pair with one pole near resonance and the other off by ~one resonance width sees `φ≈90°` (exploitable); a pair on the same flank sees `φ≈0` (a null). So whether realistic geometry populates the exploitable band depends on **station/pole spacing relative to resonance width** — a sweepable parameter.

**Tier 1 additionally produces the calibration curves (new, 2026‑07‑01):** on the desk‑check two‑source scenario, map **both** statistics — `κ` (residual eigenvector complexity) **and** `‖Im S‖/‖S‖` (rotation‑invariant CSM imaginary fraction) — against φ over the full arc. The κ curve must reproduce the desk‑check anchors (≈0 / 0.22 / 0.44 / 0.65 at 0/30/60/90°; ≈0 at 180°). The co‑primary `‖Im S‖/‖S‖` threshold for §A.4 is then **pinned by procedure** (pre‑registration §8‑iv): read it off this curve at the same effective φ values where κ crosses its 0.10 and 0.30 bands — *before* Tier 2 runs.

**Floor controls (amendment, §8‑iv sign‑off 2026‑07‑06):** the calibration runs at the pre‑registered **N=64** and must additionally produce, for *both* statistics at that N, two floor controls — (i) the **incoherent** two‑source case (desk‑check anchor: κ ≈ 0.14 at N=64 — note this sits *above* the 0.10 fail band, which is the reason this amendment exists) and (ii) the **φ=0 coherent** case. These empirical floors are what the §A.4 fail arm is referenced against. Report the same curves at N=1024 as the asymptotic secondary.

### A.3.2 Snapshot / coherence model

Across `N` snapshots, the whole FLR pattern oscillates under one shared complex driver `d_t`, plus an incoherent per‑pole background `n_t` that sets the realistic coherence magnitude:

```
s_t(x) = d_t · R(x)  +  n_t(x),     d_t ~ CN(0,1),   n_t(x) ~ CN(0, σ_bg²) i.i.d. per pole
```

⇒ source cross‑spectrum `R_s = R Rᴴ + σ_bg²·I` (a rank‑1 coherent structure on an incoherent floor). The off‑diagonal phase of `R_s` is exactly the FLR cross‑phase `φ_ij`; `σ_bg` tunes `|ρ|` (sweep to cover realistic `|ρ| ≈ 0.7–0.95`). **The realistic `|ρ|` range found here is a deliverable: B2 inherits it** (B2's coherence magnitude is no longer fixed at 0.95).

### A.3.3 Push through the real kernel (Tier 2 — the deciding tier; after Gate V)

Reuse the real transfer matrix `A` (from `transfer.py`, **DF‑only — CF excluded by theorem; see Gate V**). Simulate ground snapshots `X = A·S_grid + sensor noise`, where `S_grid` places `s_t(x)` on the SECS grid (FLR current profile in latitude, flat or realistic in longitude). Build the ground CSM `S = (1/N)·X·Xᴴ`, `eigh`, and measure the **residual eigenvector complexity** of the top‑2 eigenmodes (global phase removed — the min‑over‑θ of `‖Im(e^{iθ}u)‖/‖u‖`, i.e. `sqrt(λ_min)` of the 2×2 real/imag Gram matrix). This is the same statistic as the 2026‑06‑30 desk‑check, so results are directly comparable to the abstract φ‑curve.

**Degeneracy caveat (new, 2026‑07‑01):** eigenvector rotation is ill‑conditioned when `λ₁ ≈ λ₂` (near‑equal‑power modes), which can swing per‑vector κ. Therefore: (i) always report the top‑2 **eigenvalue gap** alongside κ; (ii) **Monte‑Carlo‑average** κ over driver/background realisations (the light MC below); (iii) report the rotation‑invariant co‑primary `‖Im S_ground‖/‖S_ground‖` in parallel — a verdict is robust only if both statistics land on the same side of their bands.

**Ionospheric spatial integration** enters automatically through `A` (each station sees a distance‑weighted sum of grid poles at the 110 km layer). To probe the washout risk explicitly, optionally add a source‑layer smoothing kernel (finite FLR azimuthal/latitudinal width) and report residual complexity **at the source** (`κ_source`, from `R_s`) versus **at the ground** (`κ_ground`, from `S`): the gap is the washout.

## A.4 Pre‑registered decision rule (pinned; §8‑iv signed off 2026‑07‑06 with the floor amendment — N=64, floor‑referenced fail arm)

**Primary statistic:** ground‑level residual eigenvector complexity `κ_ground` of the top‑2 CSM eigenmodes, at realistic `Q` (5–20), realistic `|ρ|` (0.7–0.95), and station/pole spacing matched to a real array (e.g. IMAGE ~100–250 km meridional spacing), maximised over the pole‑pair placement relative to the resonance (i.e. the *best case* a real array could sample). MC‑averaged; reported with the eigenvalue gap.

**Co‑primary statistic:** `‖Im S_ground‖/‖S_ground‖`, at the threshold pinned from the Tier‑1 calibration curve (§A.3.1, §8‑iv), evaluated above its finite‑N floor (~`1/√N`; 0.060 at N=64).

- `κ_ground ≥ 0.30` (co‑primary concurring) → **H‑A holds.** The realistic FLR regime is exploitable at the ground; B2's realistic operating point is the phase‑offset band; framing: exploitable structure exists in the physically dominant regime. Set B2's headline φ near the modelled realistic value.
- `κ_ground < 0.10` **or statistically indistinguishable from the same‑N floor controls** (co‑primary concurring in the same floor‑referenced sense; amendment 2026‑07‑06) → **H‑A fails.** Either the source is near‑real or (report which) ground integration washes the phase out. The complex eigenstructure is inert on real FLRs from ground arrays — a **headline result in its own right**; the paper's framing pivots to "phase structure does not survive to the ground CSM," with the source‑vs‑ground gap as the mechanistic figure. (Per `ROADMAP.md` node A: B2 still runs the full arc; this card never routes stubs.)
- `0.10 ≤ κ_ground < 0.30`, or the two statistics disagree → **marginal.** Report `κ_ground` vs `Q`, `|ρ|`, and spacing; feed the curve into B2 as a weighting rather than a single operating point; if the statistics disagree, report both and treat the rotation‑invariant one as the tiebreaker for the *weighting*, while flagging the disagreement explicitly.

**Secondary statistics:** (a) the source‑vs‑ground residual gap `κ_source − κ_ground` (quantifies washout, and separates the two H‑A failure modes); (b) the distribution of `φ_ij` over pole pairs vs spacing/`Q` (Tier 1); (c) `κ_ground` vs `Q` and vs `|ρ|` (robustness of the verdict); (d) the realistic `|ρ|` range (handed to B2).

## A.5 Timebox / cost

Small — a 1–2 day spike. The core is deterministic forward modelling; the only Monte Carlo is a light average over driver/background realisations for the `|ρ|` sweep and the κ degeneracy averaging. No large grids.

## A.6 Where it runs, dependencies, sequencing

- Claude Code spike in `viability-test/` (e.g. `scratch/flr_coherence.py`, gitignored), or a short runner if promoted.
- **Tier 1** needs only NumPy — it runs **today**, before any build action, immediately after the §8‑iv sign‑off.
- **Tier 2** needs the real DF‑only `A` from `transfer.py`, so it slots **after** Gate V + the transfer milestone (the first build gate) and **before** B2 is designed — its result sets B2's φ range, weighting, `|ρ|`, and the thesis framing. Order: **Card A Tier 1 (today) → Gate V → `transfer.py` → Card A Tier 2 → design B2/B6 → run Experiment B.**
- Uses the roadmap glossary verbatim (SECS, DF/CF, CSM, `R_s`, `ρ`, `φ`, `‖Im S‖/‖S‖`, residual eigenvector complexity, real kernel `A`, Gate V).

## A.7 Kill / null handling

There is no "kill" — both outcomes are publishable (H‑A holding grounds the exploitable regime physically; H‑A failing is itself a headline about ground observability of phase structure, and composes with any Experiment‑B outcome into one coherent story — see the roadmap's §4 quadrant map). The only halt: if Tier 2 depends on a real `A` that has not passed **Gate V** (the `secsy` CF‑pair probe / realness assertions), do not trust `κ_ground` until it has.

## A.8 Outputs that prove the result

- `results/A_flr_coherence/`: `.npz` with `R(x)` profile, `φ_ij` distribution, `κ_source`, `κ_ground` (MC‑averaged, with eigenvalue gaps), `‖Im S‖/‖S‖`, sweeps over `Q`/`|ρ|`/spacing, **both Tier‑1 φ‑calibration curves**; `.csv` summary; `manifest.json` (config, `secsy` version if used, git commit, seed).
- Two figures: (i) `φ_ij` vs pole spacing across the resonance (Tier 1); (ii) `κ_ground` (and `κ_source` overlaid) vs `Q`/`|ρ|` with the 0.10 / 0.30 decision bands marked.
- One paragraph: the verdict per §A.4, the realistic `|ρ|` range, and the resulting B2 φ‑range + thesis framing.

## A.9 Feedback into the roadmap (what to append after it runs)

- Append an **Experiment Log** entry (`ROADMAP.md` §8): the `κ_ground` value (with eigenvalue gap and co‑primary), the H‑A verdict, the realistic `|ρ|` range, and the resulting B2 φ headline / thesis framing.
- Update **D2 / Card B design inputs**: B2's φ range, weighting, and `|ρ|`. *(Corrected 2026‑07‑01, per the roadmap reconciliation: this card **never routes stubs** — a failing H‑A changes what the per‑mode layer's benefit means physically and pivots the paper's framing; it does not substitute for measuring D2. The earlier "route toward Stub S" phrasing is retired.)*
- Resolve the hinge open question in **§9 Status**.

## A.10 References

- Southwood, D. J. (1974). *Some features of field line resonances in the magnetosphere.* Planet. Space Sci. — the driven‑oscillator FLR model.
- Chen, L., & Hasegawa, A. (1974). *A theory of long‑period magnetic pulsations.* J. Geophys. Res.
- Samson, J. C. et al. — the ~180° phase reversal across the resonance.
- Waters, C. L., Menk, F. W., & Fraser, B. J. (1991). *Cross‑phase method for FLR eigenfrequencies* (GRL 18, 2293).
- Chi, P. J., & Russell, C. T. (1998). *Interpretation of the cross‑phase spectrum by FLR theory* (GRL 25, 4445).
- Poulter, E. M., & Allan, W. (1985); Anderson, B. J. et al. (1989); Vellante, M. et al. (2004, JGR) — ground spatial integration reducing latitudinal variation (the washout risk).
- Fukushima, N. (1976); Amm, O. (1997) — the CF+FAC null‑field theorem (cited via Gate V; not re‑tested here).

---

*Assumptions are idealisations (driven‑oscillator FLR, simple integration model). The card's job is to establish the sign and rough magnitude of the ground‑level phase survival, not a precise real‑data number — enough to fix B2's realistic regime and the thesis, honestly, before eleven weeks are committed.*
