# SPEC — Experiment B design (B1–B4, B6), with the B2/B6 confirmatory structure and the mode‑selection protocol

> **Layer:** technical SPEC (repo context) — the design layer between `ROADMAP.md` §6 Card B (the decision layer, authoritative) and the code. Written 2026‑07‑07, the same sitting as the §8‑iii/vi/vii sign‑offs. Where this SPEC and the master build brief (`GIBF_viability_BUILD_BRIEF.md`, **flagged stale**) conflict, **this SPEC and the roadmap win**; conflicts are listed in §S.9.
>
> **Status: design frozen; both Card‑A slots FILLED by Card A Tier 2 (2026‑07‑12, §S.7); fairness protocol PINNED (2026‑07‑19, §S.6)** — no design inputs remain open (roadmap §9 open question 1 closed). No Experiment‑B code exists yet; nothing here was tuned to a result. *(That last sentence is as‑written 2026‑07‑07; the module stack was built 2026‑07‑21.)*
>
> **Amended 2026‑07‑27 (geometry‑rescale pass; ROADMAP §8 2026‑07‑27, Strider's §8‑ii sign‑off):** two additions, both to *unprotected* design surface — **no pre‑registered coordinate moved** (`d = 3`, `φ = 90°`, `snr ∈ {5,10}`, `|ρ| = 0.85`, `n_snap = 64`, `τ_r = 1`, reduction ON all stand exactly as signed off). **§S.4** — the eigenvalue floor's numeral is unchanged but is now denominated in trace‑normalized units. **§S.10 (new)** — the array + SECS grid geometry, previously pinned nowhere and inherited by accident, is pinned; the SECS grid is refined so the pre‑registered `d = 3` is one station spacing rather than 1.7× the array's sampling limit.

## S.0 What is being decided, restated

D2 (roadmap §5): does per‑mode GIBF beat joint MMV‑L1 on the *same* eigenmodes of the *same* ground CSM through the *same* real DF‑only `A`? Structurally favored outcome: MMV ≥ GIBF on every clean cell (the shared‑support prior is correct there); the only principled per‑mode mechanism is B6's misspecification cells. The decision figure is the **φ‑resolved signed `GIBF−MMV` gap** (B2), adjudicated only at the pre‑registered cells below.

## S.1 Shared Monte‑Carlo skeleton (all B runs)

Per brief §9 (still valid), with the B6 arms added:

```
for cell in scenario_grid:
    for trial in range(n_trials):
        rng   = Generator(seed derived from (master_seed, cell_id, trial))
        X     = simulate_snapshots(cfg, tm, truth, rng, coherence)   # complex phasors
        S     = build_csm(X); lam, U = eigendecompose(S)
        K     = 2 (oracle: B1/B2/B4/B6b)  |  estimate_n_sources(lam, N, "mdl") (B3 record‑only, B6a solver‑fed)
        V     = eigenmodes(lam, U, K)      # sqrt(lam)-weighted; IDENTICAL to all solvers
        maps  = { l2, gibf, mmv }          # same A, same V, same solver params
        record per-trial metrics(map, truth) per method
    aggregate mean ± 95% CI per method per cell
```

Invariants (roadmap §2 / brief §0.2): `A` real DF‑only (post Gate V), `assert np.isrealobj(A)` in every solver except B4's `phantom_phase` arm; snapshots are complex frequency‑bin phasors; GIBF and MMV receive the identical `V`, `A`, `eps_frac`, `beta`, `weight_floor`, `max_iter`, `tol`, and grid‑reduction setting per run.

**CI convention (pinned here):** percentile bootstrap over trials, 2000 resamples, seeded from the cell id; one method everywhere, named in the memo. "Non‑overlapping 95 % CIs" always means the two per‑method CIs (the conservative ≈ α 0.006 reading in §5‑D2).

## S.2 Sub‑experiment designs

### B1 — incoherent separation (cost side + pilot + floor)
As brief §9‑B1 with the 07‑01 reclassification: two equal‑power DF sources, incoherent; `d ∈ {1,2,3,5,8}` cells × `snr_db ∈ {−5,0,5,10,20}`; `n_snap = 64`; oracle `K=2`; grid reduction on **and** off. Never adjudicates D2. Triple duty: signed‑gap cost measurement (penalty expected largest at small `d`), the §8‑ii pilot SD, the finite‑sample floor panel of the decision figure.

### B2 — coherent with phase (the decision experiment)
B1's grid × `φ ∈ {0,30,60,90,120,150,180}°`, coherent pairs `s2 = ρ·s1 + √(1−|ρ|²)·CN(0,P)` with `ρ = |ρ|·e^{iφ}` fixed across snapshots. `|ρ|` = **SLOT‑1** (Card A realistic range; midpoint at the confirmatory cells, range as descriptive rows). Report the signed `GIBF−MMV` gap per φ. Confirmatory cells per §8‑iii (§S.3); everything else descriptive.

### B3 — mode‑selection validity (descriptive; feeds B6a)
Two incoherent equal‑power sources; `snr_db ∈ {−5,0,5,10,20}` × `n_snap ∈ {8,16,32,64,128}`; large `n_trials`. Record **both** AIC and MDL `K̂` distributions vs truth `K=2` (comparison is B3's point), but MDL is the pinned consumer‑facing estimator (§8‑viii). Deliverable to B6a: the MDL `K̂` error‑rate table over (snr, n_snap), from which the §8‑vii procedure reads B6a's `n_snap`.

### B4 — phantom‑phase ablation (mechanism)
As brief §9‑B4, unchanged: GIBF‑only, real `A` vs `A ⊙ exp(iΦ)` geometric ramp; the only complex‑`A` site, behind the `phantom_phase` flag.

### B5 — retired (theorem; Gate V). Not built.

### B6 — prior‑misspecification stress cells (the only principled per‑mode regime)

**B6a — solver‑fed `K̂` (confirmatory arm).** Identical to a B2 cell except: per trial, `K̂ = estimate_n_sources(λ, N, "mdl")` (§8‑viii) replaces oracle `K=2`, fed **identically** to GIBF and MMV (and L2 context). Paired **oracle‑K reference arm** on the *same trials* (same seeds, same `X`): the per‑trial paired difference `metric(K̂) − metric(K=2)` per method isolates the cost of mode‑count error, and the GIBF−MMV comparison under `K̂` is the confirmatory statistic. Report additionally the `K̂`‑conditioned breakdown (trials with `K̂ < 2`, `= 2`, `> 2`) — descriptive, it explains *why* whichever method degrades. Cell: §S.3. An AIC‑fed variant may appear as a sensitivity row only (§8‑viii).

**B6b — support jitter (descriptive arm).** Oracle `K=2` (isolates the jitter axis; B6a isolates the `K̂` axis). Coherent φ=90°, SLOT‑1 `|ρ|`, snr 5 dB, `d=3`, `n_snap=64`. Primary variant: **mid‑window 1‑cell step** — snapshots 1..N/2 with source 2 at its nominal cell, N/2+1..N displaced one cell (mild drift nonstationarity; the shared‑support prior is false across the window). Secondary variant: i.i.d. per‑snapshot uniform jitter over the 3×3 neighborhood (stronger smearing). Truth for metrics = the nominal (window‑mean) cell; endpoint‑referenced Δr̄ reported as a diagnostic. No confirmatory rule (§8‑vii pins B6a only).

## S.3 Pre‑registered adjudication (pinned; §8 references authoritative)

| Item | Pin | Source |
|---|---|---|
| Win rule cells | B2, **φ = 90°**, `snr_db ∈ {5, 10}`, `d = 3`, `n_snap = 64`, `|ρ|` = SLOT‑1 midpoint | §8‑iii (07‑07) |
| Win rule reading | GIBF Δr̄ ≥ 20 % lower at `d=3`, **or** smallest `d` with `P_sep ≥ 0.8` ≥ 1 cell smaller along the full `d` axis at the confirmatory (φ, snr); non‑overlapping 95 % CIs at **both** SNR levels | §5‑D2 + §8‑iii |
| Loss rule | mirrored standard for MMV at the null cells; met at **≥ 1 of φ ∈ {0°, 180°}** (other null always reported), same SNR pair and `d` reading | §8‑vi (07‑07) |
| B6 rule | win‑rule standard at the B6a cell: φ = 90°, `d = 3`, snr = 5 dB, `n_snap` = largest of {8,16,32,64,128} where B3's MDL `K̂` error rate ≥ 20 % at 5 dB (fallback 8) — read from B3 **before** B6 runs | §8‑vii (07‑07) |
| Power | `n_trials` such that a true 20 % / 1‑cell effect clears CI non‑overlap at ≥ 80 % power; variance = max(B1 pilot SD, 50‑trial mini‑pilot SD at the φ=90°/5 dB/`d=3` cell) | §8‑ii |
| `P_sep` | correctly placed = within **1 grid cell** under min‑cost matched assignment; distinct = peaks match different true sources | §8‑v (supersedes brief's 1.5‑cell `τ_r`) |
| `K̂` estimator | MDL (Wax–Kailath), solver‑fed everywhere a single count is consumed; AIC reported in B3 and as sensitivity rows only | §8‑viii |

> **Amendment 2026‑08‑01 (§S.11).** Every `d = 3` in this table's Win rule cells / B6 rule / Power rows is superseded by **`d = 8`**, per the blind draw in §S.11 (the `d=3` cell adjudicated as a ceiling/saturation result, not a win or loss). Left as‑written above rather than silently rewritten, for the same audit‑trail reason §S.10 was added as an amendment rather than an edit to the original pins: a methods paper needs the history of what was pre‑registered when, not just the current value. `φ = 90°`, `snr_db ∈ {5,10}`, `n_snap = 64`, `|ρ|` = SLOT‑1 midpoint are unaffected.

D2's outcome map (worth‑it / free option / liability / robustness pivot / drop / inconclusive) is read off these adjudicated cells exactly as in roadmap §5‑D2.

## S.4 Mode‑selection design (consolidated)

`modeselect.estimate_n_sources(eigvals, n_snapshots, criterion="mdl")`, log‑domain Wax–Kailath per brief §6.5 with eigenvalue floor `1e‑18` **in trace‑normalized units** (amended 2026‑07‑27, see below); returns `K̂` plus the full AIC(k)/MDL(k) arrays.

> **Amendment 2026‑07‑27 — the floor is re‑expressed in trace‑normalized units (ROADMAP §8 2026‑07‑27, Strider's §8‑ii sign‑off ruling 2).** The pinned numeral `1e‑18` is **unchanged**; only the units it is denominated in change. Applied to *raw* eigenvalues it is an **absolute** floor calibrated for an O(1) unit‑power scenario (Tier 1's convention), so against Experiment B's real ground CSM (~`1e‑24` T²) every eigenvalue clipped to the same value, the log‑domain LLR flattened to 0 for every candidate `k`, and MDL/AIC reported `K̂ = 1` at 100 % of B3 cells (the 2026‑07‑21 finding). `estimate_n_sources` now trace‑normalizes internally (`λ × 2/trace`, the repo‑wide convention) **before** applying the floor, making it scale‑free by construction. This is content‑preserving: the Wax–Kailath LLR `N·p·(mean log λ_tail − log mean λ_tail)` is provably invariant to `λ → cλ` for `c > 0`, and the penalty terms contain no eigenvalues — so the normalization is a no‑op on any full‑rank spectrum and changes only *which* eigenvalues the floor catches. The floor's sole remaining job is keeping `ln` finite on the exact zeros of a rank‑deficient sample CSM (`n_snap < n_channels`), which is exactly what it was for. **Why in the units and not at the call site:** this was the third mis‑fire of one root cause (Card A Tier 2's gap‑conditioning statistic → §S.6.1's solver scale convention → §S.4), and the same sitting's grid refinement (§S.10) moves the CSM scale again. `tests/test_modeselect.py` pins the invariance across 48 decades including the rank‑deficient case; the 2026‑07‑21 dual raw/normalized reporting path is retired (reproducible at commit `12ff377`). The brief's "AIC or MDL" ambiguity is resolved: **MDL is primary** (§8‑viii); `criterion` stays an argument so B3 can score both. Consumers: B3 (record‑only, both criteria), B6a (solver‑fed, MDL), downstream D3 diagnostics (MDL). `K̂` is clipped to `[1, K_max]` with `K_max = 6` (guards the pathological all‑modes case; clipping events counted and reported).

## S.5 Outputs

`results/B_viability/`: per‑cell `.npz` (per‑trial metrics, per‑method) + `.csv` summaries + `manifest.json` (config hash, git commit, `secsy` SHA, master seed, n_trials with its power‑calc provenance). Figures: (i) the **φ‑resolved signed‑gap decision figure** with B1 floor panel and the confirmatory cells marked; (ii) B4 phantom‑phase; (iii) **B6 misspecification panel** (B6a paired‑difference + `K̂` breakdown; B6b variants); (iv) regularization‑sensitivity panel (§S.6); (v) B3 `K̂` recovery surfaces (AIC + MDL). Every figure's underlying arrays saved.

## S.6 Fairness protocol (PINNED 2026‑07‑19)

> Draft → **pinned before any Experiment‑B code or run exists** (timing ruling: ROADMAP §8, 2026‑07‑19 amendment — pre‑registration must precede confirmatory runs; "at manifesto lock" was written before PR #1 cleared the runway, and §9 itself permits "before"). Each component carries its pin date + rationale. The protocol is built to defeat the two standing reviewer attacks by construction: **"MMV was under‑tuned"** (components 1, 5) and **"that isn't Suzuki's GIBF"** (component 2).

1. **Matched regularization (pinned 2026‑07‑19).** One shared literal config object for both sparse solvers: `eps_frac = 0.01`, `p_norm = 1.0`, `weight_floor = 1e‑10`, `beta = 0.9`, `max_iter = 50`, `tol = 1e‑4`, `min_active_factor = 2`; a pytest asserts both solvers read the identical object. **The μ tie is tie‑by‑rule, not tie‑by‑value:** neither solver carries an explicit `μ` — each realizes its penalty through `eps = eps_frac · λ_max(A W Aᵀ)` (underdetermined branch; `eps_frac · λ_max(AᵀA)` overdetermined) computed on its **own current weighted system** (per mode per IRLS iteration for GIBF; once per iteration on the row‑norm‑weighted system for MMV). What is matched is the rule: it is scale‑covariant, which is what "equally tuned" means across structurally different priors — matching raw ε across solvers would mismatch effective penalty strength. The realized ε is recorded per solve into the §S.5 outputs. The L2 baseline's ridge uses the same rule on the unweighted system.
   **Scale convention (pinned 2026‑07‑19):** every solver receives `V/s` with `s = √λ₁` of the cell's CSM; recovered amplitudes are rescaled by `s` (intensities by `s²`) before metrics. This preserves inter‑mode ratios and makes `weight_floor` and `tol` scale‑free — without it, tesla‑scale amplitudes (`|a| ~ 1e‑12`) would sit below the `1e‑10` floor and the floor would dominate the weights. Precedent: the Tier‑2 trace‑normalization lesson (repo `CLAUDE.md`; §8 2026‑07‑19 gap‑conditioning entry).
2. **Symmetric grid reduction (pinned 2026‑07‑19).** Same `beta = 0.9` and same on/off setting per run. GIBF prunes per mode by `|aᵢ|`, MMV by row norm — that asymmetry is inherent to the methods, so symmetry means *same β, same setting*, not identical retained sets. **Confirmatory cells run reduction ON** (Suzuki's published GIBF includes grid reduction; the "that isn't Suzuki's GIBF" defense requires running his algorithm as published), with an OFF pair at the same `n_trials` reported as descriptive rows. B1 still runs both ways. Pinned consequence: the §8‑ii power‑calc SD inputs (B1 pilot SD and the 50‑trial mini‑pilot SD) are read from **reduction‑ON** rows, and the mini‑pilot runs reduction ON.
3. **L2 baseline (pinned 2026‑07‑19).** Computed at every cell, same `eps_frac` rule on the unweighted system, same scale convention; a context row only — never adjudicated, never tuned.
4. **Single‑source exact‑recovery gate (pinned 2026‑07‑19).** Before any comparison cell counts, **all three solvers** (roadmap §4 item 1 governs over this section's earlier "both sparse solvers" draft — the stricter superset) must localise one isolated DF source, placed at the grid cell nearest the array centroid, to the **exact cell**: global argmax of `I(j)` equals the true cell (τ = 0 — deliberately stricter than §8‑v's `τ_r = 1`). Oracle `K = 1`, `n_snap = 64`, base `eps_frac`, two tiers: **(i)** noise‑free deterministic, single run (roadmap §4's "noise‑free well‑separated" case); **(ii)** 20 dB Monte‑Carlo, 10 trials, all must pass, seeds per the §S.1 scheme. Sparse solvers run both reduction settings; L2 runs tier (i) only. **Any failure halts Experiment B** (solver bug, not science); gate artifacts archived like Gate V's.
5. **Regularization‑sensitivity panel (pinned 2026‑07‑19).** At every confirmatory cell (win cells at both SNRs, both null cells at the same SNR pair, B6a): rerun both sparse solvers at `eps_frac × {0.1, 0.3, 1, 3, 10}` — ×1 **is** the headline run — on the **same trials** (identical seeds give identical `X`/`S`/`V`/`K̂`; the data does not depend on `eps_frac`, so the marginal cost is solver time only), at full confirmatory `n_trials`, paired. Report the signed mean gap + 95 % CI per band point (percentile bootstrap, 2000 resamples, seeded from (cell id, band index)). **Fragility rule (operational, pinned pre‑data):** a cell's verdict is **regularization‑fragile** iff any band point's mean gap is opposite‑signed to the ×1 headline **and** that point's 95 % CI excludes zero — a significant reversal. Loss of significance without reversal is reported as "attenuated at band edge(s)" and is not fragile; both facets are always reported. A fragile cell's confirmatory claim is demoted to descriptive (the win rule then cannot be satisfied through that cell; the loss rule still needs ≥ 1 null; B6 has a single cell). Rationale: the draft's bare "flips sign" had no operational reading against Monte‑Carlo noise.

## S.7 Card‑A slots (the only unfrozen design inputs)

- **SLOT‑1 — `|ρ|`: FILLED (Card A Tier 2, 2026‑07‑12; `ROADMAP.md` §8).** B2/B6 coherence magnitude = **0.85 at the confirmatory cells; {0.70, 0.95} endpoints as descriptive rows.** Tier 2 confirmed the Tier‑1 sweep values as the realistic range: H‑A holds across all of |ρ| ∈ {0.70, 0.85, 0.95} (80/81 realistic cells), and κ_ground is |ρ|‑insensitive through the real kernel (grid means 0.561/0.568/0.571), so the former placeholder values are now the pinned values — Experiment B is runnable as confirmatory.
- **SLOT‑2 — φ weighting/framing: FILLED (Card A Tier 2, 2026‑07‑12; H‑A HOLDS).** Per roadmap node A's holds arm: B2's **descriptive weighting** of the φ axis centres on the realistic phase‑offset band, weighted by the archived `κ_ground(Q, |ρ|, d)` surface (`results/A_flr_coherence/tier2_results.npz`; best cell κ_ground = 0.6815 with co‑primary 0.6936, IMAGE‑chain concurrence 0.689). Paper framing: **"exploitable structure exists in the physically dominant regime"** — with the Tier‑2 washout report (κ_source − κ_ground ∈ [−0.317, +0.206]; the kernel concentrates rather than destroys the coherent structure) as the mechanistic support. The confirmatory cells are unmoved (§8‑iii, pinned data‑independent).

Sequencing gate restated: **Gate V → `transfer.py` → Card A Tier 2 → fill slots → Experiment B** (B1 pilot → mini‑pilot → powered B2/B6, B3/B4 alongside).

## S.8 Module deltas vs the brief (build checklist)

- `modeselect.py`: MDL‑primary default (§S.4); otherwise per brief §6.5.
- `metrics.py`: `τ_r = 1` grid cell (§8‑v), not 1.5; peak‑matching per brief otherwise.
- `simulate.py`: add the B6b truth variants (mid‑window step; i.i.d. 3×3 jitter) behind `truth.jitter: none|step|iid`; coherent pair model per brief §6.3.
- `run_experiment_B.py`: add the B6a paired oracle/K̂ arm (same seeds), the `K̂` breakdown recorder, and the sensitivity‑panel loop.
- Everything B5 deleted; Experiment‑A runner replaced by Gate V (separate SPEC/script).

## S.9 Supersessions of the stale brief (explicit)

1. Brief §9‑B1 "headline test / the figure *is* the decision" → B1 is cost/pilot/floor; the decision figure is B2's φ‑resolved signed gap (07‑01 restructure).
2. Brief §9‑B2 fixed `rho: 0.95` → SLOT‑1 (Card A).
3. Brief §6.6 `τ_r = 1.5` cells → 1 cell (§8‑v).
4. Brief §6.5 "AIC or MDL" → MDL primary (§8‑viii).
5. Brief §9‑B5 and §8 (Experiment A identifiability) → retired/replaced by Gate V (07‑01; theorem).

## S.10 Array + SECS grid geometry (pinned 2026‑07‑27)

> **New section.** Until now no absolute geometry was pinned anywhere: ROADMAP/SPEC/`EXPERIMENT_CARD_A.md` pin `φ / snr / d / |ρ| / n_snap / τ_r` in **relative** terms only, so the 2026‑07‑21 kickoff run inherited the archived brief's `config/base.yaml` and flagged it. That gap is what produced a degenerate confirmatory cell. Pinned here so it is never again inherited by accident. Authority: ROADMAP §8 2026‑07‑27 (Strider's §8‑ii sign‑off, ruling 1) — **the geometry moves, the pre‑registration does not.**

**Station array — unchanged.** 5×5 = 25 stations, lat span 12°, lon span 20°, centred 70 °N / 0 °E ("high" preset). 3 components ⇒ 75 channels. Nearest‑neighbour spacing 334.0 km (lat) / **190.4 km (lon)**.

**SECS pole grid — refined.** The rule, fixed before any re‑run and consulting no result:

| Quantity | Value | Where it comes from |
|---|---|---|
| Confirmatory coordinate `d` | **3** | §8‑iii, pre‑registered 2026‑07‑07 — **not moved** |
| `d = 3` in physical units | **190.4 km = 1 station spacing** | the scale below which a ground array cannot sample horizontal structure |
| Pole spacing `Δ` | **63.456 km**, isotropic | `= station spacing / 3` (image‑grid oversampling ×3) |
| Pole count | **33 × 17 = 561** | odd counts preserving the inherited 18°/28° imaging extents to within one cell (18.24° / 26.67°) |
| `d` axis in km | 63 / 127 / **190** / 317 / 508 | the pre‑registered `d ∈ {1,2,3,5,8}` now **spans** the array's resolution transition |
| Source placement | pair on the centre row, symmetric about the centre column | refinement‑invariant; supersedes the inherited fixed `col_a = 1`, which on a 17‑wide grid would fall outside the array footprint |
| Overcompleteness | 561 / 75 = **7.5×** (was 1.6×) | adjacent‑column correlation ≈ 0.997 |

**Unit consequence — load‑bearing for §8‑ii.** Every adjudication quantity is denominated in **grid cells**: `Δr̄`, `τ_r = 1 cell` (§8‑v), and the win rule's "resolves ≥ 1 grid cell smaller". Those numerals are pinned and unchanged, but one cell is now **63.5 km** where it was 106.6 km (lon) / 200.4 km (lat) — a 1.68× tighter physical tolerance along the separation axis, and an isotropic one, where the old grid made `metrics.py`'s Euclidean index distance mix two different physical lengths. Both units are reported everywhere.

**Why this rule and not a coherence‑based one.** The normalized column correlation of `A` only falls to 1/√2 at ~350 km, yet all three solvers recovered a 320 km pair exactly at −5 dB — coherence bounds worst‑case ℓ₁ recovery and these solvers beat that bound comfortably, so a coherence anchor would have certified the ceiling‑ing grid as already correct. The station‑sampling anchor is a property of the array, is computable before any experiment, and gives the confirmatory coordinate a physical meaning rather than an accidental one.

## S.11 New confirmatory coordinate for powered B2/B6 (pinned 2026‑08‑01, blind draw)

**Why a new pin was needed.** The §8‑ii cell‑adjudication ruling (2026‑07‑30, ROADMAP §8) found `d=3` — the original confirmatory coordinate — cannot adjudicate: both GIBF and MMV saturate there (Δr̄=0.000, P_sep=1.00 on 50/50 trials), a ceiling/ resolvability‑floor result, not a comparative one. Re‑anchoring `d=3` itself was explicitly rejected (would repeat the un‑pre‑registered‑geometry failure Ruling 1 fixed). A **new** coordinate had to be pre‑registered blind — chosen without consulting any mini‑pilot result — per the same discipline that produced the original `d=3` pin.

**Why a random draw, not Strider's own pick.** By 2026‑08‑01, Strider had already seen mini‑pilot SD/gap numbers across most of the `d` axis from the rescale pass (§S.10's execution record — e.g. `d=2`'s SD=10.5516 at 20 dB, `d=3`'s now‑informative row at 5 dB). A coordinate he chose himself at this point would carry a real risk of contaminated blindness. Strider's ruling (2026‑08‑01, live): have the session draw it via a documented, non‑result‑derived random mechanism instead — removes the exposure problem by construction, since no party with any knowledge of prior results makes the pick.

**The draw.**

| Quantity | Value |
|---|---|
| Candidate set | `{1, 2, 5, 8}` — the full pre‑registered `d ∈ {1,2,3,5,8}` axis (§S.10) minus `d=3` (retired, ceiling‑adjudicated, structurally excluded — not a results‑based exclusion) |
| Mechanism | `secrets.choice()` (Python's CSPRNG, OS entropy) over the candidate list, run in a fresh process with no access to any experiment result |
| Drawn | **`d = 8`** → **508 km** (§S.10's `d` axis: 63/127/190/317/**508** km) |
| Timestamp | 2026‑08‑01T17:34:14 UTC |
| Consulted before the draw | Nothing — no mini‑pilot, no B1/B2/B6 result, no SD table |

**Consequence.** `φ = 90°`, `snr_db ∈ {5, 10}`, `n_snap = 64` all stay pinned exactly as signed off (§S.6/§8‑iii); only the confirmatory `d` changes, from 3 to **8**. The next step is the §8‑ii mini‑pilot + power calc *at this new coordinate* — unchanged procedure, new cell. `d=8` sits at the coarse end of the axis (largest source separation), which is neither favorable nor unfavorable by construction — that is the point of drawing it blind rather than choosing it.

Runners record `simulate.geometry_record()` structurally in every manifest (`script_sha256` covers the runner only; the geometry lives in `simulate.py`).
