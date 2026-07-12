> **ARCHIVED — 2026-07-11.** This was Claude's one-time build spec from June 2026 (written pre-rename; see the terminology note below). It is **not current** and is kept only for history / methods-paper backfill. For current status, read (in this order): root `README.md`, `ROADMAP.md` (canonical, live source of truth), `EXPERIMENT_CARD_A.md`, `viability-test/SPEC_experiment_B.md`.
>
> **What's landed since this was written** (per `ROADMAP.md` / git history as of 2026-07-11): `secsy` installed, pinned, and its adapter contract verified (this doc's "Appendix C" probe + §5.2) — that work is now **Validation Gate V, PASSED 2026-07-07**. `transfer.py` (this doc's §6.2 spec) is **built and pytest-gated**. This doc's "Experiment A" (CF/DF identifiability, §2/§8) is the work now called **Gate V** — already resolved above. This doc's "Card C"/FLR-coherence work is now **Card A**, whose **Tier 1 is complete** (2026-07-06) and **Tier 2 is the current frontier** (not yet run). This doc's **Experiment B** (§9, B1–B5) has a frozen design in `viability-test/SPEC_experiment_B.md` but has not been run. Appendix B's checklist below predates all of this and uses pre-rename terminology throughout — do not tick it against current work; treat the summary in this paragraph as authoritative instead.
>
> **Terminology note:** the project was renamed 2026-07-01 after peer review. This document uses the *old* names throughout (e.g. "Experiment A" below means CF/DF identifiability = today's Gate V, not today's "Experiment A/Card A"). See `README.md`'s rename table before cross-referencing anything in this file against current docs.

---

# SECS‑GIBF Viability Test — Master Build Brief

**Repository:** `https://github.com/Stridasaurus/GIBF`
**Build location:** all work lives under a new top‑level folder `viability-test/`.
**This file's intended home:** `viability-test/PLAN.md`. It is the *single, complete, authoritative* context for the construction agent. Read it in full before writing any code. A short pointer `CLAUDE.md` (Appendix D) should sit at the repo root so this brief auto‑loads.
**Status:** specification / build brief for a viability gate that doubles as the experimental backbone of a methods paper.

---

## 0. Mission & how to work (read first)

### 0.1 Mission

You are building a small, self‑contained research codebase whose only job is to answer **two go/no‑go questions** about the larger SECS‑GIBF method *before* the full pipeline is built. The codebase is also the experimental backbone of a **methods paper** whose thesis is exactly these two questions and their answers. Optimise for **correctness, reproducibility, and a clean decisive result**, not for generality or performance.

The headline deliverable is a filled‑in **`DECISION_MEMO.md`** containing two figures and a go/no‑go recommendation. Everything else exists to produce that memo defensibly.

### 0.2 Two non‑negotiable invariants (YOU MUST honour these everywhere)

These two rules override every convenience, every default, and every habit from acoustic beamforming. Violating either silently invalidates the entire study.

1. **The transfer matrix `A` is real‑valued. There is no propagation phase.**
   If you ever find yourself multiplying `A` by `exp(i k r)`, STOP — that acoustic assumption is the exact thing this project exists to avoid. The *only* exception is the deliberate "phantom‑phase" ablation in Experiment B4, which exists precisely to quantify what that phase *would* buy. Outside B4, assert `np.isrealobj(A)` in every solver.

2. **Snapshots are complex frequency‑bin phasors, not raw time samples.**
   If you build the cross‑spectral matrix from real time samples you get a *real* covariance, real eigenvectors, and the whole premise collapses. All phase lives in the complex source phasors → complex `X` → complex Hermitian CSM → complex eigenvectors. A test guards this (§11). See §6.3–6.4.

If a detail is ever missing or ambiguous, prefer **the simplest choice that preserves (a) the realness of `A` and (b) the fairness of the GIBF‑vs‑MMV comparison**, and record the choice in `manifest.json` and the memo.

### 0.3 How to operate (agent workflow — follow this)

This project rewards a disciplined, test‑first, milestone‑by‑milestone approach.

- **Explore → plan → implement → verify → commit.** Before writing code for a module, restate its contract (from §6/§7) and its tests (§11) in your own words, then implement against the tests. Do not write large amounts of code before you can run something.
- **Think hard on the two hard parts:** the `secsy` adapter (§5.2, Appendix C) and the IRLS solver math (§7). These are where silent errors hide. Reason carefully before coding them.
- **Test‑driven where possible.** For each core module, the test in §11 is the acceptance criterion. Write/translate the test first, watch it fail, then make it pass. Use tiny grids in tests so `pytest -q` runs in well under a minute.
- **Build in the order of §12.** In particular, **land Experiment A end‑to‑end before building any Monte‑Carlo machinery** — if CF is unrecoverable, you drop a whole basis type and simplify everything downstream.
- **Verify continuously.** After every module: `pytest -q` green; after every experiment: the runner writes `.npz` + `.csv` + `manifest.json` and the figure regenerates from cache.
- **Track progress** against the checklist in Appendix B. Tick items as their acceptance criteria pass.
- **Commit in small, working increments** with messages that name the module/experiment and the test that now passes. Never commit a red test suite on `main`.
- **Pre‑register before peeking.** Fill the §2 decision thresholds into `DECISION_MEMO.md` *before* you look at any results. This is a scientific‑integrity requirement, not bureaucracy.
- **Scope discipline.** Build *only* what §3 specifies (Experiments A and B1–B5). The broader pipeline sketched in the source report (robustness sweeps, real‑data demos, the full experiment tree) is **explicitly out of scope** for this gate; see §1.6. Do not build it now.

### 0.4 Environment note

Installing `secsy` and running its kernels requires **network access and a working scientific‑Python stack**. If your Claude Code session has network egress disabled, enable it (or pre‑install `secsy`) before starting `transfer.py`; the `secsy` verification probe in Appendix C is the first thing that must run successfully. Everything downstream of `transfer.py` is pure NumPy/SciPy and needs no network.

---

## 1. Scientific context (sufficient to implement and to write the paper)

### 1.1 The physics: a real, phase‑less kernel

A time‑varying ionospheric sheet current produces a quasi‑static magnetic field at the ground. At a single frequency bin ω the sensor field is linearly related to the source‑current phasors:

```
B(ω) = A · s(ω)
```

where `A` is a **real** matrix of geometric coupling coefficients (the SECS kernel) and `s(ω)` is the **complex** current‑phasor vector. **All phase information lives in `s`, not in `A`.**

Why real? In the quasi‑static regime every sensor sees the instantaneous field of the source; there is no wave travel time across the array. Concretely, the magnetostatic Biot–Savart relation

```
B(r,t) = (μ0/4π) ∫ J(r',t) × (r−r')/|r−r'|³ d³r'   (+ Earth‑induction corrections)
```

is, at fixed ω, `B(r,ω) = G_real(r) · s(ω)` with `G_real` purely real and geometric. The phase differences that appear in the complex CSM come from the **relative temporal phases of the currents themselves**, not from propagation. This is the magnetostatic analog of acoustic beamforming **with the propagation phase removed** — which is the crux of both questions below.

**Domain of validity** (state this in the paper): the quasi‑static / real‑kernel treatment holds when array aperture `L` and signal period `T` satisfy `L/c ≪ T` (equivalently `L ≪ λ`). For ground magnetometer studies — arrays 10²–10³ km, frequencies 10⁻³–1 Hz, free‑space wavelength ~3×10⁵ km at 1 Hz — this is excellent (Pc3–5, Pi2 pulsations, substorm current wedges, FAC structures; arrays like IMAGE / SuperMAG). Full‑wave regimes (e.g. VLF > 3 kHz) would require a dyadic EM Green's function and are **out of scope**.

### 1.2 The SECS CF/DF source basis

We build `A` from **Spherical Elementary Current Systems (SECS)**, which split any horizontal sheet current into:

- a **curl‑free (CF)** part — the divergence of the ionospheric current, closing via field‑aligned currents (FACs); and
- a **divergence‑free (DF)** part — the rotational/solenoidal current closed within the sheet.

The `secsy` library provides the field kernels. For each current type, `secsy` returns three matrices `(Ge, Gn, Gu)` giving the east/north/up ground field per unit SECS pole amplitude. The full real matrix is built by stacking components vertically and current types horizontally (§6.2). The CF/DF ratio at a grid point is a physical diagnostic: CF dominance ⇒ current closes via FACs; DF dominance ⇒ solenoidal closure.

### 1.3 The two questions, and why they exist

**Question 1 — Identifiability of CF (the Fukushima problem).** The SECS literature (Vanhamäki & Juusola 2020, §2.7) is explicit: for radial field‑aligned currents, the ground magnetic disturbance is produced *solely by the divergence‑free part*; the curl‑free current plus its FACs produces **no field below the ionosphere** (Fukushima 1976; Amm 1997). CF is only approximately recoverable at auroral latitudes and "breaks down completely at lower latitudes." If that holds for our geometry, the CF columns of `A` are near‑null, the CF/DF separation the full pipeline advertises is **not recoverable from ground data**, and we must drop CF (or move to satellite data). **Experiment A measures this directly.**

**Question 2 — Does the GIBF eigen‑decomposition layer earn its keep?** In acoustic GIBF (Suzuki 2011) the steering vectors carry an `exp(i k r)` phase ramp across the array (his array sits ~10 wavelengths above the sources). That **phase diversity** is what lets eigenmode‑by‑eigenmode inversion cleanly separate sources. Our kernel has **no such phase**. So it is genuinely unknown whether decomposing the CSM into eigenmodes and inverting each independently beats simply inverting the same data with a sparse prior. If it does not, the entire GIBF layer is unnecessary complexity over plain sparse SECS. **Experiment B measures this head‑to‑head.**

### 1.4 Suzuki's algorithm (the method under test — reuse verbatim)

From Suzuki (2011), the pieces Experiment B reuses:

- CSM `S = ⟨q qᴴ⟩`, decomposed `S = U Λ Uᴴ` (Hermitian eigen‑decomposition).
- **Eigenmode includes its magnitude:** `vᵢ = √λᵢ · uᵢ` (Suzuki Eq. 3). **DO NOT drop the `√λᵢ`.** The original project report inverts the *unit* eigenvector `uᵢ`; that is a bug — carry `√λᵢ` here (see §14, gotcha 2).
- Per mode, solve `vᵢ = A aᵢ` for a sparse complex `aᵢ` via IRLS (L1), with grid reduction (factor `β = 0.9`) switching underdetermined → overdetermined as the active set shrinks.
- Underdetermined regularised solve: `a = W Aᵀ (A W Aᵀ + εI)⁻¹ v`, with `W = diag(|a|^{2−p})`, `p = 1`.
- Overdetermined solve: `a = (Aᵀ A + ε W⁻¹)⁻¹ Aᵀ v`.
- `ε` = a small fraction (0.1–10 %) of the largest eigenvalue of `A W Aᵀ` (recomputed from the *current* reduced matrix each iteration).
- Accumulate intensity across modes: `I(j) = Σᵢ |aᵢ(j)|²`, separated by source type.

Because `A` is real but `v` is complex, this is a complex‑right‑hand‑side least squares; standard NumPy solvers handle it. Real and imaginary parts couple only through the magnitude `|a|` in the weights — i.e. complex / group‑sparse recovery.

### 1.5 Mode selection (Wax–Kailath, for complex CSM)

Suzuki used a heuristic eigenvalue‑sum threshold; we replace it with principled information criteria adapted to complex data (Wax & Kailath 1985). Exact formulas in §6.5 and Appendix A. This removes noise‑dominated modes before inversion and gives a defensible "how many sources does the spectrum reveal" answer (Experiment B3).

### 1.6 The methods‑paper framing (and the scope boundary)

The codebase produces the *evidence* for a methods paper that introduces **SECS‑GIBF**: the first integration of Suzuki's mode‑by‑mode sparse beamforming with a *real, magnetostatic* SECS kernel, plus information‑theoretic mode selection. The paper's thesis is the pair of go/no‑go findings:

1. whether CF is identifiable from ground arrays (Question 1), and
2. whether the eigen‑decomposition layer beats joint sparse inversion when the kernel is phase‑less (Question 2).

`DECISION_MEMO.md` is the proto‑results section; its two figures (Experiment‑A observability, Experiment‑B `P_sep`‑vs‑separation) are the headline figures; the B4 phantom‑phase ablation is the key explanatory figure. Likely venues: JGR Space Physics, Earth Planets Space, Geophysical Journal International, Radio Science.

**Scope boundary (do not over‑build):** the source report also sketches a much larger program — robustness sweeps (sensor reduction, altitude mismatch, grid offset), an L₂/L₁‑SECS comparison suite, and a real IMAGE/SuperMAG data demonstration. **None of that is in scope for this viability gate.** Build only Experiments A and B1–B5 in §8–§9. Mention the broader program only as "future work / full pipeline" in the memo.

---

## 2. Hypotheses + pre‑registered decisions (write these into the memo BEFORE results)

The runners must emit *exactly* the quantities these rules reference.

### Experiment A — CF/DF identifiability

> **H‑A:** For a high‑latitude ground array, CF sources produce negligible ground signal relative to DF sources, so CF amplitude is not recoverable.

**Primary statistic:** median over the grid of the *observability ratio*
`ρ_obs(j) = ‖A·e_CF(j)‖ / ‖A·e_DF(j)‖`, where `e_CF(j)`, `e_DF(j)` are unit CF/DF amplitudes at grid point `j`.

**Decision rule:**
- `median ρ_obs < 1e‑2` at the target high latitude → **CF unrecoverable from ground data.** The full pipeline must run **DF‑only** for ground arrays; CF is dropped (or deferred to satellite data). Report the latitude at which `ρ_obs` rises above `1e‑1` (where CF starts to leak).
- `1e‑2 ≤ median ρ_obs < 1e‑1` → CF is **marginal**; retain only as an explicitly low‑confidence, latitude‑gated diagnostic.
- `median ρ_obs ≥ 1e‑1` → CF is **observable** here; keep it, but verify against Experiment B5 (CF/DF attribution).

### Experiment B — value of the GIBF eigen layer

> **H‑B:** For a real (phase‑less) kernel, eigenmode‑by‑eigenmode GIBF does **not** separate two incoherent sources better than a joint sparse (MMV‑L1) inversion of the *same* eigenmodes.

The comparison is razor‑sharp because GIBF and the MMV‑L1 competitor receive **identical input** (the same `K` eigenmodes `vᵢ`) and the **same real `A`**; the *only* difference is independent‑per‑mode inversion (GIBF) vs. shared‑support joint inversion (MMV). See §7.3.

**Primary statistics**, aggregated over Monte‑Carlo trials at matched SNR in the incoherent regime:
- mean two‑source localisation error `Δr̄` (matched assignment), and
- separation success rate `P_sep` (both sources resolved as distinct peaks at correct locations).

**Decision rule (incoherent regime — the regime most favourable to GIBF):**
- GIBF is **"worth it"** only if it beats MMV‑L1 by a meaningful, statistically separated margin — operationally: GIBF `Δr̄` at least **20 % lower** *or* GIBF resolves at a source separation **≥ 1 grid cell smaller**, with **non‑overlapping 95 % CIs, at two or more SNR levels.**
- If GIBF ≤ MMV‑L1 within CIs across the SNR sweep → **recommend dropping the eigen‑decomposition layer**; the full pipeline reduces to joint sparse SECS.
- The L2 min‑norm baseline establishes how much sparsity (of either flavour) buys over plain Tikhonov; it is **context, not a competitor** for the headline decision.

A secondary, paper‑grade result comes from the coherent variant (B2) and the phantom‑phase ablation (B4), which together explain *why* the answer is what it is.

---

## 3. Repository layout

```
viability-test/
├── README.md                      # quickstart, how to run, where results land, links to this PLAN
├── PLAN.md                        # this document
├── DECISION_MEMO.md               # deliverable; template in §13
├── requirements.txt
├── pyproject.toml                 # package name: viability
├── .gitignore                     # results/, *.npz, *.png, __pycache__, .ipynb_checkpoints, cache
│
├── config/
│   ├── base.yaml                  # geometry, grid, frequency, altitude, solver + MC params
│   ├── exp_A_identifiability.yaml
│   ├── exp_B1_incoherent.yaml
│   ├── exp_B2_coherent.yaml
│   ├── exp_B3_modeselect.yaml
│   ├── exp_B4_phaseablation.yaml
│   └── exp_B5_cfdf_attribution.yaml   # only run if Exp A says CF is observable
│
├── src/viability/
│   ├── __init__.py                # version string
│   ├── config_loader.py           # deep‑merge base.yaml + experiment override -> dict
│   ├── geometry.py                # build array + SECS grid; latitude presets
│   ├── transfer.py                # REAL SECS A via secsy; CF/DF blocks; row‑norm; cache
│   ├── simulate.py                # complex phasor snapshots -> sensor X -> CSM inputs
│   ├── csm.py                     # CSM build, Hermitian eigendecomp, eigenmodes √λ·u
│   ├── modeselect.py              # complex AIC / MDL (Wax–Kailath)
│   ├── identifiability.py         # Experiment A analysis (SVD, observability)
│   ├── metrics.py                 # Δr, separation, ΔI, spurious power, (opt) Wasserstein
│   ├── plotting.py                # source maps, SV spectra, metric curves, summary tables
│   └── solvers/
│       ├── __init__.py
│       ├── l2_minnorm.py          # Tikhonov min‑norm, per mode, summed
│       ├── gibf_irls.py           # Suzuki per‑mode L1 IRLS + grid reduction
│       └── mmv_l1.py              # joint L2,1 group‑sparse IRLS over modes
│
├── experiments/
│   ├── run_experiment_A.py        # --config exp_A_*.yaml
│   ├── run_experiment_B.py        # --config exp_B*.yaml  (scenario read from config)
│   └── make_report_figures.py     # regenerate all figures from cached results
│
├── results/                       # gitignored: per‑experiment .npz + .csv + .png + manifest.json
│   ├── A_identifiability/
│   └── B_viability/
│
├── notebooks/
│   ├── A_identifiability.ipynb    # loads cached A results, renders the decision figure
│   └── B_viability.ipynb          # loads cached B results, renders the decision figure
│
└── tests/
    ├── test_transfer.py
    ├── test_simulate.py
    ├── test_csm.py
    ├── test_modeselect.py
    ├── test_solvers.py
    └── test_metrics.py
```

**Discipline:** runners are pure Python and write cached artifacts; **notebooks only load and visualise — never recompute heavy work.** This keeps results reproducible and review‑friendly.

---

## 4. Dependencies

`requirements.txt`:

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pyyaml>=6.0
tqdm>=4.65
pytest>=7.0
secsy            # PIN the exact version you install; see §5.2
# optional, guarded by try/except:
pot              # POT, for 2-D Wasserstein map distance (optional metric)
```

`scipy` is required (Hungarian assignment via `scipy.optimize.linear_sum_assignment`; linear solves). **Pin `secsy`** to whatever version resolves at build time and record it in `requirements.txt`, `manifest.json`, and the memo — its public API has shifted across versions (§5.2).

---

## 5. Shared configuration

### 5.1 `config/base.yaml` (concrete defaults — tune in overrides)

```yaml
seed: 20240601                 # master RNG seed; per-trial seeds derive from this

geometry:
  earth_radius_km: 6371.2
  ionosphere_height_km: 110.0  # SECS layer altitude above ground
  latitude_preset: high        # one of: high, mid, low (see geometry.py presets)
  array:                       # ground magnetometer stations
    n_lat: 5                   # 5 x 5 = 25 stations
    n_lon: 5
    lat_span_deg: 12.0
    lon_span_deg: 20.0
    components: [e, n, u]      # east, north, up -> 3 rows per station
  grid:                        # SECS source grid (over the array, padded beyond it)
    n_lat: 11
    n_lon: 11
    lat_span_deg: 18.0         # wider than array footprint (external SECS padding)
    lon_span_deg: 28.0

physics:
  frequency_mhz: 15.0          # context only (single-bin); Pi2 band ~10-25 mHz
  current_types: [divfree]     # default DF-only for Exp B; Exp A uses [curlfree, divfree]
  row_normalisation: rms       # one of: none, rms (1/mean|row|), noise_sigma
  singularity_limit_km: 50.0   # secsy singularity smoothing; verify kwarg name

csm:
  n_snapshots: 64              # complex frequency-bin phasor snapshots
  snr_db: 10.0

solver:
  p_norm: 1.0                  # L1
  eps_frac: 0.01               # ε as fraction of max eigenvalue of (A W Aᵀ)
  beta: 0.9                    # grid-reduction factor
  weight_floor: 1.0e-10        # added to |a| before weighting
  max_iter: 50
  tol: 1.0e-4                  # relative change stop
  min_active_factor: 2         # stop when N_active < min_active_factor * n_channels
  grid_reduction: true         # B1 runs this both true and false

monte_carlo:
  n_trials: 50                 # per scenario cell
```

### 5.2 ⚠️ `secsy` API — verify, do not assume (gating step)

The source report uses `get_SECS_B_G_matrices(..., current_type='curlfree')` returning `(Ge, Gn, Gu)`. **This signature varies by `secsy` version.** Before writing `transfer.py`, run the verification probe in **Appendix C** and confirm:

1. `import secsy; print(secsy.__version__)` — and pin it.
2. The actual function/method name and argument order.
3. The **keyword for current type** — `'curl_free'` / `'divergence_free'` / `'curlfree'` / `'divfree'` differ across versions.
4. The **component return order** — `(Ge, Gn, Gu)` vs `(Gn, Ge, Gu)`.
5. Units and sign conventions.
6. Radial‑FAC handling for the CF kernel — **this is exactly what Experiment A probes**, so get it right and document it.

Encode whatever you confirm as a **thin adapter at the top of `transfer.py`** with a docstring stating the verified version and conventions, so the rest of the code is insulated from `secsy` churn. Map the config keywords (`curlfree`/`divfree`) to the verified `secsy` keywords inside the adapter only.

---

## 6. Core module specifications

Signatures below are contracts. Keep functions pure where possible (inputs → arrays); push all randomness through an explicit `numpy.random.Generator`.

### 6.0 Fixed conventions (resolve once, use everywhere)

- **Channel order:** all‑east, then all‑north, then all‑up — i.e. `vstack([Ge, Gn, Gu])`. So channel block 0 = the 25 east components, block 1 = the 25 north, block 2 = the 25 up. `n_channels = 3 · n_stations = 75` at defaults.
- **Column order:** current types in the order listed in `current_types`. For Exp A (`[curlfree, divfree]`) the first `n_grid` columns are CF, the next `n_grid` are DF. `tm.col_slice(type)` returns the slice for a type.
- **Grid flattening:** row‑major (`reshape(n_lat, n_lon)` for plotting). Document and reuse.

### 6.1 `geometry.py`

```python
def build_array(cfg) -> Array:
    """Station lats/lons (deg) + channel index map. 3*n_stations channels in the
    fixed order all-east, all-north, all-up (see §6.0)."""

def build_source_grid(cfg) -> Grid:
    """SECS pole lats/lons (deg), flattened row-major, with an (n_lat, n_lon)
    reshape map for plotting. Grid is wider than the array (external SECS padding)."""

LATITUDE_PRESETS = {
    "high": 70.0,   # auroral; Fukushima ~ exact
    "mid":  55.0,
    "low":  35.0,   # tilted field lines; CF expected to leak
}
def grid_center_latitude(cfg) -> float: ...
```

Stations and SECS poles sit on regular lat/lon grids centred on `grid_center_latitude(cfg)`. The source grid is intentionally larger than the array footprint (distant sources / edge effects require padding the grid beyond the data region).

### 6.2 `transfer.py`

```python
def build_transfer_matrix(cfg, current_types=None) -> TransferMatrix:
    """
    Build the REAL transfer matrix A mapping unit SECS amplitudes -> magnetic field
    at all station components.

    For each requested current type ('curlfree','divfree'), call the verified secsy
    adapter to get (Ge, Gn, Gu) [n_stations x n_grid], then:

        block_t = vstack([Ge_t, Gn_t, Gu_t])       # (3*n_stations, n_grid)
        A = hstack([block_t for t in current_types]) # real, (3*n_stations, n_types*n_grid)

    Apply row normalisation per cfg.physics.row_normalisation (A stays real).
    Return an object exposing:
      .A          : float64 ndarray (REAL)
      .n_channels : 3*n_stations
      .n_grid     : grid points
      .types      : current types in column order
      .col_slice(type) -> slice
      .row_scale  : applied normalisation vector
    Cache to results/.../A_cache_{hash}.npz keyed by hash(geometry+types config).
    """

def unit_source_vector(tm, type, grid_index) -> np.ndarray:
    """Length n_types*n_grid, 1.0 at (type, grid_index) else 0 — for observability tests."""
```

**Row normalisation** (`physics.row_normalisation`): `none`; `rms` = scale each row by `1/mean(|row|)`; `noise_sigma` = scale by `1/σ_noise` if known. The vertical (up) component can otherwise dominate. `A` stays real after normalisation. **Invariants (tests):** `A.dtype` real; `A.shape == (3*n_stations, n_types*n_grid)`; cache round‑trips; row‑norm applied consistently and idempotently.

### 6.3 `simulate.py` — complex frequency‑bin phasor snapshots (§0.2 rule 2)

```python
def make_truth(cfg, grid, source_specs) -> Truth:
    """source_specs: list of {type, grid_index, power}. Returns indices, types,
    powers, and a full-length boolean support mask for metrics."""

def simulate_snapshots(cfg, tm, truth, rng, coherence) -> Snapshots:
    """
    coherence: 'incoherent' | {'rho': <complex>} for coherent pairs.

    For each snapshot t:
      draw source phasors s_t (COMPLEX):
        incoherent: each active source ~ CN(0, power) independently
        coherent:   s2_t = rho*s1_t + sqrt(1-|rho|^2)*CN(0, power2)  (fixed rho across t)
      assemble full source phasor vector a_t (zeros except active indices)
      clean field:  x_clean_t = A @ a_t            # complex, length n_channels
    Stack -> X_clean (n_channels x N_snap).
    Add circularly-symmetric complex Gaussian sensor noise scaled to cfg.csm.snr_db,
    where SNR = mean per-channel signal power / mean per-channel noise power.
    Return X (complex) + per-source clean fields for diagnostics.
    """
```

- **Incoherent** = the sources' phasor *time series* are statistically independent across snapshots → CSM is ~rank‑2 with two well‑defined eigenmodes (the regime GIBF should win, if ever).
- **Coherent** fixes complex `rho` (e.g. `|rho|=0.95`) → eigenmodes mix; the stress test for GIBF's separation claim.
- Source `power` sets relative eigenvalue magnitudes; keep the two sources roughly equal unless a config overrides.
- **`CN(0, σ²)`** = circularly‑symmetric complex normal: real & imag each `N(0, σ²/2)`. Use this for both source phasors and sensor noise.

### 6.4 `csm.py`

```python
def build_csm(X) -> np.ndarray:
    """S = (1/N_snap) X Xᴴ. Complex Hermitian PSD, (n_channels x n_channels)."""

def eigendecompose(S):
    """numpy.linalg.eigh (Hermitian). Return eigenvalues DESC and matching eigenvectors
    (columns). Guarantee real, nonnegative eigenvalues (clip tiny <0 to 0)."""

def eigenmodes(eigvals, eigvecs, k):
    """Return V = [√λ_1 u_1, ..., √λ_k u_k]  (n_channels x k).  Suzuki Eq. 3.
    THIS IS THE INPUT BOTH gibf_irls AND mmv_l1 CONSUME — IDENTICAL for both."""
```

Because the simulator produces complex `X`, `S` is genuinely complex Hermitian and its eigenvectors are complex — preserving "phase lives in the source phasors." A test asserts `S` is *not* (near‑)real for a multi‑phase scenario. **Carry the `√λ` weighting** in `eigenmodes` (do not return unit vectors).

### 6.5 `modeselect.py` — complex Wax–Kailath

Let `M = n_channels`, `N = n_snapshots`, eigenvalues `λ_1 ≥ … ≥ λ_M > 0`. For `k = 0 … M−1`, with `p = M − k`:

```
log_geo = (1/p) * Σ_{i=k+1}^{M} ln(λ_i)          # work in LOG domain
ari     = (1/p) * Σ_{i=k+1}^{M} λ_i
LLR     = N * p * (log_geo - ln(ari))            # ≤ 0
AIC(k)  = -2 * LLR + 2 * k * (2M - k)
MDL(k)  = -1 * LLR + 0.5 * k * (2M - k) * ln(N)
```

```python
def estimate_n_sources(eigvals, n_snapshots, criterion="mdl") -> int:
    """argmin over k of AIC or MDL. Floor eigenvalues at 1e-18 before ln to keep logs
    finite. Also return full AIC(k), MDL(k) arrays for plotting."""
```

A test feeds synthetic spectra (e.g. two large eigenvalues + white floor) and asserts the recovered `k` matches the planted number across a range of `N`.

### 6.6 `metrics.py` — precise definitions

All operate on a non‑negative intensity map `I(j)` over the grid (length `n_grid`), reshaped to `(n_lat, n_lon)` for spatial ops.

- **Localisation error `Δr`** (matched): find the ≤2 strongest local maxima above a relative threshold (default 10 % of map max); solve the optimal assignment (`scipy.optimize.linear_sum_assignment`) between detected peaks and true sources; `Δr` = mean great‑circle (or grid‑cell) distance of matched pairs. Unmatched true source → penalty distance = grid diagonal.
- **Separation success `P_sep`** (per trial, boolean): both true sources have a detected peak within tolerance `τ_r` (default 1.5 grid cells) **and** the two detected peaks are distinct (separated by a local minimum between them). Aggregate to a rate over trials.
- **Intensity error `ΔI`**: `10·log10(Î_total / I_true_total)` dB, on matched support.
- **Spurious power ratio**: fraction of total estimated intensity lying outside disks of radius `τ_r` around the true sources.
- **(Optional) Wasserstein map distance**: 2‑D earth‑mover distance between L1‑normalised true and estimated maps, via POT if installed; guard with `try/except` and skip cleanly if absent.

Every metric returns a scalar per trial; runners aggregate to **mean ± 95 % CI** (bootstrap or t‑interval) per scenario cell. Use one CI method consistently and name it in the memo.

### 6.7 `plotting.py`

Render: (a) singular‑value spectra (log‑y) for Exp A; (b) observability‑ratio maps; (c) side‑by‑side true / L2 / GIBF / MMV source maps with the **10 dB contour** convention from Suzuki; (d) metric‑vs‑separation and metric‑vs‑SNR curves with CI bands; (e) AIC/MDL‑vs‑k curves; (f) a one‑page summary table. Consistent colormap; save **PNG + the underlying arrays**.

---

## 7. The three solvers (the heart of Experiment B)

All three consume the **same** eigenmode matrix `V` (n_channels × K) from `csm.eigenmodes`, and the **same** real `A`. They differ only in how they invert. Each returns a full‑length intensity map `I(j)` and (for CF/DF variants) per‑type sub‑maps. **Every solver asserts `np.isrealobj(A)` unless a `phantom_phase` flag is set (B4 only).**

### 7.1 `solvers/l2_minnorm.py` (baseline — context, not competitor)

Per mode `i`: `a = Aᵀ (A Aᵀ + εI)⁻¹ vᵢ`, with `ε = eps_frac · λ_max(A Aᵀ)`. Accumulate `I(j) = Σᵢ |aᵢ(j)|²`. No iteration, no sparsity. (For a reduced/tall `A`, use the overdetermined normal‑equation form; for the full grid it is underdetermined.)

### 7.2 `solvers/gibf_irls.py` (Suzuki, per‑mode L1)

```
function solve_gibf_mode(A, v, cfg):
    N_active = A.shape[1]                       # all columns
    cols = arange(N_active)                     # active column indices into full grid
    a = l2_init(A, v, cfg)                      # Eq. 9 underdetermined init
    prev_cost = +inf
    for it in range(cfg.max_iter):
        w = (abs(a) + cfg.weight_floor) ** (2 - cfg.p_norm)   # |a|^{2-p}
        W = diag(w)
        if 2 * len(cols) > A.shape[0]:          # underdetermined (heuristic factor 2)
            M = A @ W @ A.conj().T               # A real, but keep .conj().T for safety
            eps = cfg.eps_frac * max_eig(M)
            a = W @ A.T @ solve(M + eps*I, v)
        else:                                    # overdetermined
            eps = cfg.eps_frac * max_eig(A.T @ A)
            a = solve(A.T @ A + eps * diag(1/w), A.T @ v)
        keep = argsort(abs(a))[::-1][: int(cfg.beta * len(cols))]   # grid reduction
        cols, A, a = cols[keep], A[:, keep], a[keep]
        cost = sum(abs(a))                       # L1 cost
        if cost > prev_cost: break               # termination: cost non-decreasing
        if len(cols) < cfg.min_active_factor * A.shape[0]: break
        if rel_change(a, a_prev) < cfg.tol: break
        prev_cost = cost
    return scatter_to_full_grid(a, cols, full_len)   # zero-fill pruned points
```

Run per mode, then `I(j) = Σᵢ |aᵢ(j)|²`. **Carry `v = √λᵢ uᵢ` (Suzuki Eq. 3); do not invert the unit `uᵢ`.**

> **Pruning caveat (honour in code & memo):** grid reduction is **irreversible**. Because this kernel is ill‑conditioned (no propagation phase, possible near‑null CF), a source mislocated in early iterations can be pruned permanently. Provide the config switch `solver.grid_reduction: true|false`; run B1 **both ways** and report whether pruning helps or hurts. That itself is a finding.

### 7.3 `solvers/mmv_l1.py` (joint sparse competitor — THE key comparison)

Same input `V` (all `K` modes at once), but solve for a **source matrix** `Ā` (n_grid × K) with a **shared support** across modes via an L2,1 (row‑sparse) prior:

```
minimise  ‖V − A Ā‖_F²  +  μ · Σ_j ‖Ā[j,:]‖_2
```

IRLS with **row** weights (group / FOCUSS‑MMV):

```
function solve_mmv(A, V, cfg):
    cols = arange(A.shape[1]); A̅ = l2_init_matrix(A, V, cfg)
    for it in range(cfg.max_iter):
        rownorm = norm(A̅, axis=1)                        # ‖Ā[j,:]‖_2
        w = (rownorm + cfg.weight_floor) ** (2 - cfg.p_norm)
        W = diag(w)
        if 2*len(cols) > A.shape[0]:
            M = A @ W @ A.conj().T; eps = cfg.eps_frac*max_eig(M)
            A̅ = W @ A.T @ solve(M + eps*I, V)             # MATRIX RHS
        else:
            eps = cfg.eps_frac*max_eig(A.T @ A)
            A̅ = solve(A.T @ A + eps*diag(1/w), A.T @ V)
        keep = argsort(rownorm)[::-1][: int(cfg.beta*len(cols))]
        cols, A, A̅ = cols[keep], A[:,keep], A̅[keep,:]
        if l21_cost increases or active set too small or converged: break
    return scatter_rows_to_full_grid(A̅, cols)
```

Then `I(j) = Σ_k |Ā[j,k]|² = ‖Ā[j,:]‖²` — the **same accumulation** as GIBF. Therefore the only experimental difference between GIBF and MMV is *independent‑per‑mode* vs *shared‑support* inversion. **That isolation is the entire point.** A test asserts MMV produces a single shared support set while GIBF's per‑mode supports may differ.

> **Fair‑comparison discipline (YOU MUST):** GIBF and MMV consume the *identical* `V` and `A`, the same `eps_frac`, `beta`, `weight_floor`, `max_iter`, `tol`, and the same grid‑reduction setting per run. If you ever feed them different inputs, the headline result is invalid.

---

## 8. Experiment A — CF/DF identifiability

**Config:** `exp_A_identifiability.yaml` overrides `physics.current_types: [curlfree, divfree]` and sweeps `geometry.latitude_preset` over `[high, mid, low]`.

**`run_experiment_A.py` procedure** (no Monte Carlo — deterministic linear algebra). For each latitude preset:

1. Build `A` with both CF and DF blocks (`transfer.build_transfer_matrix`).
2. **Observability map:** for every grid point `j`, `ρ_obs(j) = ‖A·e_CF(j)‖ / ‖A·e_DF(j)‖`. Save the map and its median/percentiles.
3. **SVD of full `A`:** singular‑value spectrum; effective rank at tolerance `τ = 1e‑6·σ_max`; CF‑block energy in the numerical null space, `‖(I − P_r) C‖_F / ‖C‖_F`, where `C` = CF columns and `P_r` projects onto the top‑r left singular vectors.
4. **Conditioning:** `cond(A_full)` vs `cond(A_DF_only)`.

**Outputs** (`results/A_identifiability/`): per‑latitude `.npz` (singular values, observability map, scalars) + `.csv` summary + figures (SV‑spectrum overlay across latitudes; observability map per latitude) + `manifest.json`.

**Decision:** apply the §2 rule to `median ρ_obs` at `high` latitude. Theory expectation: at high latitude with radial FAC the CF ground field is ~zero (Fukushima), so `ρ_obs` ≈ machine precision and CF energy sits almost entirely in the null space; CF observability should grow as latitude falls. **Let the numbers decide** — `secsy`'s exact CF handling is what you are measuring, so do **not** hard‑code the expected outcome; report it.

---

## 9. Experiment B — does the GIBF layer earn its keep?

All B scenarios share the Monte‑Carlo skeleton in `run_experiment_B.py`:

```
for cell in scenario_grid:                  # e.g. (separation, snr) or (snr, n_snap)
    for trial in range(n_trials):
        rng = Generator(seed derived from (master_seed, cell, trial))
        X  = simulate_snapshots(...)         # DF-only sources unless B5
        S  = build_csm(X); λ,U = eigendecompose(S)
        K  = (known count for B1/B2/B4) or estimate_n_sources(λ,N) (B3)
        V  = eigenmodes(λ, U, K)             # IDENTICAL input to all solvers
        maps = {
          "l2":   l2_minnorm(A, V),
          "gibf": gibf(A, V),
          "mmv":  mmv_l1(A, V),
        }
        record metrics(maps[m], truth) for each m
    aggregate mean ± 95% CI per method per cell
save .npz + .csv + manifest.json; plotting renders decision figures
```

### B1 — Incoherent separation (the headline test)
- **Sources:** two DF blobs of equal power, incoherent.
- **Sweep:** separation `d ∈ {1,2,3,5,8}` grid cells × `snr_db ∈ {−5,0,5,10,20}`, `n_snapshots = 64`, `K = 2` (known).
- **Also:** run with `grid_reduction` **on and off**.
- **Reads out:** `Δr̄`, `P_sep`, `ΔI`, spurious ratio per method per cell.
- **Figure:** `P_sep` vs separation (one curve per method, per SNR panel) and `Δr̄` vs SNR at fixed `d`. **This figure *is* the Experiment‑B decision.**

### B2 — Coherent stress test
Identical to B1 but `coherence = {rho: 0.95}`. Eigenmodes mix, so GIBF's per‑mode independence is least valid here. Report whether GIBF degrades faster than MMV. Paper‑grade explanatory result.

### B3 — Mode‑selection validity
Fixed geometry, two incoherent sources, sweep `snr_db ∈ {−5,0,5,10,20}` × `n_snapshots ∈ {8,16,32,64,128}`, large `n_trials`. Record AIC and MDL estimated source count vs truth (=2). **Figure:** estimated `K` (mean + histogram/mode) vs `n_snapshots`, per SNR. Probes the stationarity / low‑DOF concern: how many independent snapshots before the spectrum even reveals two sources.

### B4 — Phantom‑phase ablation (why, not just whether)
Re‑run B1's incoherent scenario, but build a *fake* complex steering matrix `A_phase = A ⊙ exp(i·Φ)`, where `Φ` is a synthetic geometric phase ramp emulating a wave propagating at light speed across the aperture at `frequency_mhz`. Run **GIBF only** with `A_phase` vs the true real `A`. This isolates how much of acoustic‑GIBF's separation power came from phase diversity the magnetostatic problem lacks. **Figure:** `P_sep` vs separation for `A` (real) vs `A_phase` (phantom). Expectation: a large gap. **This is the gold explanatory figure.** (This is the *only* place complex `A` is permitted; gate it behind the `phantom_phase` flag and skip the realness assertion there.)

### B5 — CF/DF attribution (conditional)
**Run only if Experiment A finds CF observable** (`median ρ_obs ≥ 1e‑1`) at the chosen latitude. One CF source + one DF source, incoherent; measure cross‑type leakage (DF power misattributed to CF and vice versa) per method. If Exp A says CF is unrecoverable, **skip B5** and note in the memo that attribution is moot for ground arrays.

---

## 10. Numerical‑stability & reproducibility requirements

- Use `numpy.linalg.eigh` for the Hermitian CSM (not `eig`). Clip tiny negative eigenvalues to 0.
- AIC/MDL: compute in the log domain; floor eigenvalues at `1e‑18` before `ln`.
- IRLS weights: always add `weight_floor` before exponentiating; never invert a raw `|a|`.
- Regularisation `ε` recomputed from the *current* (possibly reduced) matrix each iteration.
- All randomness via explicit `numpy.random.Generator`; derive per‑trial seeds deterministically from `(master_seed, scenario_cell, trial_index)` (e.g. `np.random.default_rng([master_seed, *cell_ints, trial])`) so any cell is independently reproducible.
- `A` stays real except in B4. Assert `np.isrealobj(A)` in the solvers unless `phantom_phase` is set.
- Cache `A` (keyed by geometry+type hash) and CSMs where reuse is possible; cached files are gitignored.
- Every runner writes a `manifest.json` capturing the merged config, `secsy` version, git commit, and seed.

---

## 11. Tests (pytest) — these are the acceptance criteria

- `test_transfer.py` — shape; realness; row‑norm idempotence; cache round‑trip; **CF≈null sanity:** at the `high` preset, assert CF‑block field norm ≪ DF‑block (the codified Fukushima expectation). Make the threshold a **documented constant**, not a silent pass — if it fails, that itself is a flagged result.
- `test_simulate.py` — clean field reconstructs `A·a_true`; incoherent phasor series are empirically near‑uncorrelated; CSM rank ≈ number of sources in the noise‑free limit.
- `test_csm.py` — `S` Hermitian PSD; eigenvalues sorted desc; `S` is **not** near‑real for a 2‑phase scenario.
- `test_modeselect.py` — AIC/MDL recover planted source counts on synthetic spectra across `N`.
- `test_solvers.py` — each solver recovers a single source exactly in the noise‑free, well‑separated case; GIBF L1 cost is non‑increasing until termination; MMV yields one shared support; L2 baseline matches the closed‑form min‑norm.
- `test_metrics.py` — `Δr` zero for perfect maps; `P_sep` true/false on constructed examples; `ΔI` sign correct for over/under‑estimates.

**CI target:** `pytest -q` green, runs in well under a minute (tiny grids in tests).

---

## 12. Suggested build order (with verification gates)

1. `config_loader`, `geometry`, `transfer` (+ tests) — get a real `A` you trust. **Gate:** `test_transfer.py` green; `secsy` adapter docstring records verified version + conventions.
2. **Experiment A end‑to‑end** — needs only `transfer` + `identifiability` + `plotting`. **Land the CF/DF decision first**; it may simplify everything downstream (DF‑only). **Gate:** `results/A_identifiability/` populated; observability figure renders from cache; memo Question‑1 fields filled.
3. `simulate`, `csm`, `modeselect` (+ tests). **Gate:** those three test files green; CSM provably complex.
4. The three solvers (+ tests). **Gate:** `test_solvers.py` green; single‑source exact recovery for all three.
5. `metrics`, `plotting`. **Gate:** `test_metrics.py` green.
6. **Experiment B1** end‑to‑end, then B3, B2, B4, (B5 only if Exp A says CF observable). **Gate:** decision figure renders; memo Question‑2 table filled.
7. Notebooks + `DECISION_MEMO.md` finalised.

Landing Experiment A first is deliberate: if CF is unrecoverable, you drop a whole basis type before building the Monte‑Carlo machinery.

---

## 13. Deliverables

1. The codebase under `viability-test/` as specified, `pytest` green.
2. Cached results + figures for Experiments A and B in `results/`.
3. Two decision figures: the Experiment‑A SV/observability figure, and the Experiment‑B `P_sep`‑vs‑separation figure.
4. A completed **`DECISION_MEMO.md`** using the template below.

### `DECISION_MEMO.md` template

```markdown
# Viability Test — Decision Memo

## Configuration
- secsy version: …   | git commit: …   | master seed: …
- Geometry: latitude preset …, array …×…, grid …×…, altitude … km
- Frequency bin: … mHz | snapshots: … | SNR sweep: …
- CI method: … (bootstrap / t-interval)

## Question 1 — Is CF recoverable from ground data?
- median observability ratio ρ_obs @ high latitude: …
- CF energy in numerical null space: … %
- cond(A_full) vs cond(A_DF): … vs …
- Latitude at which ρ_obs crosses 1e-1: …
- **Decision:** [ DF-only / CF marginal / CF retained ]  ← applies §2 rule
- Figure: results/A_identifiability/sv_observability.png

## Question 2 — Does the GIBF eigen layer beat joint sparse L1?
Incoherent regime (B1):
| SNR | metric | L2 | GIBF | MMV-L1 | GIBF − MMV (95% CI) |
|-----|--------|----|------|--------|----------------------|
| …   | Δr̄     | …  | …    | …      | …                    |
| …   | P_sep  | …  | …    | …      | …                    |

- Smallest resolved separation (P_sep ≥ 0.8): GIBF … cells, MMV … cells.
- Grid reduction on vs off: …
- Coherent regime (B2) summary: …
- Phantom-phase ablation (B4): real-A P_sep … vs phantom-phase P_sep … (gap = …)
- Mode selection (B3): snapshots needed to recover K=2 reliably: …
- **Decision:** [ GIBF worth it / drop eigen layer ]  ← applies §2 rule
- Figure: results/B_viability/psep_vs_separation.png

## Recommendation to the full pipeline
- CF/DF: …
- Eigen-decomposition layer: keep / drop, because …
- Next step: …
```

---

## 14. Gotchas the agent must not trip on

1. **Real‑CSM trap.** Generate complex phasor snapshots (§6.3). A real `X` ⇒ real `S` ⇒ real eigenvectors ⇒ the method is meaningless. A test guards this.
2. **Dropping `√λᵢ`.** Carry the eigenvalue weighting (Suzuki Eq. 3). The source report omits it; do **not** copy that.
3. **Phantom phase only in B4.** Everywhere else `A` is real; assert it.
4. **`secsy` API drift.** Verify and pin (§5.2, Appendix C). The CF current‑type keyword and component return order are the usual breakage points.
5. **Irreversible pruning.** Offer grid reduction on/off; report both (B1).
6. **Fair‑comparison discipline.** GIBF and MMV must consume the *identical* `V` and `A`. If you ever feed them different inputs, the headline result is invalid.
7. **Pre‑register the thresholds.** Fill the §2 decision rules into the memo before reading results.
8. **Don't over‑build.** Only Experiments A and B1–B5. No robustness suite, no real‑data demo (§1.6).
9. **Notebooks don't compute.** Heavy work lives in runners → cache; notebooks only load + plot.

---

## 15. References

1. Suzuki, T. (2011). *L₁ generalized inverse beam‑forming algorithm resolving coherent/incoherent, distributed and multipole sources.* J. Sound Vib. 330, 5835–5851. *(Algorithm under test; Eq. 3 = the √λ eigenmode; Eq. 9 = the L2 init; the IRLS + grid‑reduction recipe.)*
2. Vanhamäki, H., & Juusola, L. (2020). *Introduction to Spherical Elementary Current Systems.* In Ionospheric Multi‑Spacecraft Analysis Tools, ISSI SR‑17, Springer. *(§2.7 = the Fukushima/DF‑dominance result; §2.10.1 = grid padding guidance.)*
3. Wax, M., & Kailath, T. (1985). *Detection of signals by information theoretic criteria.* IEEE TASSP 33(2), 387–392. *(AIC/MDL mode selection.)*
4. Fukushima, N. (1976). *Generalized theorem for no ground magnetic effect of vertical currents connected with Pedersen currents in the uniform‑conductivity ionosphere.* Rep. Ionos. Space Res. Japan. *(The no‑ground‑effect theorem behind Question 1.)*
5. Amm, O. (1997). *Ionospheric elementary current systems in spherical coordinates and their application.* J. Geomag. Geoelectr. *(CF/DF SECS formulation.)*
6. Samson, J. C. (1983). *The spectral matrix, eigenvalues, and principal components in the analysis of multichannel geophysical data.* Geophys. J. Int. 72(3), 679–702. *(Spectral‑matrix analysis of geophysical arrays.)*
7. Pinçon, J.‑L., & Motschmann, U. (1998). *Multi‑spacecraft filtering: general framework.* In Analysis Methods for Multi‑Spacecraft Data, ISSI SR‑001.
8. secsy library: https://github.com/klaundal/secsy

Project reference PDFs (in the repo's project knowledge): the Suzuki (2011) GIBF paper and the Vanhamäki & Juusola SECS introduction — consult these for any equation or convention not pinned down here.

*This document is the full build context for the `viability-test/` codebase. If a detail is missing, prefer the simplest choice that preserves the fairness of the GIBF‑vs‑MMV comparison and the realness of `A`, and record the choice in the manifest and memo.*

---

## Appendix A — Mathematical reference sheet (clean ASCII)

```
Forward model (single frequency bin ω):
    B(ω) = A · s(ω)            A real (3·N_mic × N_types·N_grid), s complex

Transfer matrix assembly (current types t in column order):
    block_t = [ Ge_t ; Gn_t ; Gu_t ]            # vertical stack, (3·N_mic × N_grid)
    A       = [ block_t1 | block_t2 | ... ]      # horizontal stack over types

Cross-spectral matrix:
    S = (1/N_snap) · X · Xᴴ                      # complex Hermitian PSD (M×M), M = 3·N_mic

Eigen-decomposition (numpy.linalg.eigh):
    S = U Λ Uᴴ ,  λ_1 ≥ … ≥ λ_M ≥ 0

Eigenmode (Suzuki Eq. 3 — KEEP THE √λ):
    v_i = √λ_i · u_i

Mode selection (Wax–Kailath, complex; p = M − k):
    log_geo = (1/p) Σ_{i=k+1}^{M} ln λ_i
    ari     = (1/p) Σ_{i=k+1}^{M} λ_i
    LLR     = N·p·(log_geo − ln ari)             # ≤ 0
    AIC(k)  = −2·LLR + 2·k·(2M − k)
    MDL(k)  = −LLR   + (1/2)·k·(2M − k)·ln N
    N_modes = argmin_k {AIC or MDL}

L2 min-norm init (underdetermined), per mode:
    a = Aᵀ (A Aᵀ + εI)⁻¹ v ,   ε = eps_frac · λ_max(A Aᵀ)

IRLS weights (p = 1 → L1):
    w = (|a| + floor)^{2−p} ,   W = diag(w)

GIBF underdetermined update:    a = W Aᵀ (A W Aᵀ + εI)⁻¹ v
GIBF overdetermined update:     a = (Aᵀ A + ε W⁻¹)⁻¹ Aᵀ v
Grid reduction:                 keep ⌊β·N_active⌋ largest |a|,  β = 0.9

MMV objective (shared support over K modes):
    min ‖V − A·Ā‖_F²  +  μ·Σ_j ‖Ā[j,:]‖_2        # L2,1 row-sparse
    row weights:  w_j = (‖Ā[j,:]‖_2 + floor)^{2−p}

Intensity map (same accumulation for GIBF and MMV):
    I(j) = Σ_i |a_i(j)|²        (GIBF, per-mode sum)
    I(j) = Σ_k |Ā[j,k]|²        (MMV, over K columns)

Observability ratio (Experiment A):
    ρ_obs(j) = ‖A · e_CF(j)‖ / ‖A · e_DF(j)‖     # unit CF/DF amplitudes at j

Phantom phase (Experiment B4 ONLY):
    A_phase = A ⊙ exp(i·Φ) ,   Φ = synthetic light-speed phase ramp across aperture
```

---

## Appendix B — Build checklist (tick as acceptance criteria pass)

```
[ ] secsy installed + version pinned; Appendix C probe passes; conventions documented
[ ] config_loader: deep-merge base + override
[ ] geometry: array + padded SECS grid; latitude presets; fixed channel/column order (§6.0)
[ ] transfer: REAL A, CF/DF blocks, row-norm, cache round-trip   | test_transfer GREEN
[ ] identifiability: ρ_obs map, SVD/null-space, conditioning
[ ] EXPERIMENT A end-to-end: results + figure + memo Q1 filled   | CF/DF DECISION LANDED
[ ] simulate: complex CN phasors, coherent/incoherent, SNR scaling | test_simulate GREEN
[ ] csm: build + eigh + eigenmodes(√λ·u); CSM provably complex   | test_csm GREEN
[ ] modeselect: complex AIC/MDL in log domain                    | test_modeselect GREEN
[ ] l2_minnorm solver                                            |
[ ] gibf_irls solver (grid-reduction on/off switch)             |
[ ] mmv_l1 solver (shared support)                              | test_solvers GREEN
[ ] metrics: Δr, P_sep, ΔI, spurious, (opt) Wasserstein         | test_metrics GREEN
[ ] plotting: SV spectra, obs maps, 10dB source maps, CI curves, AIC/MDL, summary
[ ] EXPERIMENT B1 (incoherent, headline) + grid-reduction on/off
[ ] EXPERIMENT B3 (mode selection)
[ ] EXPERIMENT B2 (coherent)
[ ] EXPERIMENT B4 (phantom-phase ablation, GIBF-only)
[ ] EXPERIMENT B5 (CF/DF attribution) — ONLY IF Exp A says CF observable
[ ] notebooks load cache + render the two decision figures
[ ] manifest.json written by every runner (config, secsy ver, git commit, seed)
[ ] DECISION_MEMO.md: thresholds pre-registered, both decisions filled, recommendation written
[ ] pytest -q GREEN in < 1 minute; README quickstart works from a clean clone
```

---

## Appendix C — `secsy` verification probe (run before `transfer.py`)

Write and run a throwaway script `scratch/secsy_probe.py` (gitignored). Confirm every item, then delete it and encode the answers in the `transfer.py` adapter docstring.

```python
import secsy, numpy as np
print("secsy version:", getattr(secsy, "__version__", "UNKNOWN"))

# 1) Find the kernel function and its exact name/signature in THIS version.
print([n for n in dir(secsy) if "G_matri" in n.lower() or "SECS" in n])

# 2) Build a tiny known geometry: a few stations, a few SECS poles, fixed altitude.
#    Call the kernel for BOTH current types using each candidate keyword until one works:
#       'curlfree' / 'curl_free' / 'divfree' / 'divergence_free'
#    Record which keyword strings this version accepts.

# 3) Confirm the RETURN ORDER of the three component matrices (Ge,Gn,Gu vs Gn,Ge,Gu)
#    by checking which component is largest for a pole directly under a station
#    (the radial/up field has a characteristic sign/magnitude pattern).

# 4) Confirm units and sign conventions; note the singularity-smoothing kwarg name
#    (cfg uses singularity_limit_km — verify the real keyword).

# 5) CF radial-FAC sanity (the Experiment-A premise): at a HIGH-latitude geometry,
#    the CF-block ground-field norm should be MUCH smaller than the DF-block norm.
#    Print ‖CF block‖_F / ‖DF block‖_F. Do NOT hard-code the result — just observe it.
```

Acceptance: you can name the function, the working current‑type keywords, the component order, the units/sign, and the singularity kwarg — all written into the `transfer.py` adapter docstring. If `secsy` install/import fails, resolve the environment (network + scientific stack) before proceeding; nothing downstream can be trusted without a correct `A`.

---

## Appendix D — Drop‑in root `CLAUDE.md` (so this brief auto‑loads)

Place this at the repository root so Claude Code always has the non‑negotiables and the commands in context, with a pointer to the full brief.

```markdown
# GIBF — agent context

Primary spec: `viability-test/PLAN.md`. READ IT IN FULL before coding.

## Two invariants (never violate)
1. Transfer matrix `A` is REAL. No `exp(ikr)` anywhere except Experiment B4. Assert `np.isrealobj(A)` in solvers.
2. Snapshots are COMPLEX frequency-bin phasors → complex CSM → complex eigenvectors. Never build the CSM from real time samples.

## Carry the √λ
Eigenmode is `v_i = √λ_i · u_i` (Suzuki Eq. 3). Do NOT invert the unit eigenvector.

## Fair comparison
GIBF and MMV-L1 must receive the IDENTICAL `V` and `A` and the same solver params.

## Commands
- Install:  `pip install -r viability-test/requirements.txt`
- Tests:    `cd viability-test && pytest -q`        (must be green, < 1 min)
- Single test: `pytest -q tests/test_transfer.py`
- Exp A:    `python experiments/run_experiment_A.py --config config/exp_A_identifiability.yaml`
- Exp B:    `python experiments/run_experiment_B.py --config config/exp_B1_incoherent.yaml`
- Figures:  `python experiments/make_report_figures.py`

## Workflow
Explore → plan → test-first → implement → verify → commit, in the order of PLAN.md §12.
Land Experiment A (the CF/DF decision) before building Monte-Carlo machinery.
Pre-register §2 thresholds in DECISION_MEMO.md before looking at any results.
Build ONLY Experiments A and B1–B5 — no robustness suite, no real-data demo.
```
