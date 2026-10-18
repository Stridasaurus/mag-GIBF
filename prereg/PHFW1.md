# PHFW-1: pre-registration of the post-hoc and future-work phase

> **STATUS (amended 2026-10-18, same day, by Strider's ruling in chat): PARKED DRAFT.** Strider ruled that
> future work is not to be pre-registered yet: the paper's pre-registered base is the frozen SPEC plus the S.17
> completion, and future directions are a separate decision that will be specified later, when the paper's writing
> clarifies what is wanted. **Nothing in this file is authorized to run.** It stays as a record of what was
> considered, with its seeds drawn and unused. Its expectations are not binding; if any item is revived it is
> re-read, re-justified and re-tagged first. The headline standard (frozen matched-by-rule regularization, SPEC S.6,
> pinned 2026-07-19) is not reopened by anything here.

Fixed 2026-10-18, **before any PHFW-1 code is written and before any result of the
completion run (SPEC §S.17) has been read**. Everything here is exploratory with
respect to the sealed record (SPEC §S.16): it cannot change a sealed verdict, and
any text that cites it says "post hoc". What this document adds is order: the
question, the analysis and the expectation are fixed first, the data are looked at
second, and every registered item is reported whatever it shows.

This file is a specification, not paper prose. The paper is written by hand, outside
this repository.

## 1. Findings that motivate the items (sealed record; each is a fact about the sealed data)

1. The GIBF-MMV gap at phi = 90 lives at one regularization setting: at one tenth of
   the matched epsilon the two tie; at ten times it is also ~0 (sealed regularization panel).
2. With grid reduction OFF, GIBF and MMV are both near-perfect at phi = 90 (sealed OFF arm).
3. phi = 0 fails for every method; phi = 180 is easy even for L2.
4. GIBF is indifferent to an over-specified K; MMV degrades (B6a).
5. B4: one wavelength of phantom phase hurts GIBF; 2 to 5 wavelengths help; the literal 15 mHz value is
   indistinguishable from the real kernel. B4 never ran MMV.
6. B6b: support jitter moves the gap toward GIBF in every variant.
7. MMV's d_min = 1 at phi = 180 rests on exact on-grid sparse recovery (the LOSS verdict's d_min arm).
8. The MDL estimate of K is 6 (the ceiling) in all 10,000 B6a trials.

## 2. Rules for every item

1. **Lanes.** Code in `viability-test/post_hoc/` (new modules only; sealed modules are imported, never
   edited). Output in `results/post_hoc/<item>/`, with `manifest.json` carrying `"status": "post_hoc"`,
   `"sealed_commit"`, the item id and its seed (`tests/test_seal.py` enforces the first two).
2. **Seeds.** Drawn now, in one isolated interpreter (`python -I`) at 2026-10-18T15:37:52Z, one per
   randomised item, distinct from every prior seed. Table in §5. A seed is used by exactly one item.
3. **No selective reporting.** Every registered item is reported, including nulls and failures of its
   expectation. An item that cannot be built is dropped *in the ledger with the reason*, never silently.
4. **No re-adjudication.** The D2 rules are not applied to post-hoc data. Where a post-hoc quantity is a
   relative gap, it is reported with its CI and the word "post hoc", and is not compared against the 20% line
   as a verdict.
5. **Expectations** are stated before data, scored afterwards as held / not held, and never edited. They are
   not rules.
6. **Confirmatory route.** A post-hoc finding becomes a claim only through a new pre-registered run on a new
   seed and ideally a new geometry (parked item P1), with its own rule written first. Until then it is a
   finding or an open question.
7. **Per-item definition of done.** An item is done only when all of: spec in this file (tagged), module
   committed clean, tests passing, smoke on seed 1, full run, report, ledger row updated with commit and
   path, and the run's manifest records the module's git blob. `prereg/LEDGER.md` is the single list; a
   test fails if an item here has no ledger row or a ledger row has no status.
8. **Agent-chosen values** are marked "(agent-chosen)" with their justification.

## 3. Registered items

### Free (sealed arrays only; no trials; no seed)

All read the confirmatory run's stored arrays (`results/B_viability/powered_d2/`). All are scored over the
six confirmatory cells at d = 2 unless stated.

- **A1 (L70). Why phi = 0 fails and phi = 180 works.** Per cell, the distribution of lambda2/lambda1 (from the
  stored top-eight trace-normalised eigenvalues) and, per method, the failure rate within lambda2/lambda1
  quartiles. *Expectation:* the median lambda2/lambda1 is smallest at phi = 0 and largest at phi = 180
  (at both SNRs); at phi = 90 each method's failure rate falls with lambda2/lambda1 in at least three of the
  four adjacent quartile steps.
- **A2 (L71). Are GIBF's and MMV's failures the same trials?** At phi = 90, the 2x2 table of per-trial
  correct / not-correct, and the odds ratio with a bootstrap CI. *Expectation:* odds ratio > 1 (failures
  positively associated).
- **A3 (L72). Off-grid dose-response.** From the six sealed off-grid cells: Delta r-bar and P_sep by the
  pre-specified offset-magnitude bins and by north and east components. *Expectation:* both methods'
  Delta r-bar is non-decreasing across the offset-magnitude bins; a north offset is milder than an east
  offset (stated verbatim in the 2026-10-11 notes, never tested).
- **A4 (L73). Stop reason against outcome.** Per method and cell, the failure rate by stop reason.
  *Expectation:* GIBF trials that exit on cost increase fail more often than those that exit on relative tolerance.
- **A5 (L74). Reduction OFF against ON on the same trials.** Per method, the 2x2 transition table of correct
  / not-correct between the ON and OFF arms. *Expectation:* at phi = 90 OFF repairs more ON-failures than it
  breaks, for both methods.
- **A6 (L75). Monte-Carlo uncertainty.** The standard error of every headline mean and the interval an
  independent 10,000-trial sample would give. No expectation.

### Cheap (fresh seeds; hours of compute)

All use the shared fairness configuration, oracle K = 2 (except where stated), n_snap = 64, |rho| = 0.85,
10,000 trials per cell, grid reduction ON, and the sealed runner's peak rule, tau_r and bootstrap. L2, GIBF
and MMV on the same trials unless stated. Arms within a cell share data draws.

- **B1 (L76). Dense epsilon sweep.** Seed 3602037586. The six confirmatory cells; epsilon multiples of the
  matched rule in {0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 20} (agent-chosen: log-spaced
  around the rule, bracketing the five-point sealed band). Quantities: P_sep and Delta r-bar per method versus
  epsilon; each method's best multiple (the log-centroid of the multiples within 0.02 of the maximum P_sep);
  the ratio of GIBF's to MMV's best multiple; each method's best P_sep. The maxima are in-sample (selected on the
  evaluation draws); at this n and a smooth curve the bias is negligible, and the tuned comparison proper is C2.
  *Expectations:* (E-B1a) at phi = 90, GIBF's best P_sep is within 0.05 of MMV's best P_sep at both SNRs;
  (E-B1b) the ratio of best multiples is within a factor of 3 across the six cells.
- **B2 (L77). Off-grid at d = 1 and at 20 dB.** Seed 3449719975. Cells (phi, SNR, d): phi in {90, 180} x
  SNR in {5, 10, 20} x d = 1, plus phi in {90, 180}, d = 2, SNR 20 (eight cells); each run on-grid and
  off-grid (rigid per-trial offset uniform over one cell, as sealed A4) on the same draws, so the control is
  re-run under this seed. Quantities: P_sep and Delta r-bar per method, on-grid and off-grid.
  *Expectation:* off-grid, MMV's P_sep at d = 1 is below 0.8 at phi = 180 and at phi = 90 for every SNR
  (MMV's on-grid d_min = 1 does not survive an off-grid source).
- **B3 (L78). Altitude mismatch.** Seed 2883321744. Data generated with the shell at 90, 110 (control) and 130 km,
  inverted with the 110 km kernel; six confirmatory cells (18 jobs). If the existing transfer builder cannot
  build a kernel at another shell height without editing a sealed module, the item is dropped in the ledger
  with that reason. *Expectation:* P_sep is non-increasing in |h - 110| for both methods, and the
  MMV-minus-GIBF ordering at phi = 90 does not change sign.
- **B4 (L79). Unequal powers.** Seed 758546561. Source powers (1, p), p in {1, 0.3, 0.1}; phi in {0, 90, 180};
  SNR in {5, 10}; d = 2 (18 cells; p = 1 re-run as control). Both true sources must be matched for P_sep.
  K > 2 sources are not registered here (parked). *Expectation:* P_sep is non-increasing as p falls for both
  methods, and the ordering at phi = 90 does not change sign.
- **B5 (L80). Snapshot-count sweep.** Seed 3922815376. n_snap in {16, 32, 64, 128, 256}; phi = 90, d = 2,
  SNR in {5, 10}; oracle K (ten cells; 64 re-run as control). *Expectation:* P_sep is non-decreasing in
  n_snap for both methods, and the sign of the P_sep gap does not change.
- **B6 (L81). Kernel coherence, no trials, no seed.** For the real kernel and the four phantom kernels, the
  normalised inner product of the two source columns on the centre row against separation d = 1..8, and its
  Spearman correlation with the sealed B4 P_sep(d) across the cells. *Expectations:* the real-kernel
  coherence decreases with d (d = 1 > 2 > 3); the rank correlation with GIBF's sealed B4 P_sep is negative
  (more coherent columns, lower P_sep). No expectation on the one-wavelength dip (hypothesis-generating).

### Medium (new solver code; one new module each)

- **C1 (L82). Complex MMV on the phantom kernels.** Seed 1513848041. A complex-capable MMV in its own module
  (the standard solvers keep asserting a real kernel); tested equal to the standard MMV on a real kernel
  (indices, iterations, stop reasons). B4's incoherent scenario, SNR in {5, 10}, d in {1, 2, 3}, kernels
  real / lit / w1 / w2 / w5, GIBF and MMV on the same draws, reduction ON (six cells x five kernels).
  Incoherent sources only (a coherent phantom-kernel scenario is parked). *Expectations:* MMV's P_sep at
  d = 2 is at least GIBF's for every kernel (the ordering is not changed by phase); the MMV-minus-GIBF P_sep
  gap at d = 2 is smaller at w5 than at the real kernel.
- **C2 (L83). Tuned best-versus-best.** Seed 2896748822. Each method's epsilon multiple is chosen on tuning
  trials and evaluated on disjoint test trials. Grid: the B1 multiples. Within each cell, trials 0-2,999 are
  tuning and 3,000-9,999 test (agent-chosen split). One scalar multiple per method is chosen to maximise mean
  P_sep pooled over the six confirmatory cells' tuning trials; it is then applied unchanged to the test
  trials of all six. Report P_sep and Delta r-bar per method with CIs, the relative gap, and the chosen
  multiples. *Expectation:* tuned GIBF's relative Delta r-bar disadvantage at phi = 90 is smaller than the
  sealed matched-rule disadvantage at both SNRs (about 50%); the P_sep ordering at phi = 90 does not change sign.

## 4. Parked (not registered; no run until an addendum with its own seed and expectations is committed first)

P1 a second geometry (the natural venue for a blind confirmatory follow-up); P2 distributed sources;
P3 K > 2 and coherent multi-source structure; P4 M-SBL and other tuning-free comparators; P5 random-matrix
K estimators (motivated by finding 8); P6 ordinary-tool comparison (MUSIC, Capon, regularised LS, CLEAN-SC);
P7 a noise-matched epsilon rule; P8 coherent sources on phantom kernels; P9 a scorer check of the peak-finding
step by regenerating trials and scoring with an independently written scorer (recheck finding, LEDGER L62);
P10 the real-data Pi2 proxy; P11 dispersion and wave vectors; P12 theory of d_min; P13 MEG transfer;
P14 learned inversion and a benchmark suite; P15 the information-decomposition cross-link.

## 5. Seeds (drawn 2026-10-18T15:37:52Z, isolated interpreter)

| Item | Seed |
|---|---|
| B1 dense epsilon sweep | 3602037586 |
| B2 off-grid d = 1 and 20 dB | 3449719975 |
| B3 altitude mismatch | 2883321744 |
| B4 unequal powers | 758546561 |
| B5 snapshot sweep | 3922815376 |
| C1 complex MMV | 1513848041 |
| C2 tuned best-versus-best | 2896748822 |
| reserved (second geometry, P1) | 3451550350 |

## 6. Decisions that are Strider's (flagged, not decided here)

1. Whether PHFW-1 runs as written, in part, or not at all; which of the Medium items to run.
2. Applying the D2 outcome map and the kill clause (the evidence table is produced in SPEC §S.17 item 8).
3. Whether `DECISION_MEMO.md` is a record or his writing; which downstream stub is expanded.
4. Wiping `PARAGRAPH.md` from git history, and the wider Claude-surface wipe.
5. Whether to promote any post-hoc finding to a confirmatory follow-up (§2 item 6).
