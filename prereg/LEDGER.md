# Pre-registration completion ledger

Every item the pre-registration promised, with its status and where the evidence is.
Written 2026-10-18 after the results were sealed (`seal/SEAL.md`), when an audit found
that sealing had been called finished while items were still owed. A row leaves the
"open" statuses only with a path or commit and a date. Any new pre-registered item is
added here the day it is written.

Statuses: **DONE** (data and report exist, sealed); **DONE-UNRENDERED** (data exist,
the promised table or figure does not); **PARTIAL** (run smaller than specified);
**NOT-RUN**; **STRIDER** (a decision or writing that is his, not an agent's);
**HOUSEKEEPING** (not an experiment; blocks or polishes submission).

## 1. Experiments and runs

| ID | Source | Item | Status at 2026-10-18 | Evidence / action |
|---|---|---|---|---|
| L01 | ROADMAP §6; SPEC §S.7 | Gate V kernel validation | DONE | `results/V_kernel_validation/` |
| L02 | EXPERIMENT_CARD_A | Card A Tier 1, Tier 2, floors, IMAGE chain; both promised figures | DONE | `results/A_flr_coherence/`; `fig_tier2_kappa_surface.png` carries the kappa_source overlay and the 0.30/0.10 bands (an earlier evidence packet's "no such figure" was wrong) |
| L03 | SPEC §S.6(4) | Single-source exact-recovery gate | DONE | `results/B_viability/gate_result.json` |
| L04 | brief §9-B1; SPEC §S.2-B1 | B1 incoherent grid, both reductions, "mean +/- 95% CI per method per cell", floor panel | **PARTIAL** | run at **50 trials per cell** (a documented pilot choice in `run_experiment_B.py`); powered re-run specified in SPEC §S.17 item 4 |
| L05 | SPEC §S.2-B3 | B3 mode selection, "large n_trials", AIC and MDL | **PARTIAL** | **50 trials per cell**; powered re-run in SPEC §S.17 item 5 |
| L06 | SPEC §S.2-B2; ROADMAP §5-D2 "report the SIGNED gap at each phi" | B2 coherent grid: phi in {0,30,...,180} x d x SNR | **PARTIAL** | 30 of 175 cells sealed (phi in {0,90,180}, SNR in {5,10}, d axis). **145 not run.** No ruling dropped them; the SPEC §S.14 plan, assembled 2026-09-29, omitted them. SPEC §S.17 item 2 |
| L07 | SPEC §S.7 SLOT-1 | |rho| endpoints {0.70, 0.95} as descriptive rows | **NOT-RUN** | SPEC §S.17 item 3 |
| L08 | SPEC §S.14 | Confirmatory powered B2/B6a at d=2 and its adjudication | DONE | `results/B_viability/powered_d2/`, sealed |
| L09 | SPEC §S.6(5) | Regularization-sensitivity panel | DONE | six d=2 cells, B6a, and the d=1 descriptive panel |
| L10 | SPEC §S.14 | Off-grid row (A4) | DONE | `B2offgrid_*` in `powered_d2/` |
| L11 | SPEC §S.14 item 18 | Peak-rule rescoring | DONE | `results/B_viability/peak-rule-rescoring/` |
| L12 | SPEC §S.2-B4; §S.15 | B4 phantom-phase ablation | DONE | `results/B_viability/b4_b6b/`, sealed |
| L13 | SPEC §S.2-B6b; §S.15 | B6b support jitter | DONE | `results/B_viability/b4_b6b/`, sealed |
| L14 | ROADMAP §0 | B5 CF/DF attribution | N/A | retired by theorem |

## 2. Reporting promised but not rendered

| ID | Source | Item | Status | Evidence / action |
|---|---|---|---|---|
| L20 | SPEC §S.14 item 6 | Signed gap and CI against the 20% line; range of thresholds over which each verdict holds | DONE-UNRENDERED | numbers computed ad hoc 2026-10-12, no table; SPEC §S.17 item 6 |
| L21 | SPEC §S.14 P15 | Delta r-bar under other miss penalties; conditional correct-trial error with CIs; miss / correct / mislocated partition | DONE-UNRENDERED | arrays stored; SPEC §S.17 item 6 |
| L22 | SPEC §S.2-B6; P13 | K-hat breakdown, AIC lookup, forced-K sweeps | DONE-UNRENDERED | in the manifest; no figure |
| L23 | SPEC §S.14 P3 | d_min at 0.7 and 0.9, monotonicity flag, uneven d spacing | DONE | `adjudication.json` |
| L24 | SPEC §S.14 P4 | Arm that fires at each SNR; mixed-arm flag | DONE | `adjudication.json` |
| L25 | SPEC §S.14 stop behaviour | Stop-reason counts and non-convergence rates | DONE | manifest |
| L26 | SPEC §S.14 item 6 | The rule's operating characteristic at n = 10,000 (the "about 23%" smallest reliably detectable effect) | NOT RENDERED | SPEC §S.17 item 6 |
| L27 | SPEC §S.5 (i)-(v) | Five Experiment-B figures: phi-resolved decision figure with B1 floor panel; B4; B6 panel; regularization panel; B3 surfaces | **NOT PRODUCED** | only Card A figures exist; SPEC §S.17 item 7 |
| L28 | SPEC §S.5 | "Every figure's underlying arrays saved" | NOT DONE | SPEC §S.17 item 7 |
| L29 | ROADMAP §5-D2 kill clause | Evidence table for the four-part kill condition (facts only) | NOT RENDERED | SPEC §S.17 item 8 |

## 3. Decisions and writing that are Strider's

| ID | Source | Item | Status |
|---|---|---|---|
| L40 | ROADMAP §5-D2, §4 | Reading of the D2 outcome map and application of the kill clause | **STRIDER** (an agent supplies the evidence table, L29) |
| L41 | ROADMAP §4.4 | `DECISION_MEMO.md` | **STRIDER**: he decides whether it is a research record or his own writing; not created |
| L42 | ROADMAP §4, §7 | Which downstream stub to expand into Manifesto #2 | **STRIDER** |
| L43 | ROADMAP §4.5 | The paper | **STRIDER**, outside this repo |
| L44 | 2026-10-11 | The agent-written Gate V paragraph | RESOLVED: removed from the public tree; private copy in the vault |
| L45 | format study | A deviations register for the paper's deviations table | offered, private vault, not started |

## 4. Housekeeping

| ID | Item | Status |
|---|---|---|
| L50 | Zenodo archive and software citation (JGR Open Research) | open |
| L51 | Public-repo prose corrections from the 2026-09-29 debrief: washout wording (ROADMAP §8 07-12, SPEC §S.7); "isotropic" and "0.997" (SPEC §S.10); overdetermined-branch descriptions; repo `CLAUDE.md` python version; `EXPERIMENT_CARD_A.md` citations (Anderson 1989, Waters title) | open |
| L52 | Citation audit; add Chen and Huo 2006, Cotter et al. 2005 | open |
| L53 | README "how to read this repo" | open |
| L54 | Stale vault notes (`secs-gibf-viability`, active-projects entries) | open |
| L55 | The owner's other eleven public repositories not audited for agent-written paper prose | open |

## 5. Added 2026-10-18 from an independent recheck report (pasted by Strider; evidence class: third-party synthesis, secondary)

| ID | Item | Status |
|---|---|---|
| L60 | Recheck reproduced all four D2 verdicts and every mean; CI endpoints only statistically (worst 0.07 of a half-width on a verdict-feeding endpoint) | recorded; the check's scripts are not in this repo's record yet (L63) |
| L61 | The bootstrap seeding convention (generator, draw order) is not reproducible from the SPEC's wording ("crc32(cell_id)") | **open**: document the exact convention in `seal/ADDENDA.md` (an addendum, not an edit of the sealed record); candidate for the deviations table |
| L62 | Saved files hold no raw source maps, so the peak-finding and scoring step is not checkable from saved data | **post hoc** (regenerate trials from the seed, rescore with an independently written scorer); goes to the next-phase pre-registration |
| L63 | The recheck's scripts and outputs live in a temporary scratchpad | **open**: copy to the vault GIBF project as a durable record |
| L64 | Solver, simulator, kernel and mode-count estimator were not independently checked | disclosed; rests on the single-source gate, tests and code read; next-phase item |
| L65 | The mirrored MMV-favoured standard is also met at phi = 90 at both SNRs (unregistered) | descriptive only; to be reported as such and never adjudicated |

## 6. PHFW-1 (post hoc and future work; specified in `prereg/PHFW1.md`, fixed 2026-10-18)

None of these may run before its spec is tagged; each is done only per PHFW1 §2 item 7.

| ID | Item | Status |
|---|---|---|
| L70 | A1 why phi=0 fails / phi=180 works (lambda2/lambda1) | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L71 | A2 are GIBF and MMV failures the same trials | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L72 | A3 off-grid dose-response | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L73 | A4 stop reason against outcome | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L74 | A5 reduction OFF vs ON transitions | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L75 | A6 Monte-Carlo uncertainty | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L76 | B1 dense epsilon sweep (seed 3602037586) | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L77 | B2 off-grid at d=1 and 20 dB (seed 3449719975) | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L78 | B3 altitude mismatch (seed 2883321744) | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L79 | B4 unequal powers (seed 758546561) | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L80 | B5 snapshot-count sweep (seed 3922815376) | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L81 | B6 kernel coherence, no trials | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L82 | C1 complex MMV on phantom kernels (seed 1513848041) | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L83 | C2 tuned best-versus-best (seed 2896748822) | PARKED (Strider, 2026-10-18: no future-work pre-registration yet) |
| L90 | Parked items P1-P15 (PHFW1 §4) | PARKED: need an addendum with their own seed and expectations before any run |
