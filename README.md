# mag-GIBF

A pre-registered simulation study of **Generalized Inverse Beamforming (GIBF)**, a per-eigenmode sparse imaging method from aeroacoustics (Suzuki 2011), applied to imaging ionospheric currents from ground magnetometer arrays with the **SECS** (Spherical Elementary Current Systems) source basis. GIBF is compared with joint-sparse **MMV-ℓ1** inversion, with L2 minimum-norm imaging as a reference, on the same simulated data and the same ground kernel.

A paper describing the study is in preparation.

## Questions

1. **Card A: does source phase reach the ground?** Does a realistic field-line resonance give SECS source elements a coherence whose phase is well away from 0° and 180°, and does that phase survive through the ground kernel into the measured cross-spectral matrix? Design: [`EXPERIMENT_CARD_A.md`](EXPERIMENT_CARD_A.md).
2. **Experiment B: per-mode or joint sparsity?** Given the same eigenmodes of the same ground cross-spectral matrix and the same kernel, does per-mode GIBF separate and locate two coherent sources better than joint MMV-ℓ1? Design: [`viability-test/SPEC_experiment_B.md`](viability-test/SPEC_experiment_B.md).

Before either, **Validation Gate V** checks the ground kernel built with [`secsy`](https://github.com/klaundal/secsy).

## Results

Verdicts under the pre-registered rules, as recorded. Interpretation is left to the paper.

| | Verdict | Record |
|---|---|---|
| Gate V | PASS | `results/V_kernel_validation/gateV_summary.json` |
| Card A | Hypothesis H-A holds in 80 of 81 realistic cells (one marginal); ground coherence κ_ground ≈ 0.39–0.68 | `results/A_flr_coherence/tier2_summary.csv` |
| Experiment B | WIN not met; LOSS met at φ = 180° (both SNRs, both routes), not at φ = 0°; B6 not met | `results/B_viability/powered_d2/adjudication.json` |

Experiment B's confirmatory run placed two coherent sources two grid cells (127 km) apart and used 64 snapshots, SNRs of 5 and 10 dB, and 10,000 trials per cell. Its rules, in brief:

- **WIN:** at φ = 90°, GIBF's mean localization error (Δr̄) is at least 20% lower than MMV-ℓ1's (the Δr̄ route), or the smallest separation GIBF reliably resolves (P_sep ≥ 0.8) is at least one grid cell smaller (the separation route), with non-overlapping 95% confidence intervals at both SNRs.
- **LOSS:** the same standard in MMV-ℓ1's favour at a null phase, φ = 0° or 180°.
- **B6:** the WIN standard applied when the number of sources is estimated from the data (MDL) rather than given.

Descriptive runs that do not enter the verdicts: B4 and B6b (`results/B_viability/b4_b6b/`) and the completion run of SPEC §S.17 (`results/prereg_completion/` and `results/reports/`). Their scorecards against expectations written before the data are in `ROADMAP.md` §8.

## How the record is kept

- **Designs before runs.** Each experiment's design was committed before its runs; the `prereg-*` tags mark the key commits, and the experiment log records the changes made along the way.
- **Sealed results.** [`seal/SEAL.md`](seal/SEAL.md) (tag `sealed-2026-10-12`) and [`seal/SEAL_2.md`](seal/SEAL_2.md) (tag `sealed-2-2026-10-18`) hash every sealed file, and `tests/test_seal.py` fails if any of them changes.
- **Post hoc work is kept apart** in `results/post_hoc/` and `viability-test/post_hoc/`; it never changes a sealed verdict.
- **Owed items.** [`prereg/LEDGER.md`](prereg/LEDGER.md) lists every item the pre-registration promised, with its status and evidence. [`prereg/PHFW1.md`](prereg/PHFW1.md) is a parked draft for post hoc and future work; nothing in it is authorized to run.
- **Decision record.** [`ROADMAP.md`](ROADMAP.md) holds the decision tree and the append-only experiment log (§8).

## Reproduce

```bash
git clone --recurse-submodules https://github.com/Stridasaurus/mag-GIBF.git
cd mag-GIBF
conda create -n mhd-env python=3.13 numpy scipy matplotlib pytest -y
conda activate mhd-env
pip install -e ./secsy

python -m pytest -q                       # test suite, including the seal check
python seal/verify_reproducibility.py     # regenerate the first trials of the sealed runs and compare
```

The sealed runs used Python 3.13.13 and NumPy 2.4.6, as recorded in their manifests (for example `results/B_viability/powered_d2/manifest.json`). The submodule pins the `secsy` version.

## Layout

| Path | Contents |
|---|---|
| `viability-test/` | Simulation, solvers, metrics and run scripts, plus the Experiment B design |
| `EXPERIMENT_CARD_A.md` | Card A design |
| `results/` | Outputs: Gate V (`V_kernel_validation/`), Card A (`A_flr_coherence/`), Experiment B (`B_viability/`; the confirmatory run is in `powered_d2/`), the completion run (`prereg_completion/`, `reports/`) and post hoc work (`post_hoc/`) |
| `prereg/` | Ledger of owed items; parked post hoc draft |
| `seal/` | Seal records, hash lists, run logs and the reproducibility check |
| `tests/` | Test suite, including the seal check |
| `ROADMAP.md` | Decision tree and append-only experiment log |
| `secsy/` | SECS library by K. M. Laundal, pinned as a submodule |
| `magnetometer-time-series-simulator/`, `mhd-ibf-reconstruction/` | Submodules from earlier groundwork, not used by these experiments |
| `legacy_mhd_notebook.ipynb` | Earlier notebook; `ROADMAP.md` cites its MDL cutoff |

## Reading older records

- Some experiments were renamed on 2026-07-01, and older records use the old names. `ROADMAP.md` §0 has the map.
- Older records also refer to `handoff.md`, `BUILD_BRIEF.md` and `CLAUDE.md`, working files for AI coding sessions that were removed on 2026-11-08. They remain in the history, for example at tag `sealed-2-2026-10-18`.
- Branch names in older records refer to branches deleted after their pull requests were merged.

## Credits

Project lead: Strider Settgast. Card A Tier 2 was run by Shane Gilbertie.

Much of the code and documentation was written with AI coding assistants (Claude, by Anthropic) under Strider's direction; commit co-author lines show which commits. Paper text is Strider's own.

## License and citation

GPL-3.0; see [`LICENSE`](LICENSE). Until the paper is out, please cite this repository at tag `sealed-2-2026-10-18`.
