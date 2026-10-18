# Addenda to the seal

Dated corrections or clarifications about sealed files. A sealed file is never
edited; a correction is appended here instead. Newest last.

*(none at sealing, 2026-10-12)*

## 2026-10-18: the bootstrap seeding convention, written out exactly

*Why.* An independent recheck (reported 2026-10-18) reproduced every verdict and mean but could not reproduce the confidence-interval endpoints bit-for-bit, because SPEC S.1's "seeded from the cell id" does not say how. This addendum states what the sealed code (`viability-test/run_experiment_B2B6_powered_d2.py`, functions `_seed_component`, `boot_rng`, `boot_indices`, `percentile_ci`, `mean_ci`, `summarize_cell`) does. It changes nothing; it documents.

1. **Seed component.** An integer passes through. Anything else (a string, a tuple) becomes `zlib.crc32(repr(x).encode())`, i.e. the CRC-32 of the Python *repr*, which for a string includes its quote characters (`'B2|phi=90|snr=5|d=2|nsnap=64'`, not the bare text).
2. **Generator.** `numpy.random.default_rng([component])` (PCG64), seeded with a one-element list holding the cell id's component. A panel point adds the band index as a second element, `[component, band_index]`. **The master seed is not part of the bootstrap seed** (SPEC S.14 P8, literal reading); it is part of the trial seeds only.
3. **Draw.** One call per cell: `rng.integers(0, n, size=(2000, n))` with n the number of trials, i.e. 2,000 resamples of n indices, row by row. This single index matrix is shared by every arm and method in the cell, so per-method CIs and the paired gap CI are computed on the same resamples.
4. **Interval.** The mean of each resample, then the 2.5th and 97.5th percentiles with numpy's default linear interpolation.
5. **Conditional (correct-trial) error CIs.** Seeded with `boot_rng(tuple(parts))` where `parts` is `[cell_id, method, "correct"]` (threshold variants: `[cell_id, method, tag, "correct"]`), so the component is `crc32(repr((cell_id, method, 'correct')))`; the indices are drawn over the selected correct trials only (n = their count), with the same `rng.integers(0, n, size=(2000, n))` call.
6. **d_min intervals** (`d_axis_arm`). For each d, the resample indices are `boot_indices(n_d, boot_rng(cell_id_d))` with that d-cell's own id, shared by both methods. A resample's P_sep at each d is the mean over its indices; the resample's d_min is the smallest d (scanning d downward so the smallest wins) with resampled P_sep >= the threshold, else +inf. The interval is the 2.5th and 97.5th percentile of those d_min values with numpy's `inverted_cdf` method, not linear interpolation, because d_min is discrete and may be +inf.
7. **Authority.** If any point above disagrees with an attempt to reproduce an interval, the code is the authority and the disagreement is a finding about this description.
