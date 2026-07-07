# Gate V — the paper paragraph (headline calibration)

> Deliverable per `ROADMAP.md` §6 Gate V. Cite as calibration, never as a finding.

**Kernel validation.** The transfer kernel `A` is built from the `secsy` library
(v1.0.1.dev38+g6f699cbd), whose SECS magnetic-field matrices implement the
closed-form expressions of Amm & Viljanen (1999) and Vanhamäki & Juusola (2020).
For curl-free (CF) elementary systems these are the expressions for the **full
Fukushima pair** — the horizontal CF sheet current *together with* its radial
field-aligned currents — whose ground-level field vanishes identically
(Fukushima 1976; Amm 1997): the library returns an exactly zero CF ground block
at every latitude preset (70°/55°/35°), independent of the singularity-smoothing
setting, so DF-only imaging from ground arrays enters this pipeline as a theorem,
not a threshold. Because that zero is analytic-by-construction, we confirmed the
physics independently by Biot–Savart quadrature: the three CF-pair constituents
(sheet current, polar line FAC, distributed return FACs), integrated on a
~10⁷-node grid, cancel at ground level to within quadrature error — residual
≤ 4.5×10⁻⁴ of the largest constituent, which is itself ~2.4×10⁻¹³ T per ampere
of SECS amplitude — while above the current sheet the same quadrature reproduces
the library's CF field to 3×10⁻⁶. The divergence-free block, on which every
downstream number rests, was validated against the same independent quadrature
(relative error ≤ 1.6×10⁻⁴ across test geometries) and against the analytic
under-pole radial field (3.5×10⁻⁴, consistent with the library's internal
θ-clipping at ~0.026°). All G-matrices are real float64; the verified
conventions (return order `(Ge, Gn, Gr)`; accepted keywords
`'divergence_free'`/`'curl_free'`; positive DF amplitude = eastward/
counterclockwise sheet current about the pole; positive CF amplitude = sheet
current away from the pole fed by a line FAC into the shell) are encoded in the
`transfer.py` adapter. One adapter contract emerged from the probe: a SECS pole
sharing a station's latitude–longitude produces NaN columns (a 0/0 tangent
normalization) that poison even the hard-coded CF zero, so the adapter must
forbid station–pole coincidence and assert finiteness of `A`.

Artifacts: `gateV_summary.json`, `gateV_fields.npz`, `manifest.json` (this
directory); probe source `viability-test/gateV_kernel_validation.py`.
