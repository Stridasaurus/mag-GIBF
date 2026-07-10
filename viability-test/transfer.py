"""Real SECS transfer matrix A via secsy — the Gate-V-pinned adapter.

ADAPTER CONTRACT — every fact below was pinned by Gate V (ROADMAP §8
2026-07-07; artifacts in results/V_kernel_validation/), not assumed:

  function        secsy.utils.get_SECS_B_G_matrices(glat, glon, r, plat,
                  plon, current_type=..., RI=...)
  version         1.0.1.dev38+g6f699cbd (editable install of the ./secsy
                  submodule pin; recorded in the Gate V manifest)
  keywords        config 'divfree' -> 'divergence_free',
                  config 'curlfree' -> 'curl_free'
  return order    (Ge, Gn, Gr) — east, north, RADIAL(up), each
                  (n_stations, n_poles); order and units independently
                  pinned by V4's analytic under-pole radial match
  units           tesla per unit SECS amplitude [A]; agrees with
                  independent Biot-Savart quadrature to <= 1.6e-4 (V4/V5)
  hazard (V1)     station-pole (lat, lon) coincidence produces NaN in the
                  horizontal columns that poisons even the analytic CF
                  zero (0 * NaN) — this module FORBIDS coincidence and
                  asserts np.isfinite(A).all() after every build
  theorem (V3)    the CF ground block is analytically zero below the
                  shell (secsy hard-codes the full Fukushima pair), so
                  DF-only is the default and the Experiment-B invariant;
                  the CF block is exposed only for explicit probes

Row order of A: vstack([Ge, Gn, Gr]) per current type -> (3*n_stations,
n_types*n_poles), real float64 (brief §6.2). Row normalisation 'none' or
'rms' (each row scaled to unit mean |row|); A stays real.

Provenance workflow (audit A3): this module was committed before first
use; cache files are derived data (gitignored), keyed by a sha256 of the
full build configuration.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from secsy.utils import RE, get_SECS_B_G_matrices

RI_DEFAULT = RE + 110e3  # SECS pole shell radius [m] (library default; Gate V)
COINCIDENCE_TOL_DEG = 1e-8

_SECSY_KEYWORD = {"divfree": "divergence_free", "curlfree": "curl_free"}


def _secsy_G(glat, glon, r, plat, plon, current_type, RI):
    """Thin adapter: config keyword in, (Ge, Gn, Gr) out. All secsy contact
    goes through here; nothing else in the repo names a secsy keyword."""
    if current_type not in _SECSY_KEYWORD:
        raise ValueError(f"unknown current type {current_type!r}; "
                         f"expected one of {sorted(_SECSY_KEYWORD)}")
    return get_SECS_B_G_matrices(
        glat, glon, r, plat, plon,
        current_type=_SECSY_KEYWORD[current_type], RI=RI,
    )


@dataclass
class TransferMatrix:
    A: np.ndarray
    station_lat: np.ndarray
    station_lon: np.ndarray
    pole_lat: np.ndarray
    pole_lon: np.ndarray
    types: tuple
    row_scale: np.ndarray  # applied normalisation vector (A_raw * scale[:, None] = A)
    row_normalisation: str
    r: float
    RI: float
    from_cache: bool = field(default=False, compare=False)

    @property
    def n_stations(self):
        return self.station_lat.size

    @property
    def n_channels(self):
        return 3 * self.n_stations

    @property
    def n_grid(self):
        return self.pole_lat.size

    def col_slice(self, current_type):
        i = self.types.index(current_type)
        return slice(i * self.n_grid, (i + 1) * self.n_grid)


def _cache_key(station_lat, station_lon, pole_lat, pole_lon, types,
               row_normalisation, r, RI):
    h = hashlib.sha256()
    for a in (station_lat, station_lon, pole_lat, pole_lon):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    h.update(repr((tuple(types), row_normalisation, float(r), float(RI)))
             .encode())
    return h.hexdigest()[:16]


def build_transfer_matrix(station_lat, station_lon, pole_lat, pole_lon,
                          current_types=("divfree",),
                          row_normalisation="none",
                          r=RE, RI=RI_DEFAULT, cache_dir=None):
    """Build the real transfer matrix A (3*n_stations, n_types*n_poles).

    DF-only by default (Gate V / SPEC invariant). Forbids station-pole
    coincidence (V1 NaN hazard) and asserts finiteness after every build.
    If cache_dir is given, an A_cache_{hash}.npz round-trip is used.
    """
    station_lat = np.atleast_1d(np.asarray(station_lat, dtype=np.float64)).ravel()
    station_lon = np.atleast_1d(np.asarray(station_lon, dtype=np.float64)).ravel()
    pole_lat = np.atleast_1d(np.asarray(pole_lat, dtype=np.float64)).ravel()
    pole_lon = np.atleast_1d(np.asarray(pole_lon, dtype=np.float64)).ravel()
    if station_lat.size != station_lon.size:
        raise ValueError("station lat/lon length mismatch")
    if pole_lat.size != pole_lon.size:
        raise ValueError("pole lat/lon length mismatch")
    types = tuple(current_types)
    if row_normalisation not in ("none", "rms"):
        raise ValueError(f"unknown row_normalisation {row_normalisation!r}")

    # V1 hazard: exact (lat, lon) coincidence => NaN. Forbid outright.
    d_lat = np.abs(station_lat[:, None] - pole_lat[None, :])
    d_lon = np.abs(station_lon[:, None] - pole_lon[None, :])
    hits = np.argwhere((d_lat < COINCIDENCE_TOL_DEG) &
                       (d_lon < COINCIDENCE_TOL_DEG))
    if hits.size:
        i, j = hits[0]
        raise ValueError(
            f"station {i} and pole {j} coincide at "
            f"({station_lat[i]}, {station_lon[i]}) — forbidden by the Gate V "
            f"adapter contract (coincidence => NaN that poisons even the "
            f"hard-coded CF zero); offset the grids")

    key = _cache_key(station_lat, station_lon, pole_lat, pole_lon, types,
                     row_normalisation, r, RI)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"A_cache_{key}.npz"
        if cache_path.exists():
            d = np.load(cache_path)
            return TransferMatrix(
                A=d["A"], station_lat=station_lat, station_lon=station_lon,
                pole_lat=pole_lat, pole_lon=pole_lon, types=types,
                row_scale=d["row_scale"], row_normalisation=row_normalisation,
                r=r, RI=RI, from_cache=True)

    blocks = []
    for t in types:
        ge, gn, gr = _secsy_G(station_lat, station_lon, r,
                              pole_lat, pole_lon, t, RI)
        blocks.append(np.vstack([ge, gn, gr]))
    a = np.hstack(blocks).astype(np.float64, copy=False)

    assert np.isrealobj(a) and a.dtype == np.float64
    assert a.shape == (3 * station_lat.size, len(types) * pole_lat.size)
    assert np.isfinite(a).all(), \
        "non-finite entries in A despite coincidence guard — investigate " \
        "before using (Gate V adapter contract)"

    if row_normalisation == "rms":
        mean_abs = np.abs(a).mean(axis=1)
        if not (mean_abs > 0).all():
            raise ValueError("zero row in A; cannot rms-normalise")
        row_scale = 1.0 / mean_abs
    else:
        row_scale = np.ones(a.shape[0])
    a = a * row_scale[:, None]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, A=a, row_scale=row_scale)

    return TransferMatrix(
        A=a, station_lat=station_lat, station_lon=station_lon,
        pole_lat=pole_lat, pole_lon=pole_lon, types=types,
        row_scale=row_scale, row_normalisation=row_normalisation,
        r=r, RI=RI, from_cache=False)


def unit_source_vector(tm, current_type, grid_index):
    """One-hot amplitude vector (length n_types*n_grid) for observability
    probes: 1.0 at (current_type, grid_index), 0 elsewhere."""
    if not 0 <= grid_index < tm.n_grid:
        raise IndexError(f"grid_index {grid_index} out of range 0..{tm.n_grid - 1}")
    v = np.zeros(len(tm.types) * tm.n_grid)
    v[tm.col_slice(current_type).start + grid_index] = 1.0
    return v
