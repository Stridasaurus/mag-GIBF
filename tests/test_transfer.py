"""transfer.py pytest gate (brief §6.2 invariants + Gate V adapter contract).

Run: conda run -n mhd-env python -m pytest tests/test_transfer.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from transfer import (RI_DEFAULT, build_transfer_matrix,  # noqa: E402
                      unit_source_vector)

# Codified Fukushima expectation (brief §7): at the ground the CF block is
# analytically zero in secsy (hard-coded full pair, Gate V V3), so its norm
# must be vanishingly small next to the DF block. Documented constant, not a
# silent pass — if this fails, that is itself a flagged result.
CF_DF_NORM_RATIO_MAX = 1e-12

# Small offset grids (station points deliberately off the pole points — the
# coincidence guard forbids exact overlap per Gate V V1).
ST_LAT, ST_LON = np.meshgrid(np.array([69.37, 70.37]),
                             np.array([-4.53, 0.47, 4.47]), indexing="ij")
PO_LAT, PO_LON = np.meshgrid(np.arange(67.0, 74.0, 2.0),
                             np.arange(-8.0, 9.0, 8.0), indexing="ij")
GEOM = dict(station_lat=ST_LAT.ravel(), station_lon=ST_LON.ravel(),
            pole_lat=PO_LAT.ravel(), pole_lon=PO_LON.ravel())
N_ST, N_PO = ST_LAT.size, PO_LAT.size


def test_shape_realness_dtype():
    tm = build_transfer_matrix(**GEOM)
    assert tm.types == ("divfree",)
    assert tm.A.shape == (3 * N_ST, N_PO)
    assert np.isrealobj(tm.A) and tm.A.dtype == np.float64
    assert np.isfinite(tm.A).all()
    assert tm.n_channels == 3 * N_ST and tm.n_grid == N_PO


def test_coincidence_guard_raises():
    bad = dict(GEOM)
    bad["station_lat"] = np.append(GEOM["station_lat"], PO_LAT.ravel()[0])
    bad["station_lon"] = np.append(GEOM["station_lon"], PO_LON.ravel()[0])
    with pytest.raises(ValueError, match="coincide"):
        build_transfer_matrix(**bad)


def test_row_normalisation_rms():
    tm = build_transfer_matrix(**GEOM, row_normalisation="rms")
    assert np.allclose(np.abs(tm.A).mean(axis=1), 1.0)
    # row_scale recorded such that raw * scale = A: re-derive raw and check
    raw = tm.A / tm.row_scale[:, None]
    tm_none = build_transfer_matrix(**GEOM)
    assert np.allclose(raw, tm_none.A)
    assert np.all(tm_none.row_scale == 1.0)


def test_cache_round_trip(tmp_path):
    tm1 = build_transfer_matrix(**GEOM, row_normalisation="rms",
                                cache_dir=tmp_path)
    assert not tm1.from_cache
    assert len(list(tmp_path.glob("A_cache_*.npz"))) == 1
    tm2 = build_transfer_matrix(**GEOM, row_normalisation="rms",
                                cache_dir=tmp_path)
    assert tm2.from_cache
    assert np.array_equal(tm1.A, tm2.A)
    assert np.array_equal(tm1.row_scale, tm2.row_scale)
    # different config -> different cache entry, not a stale hit
    tm3 = build_transfer_matrix(**GEOM, cache_dir=tmp_path)
    assert not tm3.from_cache
    assert len(list(tmp_path.glob("A_cache_*.npz"))) == 2


def test_cf_block_is_fukushima_null():
    tm = build_transfer_matrix(**GEOM, current_types=("divfree", "curlfree"))
    assert tm.A.shape == (3 * N_ST, 2 * N_PO)
    df = tm.A[:, tm.col_slice("divfree")]
    cf = tm.A[:, tm.col_slice("curlfree")]
    assert np.linalg.norm(cf) <= CF_DF_NORM_RATIO_MAX * np.linalg.norm(df)


def test_unit_source_vector():
    tm = build_transfer_matrix(**GEOM, current_types=("divfree", "curlfree"))
    v = unit_source_vector(tm, "curlfree", 3)
    assert v.shape == (2 * N_PO,)
    assert v.sum() == 1.0 and v[N_PO + 3] == 1.0
    with pytest.raises(IndexError):
        unit_source_vector(tm, "divfree", N_PO)


def test_unknown_keywords_rejected():
    with pytest.raises(ValueError, match="current type"):
        build_transfer_matrix(**GEOM, current_types=("curl_free",))
    with pytest.raises(ValueError, match="row_normalisation"):
        build_transfer_matrix(**GEOM, row_normalisation="rms2")


def test_ri_default_matches_gateV():
    from secsy.utils import RE
    assert RI_DEFAULT == RE + 110e3
