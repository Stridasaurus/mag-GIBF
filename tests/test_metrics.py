"""metrics.py pytest gate (SPEC §S.3 / ROADMAP §8-v tau_r=1 pin)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viability-test"))

from metrics import TAU_R, delta_r_bar, find_peaks_2d, p_sep  # noqa: E402


def _perfect_map(shape, cells, val=1.0):
    I = np.zeros(shape)
    for (r, c) in cells:
        I[r, c] = val
    return I


def test_tau_r_pin():
    assert TAU_R == 1.0


def test_delta_r_bar_zero_for_perfect_map():
    shape = (11, 11)
    true_cells = [(5, 1), (5, 4)]
    I = _perfect_map(shape, true_cells)
    assert delta_r_bar(I, true_cells) == 0.0


def test_delta_r_bar_penalises_missing_source():
    shape = (11, 11)
    true_cells = [(5, 1), (5, 4)]
    I = _perfect_map(shape, [(5, 1)])  # only one peak found
    dr = delta_r_bar(I, true_cells)
    grid_diag = np.hypot(10, 10)
    assert dr == grid_diag / 2  # one matched at 0, one unmatched at grid_diag


def test_p_sep_true_within_tolerance():
    shape = (11, 11)
    true_cells = [(5, 1), (5, 4)]
    # detected peaks 1 cell off from truth -> within tau_r=1
    I = _perfect_map(shape, [(5, 2), (5, 5)])
    assert p_sep(I, true_cells) is True


def test_p_sep_false_outside_tolerance():
    shape = (11, 11)
    true_cells = [(5, 1), (5, 4)]
    I = _perfect_map(shape, [(5, 3), (5, 4)])  # source 1's peak is 2 cells off
    assert p_sep(I, true_cells) is False


def test_p_sep_false_when_peaks_collapse_to_one_source():
    shape = (11, 11)
    true_cells = [(5, 1), (5, 4)]
    I = _perfect_map(shape, [(5, 4)])  # only one detected peak for two sources
    assert p_sep(I, true_cells) is False


def test_find_peaks_ignores_below_threshold():
    I = np.zeros((5, 5))
    I[2, 2] = 1.0
    I[0, 0] = 0.01  # below default 10% relative threshold
    peaks = find_peaks_2d(I, rel_thresh=0.10)
    assert len(peaks) == 1
    assert (peaks[0][0], peaks[0][1]) == (2, 2)
