"""metrics.py — localisation error and separation-success rate.

Per SPEC §S.3 / ROADMAP §8-v: tau_r = 1 grid cell (supersedes the archived
brief's 1.5-cell tau_r); "correctly placed" = within 1 grid cell of a true
source under minimum-cost matched assignment; "distinct" = the two reported
peaks match different true sources (guaranteed by the one-to-one assignment
so long as the assignment uses >= n_true distinct detected peaks).

Delta-r-bar unmatched-source penalty = grid diagonal (archived brief §6.6).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

TAU_R = 1.0  # pinned ROADMAP §8-v


def find_peaks_2d(I2d, rel_thresh=0.10, max_peaks=8):
    """Local maxima (8-connected) at or above rel_thresh * global max,
    sorted by intensity descending."""
    max_val = I2d.max()
    if max_val <= 0:
        return []
    thresh = rel_thresh * max_val
    nlat, nlon = I2d.shape
    peaks = []
    for i in range(nlat):
        for j in range(nlon):
            v = I2d[i, j]
            if v < thresh:
                continue
            is_max = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nlat and 0 <= jj < nlon and I2d[ii, jj] > v:
                        is_max = False
                        break
                if not is_max:
                    break
            if is_max:
                peaks.append((i, j, float(v)))
    peaks.sort(key=lambda t: -t[2])
    return peaks[:max_peaks]


def _grid_diag(shape):
    return float(np.hypot(shape[0] - 1, shape[1] - 1))


def _matched_assignment(true_cells, det_cells):
    cost = np.empty((len(true_cells), len(det_cells)))
    for i, (tr, tc) in enumerate(true_cells):
        for j, (dr, dc) in enumerate(det_cells):
            cost[i, j] = np.hypot(tr - dr, tc - dc)
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind, cost


def delta_r_bar(I2d, true_cells, rel_thresh=0.10):
    """Mean matched localisation error, in grid cells. Unmatched true
    sources (fewer detections than true sources) penalised at the grid
    diagonal (archived brief §6.6)."""
    grid_diag = _grid_diag(I2d.shape)
    n_true = len(true_cells)
    det_cells = [(p[0], p[1]) for p in find_peaks_2d(I2d, rel_thresh)]
    if not det_cells:
        return grid_diag
    row_ind, col_ind, cost = _matched_assignment(true_cells, det_cells)
    matched_cost = float(cost[row_ind, col_ind].sum())
    n_matched = len(row_ind)
    penalty = (n_true - n_matched) * grid_diag
    return (matched_cost + penalty) / n_true


def p_sep(I2d, true_cells, rel_thresh=0.10, tau_r=TAU_R):
    """True iff every true source has a matched detected peak within
    tau_r grid cells AND the matched peaks are pairwise distinct grid
    cells (structural for a one-to-one assignment onto distinct detections).
    """
    n_true = len(true_cells)
    det_cells = [(p[0], p[1]) for p in find_peaks_2d(I2d, rel_thresh)]
    if len(det_cells) < n_true:
        return False
    row_ind, col_ind, cost = _matched_assignment(true_cells, det_cells)
    if len(row_ind) < n_true:
        return False
    if not np.all(cost[row_ind, col_ind] <= tau_r):
        return False
    matched_det = [det_cells[c] for c in col_ind]
    return len(set(matched_det)) == n_true
