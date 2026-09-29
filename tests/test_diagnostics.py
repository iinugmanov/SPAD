"""Plateau helper, Jaccard scores and the exceedance formulas."""

import numpy as np
import pandas as pd

from spad.diagnostics import (
    Q_MAX_PATCHES,
    Q_PLATEAU,
    apply_plateau,
    best_jaccard,
    empirical_p,
    exceedance_table,
    exceedance_z,
    max_patches,
    q_grid,
)


def _solution(q, labels):
    labels = np.asarray(labels, dtype=np.int16)
    n_patches = int(labels.max()) + 1 if (labels >= 0).any() else 0
    return {
        "q": float(q),
        "r": 1.0,
        "n_old": n_patches,
        "ev_old": int((labels >= 0).sum()),
        "peak_diff": 0.0,
        "n": n_patches,
        "core": labels,
    }


def test_grids_match_the_study():
    assert len(q_grid()) == 31
    assert np.array_equal(q_grid(), Q_PLATEAU)
    assert np.array_equal(q_grid(-4.0, -1.0, 0.2), Q_MAX_PATCHES)
    assert Q_PLATEAU[0] == -4.0 and Q_PLATEAU[-1] == -1.0
    assert Q_MAX_PATCHES[0] == -4.0 and Q_MAX_PATCHES[-1] == -1.0


def test_jaccard_identical_and_disjoint():
    labels = np.array([0, 0, 1, 1, -1])
    scores = best_jaccard(labels, labels)
    assert np.allclose(scores, [1.0, 1.0])
    other = np.array([1, 1, 0, 0, -1])
    swapped = best_jaccard(labels, other)
    assert np.allclose(swapped, [1.0, 1.0])
    disjoint = np.array([-1, -1, -1, -1, 0])
    # Patch 0 of `labels` does not meet the only patch of `disjoint`.
    assert best_jaccard(labels, disjoint)[0] == 0.0
    assert best_jaccard(np.array([-1, -1]), labels).size == 0


def test_plateau_picks_upper_middle_of_the_longest_run():
    # Six identical solutions: one plateau, even length, upper middle.
    labels = np.array([0, 0, 0, 1, 1, -1])
    q_values = [-1.6, -1.5, -1.4, -1.3, -1.2, -1.1]
    solutions = [_solution(q, labels) for q in q_values]
    result = apply_plateau(solutions)
    assert result["status"] == "OK"
    assert result["L"] == 6
    assert result["q"] == -1.3
    assert result["n"] == 2
    assert result["edge"] is True
    # Two plateaus of equal length. The tie goes to the larger q.
    # Even length then takes the upper of the two middle points.
    short = [_solution(q, labels) for q in (-2.0, -1.9)]
    gap = [_solution(-1.8, np.array([0, 0, 0, 0, 0, 0]))]
    again = [_solution(q, labels) for q in (-1.7, -1.6)]
    mixed = apply_plateau(short + gap + again)
    assert mixed["status"] == "OK"
    assert mixed["L"] == 2
    assert mixed["q_lo"] == -1.7
    assert mixed["q"] == -1.6


def test_empty_solutions_do_not_form_a_plateau():
    empty = np.array([-1, -1, -1])
    result = apply_plateau([_solution(q, empty) for q in (-2.0, -1.9, -1.8)])
    assert result["status"] == "NONE"
    assert result["n"] == 0


def test_max_patches_tie_break():
    base = _solution(-3.8, np.array([0, 0, 1, 1, -1]))
    richer = dict(base)
    richer["q"] = -3.6
    richer["ev_old"] = base["ev_old"] + 1
    poorer = dict(base)
    poorer["q"] = -4.0
    poorer["n_old"] = base["n_old"] - 1
    # -3.5 is not on the step-0.2 grid and must be ignored.
    off = dict(base)
    off["q"] = -3.5
    off["n_old"] = 100
    chosen = max_patches([poorer, base, richer, off])
    assert chosen["q"] == -3.6


def test_exceedance_z_and_p():
    null = np.array([0.0, 0.0, 1.0])
    real = 1.0
    mean = float(null.mean())
    sd = float(null.std(ddof=1))
    assert exceedance_z(null, real) == (real - mean) / sd
    p_upper, p_lower = empirical_p(null, real)
    assert p_upper == (1 + 1) / 4
    assert p_lower == (1 + 3) / 4
    assert exceedance_z(np.array([2.0, 2.0]), 2.0) == 0.0
    assert exceedance_z(np.array([2.0, 2.0]), 3.0) == np.inf
    rows = []
    samples = ((1.0, "real", 0), (0.0, "shift50", 0), (1.0, "shift50", 1))
    for value, kind, index in samples:
        rows.append(
            {
                "kind": kind,
                "i": index,
                "variant": "sameq",
                "q": -2.0,
                "r": 10.0,
                "n": 10,
                "filt_n": value,
                "filt_core_frac": value,
                "filt_n_old": value,
                "filt_birthold_frac": value,
            }
        )
    table = exceedance_table(pd.DataFrame(rows), metrics=("filt_n",))
    assert len(table) == 1
    assert table.iloc[0]["n_null"] == 2
    assert table.iloc[0]["p_upper"] == empirical_p([0.0, 1.0], 1.0)[0]
