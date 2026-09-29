"""Filtration invariants and a frozen scan of a small catalog."""

import numpy as np

from spad.diagnostics import apply_plateau, max_patches
from spad.distance import pairwise_distance, positive_distances
from spad.filtration import (
    condensed_tree,
    filtration_levels,
    scan_q,
    select_eom,
    select_rule,
    solve_q,
)


def _synthetic():
    """Two pairs of tight groups, with a few bridge events.

    Coordinates come from ``numpy.random.default_rng(2)`` and the
    construction below. The expected scan is frozen against that catalog.
    """
    rng = np.random.default_rng(2)

    def blob(lat0, lon0, depth0, n_points=10, scale=0.008):
        return (
            lat0 + rng.normal(0, scale, n_points),
            lon0 + rng.normal(0, scale, n_points),
            depth0 + rng.normal(0, 0.4, n_points),
        )

    parts = [
        blob(50.0, 150.0, 20),
        blob(50.08, 150.08, 20),
        blob(52.0, 153.0, 40),
        blob(52.08, 153.08, 40),
        (
            np.array([50.04, 50.04, 50.045]),
            np.array([150.04, 150.05, 150.035]),
            np.array([20.0, 21.0, 19.5]),
        ),
        (
            np.array([52.04, 52.03]),
            np.array([153.04, 153.05]),
            np.array([40.0, 41.0]),
        ),
    ]
    lat = np.concatenate([part[0] for part in parts])
    lon = np.concatenate([part[1] for part in parts])
    depth = np.concatenate([part[2] for part in parts])
    return lat, lon, depth


def _matrix():
    lat, lon, depth = _synthetic()
    assert abs(lat[0] - 50.00151242705435) < 1e-12
    distance = pairwise_distance(
        np.ascontiguousarray(lat),
        np.ascontiguousarray(lon),
        np.ascontiguousarray(depth),
    )
    return distance, positive_distances(distance)


# q, r, n_old, birth events, n_rule, core sizes
_EXPECTED = (
    (-1.2, 3.7968109256787153, 4, 43, 0, ()),
    (-1.0, 4.757612994939465, 4, 40, 4, (10, 10, 10, 10)),
    (-0.8, 6.248215096909291, 4, 40, 4, (10, 10, 10, 10)),
    (-0.6, 8.764498238830239, 2, 44, 0, ()),
)


def test_qscan_regression():
    distance, positive = _matrix()
    grid = [row[0] for row in _EXPECTED]
    solutions = scan_q(distance, positive, grid)
    assert len(solutions) == len(_EXPECTED)
    for sol, expected in zip(solutions, _EXPECTED):
        q, radius, n_old, n_events, n_rule, cores = expected
        assert sol["q"] == q
        assert sol["r"] == radius
        assert sol["n_old"] == n_old
        assert sol["ev_old"] == n_events
        assert sol["n"] == n_rule
        assert tuple(item["n_core"] for item in sol["info"]) == cores
        assert sol["nested_ok"] is True
    plateau = apply_plateau(solutions)
    assert plateau["status"] == "OK"
    assert plateau["q"] == -0.8
    assert plateau["L"] == 2
    assert plateau["n"] == 4
    # On this grid the step-0.2 helper sees q=-1.2 and q=-1.0.
    # Both have 4 birth-set patches; -1.2 has more events.
    assert max_patches(solutions)["q"] == -1.2


def test_tree_invariants_and_fixed_radius():
    distance, positive = _matrix()
    filtration = filtration_levels(distance, positive, -1.0)
    alpha = filtration["alpha"]
    assert np.all(np.diff(alpha) >= -1e-15)
    assert len(filtration["masks"]) == 60
    nodes, labels, _ncomp = condensed_tree(filtration, 5)
    for node in nodes:
        assert node["death"] > node["birth"]
        assert node["pers"] == node["a_death"] - node["a_birth"]
        assert len(node["lv"]) == node["death"] - node["birth"]
        assert len(node["sizes"]) == len(node["lv"])
        assert node["size"] >= 5
    eom = select_eom(nodes, "pers")
    covered = np.zeros(filtration["n"], dtype=int)
    for node_id in eom:
        covered[nodes[node_id]["pts_birth"]] += 1
    assert covered.max() <= 1
    selected = select_rule(nodes, 3)
    sol = solve_q(distance, positive, -1.0)
    assert sol["n"] == len(selected)
    assert sol["n_old"] == len(eom)
    # Cores of one solution are disjoint by construction of the labels.
    assert len(np.unique(sol["core"][sol["core"] >= 0])) == sol["n"]
    fixed = solve_q(distance, positive, -1.0, r_fixed=12.0)
    assert fixed["r"] == 12.0
    assert fixed["r"] != sol["r"]
    # The level labels used for a core exist on that node.
    for item in sol["info"]:
        node = nodes[item["node"]]
        assert labels[item["kmed"]][sol["core"] == item["patch"]].min() >= 0
        assert item["kmed"] == node["birth"] + (node["death"] - 1 - node["birth"]) // 2
