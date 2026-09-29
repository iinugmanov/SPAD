"""Alpha-filtration of the canonical DPS density.

For a negative exponent ``q`` the radius ``r(q)`` is the power mean of
the positive pairwise distances. The density at an event is the sum of
kernel weights ``1 - d/r`` over events inside the radius, including the
event itself (weight 1), divided by the maximum of that sum.

An alpha-perfect set is obtained by iterative trimming: a point stays
only while its density, recomputed on the surviving set and still
normalised by the full-catalog maximum, is at least alpha. Sixty uniform
alpha levels run from the minimum density up to the largest alpha with
a non-empty perfect set.

Components are the connected components of the radius graph (a single
linkage cut at ``r``). Components smaller than ``m_min`` (default 5)
are dropped. The condensed tree follows a component while it has exactly
one large child, and ends it when it splits into two or more large
children or when it disappears. Persistence is the alpha interval of
that life.

Two readings of the same tree are provided. Excess of mass on
persistence, including components born at the first level, with patches
equal to the component at birth, is the max-patches score. The plateau
helper restricts that selection to nodes that live at least three levels
and are not born at the first level, and takes the component at the
median level of its life as the core. Neither reading is an automatic
choice of ``q``. ``q`` is an input.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.signal import find_peaks
from scipy.sparse.csgraph import connected_components

from spad.distance import power_mean_radius

N_LEVELS = 60
M_MIN = 5
MIN_STEPS_PER_60 = 3


def weight_matrix(distance, radius):
    """Kernel weights and the off-diagonal radius graph.

    ``W_xy = 1 - d/r`` when ``d <= r``, including the diagonal (weight 1).
    The adjacency matrix has those pairs with the diagonal removed.
    Both are CSR.
    """
    rows, cols = np.nonzero(distance <= radius)
    weights = 1.0 - distance[rows, cols] / radius
    n_events = distance.shape[0]
    kernel = sparse.csr_matrix(
        (weights, (rows, cols)), shape=(n_events, n_events)
    )
    off = rows != cols
    adjacency = sparse.csr_matrix(
        (np.ones(int(off.sum()), dtype=np.int8), (rows[off], cols[off])),
        shape=(n_events, n_events),
    )
    return kernel, adjacency


def alpha_perfect_set(kernel, norm, alpha, start=None):
    """Canonical alpha-perfect set.

    Start from ``start`` (default: every event). Repeatedly keep events
    whose density on the current set is at least ``alpha``. Density is
    ``(kernel @ membership) / norm``, and ``norm`` is the maximum row
    sum of the full catalog, not of the current set.
    """
    n_events = kernel.shape[0]
    if start is None:
        member = np.ones(n_events, dtype=bool)
    else:
        member = start.copy()
    while True:
        density = kernel @ member.astype(float) / norm
        keep = member & (density >= alpha)
        if keep.sum() == member.sum():
            return member
        member = keep


def filtration_levels(distance, positive, q, n_levels=N_LEVELS, r_override=None):
    """Density, alpha grid and alpha-perfect sets for one ``q``.

    ``r_override``, when given, replaces ``r(q)``. The alpha grid and the
    perfect sets are then built at that fixed radius. This is the
    fixed-radius null comparison: null catalogs are scored at the real
    catalog's radius instead of at their own ``r(q)``.

    Alpha levels used as sets are ``alpha[0]`` ... ``alpha[n_levels - 1]``.
    ``alpha[n_levels]`` is the terminal value (the supremum) and is not
    itself a set in the loop.
    """
    if r_override is None:
        radius = power_mean_radius(positive, q)
    else:
        radius = r_override
    kernel, adjacency = weight_matrix(distance, radius)
    row_sum = np.asarray(kernel.sum(axis=1)).ravel()
    norm = row_sum.max()
    density = row_sum / norm
    n_events = len(density)
    lo = density.min()
    hi = 1.0
    start = alpha_perfect_set(kernel, norm, lo)
    if alpha_perfect_set(kernel, norm, hi, start).any():
        lo = hi
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if alpha_perfect_set(kernel, norm, mid, start).any():
                lo = mid
            else:
                hi = mid
    alpha_sup = lo
    alpha = np.linspace(density.min(), alpha_sup, n_levels + 1)
    masks = []
    raw_labels = []
    nested = True
    previous = None
    for level in range(n_levels):
        member = alpha_perfect_set(kernel, norm, alpha[level])
        if previous is not None and (member & ~previous).any():
            nested = False
        previous = member
        index = np.nonzero(member)[0]
        labels = np.full(n_events, -1)
        if len(index):
            _, comp = connected_components(
                adjacency[index][:, index], directed=False
            )
            labels[index] = comp
        masks.append(member)
        raw_labels.append(labels)
    return {
        "q": q,
        "r": radius,
        "C": norm,
        "P": density,
        "alpha": alpha,
        "a_sup": alpha_sup,
        "masks": masks,
        "raw_lab": raw_labels,
        "nested_ok": nested,
        "n": n_events,
    }


def _large_components(raw_labels, m_min):
    """Relabel components, dropping those smaller than ``m_min``."""
    labels = []
    for raw in raw_labels:
        relabeled = np.full(raw.shape[0], -1)
        present = raw >= 0
        if present.any():
            sizes = np.bincount(raw[present])
            keep = sizes >= m_min
            remap = -np.ones(len(sizes), dtype=int)
            remap[keep] = np.arange(int(keep.sum()))
            relabeled[present] = remap[raw[present]]
        labels.append(relabeled)
    return labels


def condensed_tree(filtration, m_min=M_MIN):
    """Condensed tree of large components over alpha.

    A node continues while its points carry exactly one component label
    of size at least ``m_min``. It dies when that split produces two or
    more such labels (those children are born) or when none remain.
    Nodes still alive at the last level die at the terminal alpha.

    Each node stores the component label at every level of its life
    (``lv``), the points at birth (``pts_birth``) and the size at every
    level (``sizes``).
    """
    alpha = filtration["alpha"]
    n_levels = len(filtration["masks"])
    labels = _large_components(filtration["raw_lab"], m_min)
    nodes = []

    def new_node(parent, level, comp_label, points):
        node_id = len(nodes)
        nodes.append(
            {
                "id": node_id,
                "parent": parent,
                "children": [],
                "birth": level,
                "death": None,
                "lv": [comp_label],
                "sizes": [len(points)],
                "pts_birth": points,
                "pts_last": points,
            }
        )
        if parent is not None:
            nodes[parent]["children"].append(node_id)
        return node_id

    alive = {}
    first = labels[0]
    for comp_label in range(int(first.max()) + 1):
        points = np.nonzero(first == comp_label)[0]
        alive[comp_label] = new_node(None, 0, comp_label, points)
    for level in range(1, n_levels):
        current = labels[level]
        nxt = {}
        for comp_label, node_id in alive.items():
            seen = np.unique(current[nodes[node_id]["pts_last"]])
            seen = seen[seen >= 0]
            if len(seen) == 1:
                child = int(seen[0])
                nxt[child] = node_id
                points = np.nonzero(current == child)[0]
                nodes[node_id]["lv"].append(child)
                nodes[node_id]["sizes"].append(len(points))
                nodes[node_id]["pts_last"] = points
            else:
                nodes[node_id]["death"] = level
                if len(seen) >= 2:
                    for child in seen:
                        child = int(child)
                        points = np.nonzero(current == child)[0]
                        nxt[child] = new_node(node_id, level, child, points)
        alive = nxt
    for node_id in alive.values():
        nodes[node_id]["death"] = n_levels
    for node in nodes:
        node["a_birth"] = alpha[node["birth"]]
        node["a_death"] = alpha[node["death"]]
        node["pers"] = node["a_death"] - node["a_birth"]
        d_alpha = np.diff(alpha)[node["birth"]:node["death"]]
        node["mass"] = float(np.sum(np.asarray(node["sizes"], dtype=float) * d_alpha))
        node["size"] = len(node["pts_birth"])
    n_components = np.array([int(level.max()) + 1 for level in labels])
    return nodes, labels, n_components


def _extend(parts):
    """Concatenate lists, preserving order."""
    merged = []
    for part in parts:
        merged.extend(part)
    return merged


def select_eom(nodes, key="pers"):
    """Non-overlapping nodes maximising the sum of ``key``.

    Ties keep the parent: children replace the parent only when their
    total is strictly greater. The returned ids are sorted.
    """
    best = {}
    chosen = {}
    for node in reversed(nodes):
        children = node["children"]
        if not children:
            best[node["id"]] = node[key]
            chosen[node["id"]] = [node["id"]]
        else:
            total = sum(best[child] for child in children)
            if total > node[key]:
                best[node["id"]] = total
                chosen[node["id"]] = _extend(chosen[child] for child in children)
            else:
                best[node["id"]] = node[key]
                chosen[node["id"]] = [node["id"]]
    roots = [node["id"] for node in nodes if node["parent"] is None]
    return sorted(_extend(chosen[node_id] for node_id in roots))


def select_rule(nodes, min_steps):
    """Plateau-rule selection on one tree.

    A node is eligible when it lives at least ``min_steps`` levels and
    was not born at level 0. Eligible nodes compete by persistence.
    Ties keep the parent. A node that is not eligible does not compete;
    its eligible descendants still can. Returned ids are sorted.
    """
    best = {}
    chosen = {}
    for node in reversed(nodes):
        total = sum(best[child] for child in node["children"])
        child_sel = _extend(chosen[child] for child in node["children"])
        eligible = (node["death"] - node["birth"] >= min_steps) and (
            node["birth"] > 0
        )
        if eligible and node["pers"] >= total:
            best[node["id"]] = node["pers"]
            chosen[node["id"]] = [node["id"]]
        else:
            best[node["id"]] = total
            chosen[node["id"]] = child_sel
    roots = [node["id"] for node in nodes if node["parent"] is None]
    return sorted(_extend(chosen[node_id] for node_id in roots))


def min_life_steps(n_levels):
    """Life threshold: 3 levels on a 60-level grid, scaled for other grids."""
    return MIN_STEPS_PER_60 * int(n_levels) // 60


def fd_bins(data):
    """Freedman-Diaconis bin count. Returns 10 when the width is zero."""
    q1, q3 = np.percentile(data, [25, 75])
    width = 2 * (q3 - q1) / (len(data) ** (1 / 3))
    if width == 0:
        return 10
    return int(np.ceil((max(data) - min(data)) / width))


def peak_similarity(all_density, subset_density):
    """Mean bin-distance between histogram peaks.

    Peaks of the density histogram of all events are compared with peaks
    of the histogram of a subset. The histograms share bins built on the
    concatenated sample. The value is +infinity when either histogram has
    no peak. This is a tie-break for the max-patches helper, not a
    density property of DPS.
    """
    combined = np.concatenate((all_density, subset_density))
    bins = np.histogram_bin_edges(combined, bins=fd_bins(combined))
    hist_all, _ = np.histogram(all_density, bins=bins)
    hist_sub, _ = np.histogram(subset_density, bins=bins)
    peaks_all, _ = find_peaks(hist_all)
    peaks_sub, _ = find_peaks(hist_sub)
    if len(peaks_all) > 0 and len(peaks_sub) > 0:
        gaps = [np.min(np.abs(peaks_sub - peak)) for peak in peaks_all]
        return float(np.mean(gaps))
    return float("inf")


def solve_q(distance, positive, q, r_fixed=None, n_levels=N_LEVELS, m_min=M_MIN):
    """Both readings of the tree at one ``q``.

    ``n_old`` / ``ev_old`` / ``lab_old`` are the excess-of-mass birth
    sets (the max-patches score). ``core`` is the plateau-rule core
    labeling at the median level of each selected node. ``birth`` is the
    birth set of those same plateau-rule nodes. Labels are ``-1`` outside
    a patch and ``0 .. n-1`` inside, in order of sorted node id.
    """
    filtration = filtration_levels(
        distance, positive, q, n_levels=n_levels, r_override=r_fixed
    )
    nodes, labels, _ncomp = condensed_tree(filtration, m_min)
    n_events = filtration["n"]
    eom = select_eom(nodes, "pers")
    lab_old = np.full(n_events, -1, dtype=np.int16)
    for patch, node_id in enumerate(eom):
        node = nodes[node_id]
        lab_old[labels[node["birth"]] == node["lv"][0]] = patch
    n_birth_events = int((lab_old >= 0).sum())
    if n_birth_events >= 2:
        peak = peak_similarity(filtration["P"], filtration["P"][lab_old >= 0])
    else:
        peak = float("inf")
    selected = select_rule(nodes, min_life_steps(n_levels))
    core = np.full(n_events, -1, dtype=np.int16)
    birth = np.full(n_events, -1, dtype=np.int16)
    info = []
    for patch, node_id in enumerate(selected):
        node = nodes[node_id]
        born = node["birth"]
        died = node["death"]
        k_med = born + (died - 1 - born) // 2
        core_mask = labels[k_med] == node["lv"][k_med - born]
        birth_mask = labels[born] == node["lv"][0]
        core[core_mask] = patch
        birth[birth_mask] = patch
        info.append(
            {
                "patch": patch,
                "node": node_id,
                "birth": born,
                "death": died,
                "kmed": k_med,
                "pers": float(node["pers"]),
                "n_core": int(core_mask.sum()),
                "n_birth": int(birth_mask.sum()),
            }
        )
    return {
        "q": float(q),
        "r": float(filtration["r"]),
        "n_old": len(eom),
        "ev_old": n_birth_events,
        "peak_diff": float(peak),
        "lab_old": lab_old,
        "n": len(selected),
        "core": core,
        "birth": birth,
        "info": info,
        "n_nodes": len(nodes),
        "nested_ok": bool(filtration["nested_ok"]),
    }


def scan_q(distance, positive, q_values, r_list=None, n_levels=N_LEVELS, m_min=M_MIN):
    """``solve_q`` at each value in ``q_values``, in that order."""
    solutions = []
    for index, q in enumerate(q_values):
        r_fixed = None if r_list is None else r_list[index]
        solutions.append(
            solve_q(
                distance,
                positive,
                q,
                r_fixed=r_fixed,
                n_levels=n_levels,
                m_min=m_min,
            )
        )
    return solutions
