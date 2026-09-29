"""Diagnostics for a scan over ``q``.

The scan reports, at every ``q``, the birth-set excess-of-mass patches
and the plateau-rule cores. Null catalogs are compared at the same ``q``
(each catalog uses its own radius) and at the real catalog's radius.
Bootstrap replicas are compared at the same ``q`` by the best Jaccard
match of each real core, restricted to the subsample, against the
replica cores.

The max-patches helper and the plateau helper are summaries of that
table. They are not a rule for the best ``q``. The max-patches helper
is defined on ``q`` from -4 to -1 in steps of 0.2. The plateau helper
is defined on ``q`` from -4 to -1 in steps of 0.1.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

Q_PLATEAU = np.round(np.arange(-4.0, -0.95, 0.1), 1)
Q_MAX_PATCHES = np.round(np.arange(-4.0, -0.99, 0.2), 1)
FILT_METRICS = (
    "filt_n",
    "filt_core_frac",
    "filt_n_old",
    "filt_birthold_frac",
)


def q_grid(q_min=-4.0, q_max=-1.0, step=0.1):
    """Rounded ``q`` grid.

    The default arguments reproduce the plateau grid. Step 0.2 with the
    same endpoints reproduces the max-patches grid. Values are rounded
    to one decimal, as in those grids.
    """
    stop = float(q_max) + float(step) / 2.0
    return np.round(np.arange(float(q_min), stop, float(step)), 1)


def same_grid(values, reference) -> bool:
    """True when ``values`` is the same sequence as ``reference``."""
    values = np.asarray(list(values), dtype=float)
    reference = np.asarray(reference, dtype=float)
    return values.shape == reference.shape and np.array_equal(values, reference)


def label_sizes(labels) -> np.ndarray:
    """Sizes of labels ``>= 0``, in label order."""
    labels = np.asarray(labels)
    present = labels >= 0
    if not present.any():
        return np.array([], dtype=int)
    return np.bincount(labels[present])


def best_jaccard(labels_a, labels_b) -> np.ndarray:
    """For each patch in ``labels_a``, the best Jaccard overlap with ``labels_b``.

    Jaccard is ``|A ∩ B| / |A ∪ B|`` on event indices. An empty ``labels_a``
    returns an empty array. If ``labels_b`` has no patch, every score is 0.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    has_a = (labels_a >= 0).any()
    n_a = int(labels_a.max()) + 1 if has_a else 0
    n_b = int(labels_b.max()) + 1 if (labels_b >= 0).any() else 0
    if n_a == 0:
        return np.array([])
    if n_b == 0:
        return np.zeros(n_a)
    ok_a = labels_a >= 0
    ok_b = labels_b >= 0
    size_a = np.bincount(labels_a[ok_a], minlength=n_a)
    size_b = np.bincount(labels_b[ok_b], minlength=n_b)
    both = ok_a & ok_b
    overlap = np.zeros((n_a, n_b))
    np.add.at(overlap, (labels_a[both], labels_b[both]), 1)
    union = size_a[:, None] + size_b[None, :] - overlap
    return (overlap / union).max(axis=1)


def apply_plateau(solutions, threshold=0.8):
    """Plateau helper on solutions ordered by increasing ``q``.

    Neighbouring solutions match when they have the same number of cores
    and every core has a best Jaccard match of at least ``threshold``.
    Two empty solutions match, and still cannot form a valid plateau,
    because a valid plateau has at least two grid points and at least
    two patches.

    The chosen plateau is the longest valid one. A tie goes to the
    plateau whose first ``q`` is larger. The reported ``q`` is the middle
    grid point; for an even length it is the upper of the two middle
    points. A plateau that touches either end of the scanned grid is
    flagged ``edge``. The flag does not change the reported ``q``.
    No valid plateau yields status ``NONE``.
    """
    q_values = [sol["q"] for sol in solutions]
    counts = [sol["n"] for sol in solutions]
    same = []
    min_jaccard = []
    for left, right in itertools.pairwise(solutions):
        scores = best_jaccard(left["core"], right["core"])
        worst = float(scores.min()) if len(scores) else np.nan
        min_jaccard.append(worst)
        same.append(
            left["n"] == right["n"] and (left["n"] == 0 or worst >= threshold)
        )
    plateaus = []
    index = 0
    while index < len(solutions) - 1:
        if same[index]:
            end = index
            while end < len(same) and same[end]:
                end += 1
            plateaus.append((index, end))
            index = end
        else:
            index += 1
    valid = [
        (start, end)
        for start, end in plateaus
        if end - start + 1 >= 2 and counts[start] >= 2
    ]
    result = {
        "qs": q_values,
        "ns": counts,
        "same": same,
        "minJ": min_jaccard,
        "plateaus": [
            (q_values[start], q_values[end], counts[start])
            for start, end in plateaus
        ],
    }
    if not valid:
        result.update(
            status="NONE",
            q=np.nan,
            n=0,
            L=0,
            i0=None,
            i1=None,
            edge=False,
            sol_index=None,
        )
        return result
    start, end = max(valid, key=lambda span: (span[1] - span[0], q_values[span[0]]))
    length = end - start + 1
    if length % 2 == 0:
        middle = start + length // 2
    else:
        middle = start + (length - 1) // 2
    result.update(
        status="OK",
        q=q_values[middle],
        n=counts[middle],
        L=length,
        i0=start,
        i1=end,
        sol_index=middle,
        edge=(start == 0 or end == len(solutions) - 1),
        q_lo=q_values[start],
        q_hi=q_values[end],
    )
    return result


def max_patches(solutions):
    """Max-patches helper.

    Among solutions whose ``q``, rounded to one decimal, lies on the
    step-0.2 grid from -4 to -1, maximise the number of birth-set
    patches, then the number of events in those patches, then minimise
    the peak-similarity tie-break. Returns the chosen solution, or
    ``None`` when the scan contains none of that grid.
    """
    allowed = set(Q_MAX_PATCHES.tolist())
    candidates = [sol for sol in solutions if round(sol["q"], 1) in allowed]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda sol: (sol["n_old"], sol["ev_old"], -sol["peak_diff"]),
    )


def qscan_frame(solutions) -> pd.DataFrame:
    """One row per ``q``: birth-set score and plateau-rule cores."""
    plateau = apply_plateau(solutions)
    next_j = list(plateau["minJ"]) + [np.nan]
    rows = []
    for sol, jaccard in zip(solutions, next_j):
        core_events = int((sol["core"] >= 0).sum())
        rows.append(
            {
                "q": sol["q"],
                "r": sol["r"],
                "n_old": sol["n_old"],
                "birth_events_old": sol["ev_old"],
                "peak_diff": sol["peak_diff"],
                "n_rule": sol["n"],
                "core_events": core_events,
                "core_frac": float((sol["core"] >= 0).mean()),
                "minJ_next": jaccard,
                "nested_ok": sol.get("nested_ok", True),
            }
        )
    return pd.DataFrame(rows)


def filtration_metrics(solution, n_events) -> dict:
    """Four scores used in the exceedance table."""
    return {
        "filt_n": solution["n"],
        "filt_core_frac": float((solution["core"] >= 0).sum() / n_events),
        "filt_n_old": solution["n_old"],
        "filt_birthold_frac": solution["ev_old"] / n_events,
    }


def empirical_p(null_values, real_value):
    """Upper and lower empirical tail probabilities.

    ``p_upper = (1 + #{null >= real}) / (n + 1)`` and the same with
    ``<=`` for ``p_lower``.
    """
    null_values = np.asarray(null_values, dtype=float)
    n_null = len(null_values)
    p_upper = (1 + np.sum(null_values >= real_value)) / (n_null + 1)
    p_lower = (1 + np.sum(null_values <= real_value)) / (n_null + 1)
    return float(p_upper), float(p_lower)


def exceedance_z(null_values, real_value) -> float:
    """``(real - mean) / sample_sd``.

    The standard deviation uses ``ddof=1``. When it is zero the score is
    zero if the real value equals the null mean, and signed infinity
    otherwise.
    """
    null_values = np.asarray(null_values, dtype=float)
    sd = float(null_values.std(ddof=1))
    mean = float(null_values.mean())
    if sd > 0:
        return (real_value - mean) / sd
    if real_value != mean:
        return float(np.inf * np.sign(real_value - mean))
    return 0.0


def exceedance_table(runs, metrics=FILT_METRICS) -> pd.DataFrame:
    """Exceedance of the real catalog over each null model.

    ``runs`` is a list of dicts with ``kind``, ``variant`` (``sameq`` or
    ``fixedr``), ``q``, ``r``, ``n`` (catalog size) and the metric
    fields. The real catalog is ``kind="real"`` and ``variant="sameq"``.
    """
    frame = pd.DataFrame(runs)
    real = frame[(frame["kind"] == "real") & (frame["variant"] == "sameq")]
    rows = []
    null_kinds = [kind for kind in frame["kind"].unique() if kind != "real"]
    for kind in null_kinds:
        for variant in ("sameq", "fixedr"):
            subset = frame[(frame["kind"] == kind) & (frame["variant"] == variant)]
            if subset.empty:
                continue
            for q_value, group in subset.groupby("q"):
                real_row = real[real["q"] == q_value]
                if real_row.empty:
                    continue
                for metric in metrics:
                    null_values = group[metric].astype(float).to_numpy()
                    real_value = float(real_row[metric].iloc[0])
                    p_upper, p_lower = empirical_p(null_values, real_value)
                    rows.append(
                        {
                            "metric": metric,
                            "model": kind,
                            "variant": variant,
                            "q": q_value,
                            "r_real": float(real_row["r"].iloc[0]),
                            "r_null_median": float(group["r"].median()),
                            "n_null": len(null_values),
                            "real": real_value,
                            "null_median": float(np.median(null_values)),
                            "null_lo95": float(np.quantile(null_values, 0.025)),
                            "null_hi95": float(np.quantile(null_values, 0.975)),
                            "null_mean": float(null_values.mean()),
                            "null_sd": float(null_values.std(ddof=1)),
                            "z": exceedance_z(null_values, real_value),
                            "p_upper": p_upper,
                            "p_lower": p_lower,
                        }
                    )
    return pd.DataFrame(rows)


def metric_rows_from_run(run) -> list:
    """Flatten one stored ensemble run into exceedance rows."""
    rows = []
    n_events = run["n"]
    for sol in run["solutions"]:
        rows.append(
            {
                "kind": run["kind"],
                "i": run["i"],
                "variant": "sameq",
                "q": sol["q"],
                "r": sol["r"],
                "n": n_events,
                **filtration_metrics(sol, n_events),
            }
        )
    for sol in run.get("solutions_fixed_r") or []:
        rows.append(
            {
                "kind": run["kind"],
                "i": run["i"],
                "variant": "fixedr",
                "q": sol["q"],
                "r": sol["r"],
                "n": n_events,
                **filtration_metrics(sol, n_events),
            }
        )
    return rows


def bootstrap_reproducibility(real_solutions, boot_runs) -> pd.DataFrame:
    """Per-``q`` bootstrap match of real cores.

    For replica ``b`` with index vector ``idx``, the real core labels
    restricted by ``idx`` are compared with the replica core labels.
    ``wmean_freq05`` is the size-weighted mean, over real patches, of the
    fraction of replicas with Jaccard at least 0.5. ``frac_ge08`` is the
    fraction of real patches whose Jaccard-at-least-0.5 frequency is
    itself at least 0.8. A ``q`` with no real core is reported with
    ``n_real=0`` and no frequencies.
    """
    rows = []
    for index, sol in enumerate(real_solutions):
        if sol["n"] == 0:
            rows.append({"q": sol["q"], "n_real": 0})
            continue
        scores = np.array(
            [
                best_jaccard(
                    sol["core"][boot["idx"]], boot["solutions"][index]["core"]
                )
                for boot in boot_runs
            ]
        )
        weights = label_sizes(sol["core"])
        freq05 = (scores >= 0.5).mean(axis=0)
        freq08 = (scores >= 0.8).mean(axis=0)
        rows.append(
            {
                "q": sol["q"],
                "n_real": sol["n"],
                "mean_freq05": float(np.mean(scores >= 0.5)),
                "wmean_freq05": float(np.average(freq05, weights=weights)),
                "mean_freq08": float(np.mean(scores >= 0.8)),
                "wmean_freq08": float(np.average(freq08, weights=weights)),
                "frac_ge08": float(np.mean(freq05 >= 0.8)),
                "n_boot_median": float(
                    np.median([boot["solutions"][index]["n"] for boot in boot_runs])
                ),
            }
        )
    return pd.DataFrame(rows)


def patch_summary(lat, lon, depth, labels) -> pd.DataFrame:
    """Centroid and extent of each patch label ``>= 0``."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    depth = np.asarray(depth, dtype=float)
    labels = np.asarray(labels)
    rows = []
    if not (labels >= 0).any():
        return pd.DataFrame(rows)
    for label in range(int(labels.max()) + 1):
        mask = labels == label
        rows.append(
            {
                "id": label + 1,
                "n": int(mask.sum()),
                "lon": float(lon[mask].mean()),
                "lat": float(lat[mask].mean()),
                "depth": float(depth[mask].mean()),
                "depth_min": float(depth[mask].min()),
                "depth_max": float(depth[mask].max()),
                "lon_range": float(np.ptp(lon[mask])),
                "lat_range": float(np.ptp(lat[mask])),
            }
        )
    return pd.DataFrame(rows)
