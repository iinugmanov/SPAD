"""Inter-event attributes and mainshock, foreshock, aftershock labels.

Reads the tree catalog from :mod:`spad.form_event_trees` and keeps rows
with ``group_id != 0``. Each remaining group that has a row with
``parent == 0`` is measured against that row:

* ``time_from_head_day`` and ``dist_from_head_km`` from the parent
* ``time_diff_seq`` and ``dist_diff_seq`` between events ordered by time
  (the first event in the group is zero)

``EQtype`` is ``mainshock`` for the largest ``mag`` (the earliest row
when several share that magnitude, and any row with the same time),
``foreshock`` before that time and ``aftershock`` after it.

Expected coordinate columns are ``longitude`` and ``latitude``, as
written by declustering.
"""

import argparse

import numpy as np
import pandas as pd
from numba import jit, prange


@jit(nopython=True)
def haversine_vectorized(lon1, lat1, lon2, lat2):
    """Great-circle distance between two points.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float
        Coordinates in degrees.

    Returns
    -------
    float
        Distance in kilometres.
    """
    earth_radius_km = 6371.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = earth_radius_km * c
    return distance


@jit(nopython=True, parallel=True)
def compute_space_diff(coords):
    """Symmetric matrix of horizontal distances.

    Not used by :func:`compute_group_differences`. Sequential distances
    are computed one pair at a time.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape ``(n, 2)`` with longitude and latitude in degrees.

    Returns
    -------
    numpy.ndarray
        Distances in kilometres. The diagonal is ``1e10``.
    """
    lon1, lat1 = coords[:, 0], coords[:, 1]
    num_events = len(coords)
    space_diff = np.zeros((num_events, num_events))

    for i in prange(num_events):
        for j in prange(i, num_events):
            if i == j:
                space_diff[i, j] = 1e10
            else:
                dist = haversine_vectorized(lon1[i], lat1[i], lon1[j], lat1[j])
                space_diff[i, j] = dist
                space_diff[j, i] = dist
    return space_diff


@jit(nopython=True, parallel=True)
def compute_time_diff(time_values):
    """Time gaps between successive timestamps.

    Parameters
    ----------
    time_values : numpy.ndarray
        Timestamps in seconds, length ``n``.

    Returns
    -------
    numpy.ndarray
        Length ``n - 1``. Entry ``i`` is ``t_{i+1} - t_i`` in days.
    """
    num_events = time_values.shape[0]
    time_diff = np.zeros(num_events - 1, dtype=np.float64)
    for i in prange(num_events - 1):
        time_diff[i] = (time_values[i + 1] - time_values[i]) / 86400.0
    return time_diff


def compute_group_differences(data):
    """Attributes of each group relative to its ``parent == 0`` event.

    Groups with no parent row are omitted.

    Parameters
    ----------
    data : pandas.DataFrame
        Rows with ``group_id``, ``parent``, ``time``, ``longitude``,
        ``latitude`` and ``mag``. ``time`` must be datetime-like.

    Returns
    -------
    pandas.DataFrame
        Concatenated groups with ``time_from_head_day``,
        ``dist_from_head_km``, ``time_diff_seq``, ``dist_diff_seq`` and
        ``EQtype``.
    """
    results = []
    for _group_id, group in data.groupby("group_id"):
        parent_event = group[group["parent"] == 0]
        if parent_event.empty:
            continue

        parent_event = parent_event.iloc[0]
        parent_time = parent_event["time"].timestamp()
        parent_lon, parent_lat = parent_event["longitude"], parent_event["latitude"]

        group = group.sort_values(by="time").reset_index(drop=True)
        group["time_from_head_day"] = (
            group["time"].apply(lambda x: x.timestamp()) - parent_time
        ) / 86400.0
        group["dist_from_head_km"] = group.apply(
            lambda row, lon=parent_lon, lat=parent_lat: haversine_vectorized(
                lon,
                lat,
                row["longitude"],
                row["latitude"],
            ),
            axis=1,
        )

        if len(group) > 1:
            time_array = group["time"].astype("int64") // 10**9
            time_diff_seq = compute_time_diff(time_array.values)
            group["time_diff_seq"] = np.insert(time_diff_seq, 0, 0.0)

            dist_diff_seq = [
                haversine_vectorized(
                    group.iloc[i - 1]["longitude"],
                    group.iloc[i - 1]["latitude"],
                    group.iloc[i]["longitude"],
                    group.iloc[i]["latitude"],
                )
                for i in range(1, len(group))
            ]
            group["dist_diff_seq"] = np.insert(dist_diff_seq, 0, 0.0)
        else:
            group["time_diff_seq"] = 0.0
            group["dist_diff_seq"] = 0.0

        mainshock_event = group.loc[group["mag"].idxmax()]
        group["EQtype"] = "aftershock"
        group.loc[group["time"] < mainshock_event["time"], "EQtype"] = "foreshock"
        group.loc[group["time"] == mainshock_event["time"], "EQtype"] = "mainshock"

        results.append(group)

    return pd.concat(results, ignore_index=True)


def build_parser():
    """Command-line interface for event-type classification.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the standalone script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Add inter-group time and distance attributes and label "
            "mainshock, foreshock and aftershock."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m spad.classify_event_types \\\n"
            "      --input results/catalog_with_tree.csv \\\n"
            "      --output results/catalog_with_types.csv\n"
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV from form_event_trees, with group_id, parent, time, "
            "longitude, latitude and mag."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="CSV path for the classified groups.",
    )
    return parser


def main(argv=None):
    """Classify events inside each tree from the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, excluding the program name. ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    data = pd.read_csv(args.input)
    if "group_id" in data.columns:
        data = data[data["group_id"] != 0]

    data["time"] = pd.to_datetime(data["time"], format="mixed", utc=True)
    data = compute_group_differences(data)
    print("Columns in the output DataFrame:", data.columns.tolist())
    data.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
