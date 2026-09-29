"""SPAD: seismogenic patches from a declustered earthquake catalog.

Background events (rows whose ``Aftershock`` value is not true) form
the level set ``{P >= alpha}`` of a DPS-style density, with
``alpha = (1 - beta) * max(P)``. That is the simple choice
``X^1(alpha)`` of Agayan et al. (2014): the point itself is excluded,
``P`` is not normalised, and the script does not iterate to the maximal
alpha-perfect set. The kernel is ``1 - d/r``. The radius ``r`` is the
power mean of the positive pairwise distances at a negative exponent
``q``. The search varies ``q`` and ``beta``. Dense events are then cut
into patches by centroid-linkage hierarchical clustering at the global
mode (maximum of the Freedman-Diaconis histogram) of the pairwise
distances.

Expected columns: ``lat``, ``lon``, ``depth``, ``time``, ``magnitude``,
and ``Aftershock``. ``depth`` is in kilometres. ``time`` is an ISO 8601
timestamp such as ``2023-01-01 12:30:45.123456+00:00``.

The original scripts were written for Python 3.11.7.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit, prange
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.signal import find_peaks
from scipy.spatial.distance import squareform
from sklearn.preprocessing import LabelEncoder


@jit(nopython=True)
def haversine_vectorized(lon1, lat1, lon2, lat2):
    """Haversine distance in kilometres.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float or numpy.ndarray
        Longitudes and latitudes in degrees.

    Returns
    -------
    float or numpy.ndarray
        Great-circle distance in kilometres.
    """
    earth_radius_km = 6371.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    a = sin_dlat**2 + np.cos(lat1) * np.cos(lat2) * sin_dlon**2
    a = np.minimum(np.maximum(a, 0.0), 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    return earth_radius_km * c


@jit(nopython=True)
def compute_depth_differences(depth_array, depth_scalar):
    """Absolute depth differences in kilometres.

    Parameters
    ----------
    depth_array : numpy.ndarray
        Depths in kilometres.
    depth_scalar : float
        Reference depth in kilometres.

    Returns
    -------
    numpy.ndarray
        Absolute differences.
    """
    return np.abs(depth_array - depth_scalar)


@jit(nopython=True, parallel=True)
def compute_3d_distance(coords):
    """3D distance matrix for ``(lat, lon, depth)`` rows.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape ``(n, 3)`` with latitude, longitude and depth.

    Returns
    -------
    numpy.ndarray
        Euclidean combination of the haversine distance and the depth
        difference, in kilometres. The diagonal is zero.
    """
    num_events = len(coords)
    distance_matrix = np.zeros((num_events, num_events))

    for i in prange(num_events):
        lon1_i, lat1_i, depth1_i = coords[i, 1], coords[i, 0], coords[i, 2]
        horizontal_distances = haversine_vectorized(
            lon1_i, lat1_i, coords[:, 1], coords[:, 0]
        )
        depth_differences = compute_depth_differences(coords[:, 2], depth1_i)
        distances = np.sqrt(horizontal_distances**2 + depth_differences**2)
        distance_matrix[i, :] = distances
    return distance_matrix


def distance_histogram(distance_matrix):
    """Upper-triangle pairwise distances, excluding zeros.

    A matrix that is not symmetric within ``1e-8`` is replaced by its
    average with its transpose before the triangle is read.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square distance matrix.

    Returns
    -------
    numpy.ndarray
        Positive unique pairwise distances.
    """
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-8):
        print("Distance matrix is not symmetric. Symmetrizing it.")
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    distances = distances[distances > 0]
    return distances


def freedman_diaconis_bins(data):
    """Freedman-Diaconis bin count for a distance sample.

    Parameters
    ----------
    data : array-like
        Sample of distances.

    Returns
    -------
    int
        Number of bins.
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    bin_width = 2 * iqr / (len(data) ** (1 / 3))
    return int(np.ceil((max(data) - min(data)) / bin_width))


@jit(nopython=True, parallel=True)
def compute_density(distance_matrix, radius):
    """DPS-style density (point itself excluded, not normalised).

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square distances in kilometres.
    radius : float
        Localization radius in kilometres.

    Returns
    -------
    numpy.ndarray
        ``sum(1 - d / radius)`` over other events inside the radius.
    """
    num_events = distance_matrix.shape[0]
    densities = np.zeros(num_events)

    for i in prange(num_events):
        density = 0.0
        for j in range(num_events):
            if i != j and distance_matrix[i, j] <= radius:
                d_wg = distance_matrix[i, j]
                density += 1 - d_wg / radius
        densities[i] = density
    return densities


def fuzzy_density_comparison(densities, beta):
    """Density level ``alpha = max(P) - beta * max(P)``.

    Parameters
    ----------
    densities : numpy.ndarray
        Density at each event.
    beta : float
        Level in the search grid. ``0`` is returned when the maximum
        density is zero.

    Returns
    -------
    float
        Density level ``alpha``.
    """
    max_density = np.max(densities)
    if max_density == 0:
        return 0
    return max_density - beta * max_density


def find_dense_subsets(densities, alpha):
    """Indices whose density is at least ``alpha``.

    Parameters
    ----------
    densities : numpy.ndarray
        Density at each event.
    alpha : float
        Density level.

    Returns
    -------
    numpy.ndarray
        Integer indices.
    """
    return np.where(densities >= alpha)[0]


def analyze_peak_similarity(all_densities, dense_densities):
    """Mean distance between peaks of the two density histograms.

    Both histograms share Freedman-Diaconis edges of the pooled sample.
    The bin width is guarded against zero. An empty peak set yields
    infinity.

    Parameters
    ----------
    all_densities : numpy.ndarray
        Density of every background event.
    dense_densities : numpy.ndarray
        Density of the events that pass ``alpha``.

    Returns
    -------
    float
        Mean, over peaks of the full sample, of the distance in bins to
        the nearest peak of the dense subset.
    """

    def _freedman_diaconis_bins(data):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        bin_width = 2 * iqr / (len(data) ** (1 / 3))
        if bin_width == 0:
            return 10
        return int(np.ceil((max(data) - min(data)) / bin_width))

    all_combined = np.concatenate((all_densities, dense_densities))
    num_bins = _freedman_diaconis_bins(all_combined)
    bins = np.histogram_bin_edges(all_combined, bins=num_bins)

    hist_all, _ = np.histogram(all_densities, bins=bins)
    hist_dense, _ = np.histogram(dense_densities, bins=bins)

    peaks_all, _ = find_peaks(hist_all)
    peaks_dense, _ = find_peaks(hist_dense)

    if len(peaks_all) > 0 and len(peaks_dense) > 0:
        peak_differences = [
            np.min(np.abs(peaks_dense - peak_all)) for peak_all in peaks_all
        ]
        return np.mean(peak_differences)
    return float("inf")


def search_dense_groups(distance_matrix, all_distances):
    """Grid search of ``q`` and ``beta`` for the level set ``{P >= alpha}``.

    ``alpha = (1 - beta) * max(P)`` for a DPS-style density (the point
    itself is excluded and ``P`` is not normalised). ``q`` runs through
    ``numpy.arange(-2.9, -0.1, 0.1)`` and ``beta`` through
    ``numpy.arange(-1.0, 1.0, 0.1)``. The radius is the power mean of
    order ``q`` of the positive pairwise distances. Configurations with
    fewer than two dense events are skipped. The kept configuration
    minimizes the density-histogram peak difference, an empirical SPAD
    heuristic that is not part of DPS, and on a tie keeps the larger
    number of centroid-linkage clusters cut at that radius ``r``.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Distances between background events.
    all_distances : numpy.ndarray
        Positive pairwise distances.

    Returns
    -------
    best_params : dict or None
        ``q``, ``beta``, ``r``, ``alpha``, ``clusters`` and
        ``dense_indices`` of the selected configuration, or ``None``.
    best_clust_num : int
        Number of clusters in that configuration.
    """
    qs = np.arange(-2.9, -0.1, 0.1)
    betas = np.arange(-1.0, 1.0, 0.1)

    best_params = None
    best_clust_num = 0
    min_peak_diff = float("inf")

    for q in qs:
        for beta in betas:
            r = (
                (np.sum(all_distances**q) / len(all_distances)) ** (1 / q)
                if q != 0
                else np.exp(np.sum(np.log(all_distances)) / len(all_distances))
            )

            densities = compute_density(distance_matrix, r)
            alpha = fuzzy_density_comparison(densities, beta)
            dense_indices = find_dense_subsets(densities, alpha)

            if len(dense_indices) < 2:
                continue

            peak_diff = analyze_peak_similarity(densities, densities[dense_indices])
            dense_distance_matrix = distance_matrix[
                np.ix_(dense_indices, dense_indices)
            ]
            dense_distance_matrix = (
                dense_distance_matrix + dense_distance_matrix.T
            ) / 2
            condensed_dense_distance_matrix = squareform(dense_distance_matrix)

            linkage_matrix = linkage(condensed_dense_distance_matrix, method="centroid")
            clusters = fcluster(linkage_matrix, t=r, criterion="distance")
            num_clusters = len(np.unique(clusters))

            if peak_diff < min_peak_diff or (
                peak_diff == min_peak_diff and num_clusters > best_clust_num
            ):
                best_params = {
                    "q": q,
                    "beta": beta,
                    "r": r,
                    "alpha": alpha,
                    "clusters": clusters,
                    "dense_indices": dense_indices,
                }
                min_peak_diff = peak_diff
                best_clust_num = num_clusters

    return best_params, best_clust_num


def plot_distance_histogram(
    distance_matrix, distance_matrix_dense, output_histogram_csv, output_histogram_png
):
    """Save the pairwise-distance histogram and its bin table.

    Both curves use 40 bins. The figure is written as PDF.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Distances between background events.
    distance_matrix_dense : numpy.ndarray
        Distances between events that belong to a dense group.
    output_histogram_csv : str
        Path of the bin-center table.
    output_histogram_png : str
        Path of the figure. The file is PDF even if the name differs.

    Returns
    -------
    distances, distances_dense : numpy.ndarray
        Positive upper-triangle distances of the two matrices.
    """
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    distances_dense = distance_matrix_dense[
        np.triu_indices_from(distance_matrix_dense, k=1)
    ]

    distances = distances[distances > 0]
    distances_dense = distances_dense[distances_dense > 0]

    plt.figure(figsize=(9, 6), dpi=600)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 24})

    hist_a, bin_edges_a = np.histogram(distances, bins=40, density=True)
    hist_b, bin_edges_b = np.histogram(distances_dense, bins=40, density=True)
    bin_centers_a = (bin_edges_a[:-1] + bin_edges_a[1:]) / 2
    bin_centers_b = (bin_edges_b[:-1] + bin_edges_b[1:]) / 2

    hist_data = pd.DataFrame(
        {
            "Distance_all_km": bin_centers_a,
            "Frequency_all": hist_a,
            "Distance_dense_km": bin_centers_b,
            "Frequency_dense": hist_b,
        }
    )
    hist_data.to_csv(output_histogram_csv, index=False)
    print(f"Histogram data saved to {output_histogram_csv}")

    plt.plot(bin_centers_a, hist_a, marker="o", color="black")
    plt.plot(bin_centers_b, hist_b, marker="*", color="red")
    plt.xlabel("Distances, km", fontsize=24)
    plt.ylabel("Relative Frequency", fontsize=24)

    ax = plt.gca()
    for tick in ax.get_yticklabels():
        tick.set_fontsize(20)
        tick.set_fontname("Times New Roman")
        tick.set_position((tick.get_position()[0] - 0.01, tick.get_position()[1]))
    for tick in ax.get_xticklabels():
        tick.set_fontsize(20)
        tick.set_fontname("Times New Roman")
        tick.set_position((tick.get_position()[0] - 0.01, tick.get_position()[1]))

    plt.savefig(output_histogram_png, format="pdf", dpi=600, bbox_inches="tight")
    plt.show()
    return distances, distances_dense


def visualize_dendrogram(linkage_matrix, threshold, output_path):
    """Save the centroid-linkage dendrogram.

    Parameters
    ----------
    linkage_matrix : numpy.ndarray
        Linkage matrix of the dense events.
    threshold : float
        Horizontal line, in kilometres, at the global mode (maximum of
        the Freedman-Diaconis histogram).
    output_path : str
        Figure path. The file is written as PDF.
    """
    plt.figure(figsize=(9, 6), facecolor="#D0CFD4")
    dendrogram(
        linkage_matrix,
        color_threshold=threshold,
        p=5,
        truncate_mode="level",
    )
    plt.axhline(y=threshold, color="r", linestyle="--", label="Specific distance")
    plt.xlabel("Set index", fontsize=24, fontname="Times New Roman")
    plt.ylabel("Distance, km", fontsize=24, fontname="Times New Roman")

    x_tick_font = {"fontname": "Times New Roman", "fontsize": 12}
    y_tick_font = {"fontname": "Times New Roman", "fontsize": 20}
    plt.xticks(fontsize=x_tick_font["fontsize"], fontname=x_tick_font["fontname"])
    plt.yticks(fontsize=y_tick_font["fontsize"], fontname=y_tick_font["fontname"])
    plt.legend(fontsize=22)
    plt.savefig(output_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.show()


def _drop_nonfinite_coordinates(coordinates):
    """Drop rows with a non-finite coordinate.

    Parameters
    ----------
    coordinates : numpy.ndarray
        Coordinate array. Rows are removed here only; the caller keeps
        the original table.

    Returns
    -------
    numpy.ndarray
        Rows whose entries are all finite.
    """
    return coordinates[
        ~np.isnan(coordinates).any(axis=1) & ~np.isinf(coordinates).any(axis=1)
    ]


def build_parser():
    """Command-line interface for patch identification.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the standalone script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Identify seismogenic patches in a declustered catalog "
            "(level set {P >= alpha} of a DPS-style density, "
            "then centroid linkage at the global mode)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m spad.identify_patches \\\n"
            "      --input background.csv \\\n"
            "      --dense-events dense_events.csv \\\n"
            "      --clustered-results patches.csv \\\n"
            "      --histogram-csv histogram.csv \\\n"
            "      --histogram-plot histogram.pdf \\\n"
            "      --dendrogram-plot dendrogram.pdf\n"
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Declustered CSV with columns lat, lon, depth, time, magnitude, Aftershock."
        ),
    )
    parser.add_argument(
        "--dense-events",
        required=True,
        help="CSV of all background events with column dense_group_id.",
    )
    parser.add_argument(
        "--clustered-results",
        required=True,
        help="CSV of dense events with the patch column cluster.",
    )
    parser.add_argument(
        "--histogram-csv",
        required=True,
        help="CSV of pairwise-distance histogram bin centers.",
    )
    parser.add_argument(
        "--histogram-plot",
        required=True,
        help="Pairwise-distance figure, written as PDF.",
    )
    parser.add_argument(
        "--dendrogram-plot",
        required=True,
        help="Dendrogram figure, written as PDF.",
    )
    return parser


def main(argv=None):
    """Run SPAD patch identification from the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, excluding the program name. ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    data = pd.read_csv(args.input)

    if "Aftershock" in data.columns:
        # Same test as ``!= True``: keep False and missing values.
        data = data[data["Aftershock"].ne(True)]

    # magnitude and time are part of the catalog contract; clustering
    # uses the coordinates only.
    _ = data["magnitude"]
    _ = data["time"]

    coordinates = data[["lat", "lon", "depth"]].values
    coordinates = _drop_nonfinite_coordinates(coordinates)
    distance_matrix = compute_3d_distance(coordinates)

    all_distances = distance_histogram(distance_matrix)
    num_bins = freedman_diaconis_bins(all_distances)
    print(f"Suggested number of bins: {num_bins}")

    hist_values, bin_edges = np.histogram(all_distances, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    first_mode = bin_centers[np.argmax(hist_values)]

    best_params, best_clust_num = search_dense_groups(distance_matrix, all_distances)

    if best_params:
        print(
            "Best parameters: "
            f"q = {best_params['q']}, beta = {best_params['beta']}, "
            f"r = {best_params['r']}, alpha = {best_params['alpha']}"
        )
        print(f"Number of dense groups: {best_clust_num}")

        encoder = LabelEncoder()
        clusters_renamed = encoder.fit_transform(best_params["clusters"])
        cluster_labels = np.full(len(data), -1)
        for idx, cluster_label in zip(best_params["dense_indices"], clusters_renamed):
            cluster_labels[idx] = cluster_label

        data["dense_group_id"] = cluster_labels
        data.to_csv(args.dense_events, index=False)
    else:
        print("No suitable parameters found for clustering.")

    # Distance matrix for dense-group events only.
    data_dense = data[data["dense_group_id"] != -1].copy()
    _ = data_dense["magnitude"]
    _ = data_dense["time"]
    coordinates_dense = data_dense[["lat", "lon", "depth"]].values
    coordinates_dense = _drop_nonfinite_coordinates(coordinates_dense)
    distance_matrix_dense = compute_3d_distance(coordinates_dense)

    condensed_distance_matrix = squareform(distance_matrix_dense, checks=False)
    linkage_matrix = linkage(condensed_distance_matrix, method="centroid")

    threshold_specific_distance = first_mode
    print(
        "Specific distance of nontrivial pairwise distance distribution: "
        f"{threshold_specific_distance} km"
    )

    cluster_labels = fcluster(
        linkage_matrix, t=threshold_specific_distance, criterion="distance"
    )
    data_dense["cluster"] = cluster_labels
    data_dense.to_csv(args.clustered_results, index=False)

    plot_distance_histogram(
        distance_matrix,
        distance_matrix_dense,
        args.histogram_csv,
        args.histogram_plot,
    )
    visualize_dendrogram(
        linkage_matrix, threshold_specific_distance, args.dendrogram_plot
    )


if __name__ == "__main__":
    main()
