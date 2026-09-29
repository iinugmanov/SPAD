"""3D nearest-neighbour declustering of an earthquake catalog.

For an earlier earthquake ``i`` and a later earthquake ``j`` the proximity is

    eta = T * R,
    T = t_days * 10 ** (-b * M_i / 2),
    R = (r_km ** d) * 10 ** (-b * M_i / 2),

when ``t_days > 0``, and infinity otherwise. ``r_km`` is the 3D distance
(haversine horizontal distance combined with the depth difference). A
two-component Gaussian mixture of ``log10(eta)`` supplies the threshold
between the clustered mode and the background. Pairs that fall inside a
space-time window are preferred as the nearest neighbour when such a pair
exists.

Expected columns: ``longitude``, ``latitude``, ``z_proj`` (km), ``time``,
``mag``.
"""

import argparse
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit, prange
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def load_catalog(file_path):
    """Load a comma-separated earthquake catalog.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Catalog as read from disk.
    """
    return pd.read_csv(file_path, delimiter=",")


def prepare_catalog(catalog):
    """Parse times, sort by time, and attach a row identifier.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Catalog containing a ``time`` column.

    Returns
    -------
    pandas.DataFrame
        Copy sorted by time. ``event_id`` is the positional index after
        that sort.
    """
    catalog = catalog.copy()
    catalog["time"] = pd.to_datetime(catalog["time"], format="mixed", utc=True)
    catalog.sort_values(by="time", inplace=True)
    catalog = catalog.reset_index(drop=True)
    catalog["event_id"] = catalog.index
    return catalog


@jit(nopython=True, parallel=True, cache=True)
def compute_space_diff(coords):
    """Symmetric matrix of 3D distances.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape ``(n_events, 3)``. The columns are read as
        latitude (degrees), longitude (degrees) and depth (km).

    Returns
    -------
    numpy.ndarray
        Distances in kilometres. The diagonal is ``1e10``.
    """
    n = coords.shape[0]
    dist = np.zeros((n, n))
    earth_radius_km = 6371.0

    for i in prange(n):
        lat1, lon1, depth1 = coords[i, 0], coords[i, 1], coords[i, 2]
        lat2 = coords[:, 0]
        lon2 = coords[:, 1]
        depth2 = coords[:, 2]

        lat1r = np.radians(lat1)
        lon1r = np.radians(lon1)
        lat2r = np.radians(lat2)
        lon2r = np.radians(lon2)
        dlon = lon2r - lon1r
        dlat = lat2r - lat1r
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
        )
        a = np.clip(a, 0.0, 1.0)
        horiz_dist = 2 * earth_radius_km * np.arcsin(np.sqrt(a))

        depth_diff = np.abs(depth2 - depth1)
        dist[i, :] = np.sqrt(horiz_dist**2 + depth_diff**2)
        dist[i, i] = 1e10

    return dist


@jit(nopython=True, parallel=True)
def compute_time_diff(time_values):
    """Pairwise time differences in days.

    Parameters
    ----------
    time_values : numpy.ndarray
        Event times as integer nanoseconds (``datetime64[ns]``).

    Returns
    -------
    numpy.ndarray
        ``time_diff[j, i] = t_j - t_i`` in days.
    """
    num_events = time_values.shape[0]
    time_values = time_values.astype("int64") // 10**9
    time_diff = np.zeros((num_events, num_events), dtype=np.float64)
    for j in prange(num_events):
        for i in prange(num_events):
            time_diff[j, i] = (time_values[j] - time_values[i]) / 86400.0
    return time_diff


@jit(nopython=True, parallel=True)
def compute_eta(time_diff, space_diff, magnitudes, b_value, d_value, apply_filter):
    """Rescaled time ``T``, distance ``R`` and proximity ``eta``.

    ``T``, ``R`` and ``eta`` are filled for every pair with ``t_j > t_i``.
    When ``apply_filter`` is true and the pair lies outside the space-time
    window, ``eta`` is left at infinity while ``T`` and ``R`` are kept.

    Parameters
    ----------
    time_diff, space_diff : numpy.ndarray
        Pairwise time (days) and distance (km).
    magnitudes : numpy.ndarray
        Magnitude of each event. The earlier event's magnitude is used.
    b_value, d_value : float
        Gutenberg-Richter ``b`` and fractal dimension ``d``.
    apply_filter : bool
        If true, ``eta`` is finite only inside the space-time window.

    Returns
    -------
    eta, T, R : numpy.ndarray
        Proximity and its time and space factors.
    """
    num_events = time_diff.shape[0]
    eta = np.full((num_events, num_events), np.inf)
    t_factor = np.zeros((num_events, num_events))
    r_factor = np.zeros((num_events, num_events))

    for j in prange(num_events):
        for i in prange(j):
            raw_time_diff = time_diff[j, i]
            raw_space_diff = space_diff[j, i]

            if raw_time_diff > 0:
                t_val = raw_time_diff * 10 ** (-b_value * magnitudes[i] / 2)
                r_val = (raw_space_diff**d_value) * 10 ** (-b_value * magnitudes[i] / 2)

                t_factor[j, i] = t_val
                r_factor[j, i] = r_val
                eta_val = t_val * r_val

                if apply_filter:
                    r_thresh = 0.02 * 10 ** (0.5 * magnitudes[i])
                    t_thresh = 0.04 * 10 ** (0.55 * magnitudes[i])
                    if (raw_space_diff <= r_thresh) and (raw_time_diff <= t_thresh):
                        eta[j, i] = eta_val
                    else:
                        eta[j, i] = np.inf
                else:
                    eta[j, i] = eta_val

    return eta, t_factor, r_factor


@jit(nopython=True, parallel=True)
def compute_nearest_neighbors(
    time_values, space_diff, magnitudes, b_value, d_value, apply_filter
):
    """Nearest earlier neighbour of each event in ``eta``.

    Parameters
    ----------
    time_values : numpy.ndarray
        Integer nanosecond timestamps.
    space_diff : numpy.ndarray
        Pairwise 3D distances in kilometres.
    magnitudes : numpy.ndarray
        Event magnitudes.
    b_value, d_value : float
        Gutenberg-Richter ``b`` and fractal dimension ``d``.
    apply_filter : bool
        Passed through to :func:`compute_eta`.

    Returns
    -------
    min_eta : numpy.ndarray
        Smallest finite-or-infinite ``eta`` against earlier events.
    nearest_neighbors : numpy.ndarray
        Index of that neighbour, or ``-1`` when the event is the earliest.
    min_T, min_R : numpy.ndarray
        ``T`` and ``R`` of the selected pair.
    eta : numpy.ndarray
        Full proximity matrix.
    """
    num_events = len(time_values)
    time_diff = compute_time_diff(time_values)
    eta, t_factor, r_factor = compute_eta(
        time_diff, space_diff, magnitudes, b_value, d_value, apply_filter
    )

    min_eta = np.zeros(num_events)
    nearest_neighbors = np.zeros(num_events, dtype=np.int32)
    min_t = np.zeros(num_events)
    min_r = np.zeros(num_events)

    for j in prange(num_events):
        valid_indices = np.where(time_diff[j, :] > 0)[0]
        if valid_indices.size > 0:
            min_idx = valid_indices[np.argmin(eta[j, valid_indices])]
            min_eta[j] = eta[j, min_idx]
            nearest_neighbors[j] = min_idx
            min_t[j] = t_factor[j, min_idx]
            min_r[j] = r_factor[j, min_idx]
        else:
            min_eta[j] = np.inf
            nearest_neighbors[j] = -1
            min_t[j] = 0.0
            min_r[j] = 0.0

    return min_eta, nearest_neighbors, min_t, min_r, eta


def process_catalog_with_progress(df, b_value, d_value, apply_filter):
    """Attach nearest-neighbour columns to a catalog.

    Two neighbours are computed for every event: one without the
    space-time window and one with it. A finite filtered neighbour
    replaces the unfiltered one. ``apply_filter`` is accepted for
    compatibility and is not read; both neighbours are always computed.

    Non-finite ``min_eta`` values are replaced by one more than the
    largest finite value (or by ``1`` when none is finite).

    Parameters
    ----------
    df : pandas.DataFrame
        Catalog with ``longitude``, ``latitude``, ``z_proj``, ``time``
        and ``mag``. ``time`` must already be datetime-like. The array
        passed to :func:`compute_space_diff` is latitude, longitude,
        ``z_proj``, which is the order that routine reads.
    b_value, d_value : float
        Gutenberg-Richter ``b`` and fractal dimension ``d``.
    apply_filter : bool
        Unused. Filtered and unfiltered neighbours are both computed.

    Returns
    -------
    pandas.DataFrame
        Copy with ``min_eta``, ``nearest_neighbor``, ``min_T``, ``min_R``
        and ``passed_filter``.
    """
    del apply_filter  # both neighbours are always computed; see docstring
    coords = df[["latitude", "longitude", "z_proj"]].values
    time_values = df["time"].astype(np.int64).values
    magnitudes = df["mag"].values

    space_diff = compute_space_diff(coords)

    (
        min_eta_global,
        nearest_neighbors_global,
        min_t_global,
        min_r_global,
        _,
    ) = compute_nearest_neighbors(
        time_values,
        space_diff,
        magnitudes,
        b_value,
        d_value,
        apply_filter=False,
    )

    (
        min_eta_filtered,
        nearest_neighbors_filtered,
        min_t_filtered,
        min_r_filtered,
        _,
    ) = compute_nearest_neighbors(
        time_values,
        space_diff,
        magnitudes,
        b_value,
        d_value,
        apply_filter=True,
    )

    num_events = len(time_values)
    final_min_eta = np.empty_like(min_eta_global)
    final_nearest_neighbors = np.empty_like(nearest_neighbors_global)
    final_min_t = np.empty_like(min_t_global)
    final_min_r = np.empty_like(min_r_global)
    passed_filter = np.empty(num_events, dtype=np.bool_)

    for j in range(num_events):
        if np.isfinite(min_eta_filtered[j]):
            final_min_eta[j] = min_eta_filtered[j]
            final_nearest_neighbors[j] = nearest_neighbors_filtered[j]
            final_min_t[j] = min_t_filtered[j]
            final_min_r[j] = min_r_filtered[j]
            passed_filter[j] = True
        else:
            final_min_eta[j] = min_eta_global[j]
            final_nearest_neighbors[j] = nearest_neighbors_global[j]
            final_min_t[j] = min_t_global[j]
            final_min_r[j] = min_r_global[j]
            passed_filter[j] = False

    if np.any(np.isfinite(final_min_eta)):
        finite_eta = final_min_eta[np.isfinite(final_min_eta)]
        max_finite = np.nanmax(finite_eta) if finite_eta.size > 0 else 1.0
        fill_value = max_finite + 1.0
    else:
        fill_value = 1.0

    final_min_eta = np.nan_to_num(
        final_min_eta, nan=fill_value, posinf=fill_value, neginf=fill_value
    )

    df_result = df.copy()
    df_result["min_eta"] = final_min_eta
    df_result["nearest_neighbor"] = final_nearest_neighbors
    df_result["min_T"] = final_min_t
    df_result["min_R"] = final_min_r
    df_result["passed_filter"] = passed_filter
    return df_result


def plot_heatmap(
    processed_catalog,
    bins_strategy="quantile",
    interpolation="bilinear",
    output_path="heatmap_T_R.png",
    n_trials=10,
    random_seed=42,
    eta_histogram_path="disturb_eta.pdf",
    gmm_path="GMM.pdf",
):
    """Estimate the ``eta`` threshold and save the declustering figures.

    The ``log10(eta)`` histogram uses a Freedman-Diaconis bin count
    clipped to the range 10-100, drawn together with a cubic
    interpolation of the bin heights. ``bins_strategy`` is accepted for
    compatibility and is not used. The heatmap of ``log10(T)`` against
    ``log10(R)`` uses 80 bins.

    The threshold is the intersection, between the two component means,
    of a two-component Gaussian mixture. ``n_trials`` fits are run. The
    trial whose intersection lies closest to the most common value
    rounded to two decimals is kept. ``eta0 = 10 ** intersection``.

    Parameters
    ----------
    processed_catalog : pandas.DataFrame
        Output of :func:`process_catalog_with_progress`.
    bins_strategy : str, optional
        Unused. Retained so existing callers can still pass it.
    interpolation : str, optional
        Interpolation passed to ``imshow`` for the heatmap.
    output_path : str, optional
        Path of the ``T``-``R`` heatmap. Written as PDF.
    n_trials : int, optional
        Number of mixture fits.
    random_seed : int or None, optional
        Base seed. Trial ``k`` uses ``random_seed + k``. ``None`` leaves
        the seed unset.
    eta_histogram_path : str, optional
        Path of the ``log10(eta)`` distribution figure.
    gmm_path : str, optional
        Path of the Gaussian-mixture figure.

    Returns
    -------
    float
        Threshold ``eta0`` on a linear scale.
    """
    del bins_strategy  # unused by the active histogram; see docstring

    min_t = processed_catalog["min_T"].values
    min_r = processed_catalog["min_R"].values
    min_eta = processed_catalog["min_eta"].values

    mask = (
        np.isfinite(min_t)
        & np.isfinite(min_r)
        & np.isfinite(min_eta)
        & (min_t > 0)
        & (min_r > 0)
    )
    t_filtered = min_t[mask]
    r_filtered = min_r[mask]
    eta_filtered = min_eta[mask]
    log_eta = np.log10(eta_filtered)

    def auto_bins(data):
        """Freedman-Diaconis bin count, clipped to 10-100 bins."""
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            iqr = np.std(data) * 1.34
        bin_width = 2 * iqr / (len(data) ** (1 / 3))
        bins = int((np.max(data) - np.min(data)) / bin_width)
        return max(min(bins, 100), 10)

    plt.figure(figsize=(10, 6))
    n_bins = auto_bins(log_eta)
    hist, bin_edges = np.histogram(log_eta, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    x_grid = np.linspace(log_eta.min() - 1, log_eta.max() + 1, 10000)
    interp_func = interp1d(
        bin_centers, hist, kind="cubic", bounds_error=False, fill_value=0
    )
    smooth_curve = interp_func(x_grid)

    plt.hist(
        log_eta,
        bins=n_bins,
        density=True,
        alpha=0.5,
        color="lightblue",
        edgecolor="black",
        label=r"$\log_{10}(\eta_{ij})$",
    )
    plt.plot(x_grid, smooth_curve, "b-", linewidth=2, label="Smoothed density")
    plt.title(
        r"Distribution of $\log_{10}(\eta_{ij})$ with automatic binning",
        pad=20,
    )
    plt.xlabel(r"$\log_{10}(\eta_{ij})$")
    plt.ylabel("Probability density")
    plt.ylim(0, smooth_curve.max() * 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(eta_histogram_path, dpi=600)
    plt.tight_layout()
    plt.show()

    trial_results = []
    for trial in range(n_trials):
        seed = random_seed + trial if random_seed is not None else None
        gmm = GaussianMixture(
            n_components=2,
            random_state=seed,
            init_params="k-means++",
            covariance_type="full",
            n_init=50,
            tol=1e-6,
            max_iter=500,
        )
        gmm.fit(log_eta.reshape(-1, 1))

        sorted_indices = np.argsort(gmm.means_.flatten())
        means = gmm.means_.flatten()[sorted_indices]
        stds = np.sqrt(gmm.covariances_.flatten()[sorted_indices])
        weights = gmm.weights_.flatten()[sorted_indices]

        x = np.linspace(means[0] - 2 * stds[0], means[1] + 2 * stds[1], 100000)
        pdf1 = weights[0] * norm.pdf(x, means[0], stds[0])
        pdf2 = weights[1] * norm.pdf(x, means[1], stds[1])

        valid_region = (x > means[0]) & (x < means[1])
        if np.any(valid_region):
            intersection_log_trial = x[valid_region][
                np.argmin(np.abs(pdf1[valid_region] - pdf2[valid_region]))
            ]
        else:
            intersection_log_trial = np.mean(means)

        trial_results.append(
            {
                "intersection": intersection_log_trial,
                "x": x,
                "pdf1": pdf1,
                "pdf2": pdf2,
                "means": means,
                "stds": stds,
                "weights": weights,
            }
        )

    intersections = [res["intersection"] for res in trial_results]
    rounded_intersections = [round(val, 2) for val in intersections]
    try:
        mode_intersection = statistics.mode(rounded_intersections)
    except statistics.StatisticsError:
        mode_intersection = np.median(intersections)
    differences = [abs(val - mode_intersection) for val in intersections]
    best_trial_idx = int(np.argmin(differences))
    best_trial = trial_results[best_trial_idx]
    intersection_log_final = best_trial["intersection"]
    threshold_eta0 = 10**intersection_log_final

    print("Results of individual runs (unrounded):", intersections)
    print(
        "Most frequent (rounded) intersection value:",
        mode_intersection,
    )
    print(
        "Final intersection (intersection_log_final):",
        intersection_log_final,
    )
    print("Threshold eta0 (linear scale):", threshold_eta0)

    plt.figure(figsize=(10, 8))
    plt.hist(
        log_eta,
        bins=n_bins,
        density=True,
        alpha=0.5,
        color="lightblue",
        edgecolor="black",
        label=r"$\log_{10}(\eta_{ij})$",
    )
    plt.plot(
        x_grid,
        smooth_curve,
        "b-",
        linewidth=4,
        alpha=0.5,
        label="Smoothed density",
    )
    plt.plot(
        best_trial["x"],
        best_trial["pdf1"],
        label=(
            "Component 1 "
            f"(μ={best_trial['means'][0]:.2f}, "
            f"σ={best_trial['stds'][0]:.2f})"
        ),
        color="black",
        linewidth=2,
    )
    plt.plot(
        best_trial["x"],
        best_trial["pdf2"],
        label=(
            "Component 2 "
            f"(μ={best_trial['means'][1]:.2f}, "
            f"σ={best_trial['stds'][1]:.2f})"
        ),
        color="black",
        linewidth=3,
    )
    plt.axvline(
        intersection_log_final,
        color="black",
        linestyle="--",
        label=(
            r"Intersection "
            rf"($\log_{{10}}(\eta_{{ij}})={intersection_log_final:.2f}$)"
        ),
    )
    plt.ylim(0, smooth_curve.max() * 1.05)
    plt.legend(
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.title(r"Gaussian mixture of $\log_{10}(\eta_{ij})$")
    plt.savefig(gmm_path, dpi=600)
    plt.tight_layout()
    plt.show()

    log_t = np.log10(t_filtered)
    log_r = np.log10(r_filtered)

    n_bins = 80
    heatmap, xedges, yedges = np.histogram2d(log_t, log_r, bins=n_bins)
    heatmap = gaussian_filter(heatmap, sigma=1)

    x_grid_heat, y_grid_heat = np.meshgrid(
        np.linspace(xedges[0], xedges[-1], num=n_bins),
        np.linspace(yedges[0], yedges[-1], num=n_bins),
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.figure(figsize=(9, 6), dpi=600)
    plt.imshow(
        heatmap.T,
        extent=extent,
        origin="lower",
        cmap="seismic",
        interpolation=interpolation,
    )

    contour_levels = np.linspace(
        np.percentile(heatmap, 5),
        np.percentile(heatmap, 95),
        5,
    )
    contour = plt.contour(
        x_grid_heat,
        y_grid_heat,
        heatmap.T,
        levels=contour_levels,
        colors="black",
        linewidths=1.2,
        alpha=0.7,
    )
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.1f")

    x_line = np.linspace(log_t.min(), log_t.max(), 1000)
    y_line = intersection_log_final - x_line
    plt.plot(
        x_line,
        y_line,
        color="white",
        linestyle=":",
        linewidth=3,
        label=(
            f"GMM threshold: log10(η0)={intersection_log_final:.2f}\n"
            f"(η0={threshold_eta0:.2e})"
        ),
    )
    plt.legend()
    plt.xlim(-5, log_t.max())
    plt.ylim(-3, log_r.max())
    plt.savefig(output_path, format="pdf", dpi=600, bbox_inches="tight")
    plt.show()
    print(f"Heatmap saved at {output_path}")

    return threshold_eta0


def build_parser():
    """Command-line interface for declustering.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the standalone script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "3D nearest-neighbour declustering of an earthquake catalog. "
            "Labels the clustered mode in the column Aftershock."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m spad.declustering \\\n"
            "      --catalog catalog.csv \\\n"
            "      --b-value 1 \\\n"
            "      --d-value 1.6 \\\n"
            "      --output-dir results/decluster\n"
        ),
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help=("CSV catalog with columns longitude, latitude, z_proj, time, mag."),
    )
    parser.add_argument(
        "--b-value",
        type=float,
        required=True,
        help="Gutenberg-Richter b-value (example: 1).",
    )
    parser.add_argument(
        "--d-value",
        type=float,
        required=True,
        help="Fractal dimension d (example: 1.6).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the declustered catalog and figures.",
    )
    parser.add_argument(
        "--catalog-output",
        default=None,
        help=("Output CSV path (default: <output-dir>/catalog_with_aftershock.csv)."),
    )
    parser.add_argument(
        "--heatmap",
        default=None,
        help="T-R heatmap PDF (default: <output-dir>/Declustered_T_R.pdf).",
    )
    parser.add_argument(
        "--eta-histogram",
        default=None,
        help=("log10(eta) distribution PDF (default: <output-dir>/disturb_eta.pdf)."),
    )
    parser.add_argument(
        "--gmm-figure",
        default=None,
        help="Gaussian-mixture PDF (default: <output-dir>/GMM.pdf).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Gaussian-mixture fits (default: 30).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2,
        help="Base random seed for the mixture fits (default: 2).",
    )
    parser.add_argument(
        "--interpolation",
        default="bicubic",
        help="imshow interpolation of the T-R heatmap (default: bicubic).",
    )
    return parser


def main(argv=None):
    """Run declustering from the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, excluding the program name. ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    catalog_output = args.catalog_output or os.path.join(
        args.output_dir, "catalog_with_aftershock.csv"
    )
    heatmap_output = args.heatmap or os.path.join(
        args.output_dir, "Declustered_T_R.pdf"
    )
    eta_histogram = args.eta_histogram or os.path.join(
        args.output_dir, "disturb_eta.pdf"
    )
    gmm_figure = args.gmm_figure or os.path.join(args.output_dir, "GMM.pdf")

    catalog = prepare_catalog(load_catalog(args.catalog))
    processed_catalog = process_catalog_with_progress(
        catalog, args.b_value, args.d_value, apply_filter=True
    )
    threshold_eta = plot_heatmap(
        processed_catalog,
        bins_strategy="quantile",
        interpolation=args.interpolation,
        output_path=heatmap_output,
        n_trials=args.n_trials,
        random_seed=args.random_seed,
        eta_histogram_path=eta_histogram,
        gmm_path=gmm_figure,
    )
    print(
        "Final saved intersection (intersection threshold):",
        threshold_eta,
    )
    processed_catalog["Aftershock"] = processed_catalog["min_eta"] < threshold_eta
    processed_catalog.to_csv(catalog_output, index=False)
    print(f"Processed catalog saved in the directory: {args.output_dir}")


if __name__ == "__main__":
    main()
