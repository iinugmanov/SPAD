"""Plot one nearest-neighbour tree for each ``group_id``.

Reads the classified catalog from :mod:`spad.classify_event_types`.
The horizontal axis is ``time_from_head_day`` and the vertical axis is
``dist_from_head_km``. Arrows follow ``nearest_neighbor`` when that
event is in the same group. Marker color is ``EQtype``.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def plot_event_tree(data, group_id, save_path):
    """Draw and save the tree of one group.

    Parameters
    ----------
    data : pandas.DataFrame
        Classified events. Required columns: ``group_id``, ``event_id``,
        ``nearest_neighbor``, ``time_from_head_day``,
        ``dist_from_head_km``, ``EQtype``.
    group_id : hashable
        Value of ``group_id`` to draw.
    save_path : str
        Image path (PNG when the name ends in ``.png``).
    """
    group_data = data[data["group_id"] == group_id]

    if group_data.empty:
        print(f"Group {group_id} was not found.")
        return

    _fig, ax = plt.subplots(figsize=(9, 6))

    node_colors = {"mainshock": "red", "foreshock": "blue", "aftershock": "green"}
    node_size_map = {"mainshock": 150, "foreshock": 70, "aftershock": 70}

    for _, row in group_data.iterrows():
        parent_id = row["nearest_neighbor"]
        child_id = row["event_id"]

        if parent_id in group_data["event_id"].values and child_id != parent_id:
            parent_event = group_data[group_data["event_id"] == parent_id].iloc[0]
            child_event = group_data[group_data["event_id"] == child_id].iloc[0]

            ax.annotate(
                "",
                xy=(
                    child_event["time_from_head_day"],
                    child_event["dist_from_head_km"],
                ),
                xytext=(
                    parent_event["time_from_head_day"],
                    parent_event["dist_from_head_km"],
                ),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "grey",
                    "lw": 0.5,
                    "alpha": 0.5,
                },
            )

    for _, row in group_data.iterrows():
        ax.scatter(
            row["time_from_head_day"],
            row["dist_from_head_km"],
            s=node_size_map[row["EQtype"]],
            color=node_colors[row["EQtype"]],
            alpha=0.6,
            edgecolor="black",
            linewidth=0.5,
            marker="o",
        )
        ax.text(
            row["time_from_head_day"],
            row["dist_from_head_km"],
            str(row["event_id"]),
            fontsize=7,
            ha="center",
            va="bottom",
            color="black",
        )

    ax.set_ylabel("Distance from Head (km)", fontsize=14)
    ax.set_xlabel("Time from Head (days)", fontsize=14)
    ax.set_title(f"Event Tree for Group {group_id}", fontsize=16)

    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Mainshock",
            markersize=10,
            markerfacecolor="red",
            alpha=0.5,
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Foreshock",
            markersize=7,
            markerfacecolor="blue",
            alpha=0.5,
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Aftershock",
            markersize=7,
            markerfacecolor="green",
            alpha=0.5,
            markeredgecolor="black",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=14,
        frameon=False,
    )

    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {save_path}")


def plot_all_groups(data, save_dir):
    """Draw every distinct ``group_id`` into ``save_dir``.

    Parameters
    ----------
    data : pandas.DataFrame
        Classified catalog.
    save_dir : str
        Directory created if it does not exist. Files are named
        ``group_<id>_tree.png``.
    """
    os.makedirs(save_dir, exist_ok=True)
    unique_groups = data["group_id"].unique()

    for group_id in unique_groups:
        save_path = os.path.join(save_dir, f"group_{group_id}_tree.png")
        plot_event_tree(data, group_id, save_path)


def build_parser():
    """Command-line interface for tree figures.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the standalone script.
    """
    parser = argparse.ArgumentParser(
        description="Plot event trees for each group_id.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m spad.plot_event_trees \\\n"
            "      --input results/catalog_with_types.csv \\\n"
            "      --output-dir results/event_trees\n"
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV from classify_event_types, with group_id, event_id, "
            "nearest_neighbor, time_from_head_day, dist_from_head_km "
            "and EQtype."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for group_<id>_tree.png figures.",
    )
    return parser


def main(argv=None):
    """Plot event trees from the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, excluding the program name. ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    data = pd.read_csv(args.input)
    plot_all_groups(data, save_dir=args.output_dir)


if __name__ == "__main__":
    main()
