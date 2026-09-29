"""Form event trees from nearest-neighbour chains.

Reads a declustered catalog (``event_id``, ``nearest_neighbor``,
``passed_filter``) and writes ``tree_id``, ``parent`` and ``group_id``.

An event with ``passed_filter`` walks ``nearest_neighbor`` while the
neighbour also passed the filter. The first neighbour outside that set
is the tree root. Trees that contain only one filtered event are
dropped. The root row receives ``parent = 0`` and ``tree_id`` equal to
its own ``event_id``. Other events in a kept tree receive
``parent = nearest_neighbor``. Remaining events receive ``parent = -1``
and ``group_id = 0``. Kept trees are numbered from 1 in order of
appearance.
"""

import argparse
from collections import defaultdict

import pandas as pd


def get_root(event, event_by_id, aftershock_ids):
    """Walk a nearest-neighbour chain to the first external neighbour.

    Parameters
    ----------
    event : dict
        Starting event. Must contain ``nearest_neighbor``.
    event_by_id : dict
        Events keyed by ``event_id``.
    aftershock_ids : set
        ``event_id`` values of events with ``passed_filter``.

    Returns
    -------
    nearest-neighbour id
        ``event_id`` of the first neighbour that is not in
        ``aftershock_ids``.
    """
    current = event
    while current["nearest_neighbor"] in aftershock_ids:
        current = event_by_id[current["nearest_neighbor"]]
    return current["nearest_neighbor"]


def assign_tree_attributes(events):
    """Set ``tree_id``, ``parent`` and leave grouping to the caller.

    The list is updated in place. See the module docstring for the rules.

    Parameters
    ----------
    events : list of dict
        Catalog rows. Each dict needs ``event_id``, ``nearest_neighbor``
        and ``passed_filter``.

    Returns
    -------
    set
        Root identifiers of trees that contain more than one filtered
        event.
    """
    event_by_id = {event["event_id"]: event for event in events}
    aftershock_ids = {event["event_id"] for event in events if event["passed_filter"]}

    for event in events:
        if event["passed_filter"]:
            event["tree_id"] = get_root(event, event_by_id, aftershock_ids)
        else:
            event["tree_id"] = None

    trees = defaultdict(list)
    for event in events:
        if event["tree_id"] is not None:
            trees[event["tree_id"]].append(event)

    valid_tree_ids = {tid for tid, group in trees.items() if len(group) > 1}

    for event in events:
        if event["tree_id"] not in valid_tree_ids:
            event["tree_id"] = None

    for event in events:
        if event["event_id"] in valid_tree_ids:
            event["parent"] = 0
        elif event["tree_id"] is not None:
            event["parent"] = event["nearest_neighbor"]
        else:
            event["parent"] = -1

    for event in events:
        if event["parent"] == 0:
            event["tree_id"] = event["event_id"]

    return valid_tree_ids


def add_group_ids(df_updated):
    """Number kept trees from 1 in order of appearance.

    Parameters
    ----------
    df_updated : pandas.DataFrame
        Table containing ``tree_id``. Missing ``tree_id`` becomes
        ``group_id`` 0.

    Returns
    -------
    pandas.DataFrame
        The same table with ``group_id``.
    """
    valid_trees = df_updated[df_updated["tree_id"].notna()]["tree_id"].unique()
    tree_to_group = {tid: i + 1 for i, tid in enumerate(valid_trees)}
    df_updated["group_id"] = (
        df_updated["tree_id"].map(tree_to_group).fillna(0).astype(int)
    )
    return df_updated


def build_parser():
    """Command-line interface for tree construction.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the standalone script.
    """
    parser = argparse.ArgumentParser(
        description=("Build event trees and group ids from a declustered catalog."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m spad.form_event_trees \\\n"
            "      --input results/decluster/catalog_with_aftershock.csv \\\n"
            "      --output results/catalog_with_tree.csv\n"
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help=("Declustered CSV with event_id, nearest_neighbor and passed_filter."),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="CSV path for the catalog with tree_id, parent and group_id.",
    )
    return parser


def main(argv=None):
    """Form event trees from the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, excluding the program name. ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    df = pd.read_csv(args.input)
    events = df.to_dict("records")
    valid_tree_ids = assign_tree_attributes(events)
    df_updated = add_group_ids(pd.DataFrame(events))
    df_updated.to_csv(args.output, index=False)
    print("\nUnique tree identifiers:", valid_tree_ids)
    print("Number of unique trees:", len(valid_tree_ids))
    print(f"Catalog with trees saved to {args.output}")


if __name__ == "__main__":
    main()
