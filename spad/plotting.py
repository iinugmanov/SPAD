"""Figures for a ``q`` scan, an exceedance table, maps and a filtration tree.

Figures are written to the path that is passed in. Nothing in this
module chooses a ``q``.
"""

from __future__ import annotations

import numpy as np

from spad.filtration import N_LEVELS, condensed_tree, filtration_levels


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _colors():
    plt = _pyplot()
    tab = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()
    dark = plt.cm.Dark2(np.linspace(0, 1, 8)).tolist()
    return tab + dark


def plot_qscan(frame, path, title=None):
    """Birth-set patch count, core patch count and core fraction against ``q``."""
    plt = _pyplot()
    ordered = frame.sort_values("q")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(ordered["q"], ordered["n_old"], "o-", color="C0")
    axes[0].set_ylabel("birth-set patches (excess of mass)")
    axes[1].plot(ordered["q"], ordered["n_rule"], "o-", color="C2")
    axes[1].set_ylabel("plateau-rule patches")
    axes[2].plot(ordered["q"], ordered["core_frac"], "o-", color="C3")
    axes[2].set_ylabel("fraction of events in cores")
    for axis in axes:
        axis.set_xlabel("q")
        axis.grid(alpha=0.3)
    fig.suptitle(title or "Filtration scan")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_qscan_overlay(real_frame, group_frames, path):
    """Real ``q`` scan over per-catalog curves.

    ``group_frames`` maps a group name to a frame with columns ``q``,
    ``i``, ``n_rule`` and ``core_frac``.
    """
    plt = _pyplot()
    names = list(group_frames)
    fig, axes = plt.subplots(
        2,
        max(len(names), 1),
        figsize=(4.6 * max(len(names), 1), 7),
        squeeze=False,
    )
    real = real_frame.sort_values("q")
    for column, name in enumerate(names):
        group = group_frames[name]
        top = axes[0, column]
        bottom = axes[1, column]
        for _, part in group.groupby("i"):
            part = part.sort_values("q")
            top.plot(part["q"], part["n_rule"], "-", color="C0", alpha=0.35, lw=0.8)
            bottom.plot(
                part["q"], part["core_frac"], "-", color="C0", alpha=0.35, lw=0.8
            )
        top.plot(real["q"], real["n_rule"], "k-", lw=2.0, label="real")
        bottom.plot(real["q"], real["core_frac"], "k-", lw=2.0)
        top.set_title(name)
        top.set_ylabel("plateau-rule patches")
        bottom.set_ylabel("core fraction")
        top.legend()
        for axis in (top, bottom):
            axis.set_xlabel("q")
            axis.grid(alpha=0.3)
    fig.suptitle("Per-q curves (thin: ensemble, black: real)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_exceedance(table, path):
    """``z`` against ``q`` for each metric, model and radius variant.

    Infinite ``z`` is omitted from the line (it is still in the table).
    Points with ``p_upper <= 0.05`` are marked.
    """
    plt = _pyplot()
    metrics = list(dict.fromkeys(table["metric"].tolist()))
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(10, 3.2 * len(metrics)), squeeze=False
    )
    for axis, metric in zip(axes[:, 0], metrics):
        subset = table[table["metric"] == metric]
        for (model, variant), part in subset.groupby(["model", "variant"]):
            part = part.sort_values("q")
            z = part["z"].to_numpy(dtype=float)
            finite = np.isfinite(z)
            style = "-" if variant == "sameq" else "--"
            axis.plot(
                part["q"].to_numpy()[finite],
                z[finite],
                style,
                label=f"{model}, {variant}",
            )
            marked = finite & (part["p_upper"].to_numpy() <= 0.05)
            axis.plot(part["q"].to_numpy()[marked], z[marked], "o", ms=3.5)
        axis.axhline(0.0, color="k", lw=0.6)
        axis.set_ylabel("z")
        axis.set_xlabel("q")
        axis.set_title(metric)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_map(lon, lat, depth, labels, path, title, depth_label="depth, km"):
    """Map and longitude-depth section coloured by patch label."""
    plt = _pyplot()
    colors = _colors()
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    depth = np.asarray(depth, dtype=float)
    labels = np.asarray(labels)
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for axis, y_coord in zip(axes, (lat, depth)):
        axis.scatter(lon, y_coord, s=8, c="0.82", linewidths=0)
        if (labels >= 0).any():
            for label in range(int(labels.max()) + 1):
                mask = labels == label
                axis.scatter(
                    lon[mask],
                    y_coord[mask],
                    s=12,
                    color=colors[label % len(colors)],
                    linewidths=0,
                )
    if (labels >= 0).any():
        for label in range(int(labels.max()) + 1):
            mask = labels == label
            if int(mask.sum()) >= 2:
                axes[0].text(
                    lon[mask].mean(),
                    lat[mask].mean(),
                    str(label + 1),
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                )
    axes[0].set_aspect(1.0 / np.cos(np.radians(lat.mean())))
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    axes[0].set_title(title)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("longitude")
    axes[1].set_ylabel(depth_label)
    axes[1].set_title("depth section")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _tree_layout(nodes):
    """Horizontal spans: leaves take width equal to their birth size."""
    spans = {}
    cursor = [0.0]
    gap = 0.02 * max(node["size"] for node in nodes)

    def place(node_id):
        node = nodes[node_id]
        if not node["children"]:
            spans[node_id] = (cursor[0], cursor[0] + node["size"])
            cursor[0] += node["size"] + gap
            return
        children = sorted(node["children"], key=lambda child: -nodes[child]["size"])
        for child in children:
            place(child)
        spans[node_id] = (
            min(spans[child][0] for child in node["children"]),
            max(spans[child][1] for child in node["children"]),
        )

    roots = [node for node in nodes if node["parent"] is None]
    for node in sorted(roots, key=lambda item: -item["size"]):
        place(node["id"])
    return spans


def plot_tree(filtration, nodes, selected, path, m_min=5):
    """Condensed tree. Selected nodes are outlined.

    Bar width is the component size. The vertical axis is alpha.
    """
    from matplotlib.patches import Rectangle

    plt = _pyplot()
    colors = _colors()
    alpha = filtration["alpha"]
    spans = _tree_layout(nodes)
    fig, axis = plt.subplots(figsize=(12, 7))
    selected_index = {node_id: j for j, node_id in enumerate(selected)}

    def selected_ancestor(node_id):
        while node_id is not None:
            if node_id in selected_index:
                return selected_index[node_id]
            node_id = nodes[node_id]["parent"]
        return None

    for node in nodes:
        x0, x1 = spans[node["id"]]
        center = 0.5 * (x0 + x1)
        group = selected_ancestor(node["id"])
        if group is None:
            color = "0.35"
            alpha_face = 0.8
        else:
            color = colors[group % len(colors)]
            alpha_face = 1.0 if node["id"] in selected_index else 0.45
        for step, size in enumerate(node["sizes"]):
            level = node["birth"] + step
            axis.add_patch(
                Rectangle(
                    (center - size / 2.0, alpha[level]),
                    size,
                    alpha[level + 1] - alpha[level],
                    color=color,
                    alpha=alpha_face,
                    lw=0,
                )
            )
        if node["children"]:
            child_centers = [0.5 * sum(spans[child]) for child in node["children"]]
            axis.plot(
                [min(child_centers), max(child_centers)],
                [node["a_death"], node["a_death"]],
                color="k",
                lw=0.8,
            )
        if node["id"] in selected_index:
            axis.add_patch(
                Rectangle(
                    (center - node["size"] / 2.0, node["a_birth"]),
                    node["size"],
                    node["pers"],
                    fill=False,
                    edgecolor=color,
                    lw=1.6,
                )
            )
    xs = [value for span in spans.values() for value in span]
    axis.set_xlim(min(xs) - 1, max(xs) + 1)
    axis.set_ylim(alpha[0], alpha[-1])
    axis.set_xlabel("component size (horizontal span)")
    axis.set_ylabel("alpha")
    axis.set_xticks([])
    axis.set_title(
        f"q={float(filtration['q']):.1f}, r={float(filtration['r']):.2f} km, "
        f"m_min={m_min}, {len(selected)} selected"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_levels(filtration, n_components, nodes, selected, path):
    """Component count and perfect-set size against alpha, with selected bands."""
    plt = _pyplot()
    colors = _colors()
    alpha = filtration["alpha"][:-1]
    n_points = [int(mask.sum()) for mask in filtration["masks"]]
    fig, axis = plt.subplots(figsize=(10, 5))
    n_sel = max(len(selected), 1)
    for index, node_id in enumerate(selected):
        node = nodes[node_id]
        axis.axvspan(
            node["a_birth"],
            node["a_death"],
            ymin=index / n_sel,
            ymax=(index + 1) / n_sel,
            color=colors[index % len(colors)],
            alpha=0.35,
            lw=0,
        )
    axis.step(alpha, n_components, where="post", color="C0", lw=2, label="components")
    axis.set_xlabel("alpha")
    axis.set_ylabel("components of size >= m_min")
    twin = axis.twinx()
    twin.step(alpha, n_points, where="post", color="C3", lw=2, label="|X(alpha)|")
    twin.set_ylabel("events in the alpha-perfect set")
    axis.set_title(
        f"q={float(filtration['q']):.1f}, r={float(filtration['r']):.2f} km"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def tree_at_q(distance, positive, q, n_levels=N_LEVELS, m_min=5, r_fixed=None):
    """Filtration and condensed tree at one ``q``, for the tree figure."""
    filtration = filtration_levels(
        distance, positive, q, n_levels=n_levels, r_override=r_fixed
    )
    nodes, _labels, n_components = condensed_tree(filtration, m_min)
    return filtration, nodes, n_components
