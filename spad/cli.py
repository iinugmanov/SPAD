"""Command line for the filtration clustering stage.

Subcommands:

* ``qscan`` solves the real catalog on a ``q`` grid.
* ``ensemble`` solves null and bootstrap replicas at fixed seeds.
* ``exceedance`` builds the ``z`` and empirical-``p`` table from an
  ensemble directory, and the bootstrap reproducibility table when
  bootstrap replicas are present.
* ``patches`` extracts birth sets and cores at one user-given ``q``.

The max-patches helper and the plateau helper are reported by ``qscan``.
They do not select a best ``q``. Choose ``q`` from the scan, from the
exceedance table and from the bootstrap table.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from spad.catalog import load_catalog
from spad.diagnostics import (
    Q_PLATEAU,
    apply_plateau,
    bootstrap_reproducibility,
    exceedance_table,
    max_patches,
    metric_rows_from_run,
    patch_summary,
    q_grid,
    qscan_frame,
    same_grid,
)
from spad.distance import distance_matrix
from spad.ensemble import (
    attach_real_radius,
    ensemble_job,
    iter_jobs,
    load_record,
    member_path,
    study_seed,
)
from spad.filtration import scan_q, solve_q


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spad.cli",
        description=(
            "Alpha-filtration of the DPS density. "
            "q is chosen by the user. The max-patches and plateau "
            "summaries are optional helpers, not a best-q rule."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    qscan = sub.add_parser(
        "qscan",
        help="solve a q grid on one catalog",
        description=(
            "Solve the filtration at each q. "
            "The printed max-patches and plateau lines are helpers. "
            "They are not a selected best q. "
            "The max-patches helper uses the step-0.2 grid from -4 to -1. "
            "The plateau helper is defined on the step-0.1 grid from -4 to -1; "
            "on any other grid the same comparison is only a description "
            "of that scan."
        ),
    )
    _add_catalog_args(qscan)
    _add_grid_args(qscan)
    qscan.add_argument(
        "--output",
        default=None,
        help="CSV of the per-q table (optional)",
    )
    qscan.add_argument(
        "--plot",
        default=None,
        help="path of a q-scan figure (optional, not written unless set)",
    )

    ensemble = sub.add_parser(
        "ensemble",
        help="null and bootstrap replicas at fixed seeds",
        description=(
            "Seeds: shift 50 km = 1000+i, shift 100 km = 2000+i, "
            "uniform in the 75 km envelope = 3000+i, "
            "bootstrap 80 percent without replacement = 4000+i. "
            "Nulls are solved at their own r(q) and at the real r(q). "
            "Bootstrap replicas are solved at their own r(q). "
            "One pickle per replica is written under --out-dir. "
            "Pickles are local outputs."
        ),
    )
    _add_catalog_args(ensemble)
    _add_grid_args(ensemble)
    ensemble.add_argument("--out-dir", required=True, help="directory for pickles")
    ensemble.add_argument("--n-shift50", type=int, default=0)
    ensemble.add_argument("--n-shift100", type=int, default=0)
    ensemble.add_argument("--n-unif", type=int, default=0)
    ensemble.add_argument("--n-boot", type=int, default=0)
    ensemble.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes (spawn). 1 runs in this process",
    )
    ensemble.add_argument(
        "--skip-existing",
        action="store_true",
        help="keep a pickle that is already present",
    )

    exceed = sub.add_parser(
        "exceedance",
        help="z and empirical p from an ensemble directory",
        description=(
            "Reads the pickles written by the ensemble command. "
            "z = (real - null mean) / null standard deviation (ddof=1). "
            "p_upper = (1 + count of nulls >= real) / (n_null + 1). "
            "Bootstrap reproducibility compares plateau-rule cores at the "
            "same q. This table does not select a q."
        ),
    )
    exceed.add_argument("--runs", required=True, help="directory of ensemble pickles")
    exceed.add_argument("--output", default=None, help="exceedance CSV (optional)")
    exceed.add_argument(
        "--bootstrap-output",
        default=None,
        help="bootstrap reproducibility CSV (optional)",
    )
    exceed.add_argument("--plot", default=None, help="z-curve figure (optional)")
    exceed.add_argument(
        "--overlay-plot",
        default=None,
        help="q-scan overlay figure (optional)",
    )

    patches = sub.add_parser(
        "patches",
        help="extract patches at one q",
        description=(
            "Solve one q given by the user. "
            "Writes birth-set labels from excess of mass and core labels "
            "from the plateau rule at that same q. "
            "Passing one q does not invoke the plateau's choice of q."
        ),
    )
    _add_catalog_args(patches)
    patches.add_argument("--q", type=float, required=True, help="exponent q < 0")
    patches.add_argument(
        "--events",
        default=None,
        help="CSV of events with patch labels (optional)",
    )
    patches.add_argument(
        "--patches",
        default=None,
        help="CSV of patch summaries (optional). Default: stdout only",
    )
    patches.add_argument("--plot-map", default=None, help="map of cores (optional)")
    patches.add_argument(
        "--plot-birth",
        default=None,
        help="map of excess-of-mass birth sets (optional)",
    )
    patches.add_argument("--plot-tree", default=None, help="condensed tree (optional)")
    patches.add_argument(
        "--plot-levels",
        default=None,
        help="alpha-level figure (optional)",
    )
    return parser


def _add_catalog_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", required=True, help="CSV catalog")
    parser.add_argument(
        "--depth",
        choices=("catalog", "slab"),
        default="catalog",
        help=(
            "catalog: depth column. "
            "slab: -z_proj, and null depths from the slab-surface interpolant"
        ),
    )


def _add_grid_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--q-min", type=float, default=-4.0)
    parser.add_argument("--q-max", type=float, default=-1.0)
    parser.add_argument(
        "--q-step",
        type=float,
        default=0.1,
        help="default 0.1 (plateau grid). Use 0.2 for the max-patches grid",
    )


def _require_events(catalog) -> None:
    if len(catalog.lat) < 2:
        raise SystemExit("need at least two events with finite coordinates")


def cmd_qscan(args) -> None:
    loaded = load_catalog(args.catalog, depth=args.depth)
    _require_events(loaded)
    grid = q_grid(args.q_min, args.q_max, args.q_step)
    distance, positive = distance_matrix(loaded.lat, loaded.lon, loaded.depth)
    solutions = scan_q(distance, positive, grid)
    frame = qscan_frame(solutions)
    _print_frame(frame)
    print(
        f"\nN = {len(loaded.lat)} (rows read {loaded.n_raw}), "
        f"depth = {loaded.depth_mode}"
    )
    print(
        "Helpers below do not select a best q. "
        "Read the table, and the exceedance and bootstrap diagnostics."
    )
    chosen = max_patches(solutions)
    if chosen is None:
        print(
            "Max-patches helper: none of the step-0.2 grid "
            "(-4 to -1) is in this scan."
        )
    else:
        print(
            "Max-patches helper "
            f"(grid -4..-1 step 0.2): q={chosen['q']} "
            f"r={chosen['r']:.2f} km, {chosen['n_old']} birth-set patches, "
            f"{chosen['ev_old']} events."
        )
    plateau = apply_plateau(solutions)
    frozen = same_grid(grid, Q_PLATEAU)
    if not frozen:
        print(
            "Plateau helper on this scan (the frozen helper grid is "
            "-4 to -1 step 0.1; this scan is different):"
        )
    else:
        print("Plateau helper (grid -4..-1 step 0.1, J>=0.8, life>=3):")
    if plateau["status"] != "OK":
        print("  no valid plateau")
    else:
        sol = solutions[plateau["sol_index"]]
        sizes = [item["n_core"] for item in sol["info"]]
        print(
            f"  q={plateau['q']} r={sol['r']:.2f} km, "
            f"plateau {plateau['q_lo']}..{plateau['q_hi']} "
            f"L={plateau['L']} edge={plateau['edge']}, "
            f"{plateau['n']} patches, core sizes {sizes}, "
            f"core events {int((sol['core'] >= 0).sum())}"
        )
    if args.output:
        frame.to_csv(args.output, index=False)
        print("wrote", args.output)
    if args.plot:
        from spad.plotting import plot_qscan

        plot_qscan(frame, args.plot, title="Filtration q scan")
        print("wrote", args.plot)


def cmd_ensemble(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    grid = q_grid(args.q_min, args.q_max, args.q_step)
    real_spec = {
        "kind": "real",
        "i": 0,
        "seed": study_seed("real", 0),
        "catalog": args.catalog,
        "depth": args.depth,
        "q_values": grid,
        "path": member_path(args.out_dir, "real", 0),
        "skip_existing": args.skip_existing,
    }
    print("solving real catalog", flush=True)
    real_path = ensemble_job(real_spec)
    real = load_record(real_path)
    print(f"real N={real['n']} -> {real_path}", flush=True)
    counts = {
        "shift50": args.n_shift50,
        "shift100": args.n_shift100,
        "unif": args.n_unif,
        "boot": args.n_boot,
    }
    jobs = iter_jobs(
        args.out_dir,
        args.catalog,
        args.depth,
        grid,
        counts,
        skip_existing=args.skip_existing,
    )
    attach_real_radius(jobs, real)
    if not jobs:
        print("no replicas requested")
        return
    if args.workers <= 1:
        for job in jobs:
            path = ensemble_job(job)
            print("wrote", path, flush=True)
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=3) as pool:
        for path in pool.imap_unordered(ensemble_job, jobs):
            print("wrote", path, flush=True)


def cmd_exceedance(args) -> None:
    paths = sorted(glob.glob(os.path.join(args.runs, "*.pkl")))
    if not paths:
        raise SystemExit("no pickles in " + args.runs)
    records = [load_record(path) for path in paths]
    rows = []
    for record in records:
        rows.extend(metric_rows_from_run(record))
    table = exceedance_table(rows)
    if table.empty:
        print("no null replicas; exceedance table is empty")
    else:
        _print_frame(table)
    if args.output and not table.empty:
        table.to_csv(args.output, index=False)
        print("wrote", args.output)
    real = next((record for record in records if record["kind"] == "real"), None)
    boots = [record for record in records if record["kind"] == "boot"]
    boot_frame = None
    if real is not None and boots:
        boot_frame = bootstrap_reproducibility(real["solutions"], boots)
        print("\nbootstrap reproducibility (plateau-rule cores, same q)")
        _print_frame(boot_frame)
        if args.bootstrap_output:
            boot_frame.to_csv(args.bootstrap_output, index=False)
            print("wrote", args.bootstrap_output)
    if args.plot and not table.empty:
        from spad.plotting import plot_exceedance

        plot_exceedance(table, args.plot)
        print("wrote", args.plot)
    if args.overlay_plot and real is not None:
        from spad.plotting import plot_qscan, plot_qscan_overlay

        real_frame = qscan_frame(real["solutions"])
        groups = {}
        for kind in ("shift50", "shift100", "unif", "boot"):
            members = [record for record in records if record["kind"] == kind]
            if not members:
                continue
            parts = []
            for record in members:
                part = qscan_frame(record["solutions"])
                part = part.copy()
                part["i"] = record["i"]
                parts.append(part)
            groups[kind] = pd.concat(parts, ignore_index=True)
        if groups:
            plot_qscan_overlay(real_frame, groups, args.overlay_plot)
            print("wrote", args.overlay_plot)
        else:
            plot_qscan(real_frame, args.overlay_plot)
            print("wrote", args.overlay_plot)


def cmd_patches(args) -> None:
    loaded = load_catalog(args.catalog, depth=args.depth)
    _require_events(loaded)
    distance, positive = distance_matrix(loaded.lat, loaded.lon, loaded.depth)
    sol = solve_q(distance, positive, args.q)
    print(
        f"q={sol['q']} r={sol['r']:.4f} km  "
        f"birth-set patches {sol['n_old']} events {sol['ev_old']}  "
        f"plateau-rule patches {sol['n']} "
        f"core events {int((sol['core'] >= 0).sum())} "
        f"core fraction {(sol['core'] >= 0).mean():.4f}"
    )
    print("core sizes", [item["n_core"] for item in sol["info"]])
    birth_sizes = [item["n_birth"] for item in sol["info"]]
    print("birth-set sizes of the plateau-rule nodes", birth_sizes)
    cores = patch_summary(loaded.lat, loaded.lon, loaded.depth, sol["core"])
    births = patch_summary(loaded.lat, loaded.lon, loaded.depth, sol["lab_old"])
    if len(cores):
        print("\nplateau-rule cores at this q")
        _print_frame(cores)
    if len(births):
        print("\nexcess-of-mass birth sets at this q")
        _print_frame(births)
    if args.patches:
        cores.to_csv(args.patches, index=False)
        print("wrote", args.patches)
    if args.events:
        events = loaded.frame.copy()
        events["depth_used"] = loaded.depth
        events["eom_patch"] = sol["lab_old"].astype(int) + 1
        events["core_patch"] = sol["core"].astype(int) + 1
        events["rule_birth_patch"] = sol["birth"].astype(int) + 1
        events.to_csv(args.events, index=False)
        print("wrote", args.events)
    depth_label = (
        "slab depth (-z_proj), km" if args.depth == "slab" else "depth, km"
    )
    if args.plot_map or args.plot_birth or args.plot_tree or args.plot_levels:
        from spad.plotting import plot_levels, plot_map, plot_tree, tree_at_q

        if args.plot_map:
            plot_map(
                loaded.lon,
                loaded.lat,
                loaded.depth,
                sol["core"],
                args.plot_map,
                f"cores at q={sol['q']}, r={sol['r']:.2f} km",
                depth_label=depth_label,
            )
            print("wrote", args.plot_map)
        if args.plot_birth:
            plot_map(
                loaded.lon,
                loaded.lat,
                loaded.depth,
                sol["lab_old"],
                args.plot_birth,
                f"birth sets at q={sol['q']}, r={sol['r']:.2f} km",
                depth_label=depth_label,
            )
            print("wrote", args.plot_birth)
        if args.plot_tree or args.plot_levels:
            filtration, nodes, n_components = tree_at_q(
                distance, positive, args.q
            )
            from spad.filtration import min_life_steps, select_rule

            selected = select_rule(nodes, min_life_steps(len(filtration["masks"])))
            if args.plot_tree:
                plot_tree(filtration, nodes, selected, args.plot_tree)
                print("wrote", args.plot_tree)
            if args.plot_levels:
                plot_levels(filtration, n_components, nodes, selected, args.plot_levels)
                print("wrote", args.plot_levels)


def _print_frame(frame) -> None:
    with pd.option_context(
        "display.max_rows",
        200,
        "display.width",
        200,
        "display.max_columns",
        40,
    ):
        text = frame.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
        print(text)


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    commands = {
        "qscan": cmd_qscan,
        "ensemble": cmd_ensemble,
        "exceedance": cmd_exceedance,
        "patches": cmd_patches,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
