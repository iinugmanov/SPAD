"""Null and bootstrap ensembles.

Each replica has a fixed seed (see ``spad.nulls`` and ``spad.bootstrap``).
The real catalog is solved first. Null catalogs are solved twice: once
at their own ``r(q)`` and once at the real catalog's ``r`` on the same
``q`` grid. Bootstrap replicas are solved only at their own ``r(q)``.
One pickle is written per replica. Those files are local outputs.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from spad.bootstrap import bootstrap_seed
from spad.catalog import load_catalog
from spad.diagnostics import q_grid
from spad.distance import SlabSurface, distance_matrix
from spad.filtration import scan_q
from spad.nulls import realize_catalog, shift_seed, uniform_seed


def study_seed(kind: str, index: int) -> int:
    """Seed used for replica ``index`` of ``kind``."""
    if kind == "real":
        return 0
    if kind == "shift50":
        return shift_seed(index, 50.0)
    if kind == "shift100":
        return shift_seed(index, 100.0)
    if kind == "unif":
        return uniform_seed(index)
    if kind == "boot":
        return bootstrap_seed(index)
    raise ValueError("unknown catalog kind: " + str(kind))


def _solve_realized(lat, lon, depth, q_values, r_list=None):
    distance, positive = distance_matrix(lat, lon, depth)
    return scan_q(distance, positive, q_values, r_list=r_list)


def run_member(spec: dict) -> dict:
    """Solve one catalog described by ``spec`` and return the record.

    ``spec`` keys: ``kind``, ``i``, ``seed``, ``catalog``, ``depth``,
    ``q_values``, and optionally ``real_r`` (a list, nulls only).
    """
    loaded = load_catalog(spec["catalog"], depth=spec["depth"])
    surface = None
    if spec["depth"] == "slab" and spec["kind"] in ("shift50", "shift100", "unif"):
        surface = SlabSurface(loaded.lat, loaded.lon, loaded.depth)
    lat, lon, depth, index, outside = realize_catalog(
        spec["kind"],
        spec["seed"],
        loaded.lat,
        loaded.lon,
        loaded.generator_depth,
        loaded.depth,
        surface=surface,
    )
    q_values = np.asarray(spec["q_values"], dtype=float)
    record = {
        "kind": spec["kind"],
        "i": int(spec["i"]),
        "seed": int(spec["seed"]),
        "n": len(lat),
        "lat": np.asarray(lat),
        "lon": np.asarray(lon),
        "depth": np.asarray(depth),
        "idx": None if index is None else np.asarray(index),
        "outside": None if outside is None else np.asarray(outside),
        "solutions": _solve_realized(lat, lon, depth, q_values),
    }
    real_r = spec.get("real_r")
    if real_r is not None and spec["kind"] not in ("real", "boot"):
        record["solutions_fixed_r"] = _solve_realized(
            lat, lon, depth, q_values, r_list=list(real_r)
        )
    return record


def write_record(record: dict, path: str) -> None:
    """Write ``record`` to ``path`` via a temporary file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(record, handle, protocol=4)
    os.replace(temporary, path)


def ensemble_job(spec: dict) -> str:
    """Worker entry point. Writes ``spec["path"]`` and returns it."""
    if spec.get("skip_existing") and os.path.exists(spec["path"]):
        return spec["path"]
    record = run_member(spec)
    write_record(record, spec["path"])
    return spec["path"]


def member_path(out_dir: str, kind: str, index: int) -> str:
    """Pickle path for one replica."""
    return os.path.join(out_dir, f"{kind}_{index:03d}.pkl")


def load_record(path: str) -> dict:
    """Load one ensemble pickle."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def iter_jobs(out_dir, catalog, depth, q_values, counts, skip_existing=False):
    """Job specs. The real catalog is not included; it is run by the parent."""
    plan = (
        ("boot", counts.get("boot", 0)),
        ("shift50", counts.get("shift50", 0)),
        ("shift100", counts.get("shift100", 0)),
        ("unif", counts.get("unif", 0)),
    )
    jobs = []
    for kind, n_rep in plan:
        for index in range(int(n_rep)):
            jobs.append(
                {
                    "kind": kind,
                    "i": index,
                    "seed": study_seed(kind, index),
                    "catalog": catalog,
                    "depth": depth,
                    "q_values": np.asarray(q_values, dtype=float),
                    "path": member_path(out_dir, kind, index),
                    "skip_existing": skip_existing,
                }
            )
    return jobs


def attach_real_radius(jobs, real_record):
    """Give null jobs the real catalog's radius at each ``q``."""
    radii = [sol["r"] for sol in real_record["solutions"]]
    for job in jobs:
        if job["kind"] not in ("real", "boot"):
            job["real_r"] = list(radii)
    return jobs


def default_q_values():
    """Plateau grid, -4 to -1 in steps of 0.1."""
    return q_grid(-4.0, -1.0, 0.1)
