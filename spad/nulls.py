"""Null catalogs for the clustering stage.

Two constructions are provided.

* Epicentre shift. Each event is moved by an isotropic Gaussian
  horizontal vector (``sigma`` kilometres per component) and, in the
  generator, by a uniform depth jitter of ±10 km clipped at zero.
* Uniform. Epicentres are drawn uniformly on the sphere inside the union
  of 75 km horizontal discs about the real epicentres. The generator
  assigns a permutation of the real depths.

When a slab surface is supplied, shift and uniform depths from the
generator are discarded and replaced by the interpolated slab depth.
The generator is still called in full, so the random stream matches a
run that keeps those depths. Bootstrap replicas are not null catalogs;
they keep the depth already chosen for the real events they retain.

Seeds follow the study: shift 50 km uses ``1000 + i``, shift 100 km uses
``2000 + i``, uniform uses ``3000 + i``.
"""

from __future__ import annotations

import numpy as np

from spad.bootstrap import bootstrap_indices
from spad.distance import R_EARTH_KM, SlabSurface, destination_point

SHIFT50_SEED0 = 1000
SHIFT100_SEED0 = 2000
UNIFORM_SEED0 = 3000
SHIFT_DEPTH_KM = 10.0
UNIFORM_BUFFER_KM = 75.0


def shift_seed(index: int, sigma_km: float) -> int:
    """Study seed for one epicentre-shift replica."""
    if float(sigma_km) == 50.0:
        return SHIFT50_SEED0 + int(index)
    if float(sigma_km) == 100.0:
        return SHIFT100_SEED0 + int(index)
    raise ValueError("study seeds are defined for sigma 50 and 100 km")


def uniform_seed(index: int) -> int:
    """Study seed for one uniform-in-region replica."""
    return UNIFORM_SEED0 + int(index)


def null_shift(lat, lon, depth, sigma_km, seed, depth_jitter_km=SHIFT_DEPTH_KM):
    """Shift every epicentre by an isotropic Gaussian vector.

    ``dx`` is east and ``dy`` is north, each ``N(0, sigma_km)``. Depth is
    ``depth + U(-depth_jitter_km, depth_jitter_km)``, clipped at 0.

    When every input longitude is positive, a negative destination
    longitude is increased by 360. That keeps a positive-longitude
    catalog on one side of the antimeridian in the stored numbers.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    depth = np.asarray(depth, dtype=float)
    rng = np.random.default_rng(seed)
    dx = rng.normal(0.0, sigma_km, len(lat))
    dy = rng.normal(0.0, sigma_km, len(lat))
    shifted_lat, shifted_lon = destination_point(
        lat, lon, np.arctan2(dx, dy), np.hypot(dx, dy)
    )
    if (lon > 0).all():
        shifted_lon = np.where(shifted_lon < 0, shifted_lon + 360.0, shifted_lon)
    jitter = rng.uniform(-depth_jitter_km, depth_jitter_km, len(depth))
    shifted_depth = np.clip(depth + jitter, 0, None)
    return shifted_lat, shifted_lon, shifted_depth


def null_uniform(lat, lon, depth, seed, buffer_km=UNIFORM_BUFFER_KM):
    """Uniform epicentres inside the union of horizontal discs.

    Sampling is uniform in longitude and in ``sin(latitude)`` inside a
    bounding box, then rejected unless the haversine distance to the
    nearest real epicentre is at most ``buffer_km``. Depths are a
    permutation of the input depths.
    """
    from sklearn.neighbors import BallTree

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    depth = np.asarray(depth, dtype=float)
    rng = np.random.default_rng(seed)
    tree = BallTree(np.radians(np.column_stack((lat, lon))), metric="haversine")
    pad_lat = buffer_km / 111.0 + 0.2
    cos_scale = np.cos(
        np.radians(max(abs(lat.min()), abs(lat.max())) + pad_lat)
    )
    pad_lon = buffer_km / (111.0 * cos_scale) + 0.2
    sin_south = np.sin(np.radians(lat.min() - pad_lat))
    sin_north = np.sin(np.radians(lat.max() + pad_lat))
    n_events = len(lat)
    accepted_lat = []
    accepted_lon = []
    n_got = 0
    while n_got < n_events:
        batch = 20000
        draw_lat = np.degrees(np.arcsin(rng.uniform(sin_south, sin_north, batch)))
        draw_lon = rng.uniform(lon.min() - pad_lon, lon.max() + pad_lon, batch)
        dist_rad, _ = tree.query(
            np.radians(np.column_stack((draw_lat, draw_lon))), k=1
        )
        keep = dist_rad[:, 0] * R_EARTH_KM <= buffer_km
        accepted_lat.append(draw_lat[keep])
        accepted_lon.append(draw_lon[keep])
        n_got += int(keep.sum())
    out_lat = np.concatenate(accepted_lat)[:n_events]
    out_lon = np.concatenate(accepted_lon)[:n_events]
    return out_lat, out_lon, rng.permutation(depth)


def envelope_area(lat, lon, buffer_km=UNIFORM_BUFFER_KM, n_draw=400000, seed=0):
    """Monte Carlo area (km^2) of the union of horizontal buffers."""
    from sklearn.neighbors import BallTree

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    rng = np.random.default_rng(seed)
    tree = BallTree(np.radians(np.column_stack((lat, lon))), metric="haversine")
    pad_lat = buffer_km / 111.0 + 0.2
    cos_scale = np.cos(
        np.radians(max(abs(lat.min()), abs(lat.max())) + pad_lat)
    )
    pad_lon = buffer_km / (111.0 * cos_scale) + 0.2
    lat0 = np.radians(lat.min() - pad_lat)
    lat1 = np.radians(lat.max() + pad_lat)
    lon0 = np.radians(lon.min() - pad_lon)
    lon1 = np.radians(lon.max() + pad_lon)
    box = R_EARTH_KM**2 * (lon1 - lon0) * (np.sin(lat1) - np.sin(lat0))
    draw_lat = np.arcsin(rng.uniform(np.sin(lat0), np.sin(lat1), n_draw))
    draw_lon = rng.uniform(lon0, lon1, n_draw)
    dist_rad, _ = tree.query(np.column_stack((draw_lat, draw_lon)), k=1)
    inside = np.mean(dist_rad[:, 0] * R_EARTH_KM <= buffer_km)
    return float(box * inside)


def realize_catalog(
    kind: str,
    seed: int,
    lat,
    lon,
    generator_depth,
    kept_depth,
    surface: SlabSurface | None = None,
):
    """Build one real, null, or bootstrap catalog.

    ``generator_depth`` is the depth array passed into the null
    generators (the catalog ``depth`` column in the slab study).
    ``kept_depth`` is the depth used for the real catalog and for
    bootstrap replicas. For ``shift50``, ``shift100`` and ``unif``, a
    slab ``surface`` replaces the generator depth.

    Returns latitude, longitude, depth, bootstrap indices or ``None``,
    and an outside-hull mask or ``None``.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    generator_depth = np.asarray(generator_depth, dtype=float)
    kept_depth = np.asarray(kept_depth, dtype=float)
    if kind == "real":
        return lat, lon, kept_depth, None, None
    if kind == "boot":
        index = bootstrap_indices(len(lat), seed)
        return lat[index], lon[index], kept_depth[index], index, None
    if kind == "shift50":
        new_lat, new_lon, new_depth = null_shift(
            lat, lon, generator_depth, 50.0, seed
        )
    elif kind == "shift100":
        new_lat, new_lon, new_depth = null_shift(
            lat, lon, generator_depth, 100.0, seed
        )
    elif kind == "unif":
        new_lat, new_lon, new_depth = null_uniform(
            lat, lon, generator_depth, seed
        )
    else:
        raise ValueError("unknown catalog kind: " + str(kind))
    if surface is None:
        return new_lat, new_lon, new_depth, None, None
    slab_depth, outside = surface(new_lat, new_lon, return_flag=True)
    return new_lat, new_lon, slab_depth, None, outside
