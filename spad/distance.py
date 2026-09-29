"""Distances for the clustering stage.

The horizontal separation is a great-circle haversine on latitude and
longitude with Earth radius 6371 km. Depth is combined in quadrature.
Depth may be the catalog depth or the slab-surface depth ``-z_proj``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import jit, prange

R_EARTH_KM = 6371.0
_WGS84_A = 6378137.0
_WGS84_E = 0.0818191908426


@jit(nopython=True, parallel=True)
def pairwise_distance(lat, lon, depth):
    """Pairwise 3D distance in kilometres.

    ``d = sqrt(haversine(lat, lon)^2 + (depth_i - depth_j)^2)``.
    The haversine argument is clipped to ``[0, 1]``. ``lat`` and ``lon``
    are in degrees, ``depth`` in kilometres. The diagonal is zero.
    """
    n = lat.shape[0]
    out = np.zeros((n, n))
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    for i in prange(n):
        for j in range(n):
            dlat = (lat_rad[j] - lat_rad[i]) / 2.0
            dlon = (lon_rad[j] - lon_rad[i]) / 2.0
            hav = (
                np.sin(dlat) ** 2
                + np.cos(lat_rad[i])
                * np.cos(lat_rad[j])
                * np.sin(dlon) ** 2
            )
            hav = min(max(hav, 0.0), 1.0)
            horizontal = R_EARTH_KM * 2.0 * np.arcsin(np.sqrt(hav))
            dz = abs(depth[j] - depth[i])
            out[i, j] = np.sqrt(horizontal * horizontal + dz * dz)
    return out


def positive_distances(distance):
    """Upper-triangle entries that are strictly positive."""
    upper = distance[np.triu_indices_from(distance, k=1)]
    return upper[upper > 0]


def power_mean_radius(distances, q):
    """Power mean of positive pairwise distances at exponent ``q`` < 0.

    ``r(q) = (mean of d^q) ** (1/q)``.
    """
    return np.mean(distances**q) ** (1.0 / q)


def as_coordinate_arrays(lat, lon, depth):
    """Contiguous float arrays, the layout the distance kernel expects."""
    return (
        np.ascontiguousarray(lat, dtype=float),
        np.ascontiguousarray(lon, dtype=float),
        np.ascontiguousarray(depth, dtype=float),
    )


def distance_matrix(lat, lon, depth):
    """Pairwise distance and the positive upper-triangle distances."""
    lat_c, lon_c, depth_c = as_coordinate_arrays(lat, lon, depth)
    matrix = pairwise_distance(lat_c, lon_c, depth_c)
    return matrix, positive_distances(matrix)


def destination_point(lat, lon, bearing, distance_km):
    """Great-circle destination.

    ``lat`` and ``lon`` are degrees. ``bearing`` is radians clockwise from
    north. ``distance_km`` is kilometres. Returned longitude is in
    ``(-180, 180]``.
    """
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    angular = np.asarray(distance_km, dtype=float) / R_EARTH_KM
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular)
        + np.cos(lat1) * np.sin(angular) * np.cos(bearing)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(angular) * np.cos(lat1),
        np.cos(angular) - np.sin(lat1) * np.sin(lat2),
    )
    lon_deg = (np.degrees(lon2) + 540.0) % 360.0 - 180.0
    return np.degrees(lat2), lon_deg


def inverse_mercator_3395(x_m, y_m):
    """Inverse Web Mercator (EPSG:3395) to latitude and longitude, degrees.

    This is a coordinate check. The clustering distance uses the catalog
    latitude and longitude, not this inverse.
    """
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    lon = np.degrees(x_m / _WGS84_A)
    t = np.exp(-y_m / _WGS84_A)
    phi = np.pi / 2.0 - 2.0 * np.arctan(t)
    for _ in range(15):
        sin_phi = np.sin(phi)
        factor = ((1.0 - _WGS84_E * sin_phi) / (1.0 + _WGS84_E * sin_phi)) ** (
            _WGS84_E / 2.0
        )
        phi = np.pi / 2.0 - 2.0 * np.arctan(t * factor)
    return np.degrees(phi), lon


class SlabSurface:
    """Slab depth at an epicentre.

    Linear interpolation of the real events' ``-z_proj`` in local
    kilometres (equirectangular about the mean latitude). Outside the
    convex hull the nearest real epicentre is used. Epicentres that fall
    on the same local-kilometre point after rounding to 0.001 km are
    averaged first.
    """

    def __init__(self, lat, lon, depth_slab):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        depth_slab = np.asarray(depth_slab, dtype=float)
        self.lat0 = np.radians(np.mean(lat))
        self.lon0 = np.mean(lon)
        points = self.xy(lat, lon)
        table = pd.DataFrame(
            {
                "x": np.round(points[:, 0], 3),
                "y": np.round(points[:, 1], 3),
                "z": depth_slab,
            }
        )
        grouped = table.groupby(["x", "y"]).z.mean().reset_index()
        known = grouped[["x", "y"]].values
        self._linear = LinearNDInterpolator(known, grouped.z.values)
        self._nearest = NearestNDInterpolator(known, grouped.z.values)

    def xy(self, lat, lon):
        """Local east and north kilometres."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        east = R_EARTH_KM * np.radians(lon - self.lon0) * np.cos(self.lat0)
        north = R_EARTH_KM * np.radians(lat)
        return np.column_stack((east, north))

    def __call__(self, lat, lon, return_flag=False):
        points = self.xy(lat, lon)
        depth = self._linear(points)
        outside = ~np.isfinite(depth)
        depth = np.array(depth, dtype=float, copy=True)
        if np.any(outside):
            depth[outside] = self._nearest(points[outside])
        if return_flag:
            return depth, outside
        return depth
