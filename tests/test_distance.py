"""Distance symmetry, a known pair, and the slab-surface interpolant."""

import numpy as np

from spad.distance import SlabSurface, pairwise_distance


def test_known_pair_and_symmetry():
    lat = np.array([55.0, 56.0, 55.4])
    lon = np.array([160.0, 161.0, 160.5])
    depth = np.array([30.0, 45.0, 20.0])
    distance = pairwise_distance(lat, lon, depth)
    assert distance.shape == (3, 3)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    # Independent haversine plus depth, Earth radius 6371 km.
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = (lat_r[1] - lat_r[0]) / 2.0
    dlon = (lon_r[1] - lon_r[0]) / 2.0
    hav = (
        np.sin(dlat) ** 2
        + np.cos(lat_r[0]) * np.cos(lat_r[1]) * np.sin(dlon) ** 2
    )
    hav = float(np.clip(hav, 0.0, 1.0))
    horizontal = 6371.0 * 2.0 * np.arcsin(np.sqrt(hav))
    expected = np.sqrt(horizontal**2 + (45.0 - 30.0) ** 2)
    assert distance[0, 1] == expected
    assert abs(distance[0, 1] - 128.6669049938575) < 1e-9


def test_triangle_inequality():
    rng = np.random.default_rng(4)
    lat = 50.0 + rng.normal(0, 0.4, 8)
    lon = 160.0 + rng.normal(0, 0.4, 8)
    depth = 30.0 + rng.normal(0, 5.0, 8)
    distance = pairwise_distance(lat, lon, depth)
    for i in range(8):
        for j in range(8):
            via = distance[i, :] + distance[:, j]
            assert np.all(distance[i, j] <= via + 1e-6)


def test_slab_surface_linear_and_nearest():
    lat = np.array([0.0, 0.0, 1.0])
    lon = np.array([0.0, 1.0, 0.0])
    depth = np.array([10.0, 20.0, 30.0])
    surface = SlabSurface(lat, lon, depth)
    # The three input epicentres are recovered.
    back = surface(lat, lon)
    assert np.allclose(back, depth)
    # A point far from the hull receives a nearest-neighbour depth.
    far, outside = surface(np.array([10.0]), np.array([10.0]), return_flag=True)
    assert bool(outside[0])
    assert far[0] in (10.0, 20.0, 30.0)
