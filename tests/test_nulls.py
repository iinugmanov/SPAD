"""Null catalogs and bootstrap draws are reproducible at a fixed seed."""

import numpy as np

from spad.bootstrap import bootstrap_indices
from spad.nulls import null_shift, null_uniform, realize_catalog


def _catalog():
    rng = np.random.default_rng(5)
    lat = 52.0 + rng.normal(0, 0.2, 30)
    lon = 158.0 + rng.normal(0, 0.2, 30)
    depth = 20.0 + rng.uniform(0, 30, 30)
    return lat, lon, depth


def test_shift_and_uniform_are_seed_stable():
    lat, lon, depth = _catalog()
    for sigma, seed in ((50.0, 1000), (100.0, 2000)):
        first = null_shift(lat, lon, depth, sigma, seed)
        second = null_shift(lat, lon, depth, sigma, seed)
        for left, right in zip(first, second):
            assert np.array_equal(left, right)
    other = null_shift(lat, lon, depth, 50.0, 1001)
    assert not np.allclose(other[0], null_shift(lat, lon, depth, 50.0, 1000)[0])
    uniform_a = null_uniform(lat, lon, depth, 3000)
    uniform_b = null_uniform(lat, lon, depth, 3000)
    for left, right in zip(uniform_a, uniform_b):
        assert np.array_equal(left, right)
    # Permuted depths are the same multiset.
    assert np.array_equal(np.sort(uniform_a[2]), np.sort(depth))


def test_bootstrap_indices():
    first = bootstrap_indices(40, 4000)
    second = bootstrap_indices(40, 4000)
    assert np.array_equal(first, second)
    assert list(first) == sorted(first)
    assert len(first) == round(0.8 * 40)
    assert len(np.unique(first)) == len(first)
    assert not np.array_equal(first, bootstrap_indices(40, 4001))


def test_realize_catalog_repeats():
    lat, lon, depth = _catalog()
    for kind, seed in (("shift50", 1000), ("unif", 3000), ("boot", 4000)):
        left = realize_catalog(kind, seed, lat, lon, depth, depth)
        right = realize_catalog(kind, seed, lat, lon, depth, depth)
        for item_l, item_r in zip(left, right):
            if item_l is None:
                assert item_r is None
            else:
                assert np.array_equal(item_l, item_r)
    real = realize_catalog("real", 0, lat, lon, depth, depth)
    assert np.array_equal(real[0], lat)
    assert real[3] is None
