"""Bootstrap subsamples of a catalog.

Each replica keeps 80% of the events, drawn without replacement. Indices
are returned in increasing order. The study seeds are ``4000 + i``.
"""

from __future__ import annotations

import numpy as np

BOOTSTRAP_FRACTION = 0.8
BOOTSTRAP_SEED0 = 4000


def bootstrap_seed(index: int) -> int:
    """Seed of bootstrap replica ``index`` (``4000 + index``)."""
    return BOOTSTRAP_SEED0 + int(index)


def bootstrap_indices(
    n_events: int,
    seed: int,
    fraction: float = BOOTSTRAP_FRACTION,
) -> np.ndarray:
    """Sorted indices of one subsample without replacement."""
    rng = np.random.default_rng(seed)
    size = round(fraction * n_events)
    return np.sort(rng.choice(n_events, size, replace=False))
