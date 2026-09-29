"""Load a catalog for the clustering stage.

``depth="catalog"`` uses the ``depth`` column. ``depth="slab"`` uses
``-z_proj`` (kilometres, positive downward). Rows with ``Aftershock``
equal to true are removed. Rows with non-finite coordinates for the
chosen depth are removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Catalog:
    """Events kept for one clustering run."""

    frame: pd.DataFrame
    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray
    generator_depth: np.ndarray
    n_raw: int
    depth_mode: str


def _rename_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "lat" not in frame.columns and "latitude" in frame.columns:
        rename["latitude"] = "lat"
    if "lon" not in frame.columns and "longitude" in frame.columns:
        rename["longitude"] = "lon"
    if rename:
        frame = frame.rename(columns=rename)
    return frame


def load_catalog(path: str, depth: str = "catalog") -> Catalog:
    """Load ``path`` and apply the background-event coordinate filter.

    Parameters
    ----------
    path :
        CSV path.
    depth :
        ``"catalog"`` or ``"slab"``.
    """
    if depth not in ("catalog", "slab"):
        raise ValueError("depth must be 'catalog' or 'slab'")
    frame = pd.read_csv(path)
    n_raw = len(frame)
    if "Aftershock" in frame.columns:
        frame = frame.loc[frame["Aftershock"].ne(True)].reset_index(drop=True)
    frame = _rename_coordinates(frame)
    missing = [name for name in ("lat", "lon") if name not in frame.columns]
    if missing:
        raise ValueError("catalog is missing columns: " + ", ".join(missing))
    if depth == "slab":
        if "z_proj" not in frame.columns:
            raise ValueError("slab depth requires a z_proj column")
        finite = np.isfinite(frame[["lat", "lon", "z_proj"]].to_numpy(dtype=float))
        frame = frame.loc[finite.all(axis=1)].reset_index(drop=True)
        used = -frame["z_proj"].to_numpy(dtype=float)
        if "depth" in frame.columns:
            generator = frame["depth"].to_numpy(dtype=float)
        else:
            generator = used.copy()
    else:
        if "depth" not in frame.columns:
            raise ValueError("catalog depth requires a depth column")
        finite = np.isfinite(frame[["lat", "lon", "depth"]].to_numpy(dtype=float))
        frame = frame.loc[finite.all(axis=1)].reset_index(drop=True)
        used = frame["depth"].to_numpy(dtype=float)
        generator = used.copy()
    lat = frame["lat"].to_numpy(dtype=float)
    lon = frame["lon"].to_numpy(dtype=float)
    return Catalog(
        frame=frame,
        lat=lat,
        lon=lon,
        depth=used,
        generator_depth=generator,
        n_raw=n_raw,
        depth_mode=depth,
    )
