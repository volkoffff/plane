"""Geometric collision queries, independent of simulation and rendering."""

from __future__ import annotations

import numpy as np


def segment_box_entry(
    start: np.ndarray,
    end: np.ndarray,
    half_extents: tuple[float, float, float],
) -> float | None:
    """Return the first segment fraction in [0, 1] inside a box at the origin.

    Coordinates must be expressed in the box's local frame. Faces are included.
    Parallel segments are checked separately to avoid divisions by zero.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    extents = np.asarray(half_extents, dtype=float)
    if any(vector.shape != (3,) for vector in (start, end, extents)):
        raise ValueError("Collision vectors must have three components")
    if not all(np.isfinite(vector).all() for vector in (start, end, extents)):
        raise ValueError("Collision vectors must be finite")
    if np.any(extents <= 0.0):
        raise ValueError("Box half-extents must be positive")

    entry, exit = 0.0, 1.0
    for origin, delta, extent in zip(start, end - start, extents):
        if abs(delta) < 1e-12:
            if origin < -extent or origin > extent:
                return None
            continue

        first = (-extent - origin) / delta
        second = (extent - origin) / delta
        entry = max(entry, min(first, second))
        exit = min(exit, max(first, second))
        if entry > exit:
            return None

    return float(entry)
