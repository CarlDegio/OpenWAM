"""Piper joint14 <-> padded joint20 contract (not an EEF representation)."""

from __future__ import annotations

import numpy as np

REPRESENTATION = "piper_joint20_v1"
FEATURE_NAMES = tuple(
    f"{arm}_{joint}" for arm in ("left", "right") for joint in (*[f"j{i}" for i in range(1, 7)], "gripper")
)
# Native [L q1..q6, L grip, R q1..q6, R grip] -> padded two 10-D blocks.
JOINT14_IN_20 = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19], dtype=np.int64)
PADDING20 = np.array([6, 7, 8, 16, 17, 18], dtype=np.int64)
JOINT20_IN_80 = np.array([*range(10), *range(34, 44)], dtype=np.int64)
DIM_MASK20 = np.zeros(20, dtype=bool)
DIM_MASK20[JOINT14_IN_20] = True


def pack_joint14(values) -> np.ndarray:
    """Pad native joint targets/state, preserving units and signed gripper values."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0 or arr.shape[-1] != 14:
        raise ValueError(f"Expected (..., 14) Piper joint values, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Piper joint values must be finite")
    padded = np.zeros((*arr.shape[:-1], 20), dtype=np.float32)
    padded[..., JOINT14_IN_20] = arr
    return padded


def unpack_joint20(values) -> np.ndarray:
    """Extract native joint14 commands from a physical (unnormalized) joint20 output."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0 or arr.shape[-1] != 20:
        raise ValueError(f"Expected (..., 20) padded Piper joint values, got {arr.shape}")
    return arr[..., JOINT14_IN_20].copy()
