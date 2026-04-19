"""Shared vector math utilities for the RAG package.

Both :mod:`gemma.rag.indexer` and :mod:`gemma.rag.retrieval` need
L2-normalisation before storing / querying embeddings. This module is the
single canonical definition so the two files never drift apart (#12).
"""

from __future__ import annotations

import numpy as np


def normalise(vec: np.ndarray) -> np.ndarray:
    """Return *vec* L2-normalised as float32.

    Zero-norm vectors pass through unchanged so callers never receive NaN.

    Args:
        vec: Input array of any dtype; cast to float32 before normalising.

    Returns:
        float32 ndarray with unit L2 norm, or the cast input when the
        original norm is zero.
    """
    v = vec.astype(np.float32)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm
