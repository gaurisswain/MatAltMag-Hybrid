from __future__ import annotations

import numpy as np


def precision_at_k(y_true, y_score, k: int) -> float:
    order = np.argsort(-np.asarray(y_score))[:k]
    return float(np.asarray(y_true)[order].mean()) if len(order) else 0.0

