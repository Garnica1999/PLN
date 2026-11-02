"""El nivel ajustado debe producir intervalos más anchos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.scripts.metrics_breakdown import paired_bootstrap


def test_alpha_menor_ensancha_el_intervalo() -> None:
    variant = pd.Series([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
    baseline = pd.Series([0, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    _, low_95, high_95 = paired_bootstrap(variant, baseline, np.random.default_rng(42), 0.05)
    _, low_99, high_99 = paired_bootstrap(variant, baseline, np.random.default_rng(42), 0.01)
    assert (high_99 - low_99) >= (high_95 - low_95)
