import pandas as pd
import numpy as np

from src.models.pca import PCAFactor
from src.models.carry_momo_season import CarryMomentumSeasonality


def _toy_X():
    idx = pd.bdate_range("2020-01-01", periods=200)
    data = {
        "e2-e1": np.sin(np.linspace(0, 10, len(idx))) + np.random.normal(0, 0.05, len(idx)),
        "e3-e2": np.cos(np.linspace(0, 8, len(idx))) + np.random.normal(0, 0.05, len(idx)),
        "e4-e3": np.random.normal(0, 0.1, len(idx)),
    }
    X = pd.DataFrame(data, index=idx)
    return X


def test_pca_fit_transform_health():
    X = _toy_X()
    p = PCAFactor(n_components=2, ewma_span=30)
    p.fit(X)
    Z = p.transform(X)
    assert Z.shape[1] == 2
    S = p.signal(Z)
    assert S.shape == Z.shape
    rep = p.health(Z)
    assert rep.ok is True


def test_cms_transform_signal():
    X = _toy_X()
    m = CarryMomentumSeasonality(k_momo=10, ewma_span=30, seasonality=False)
    m.fit(X)
    Z = m.transform(X)
    assert any(c.startswith("carry_") for c in Z.columns)
    assert any(c.startswith("momo10_") for c in Z.columns)
    S = m.signal(Z)
    assert S.shape == Z.shape
