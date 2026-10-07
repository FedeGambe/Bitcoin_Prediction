import numpy as np
import pandas as pd
import pytest

from cripto_trend.config import DATETIME, DATA_ZIP, TARGET_PRICE
from cripto_trend.data import GROUPS, clean, market_trend_frame, rename_close_columns, sentiment_frame
from cripto_trend.features import base_100, make_sequences, pca_reduce, returns


def test_make_sequences_single_step():
    data = np.arange(30, dtype=float).reshape(10, 3)  # target in ultima colonna
    X, y = make_sequences(data, length=4, target_index=2)
    assert X.shape == (6, 4, 2)
    assert y[0] == data[4, 2]
    assert np.array_equal(X[0], data[0:4, :2])


def test_make_sequences_multi_step():
    data = np.arange(60, dtype=float).reshape(20, 3)
    X, y = make_sequences(data, length=4, target_index=2, horizon=5)
    assert X.shape == (11, 4, 2)
    assert y.shape == (11, 5)
    assert np.array_equal(y[0], data[4:9, 2])


def test_base_100_and_returns():
    s = pd.Series([50.0, 100.0, 75.0])
    assert base_100(s).tolist() == [100.0, 200.0, 150.0]
    r = returns(pd.DataFrame({"BTC": s}), ["BTC"], {"1h": 1})
    assert r["BTC_1h"].iloc[1] == pytest.approx(1.0)


def test_pca_reaches_variance():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 2))
    X = pd.DataFrame(np.hstack([base, base + rng.normal(scale=0.01, size=(200, 2))]), columns=list("abcd"))
    res = pca_reduce(X, variance=0.95)
    assert res.n_components == 2
    assert res.explained_variance[:2].sum() >= 0.95


@pytest.fixture(scope="module")
def dataset():
    if not DATA_ZIP.exists():
        pytest.skip("dataset non presente")
    from cripto_trend.data import load_raw
    return clean(load_raw())


def test_dataset_quality(dataset):
    assert len(dataset) == 17515
    assert dataset[DATETIME].is_monotonic_increasing
    assert not dataset.isna().any().any()


def test_column_selections(dataset):
    mt = market_trend_frame(dataset)
    assert TARGET_PRICE in mt and "open_interest" not in mt
    assert mt.shape[1] - 2 == 28  # feature per la PCA (senza Datetime e target)
    assert not any("volume" in c.lower() for c in mt.columns)
    assert any("volume" in c.lower() for c in sentiment_frame(dataset).columns)


def test_rename_close_columns(dataset):
    close = rename_close_columns(dataset)
    for cols in GROUPS.values():
        assert set(cols) <= set(close.columns)
    assert close.columns.is_unique
