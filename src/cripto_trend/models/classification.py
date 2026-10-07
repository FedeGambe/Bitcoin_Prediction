"""Classificazione della direzione di BTC (sale/scende) con LightGBM e walk-forward con purging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

from ..config import DATETIME, SEED, SETTIMANA, TARGET_PRICE

TARGET = "BTC_+1s"
LGBM_PARAMS = dict(
    n_estimators=100, learning_rate=0.05, max_depth=5, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1,
)


def direction_dataset(df: pd.DataFrame, horizon: int = SETTIMANA) -> pd.DataFrame:
    """Target binaria: 1 se il prezzo tra `horizon` ore è più alto di ora.

    Feature: tutte le colonne di `df` (tranne il prezzo grezzo) più il rendimento
    dell'ultima settimana. Nomi senza spazi per LightGBM.
    """
    out = df.copy()
    future = out[TARGET_PRICE].pct_change(horizon).shift(-horizon)
    out["BTC_-1s"] = out[TARGET_PRICE].pct_change(SETTIMANA)
    out[TARGET] = future
    out = out.dropna().drop(columns=[TARGET_PRICE])
    out[TARGET] = (out[TARGET] > 0).astype(int)
    out.columns = out.columns.str.replace(" ", "_")
    return out.reset_index(drop=True)


@dataclass
class ClassificationResult:
    accuracy: float
    balanced_accuracy: float
    average_precision: float
    confusion: np.ndarray
    predictions: pd.DataFrame  # Datetime, true, pred


def _scores(y_true, y_pred) -> dict:
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        average_precision=average_precision_score(y_true, y_pred),
    )


def walk_forward(data: pd.DataFrame, gap: int = SETTIMANA, step: int = SETTIMANA,
                 min_train_ratio: float = 0.5) -> ClassificationResult:
    """Riaddestramento settimanale con `gap` ore di purging tra train e test.

    Il gap evita che le ultime righe di train (la cui target guarda una settimana avanti)
    si sovrappongano al periodo di test.
    """
    from lightgbm import LGBMClassifier

    features = [c for c in data.columns if c not in (DATETIME, TARGET)]
    tscv = TimeSeriesSplit(n_splits=len(data) // step, gap=gap)
    min_train = int(len(data) * min_train_ratio)

    rows = []
    for train_idx, test_idx in tscv.split(data):
        if len(train_idx) < min_train:
            continue
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        model = LGBMClassifier(**LGBM_PARAMS).fit(train[features], train[TARGET])
        rows.append(pd.DataFrame({
            DATETIME: test[DATETIME].values,
            "true": test[TARGET].values,
            "pred": model.predict(test[features]),
        }))

    preds = pd.concat(rows, ignore_index=True)
    return ClassificationResult(
        **_scores(preds["true"], preds["pred"]),
        confusion=confusion_matrix(preds["true"], preds["pred"]),
        predictions=preds,
    )
