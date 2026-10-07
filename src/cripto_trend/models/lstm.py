"""LSTM per il prezzo di BTC: single-step (1h) e multi-step (336h).

TensorFlow viene importato solo dentro le funzioni che lo usano, così il resto
del pacchetto funziona anche senza l'extra `[dl]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..config import SEED, SETTIMANA, TARGET_PRICE
from ..features import make_sequences


@dataclass
class SequenceData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    target_index: int
    n_columns: int
    train_size: int


def prepare_sequences(
    df: pd.DataFrame,
    target: str = TARGET_PRICE,
    length: int = SETTIMANA,
    horizon: int = 1,
    train_ratio: float = 0.8,
) -> SequenceData:
    """Split cronologico 80/20, MinMax fittato solo sul train, finestre di `length` ore."""
    features = [c for c in df.columns if c != target]
    ordered = df[features + [target]]
    train_size = int(len(ordered) * train_ratio)

    scaler = MinMaxScaler()
    train = scaler.fit_transform(ordered.iloc[:train_size])
    test = scaler.transform(ordered.iloc[train_size:])

    target_index = len(features)
    X_train, y_train = make_sequences(train, length, target_index, horizon)
    X_test, y_test = make_sequences(test, length, target_index, horizon)
    return SequenceData(X_train, y_train, X_test, y_test, scaler, target_index, train.shape[1], train_size)


def inverse_target(values: np.ndarray, seq: SequenceData) -> np.ndarray:
    """Riporta la target in dollari (lo scaler si aspetta tutte le colonne)."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    full = np.zeros((flat.size, seq.n_columns))
    full[:, seq.target_index] = flat
    return seq.scaler.inverse_transform(full)[:, seq.target_index].reshape(np.shape(values))


def build_model(length: int, n_features: int, n_outputs: int = 1):
    """LSTM(64) -> Dense(32) -> Dense(16) -> Dense(n_outputs), come nei notebook."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = Sequential([
        Input(shape=(length, n_features)),
        LSTM(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(n_outputs),
    ])
    metrics = [tf.keras.metrics.R2Score(), tf.keras.metrics.RootMeanSquaredError()] if n_outputs == 1 else []
    model.compile(optimizer="adam", loss="mse", metrics=metrics)
    return model


def train(model, seq: SequenceData, epochs: int = 50, batch_size: int = 30):
    """Addestramento con early stopping sulla validation (10% finale del train)."""
    from tensorflow.keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=5, restore_best_weights=True)
    history = model.fit(
        seq.X_train, seq.y_train,
        epochs=epochs, batch_size=batch_size, validation_split=0.1,
        callbacks=[early_stop], verbose=1,
    )
    return history.history


def load(path):
    """Carica un modello salvato in `models/`."""
    from tensorflow.keras.models import load_model

    return load_model(path)


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Scarto medio assoluto tra reale e previsto, in dollari.

    Nei notebook viene sommato alle previsioni ("y_corretta"). Attenzione: è calcolato
    sul test set, quindi la correzione usa informazione del futuro e va letta solo
    come diagnostica della sottostima, non come performance del modello.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def forecast_autoregressive(model, seq: SequenceData, last_window: np.ndarray, steps: int) -> np.ndarray:
    """Previsione ricorsiva: ogni output rientra come target nella finestra successiva.

    `last_window` è una finestra scalata (length, n_columns) con la target in ultima colonna.
    Le feature esogene restano congelate all'ultimo valore osservato, quindi l'errore
    si accumula con l'orizzonte.
    """
    window = last_window.copy()
    preds = []
    for _ in range(steps):
        nxt = float(model.predict(window[np.newaxis, :, :-1], verbose=0)[0, 0])
        preds.append(nxt)
        step = window[-1].copy()
        step[seq.target_index] = nxt
        window = np.vstack([window[1:], step])
    return inverse_target(np.array(preds), seq)
