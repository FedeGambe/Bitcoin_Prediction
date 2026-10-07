"""Trasformazioni: rendimenti, multicollinearità (VIF), PCA e finestre temporali per LSTM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import GIORNO, MESE, SETTIMANA

RETURN_PERIODS = {"1d": GIORNO, "1w": SETTIMANA, "1m": MESE, "6m": 6 * MESE, "1y": 365 * GIORNO}


def base_100(series: pd.Series) -> pd.Series:
    """Serie normalizzata a 100 sul primo valore (per confrontare asset con scale diverse)."""
    first = series.iloc[0]
    return series / first * 100 if first != 0 else series


def returns(df: pd.DataFrame, columns: list[str], periods: dict[str, int] = RETURN_PERIODS) -> pd.DataFrame:
    """Rendimenti percentuali su più orizzonti: colonne `<asset>_<periodo>`."""
    out = pd.DataFrame(index=df.index)
    for col in columns:
        for name, hours in periods.items():
            out[f"{col}_{name}"] = df[col].pct_change(periods=hours)
    return out


def vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per ogni colonna (con costante)."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xc = sm.add_constant(X).values
    values = [variance_inflation_factor(Xc, i + 1) for i in range(X.shape[1])]
    return pd.DataFrame({"feature": X.columns, "VIF": values}).sort_values("VIF", ascending=False)


@dataclass
class PCAResult:
    components: pd.DataFrame       # punteggi delle prime n componenti
    loadings: pd.DataFrame         # pesi delle variabili (righe = componenti)
    explained_variance: np.ndarray # varianza spiegata da tutte le componenti
    n_components: int
    scaler: StandardScaler
    pca: PCA


def pca_reduce(X: pd.DataFrame, variance: float = 0.95, names: list[str] | None = None) -> PCAResult:
    """Standardizza, applica la PCA e tiene le componenti che spiegano `variance` della varianza."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA().fit(X_scaled)
    explained = pca.explained_variance_ratio_
    n = int(np.argmax(explained.cumsum() >= variance) + 1)

    cols = names[:n] if names else [f"PC{i + 1}" for i in range(n)]
    scores = pd.DataFrame(pca.transform(X_scaled)[:, :n], columns=cols, index=X.index)
    loadings = pd.DataFrame(pca.components_[:n], columns=X.columns, index=cols)
    return PCAResult(scores, loadings, explained, n, scaler, pca)


def top_loadings(loadings: pd.DataFrame, k: int = 5) -> dict[str, list[tuple[str, float]]]:
    """Le k variabili con peso assoluto più alto per ogni componente."""
    out = {}
    for comp, row in loadings.iterrows():
        top = row.abs().sort_values(ascending=False).head(k).index
        out[comp] = [(var, float(row[var])) for var in top]
    return out


def make_sequences(data: np.ndarray, length: int, target_index: int, horizon: int = 1):
    """Finestre scorrevoli per LSTM.

    `data` ha la target nell'ultima colonna (`target_index`). Ogni X contiene le feature
    (tutte le colonne tranne la target) delle `length` ore precedenti; y è la target
    all'ora successiva (`horizon=1`) o le `horizon` ore successive (multi-step).
    """
    n = len(data) - length - (horizon if horizon > 1 else 0)
    X =np.stack([data[i:i + length, :-1] for i in range(n)])
    if horizon == 1:
        y = np.array([data[i + length, target_index] for i in range(n)])
    else:
        y = np.stack([data[i + length:i + length + horizon, target_index] for i in range(n)])
    return X, y
