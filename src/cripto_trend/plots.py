"""Grafici Plotly riutilizzabili (tema scuro, come nei notebook)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import DATETIME
from .features import base_100

TEMPLATE = "plotly_dark"


def _title(fig: go.Figure, text: str) -> go.Figure:
    fig.update_layout(title={"text": text, "x": 0.5, "xanchor": "center"}, template=TEMPLATE)
    return fig


def group_vs_btc(df: pd.DataFrame, columns: list[str], name: str) -> go.Figure:
    """Andamento base 100 di un gruppo di asset (nomi brevi) insieme a BTC."""
    fig = go.Figure()
    for col in dict.fromkeys(["BTC", *columns]):
        fig.add_trace(go.Scatter(x=df[DATETIME], y=base_100(df[col]), mode="lines", name=col))
    return _title(fig, f"{name} vs BTC (base 100)")


def correlation_heatmap(df: pd.DataFrame, title: str = "Matrice di correlazione") -> go.Figure:
    fig = px.imshow(df.corr(numeric_only=True), text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
    return _title(fig, title)


def pca_variance(explained: np.ndarray, threshold: float) -> go.Figure:
    x = np.arange(1, len(explained) + 1)
    fig = go.Figure([
        go.Bar(x=x, y=explained, name="Varianza spiegata"),
        go.Scatter(x=x, y=explained.cumsum(), name="Varianza cumulata", mode="lines+markers"),
    ])
    fig.add_hline(y=threshold, line_dash="dash", line_color="gray")
    return _title(fig, "PCA - varianza per componente")


def forecast_vs_actual(dates, actual, predicted, title: str = "Previsione vs reale") -> go.Figure:
    fig = go.Figure([
        go.Scatter(x=dates, y=actual, mode="lines", name="Reale", line=dict(color="white")),
        go.Scatter(x=dates, y=predicted, mode="lines", name="Previsto", line=dict(color="lightsalmon")),
    ])
    return _title(fig, title)


def predicted_vs_actual(actual, predicted, title: str = "Reale vs predetto") -> go.Figure:
    lo, hi = float(np.min([actual, predicted])), float(np.max([actual, predicted]))
    fig = go.Figure([
        go.Scatter(x=actual, y=predicted, mode="markers", name="Predizioni", marker=dict(opacity=0.5)),
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", line=dict(dash="dash", color="white")),
    ])
    return _title(fig, title)


def training_curves(history: dict) -> go.Figure:
    fig = go.Figure([go.Scatter(y=v, mode="lines", name=k) for k, v in history.items() if "loss" in k])
    fig.update_xaxes(title="Epoca")
    return _title(fig, "Loss di training e validation")


def confusion(cm: np.ndarray, labels=("Giù (0)", "Su (1)")) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=cm, x=list(labels), y=list(labels), colorscale="Blues", text=cm, texttemplate="%{text}"))
    fig.update_layout(xaxis_title="Predetto", yaxis_title="Reale")
    return _title(fig, "Matrice di confusione")


def shape_grid(X: pd.DataFrame, y: pd.Series, curves: dict[str, dict]) -> go.Figure:
    """Nuvola di punti + polinomiale + LOWESS per ogni componente (griglia 2x3)."""
    fig = make_subplots(rows=2, cols=3, subplot_titles=list(X.columns))
    for i, col in enumerate(X.columns):
        r, c, first = i // 3 + 1, i % 3 + 1, i == 0
        cv = curves[col]
        fig.add_trace(go.Scatter(x=X[col], y=y, mode="markers", marker=dict(size=3, opacity=0.3),
                                 name="Dati", showlegend=first), row=r, col=c)
        fig.add_trace(go.Scatter(x=cv["poly_x"], y=cv["poly_y"], mode="lines", name="Polinomiale (grado 3)",
                                 line=dict(color="orange"), showlegend=first), row=r, col=c)
        fig.add_trace(go.Scatter(x=cv["lowess_x"], y=cv["lowess_y"], mode="lines", name="LOWESS",
                                 line=dict(color="white"), showlegend=first), row=r, col=c)
    return _title(fig, "Fear & Greed vs componenti: polinomiale e LOWESS")
