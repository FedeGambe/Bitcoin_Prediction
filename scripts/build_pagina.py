"""Rigenera i dati dei grafici di docs/index.html (pagina GitHub Pages).

Esegue le stesse analisi degli script run_* (con i modelli LSTM salvati), riduce le serie
per il web e scrive il JSON nel tag <script id="dati"> della pagina. Cover e favicon non
vengono toccate: la pagina le legge da docs/img/.

Uso: python scripts/build_pagina.py
Richiede l'extra `dl` (tensorflow, lightgbm).
"""

import json
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from cripto_trend.config import DATETIME, DOCS_DIR, MODELS_DIR, SEED, SETTIMANA, TARGET_PRICE, TARGET_SENTIMENT
from cripto_trend.data import load_dataset, market_trend_frame, rename_close_columns, sentiment_frame, short_name
from cripto_trend.features import base_100, pca_reduce, top_loadings
from cripto_trend.models import classification, lstm, sentiment

PAGE = DOCS_DIR / "index.html"
N_FUTURE = 24 * 14
RNG = np.random.default_rng(SEED)


def r(values, nd=2):
    """Lista JSON compatta: arrotonda e converte NaN in null."""
    return [None if pd.isna(v) else round(float(v), nd) for v in values]


def dates(values, fmt="%Y-%m-%d %H:%M"):
    return [pd.Timestamp(v).strftime(fmt) for v in values]


def sample_idx(n, k):
    return np.sort(RNG.choice(n, size=min(n, k), replace=False))


def pca_payload(pca, n_var):
    """Varianza per componente, cumulata e pesi delle variabili (per grafici e heatmap)."""
    return {"var": r(pca.explained_variance * 100, 2), "cum": r(pca.explained_variance.cumsum() * 100, 1),
            "n": pca.n_components, "nvar": n_var,
            "comp": list(pca.loadings.index), "variabili": [short_name(c) for c in pca.loadings.columns],
            "pesi": [r(row, 2) for row in pca.loadings.to_numpy()]}


def descrittiva(df):
    close = rename_close_columns(df).set_index(DATETIME)
    daily = close.resample("D").last().dropna()
    yearly = close.pct_change(365 * 24).resample("D").last() * 100
    btc_30g = close["BTC"].pct_change(30 * 24)
    corr = close.corr()["BTC"].drop("BTC").sort_values()
    return {
        "meta": {"righe": len(df), "colonne": df.shape[1], "inizio": str(df[DATETIME].min().date()),
                 "fine": str(df[DATETIME].max().date()),
                 "giorni": (df[DATETIME].max() - df[DATETIME].min()).days},
        "prezzi": {"date": dates(daily.index, "%Y-%m-%d"),
                   "serie": {c: r(base_100(daily[c]), 1) for c in daily.columns
                             if c not in ("fear greed", "funding rt")},
                   "btc": r(daily["BTC"], 0), "fg": r(daily["fear greed"], 0),
                   "btc30": r(btc_30g.resample("D").last().reindex(daily.index) * 100, 1),
                   "corr_fg": {"prezzo": round(float(close["fear greed"].corr(close["BTC"])), 2),
                               "rend30": round(float(close["fear greed"].corr(btc_30g)), 2)}},
        "corr": {"nomi": list(corr.index), "valori": r(corr)},
        "rend1y": {"date": dates(yearly.dropna(subset=["BTC"]).index, "%Y-%m-%d"),
                   "serie": {c: r(yearly.dropna(subset=["BTC"])[c], 1) for c in ["BTC", "SP", "NASDAQ", "gold"]}},
    }


def market_trend(df):
    mt = market_trend_frame(df)
    X = mt.drop(columns=[DATETIME, TARGET_PRICE])
    pca = pca_reduce(X, variance=0.95)
    reduced = pca.components.assign(**{TARGET_PRICE: mt[TARGET_PRICE].values})
    test_dates = mt[DATETIME].iloc[int(len(mt) * 0.8):].reset_index(drop=True)

    seq = lstm.prepare_sequences(reduced)
    model = lstm.load(MODELS_DIR / "modello_completo.keras")
    y_true = lstm.inverse_target(seq.y_test, seq)
    y_pred = lstm.inverse_target(model.predict(seq.X_test, verbose=0).ravel(), seq)
    bias = lstm.mean_bias(y_true, y_pred)
    idx = np.arange(0, len(y_true), 2)  # una osservazione ogni 2 ore
    d1 = test_dates.iloc[SETTIMANA:].reset_index(drop=True)

    seq_m = lstm.prepare_sequences(reduced, horizon=N_FUTURE)
    model_m = lstm.load(MODELS_DIR / "modello_completo_multi.keras")
    ym_true = lstm.inverse_target(seq_m.y_test, seq_m)
    ym_pred = lstm.inverse_target(model_m.predict(seq_m.X_test, verbose=0), seq_m)
    bias_m = lstm.mean_bias(ym_true, ym_pred)
    last = len(ym_true) - 1
    d_last = test_dates.iloc[SETTIMANA + last:SETTIMANA + last + N_FUTURE]

    with open(MODELS_DIR / "history.pkl", "rb") as f:
        hist = pickle.load(f)
    with open(MODELS_DIR / "history_multi.pkl", "rb") as f:
        hist_m = pickle.load(f)

    clf = classification.walk_forward(classification.direction_dataset(mt))
    p = clf.predictions.set_index(DATETIME)
    monthly = (p["true"] == p["pred"]).resample("MS").mean()

    from sklearn.metrics import r2_score
    return {
        "pca_mt": pca_payload(pca, X.shape[1]),
        "lstm1": {"date": dates(d1.iloc[idx]), "reale": r(y_true[idx], 0), "previsto": r(y_pred[idx], 0),
                  "corretto": r(y_pred[idx] + bias, 0), "mae": round(bias, 2),
                  "r2": round(float(r2_score(y_true, y_pred)), 2),
                  "r2_corr": round(float(r2_score(y_true, y_pred + bias)), 2)},
        "lstm_multi": {"date": dates(d_last), "reale": r(ym_true[last], 0), "previsto": r(ym_pred[last], 0),
                       "corretto": r(ym_pred[last] + bias_m, 0), "mae": round(bias_m, 2)},
        "storia": {"loss": r(hist["loss"], 5), "val_loss": r(hist["val_loss"], 5),
                   "loss_m": r(hist_m["loss"], 5), "val_loss_m": r(hist_m["val_loss"], 5)},
        "clf": {"accuracy": round(clf.accuracy, 3), "balanced": round(clf.balanced_accuracy, 3),
                "ap": round(clf.average_precision, 3), "cm": clf.confusion.tolist(), "n": len(p),
                "mesi": dates(monthly.index, "%Y-%m"), "acc_mese": r(monthly * 100, 1),
                "quota_su": round(float(p["true"].mean() * 100), 1)},
    }


def fear_greed(df):
    sf = sentiment_frame(df)
    y = sf[TARGET_SENTIMENT]
    X = sf.drop(columns=[DATETIME, TARGET_SENTIMENT])
    X = X[[c for c in X.columns if "volume" not in c.lower()]]
    pca = pca_reduce(X, variance=0.90, names=sentiment.COMPONENT_NAMES)
    comps = pca.components

    X_train, X_test, y_train, y_test = train_test_split(comps, y, test_size=0.2, random_state=343)
    ranking = sentiment.compare_regressors(X_train, y_train)
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, n_estimators=200, min_samples_leaf=2).fit(X_train, y_train)
    y_hat = rf.predict(X_test)
    rf_m = sentiment.regression_metrics(y_test, y_hat)
    robust = sentiment.split_robustness(comps, y, sf[DATETIME], n_estimators=200, min_samples_leaf=2)
    sel = sample_idx(len(y_test), 1500)

    curve, punti = {}, sample_idx(len(comps), 1500)
    for col in comps.columns:
        cv = sentiment.shape_curves(comps[col].to_numpy(), y.to_numpy())
        step = max(1, len(cv["lowess_x"]) // 200)
        curve[col] = {"x": r(comps[col].to_numpy()[punti], 3), "y": r(y.to_numpy()[punti], 0),
                      "px": r(cv["poly_x"][::3], 3), "py": r(cv["poly_y"][::3], 2),
                      "lx": r(cv["lowess_x"][::step], 3), "ly": r(cv["lowess_y"][::step], 2)}

    return {
        "pca_fg": {**pca_payload(pca, X.shape[1]),
                   "top": {k: [[short_name(v), round(w, 2)] for v, w in t]
                           for k, t in top_loadings(pca.loadings, 4).items()}},
        "rank": {"modelli": [m.replace(" Regression", "") for m in ranking["Modello"]],
                 "r2": r(ranking["R2 medio"], 3), "mse": r(ranking["MSE medio"], 2)},
        "rf": {"reale": r(np.asarray(y_test)[sel], 0), "previsto": r(y_hat[sel], 1),
               **{k: round(v, 3) for k, v in rf_m.items()}},
        "curve": curve,
        "split": {"nomi": list(robust["Split"]), "r2": r(robust["R2"], 3), "mae": r(robust["MAE"], 2)},
    }


def main():
    df = load_dataset()
    data = {**descrittiva(df), **market_trend(df), **fear_greed(df)}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    html = PAGE.read_text(encoding="utf-8")
    pattern = re.compile(r'(<script id="dati" type="application/json">).*?(</script>)', re.S)
    if not pattern.search(html):
        raise SystemExit(f'Tag <script id="dati"> non trovato in {PAGE}')
    PAGE.write_text(pattern.sub(lambda m: m.group(1) + payload + m.group(2), html, count=1), encoding="utf-8")
    print(f"Dati aggiornati in {PAGE} ({len(payload) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
