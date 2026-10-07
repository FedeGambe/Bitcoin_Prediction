"""Fase 2 - Market trend di BTC: PCA, LSTM single-step e multi-step, classificazione LightGBM.

Uso:
    python scripts/run_market_trend.py            # usa i modelli salvati in models/
    python scripts/run_market_trend.py --train    # riaddestra le LSTM (lento)
Richiede l'extra `dl` (tensorflow, lightgbm). Le metriche vanno in reports/metriche_market_trend.json.
"""

import argparse
import json

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from cripto_trend.config import DATETIME, MODELS_DIR, REPORTS_DIR, TARGET_PRICE
from cripto_trend.data import load_dataset, market_trend_frame
from cripto_trend.features import pca_reduce
from cripto_trend.models import classification, lstm

N_FUTURE = 24 * 14  # orizzonte multi-step: due settimane


def evaluate(y_true, y_pred) -> dict:
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "R2": float(r2_score(y_true, y_pred))}


def main(retrain: bool) -> None:
    df = market_trend_frame(load_dataset())
    X = df.drop(columns=[DATETIME, TARGET_PRICE])

    pca = pca_reduce(X, variance=0.95)
    print(f"PCA: {pca.n_components} componenti spiegano il 95% della varianza ({X.shape[1]} variabili)")
    reduced = pca.components.assign(**{TARGET_PRICE: df[TARGET_PRICE].values})

    metrics = {"pca_components": pca.n_components}

    # LSTM single-step: prossima ora da una settimana di storia
    seq = lstm.prepare_sequences(reduced)
    if retrain:
        model = lstm.build_model(seq.X_train.shape[1], seq.X_train.shape[2])
        lstm.train(model, seq)
        model.save(MODELS_DIR / "modello_completo.keras")
    else:
        model = lstm.load(MODELS_DIR / "modello_completo.keras")
    y_true = lstm.inverse_target(seq.y_test, seq)
    y_pred = lstm.inverse_target(model.predict(seq.X_test, verbose=0).ravel(), seq)
    bias = lstm.mean_bias(y_true, y_pred)
    metrics["lstm_single_step"] = {**evaluate(y_true, y_pred), "mean_bias_usd": bias,
                                   "R2_dopo_correzione": float(r2_score(y_true, y_pred + bias))}
    print("LSTM single-step:", metrics["lstm_single_step"])

    # LSTM multi-step: 336 ore in un colpo solo
    seq_m = lstm.prepare_sequences(reduced, horizon=N_FUTURE)
    if retrain:
        model_m = lstm.build_model(seq_m.X_train.shape[1], seq_m.X_train.shape[2], n_outputs=N_FUTURE)
        lstm.train(model_m, seq_m, epochs=20, batch_size=32)
        model_m.save(MODELS_DIR / "modello_completo_multi.keras")
    else:
        model_m = lstm.load(MODELS_DIR / "modello_completo_multi.keras")
    y_true_m = lstm.inverse_target(seq_m.y_test, seq_m)
    y_pred_m = lstm.inverse_target(model_m.predict(seq_m.X_test, verbose=0), seq_m)
    # Metriche in dollari su tutti gli step (nel notebook R² mescolava scala 0-1 e dollari)
    metrics["lstm_multi_step"] = {**evaluate(y_true_m.ravel(), y_pred_m.ravel()),
                                  "mean_bias_usd": lstm.mean_bias(y_true_m, y_pred_m),
                                  "step1": evaluate(y_true_m[:, 0], y_pred_m[:, 0])}
    print("LSTM multi-step:", metrics["lstm_multi_step"])

    # Classificazione: BTC tra una settimana sarà più alto? (walk-forward con purging)
    clf = classification.walk_forward(classification.direction_dataset(df))
    metrics["lightgbm_purging"] = {"accuracy": clf.accuracy, "balanced_accuracy": clf.balanced_accuracy,
                                   "average_precision": clf.average_precision,
                                   "confusion": clf.confusion.tolist()}
    print("LightGBM walk-forward:", {k: round(v, 3) for k, v in metrics["lightgbm_purging"].items() if k != "confusion"})

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "metriche_market_trend.json"
    out.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nMetriche salvate in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="riaddestra le LSTM invece di caricarle")
    main(parser.parse_args().train)
