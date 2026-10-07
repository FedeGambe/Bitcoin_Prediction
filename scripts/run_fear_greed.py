"""Fase 3 - Fear & Greed Index: PCA (90%), confronto regressori, Random Forest, curve LOWESS.

Uso: python scripts/run_fear_greed.py [--grid]
Con --grid esegue la grid search completa del Random Forest (lenta); altrimenti usa i
migliori parametri trovati nel notebook. Metriche in reports/metriche_fear_greed.json.
"""

import argparse
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from cripto_trend.config import DATETIME, REPORTS_DIR, SEED, TARGET_SENTIMENT
from cripto_trend.data import load_dataset, sentiment_frame
from cripto_trend.features import pca_reduce, top_loadings
from cripto_trend.models import sentiment

BEST_RF_PARAMS = dict(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=2, bootstrap=True)


def main(grid: bool) -> None:
    df = sentiment_frame(load_dataset())
    y = df[TARGET_SENTIMENT]
    X = df.drop(columns=[DATETIME, TARGET_SENTIMENT])
    X = X[[c for c in X.columns if "volume" not in c.lower()]]  # la PCA usa solo i livelli

    pca = pca_reduce(X, variance=0.90, names=sentiment.COMPONENT_NAMES)
    print(f"PCA: {pca.n_components} componenti spiegano il 90% della varianza")
    for comp, top in top_loadings(pca.loadings, 3).items():
        print(f"  {comp}: {', '.join(v for v, _ in top)}")

    # Split casuale come nel notebook: con dati orari autocorrelati le metriche sono ottimistiche
    X_train, X_test, y_train, y_test = train_test_split(pca.components, y, test_size=0.2, random_state=343)

    ranking = sentiment.compare_regressors(X_train, y_train)
    print("\nConfronto regressori (5-fold):\n", ranking.round(4).to_string(index=False))

    rf = (sentiment.tune_random_forest(X_train, y_train) if grid
          else RandomForestRegressor(random_state=SEED, n_jobs=-1, **BEST_RF_PARAMS).fit(X_train, y_train))
    rf_metrics = sentiment.regression_metrics(y_test, rf.predict(X_test))
    print("\nRandom Forest sul test:", {k: round(v, 3) for k, v in rf_metrics.items()})

    robust = sentiment.split_robustness(pca.components, y, df[DATETIME], **BEST_RF_PARAMS)
    print("\nRandom Forest con split diversi:\n", robust.round(3).to_string(index=False))

    out = REPORTS_DIR / "metriche_fear_greed.json"
    out.write_text(json.dumps({
        "pca_components": pca.n_components,
        "ranking": ranking.to_dict(orient="records"),
        "random_forest": rf_metrics,
        "split_robustness": robust.to_dict(orient="records"),
    }, indent=2, default=float))
    print(f"\nMetriche salvate in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", action="store_true", help="grid search completa del Random Forest")
    main(parser.parse_args().grid)
