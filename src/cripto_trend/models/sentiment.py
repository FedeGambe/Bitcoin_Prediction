"""Stima del Fear & Greed Index dalle componenti principali del mercato."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, QuantileRegressor,
    RANSACRegressor, Ridge, TheilSenRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ..config import SEED

# Nomi interpretativi delle 6 componenti (dai pesi della PCA, vedi reports/fear_greed.md)
COMPONENT_NAMES = [
    "Global Market Movement",
    "Market Sentiment and Funding Rates",
    "Crypto Search Trends and Sentiment",
    "Commodity and Equity Market Dynamics",
    "Commodities Price Movements",
    "Emerging Market and Macro Trends",
]

REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net Regression": ElasticNet(),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=SEED),
    "Random Forest Regression": RandomForestRegressor(random_state=SEED),
    "Gradient Boosting Regression": GradientBoostingRegressor(random_state=SEED),
    "AdaBoost Regression": AdaBoostRegressor(random_state=SEED),
    "Support Vector Regression": SVR(),
    "K-Nearest Neighbors Regression": KNeighborsRegressor(),
    "MLP Regression": MLPRegressor(random_state=SEED),
    "Huber Regression": HuberRegressor(),
    "Quantile Regression": QuantileRegressor(),
    "RANSAC Regression": RANSACRegressor(random_state=SEED),
    "TheilSen Regression": TheilSenRegressor(random_state=SEED),
    "Bayesian Ridge Regression": BayesianRidge(),
}

RF_GRID = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [None, 10, 20, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [2, 5, 10],
    "bootstrap": [True],
}


def compare_regressors(X, y, models: dict = REGRESSORS, n_folds: int = 5) -> pd.DataFrame:
    """R² e MSE medi in K-fold (pipeline con StandardScaler), ordinati per MSE."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        cv = cross_validate(make_pipeline(StandardScaler(), model), X, y, cv=kfold,
                            scoring=("r2", "neg_mean_squared_error"), n_jobs=-1)
        rows.append({"Modello": name, "R2 medio": cv["test_r2"].mean(),
                     "MSE medio": -cv["test_neg_mean_squared_error"].mean()})
    return pd.DataFrame(rows).sort_values("MSE medio").reset_index(drop=True)


def tune_random_forest(X_train, y_train, grid: dict = RF_GRID, cv: int = 3) -> RandomForestRegressor:
    """Grid search sul Random Forest (scoring R²)."""
    search = GridSearchCV(RandomForestRegressor(random_state=SEED), grid, cv=cv, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)), "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


def shape_curves(x: np.ndarray, y: np.ndarray, degree: int = 3, frac: float = 0.3, n_points: int = 300):
    """Curve interpretative di y rispetto a una componente: polinomiale e LOWESS."""
    import statsmodels.api as sm

    poly = PolynomialFeatures(degree=degree)
    reg = LinearRegression().fit(poly.fit_transform(x.reshape(-1, 1)), y)
    grid = np.linspace(x.min(), x.max(), n_points).reshape(-1, 1)
    poly_y = reg.predict(poly.transform(grid))
    lowess = sm.nonparametric.lowess(y, x, frac=frac)
    return {"poly_x": grid.ravel(), "poly_y": poly_y, "lowess_x": lowess[:, 0], "lowess_y": lowess[:, 1]}


def split_robustness(X, y, dates: pd.Series, **rf_params) -> pd.DataFrame:
    """R² e MAE del Random Forest con split via via più severi.

    Il Fear & Greed ha un valore al giorno ripetuto su 24 ore: con lo split casuale ore
    dello stesso giorno finiscono in train e test. Raggruppare per giorno o settimana,
    o dividere in ordine cronologico, misura quanto il modello generalizza davvero.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    X, y = np.asarray(X), np.asarray(y)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    idx = np.arange(len(y))
    n = int(len(y) * 0.8)
    splits = {
        "Casuale (notebook)": train_test_split(idx, test_size=0.2, random_state=343),
        "Per giorno": next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, dates.dt.date)),
        "Per settimana": next(GroupShuffleSplit(1, test_size=0.2, random_state=0)
                              .split(X, y, dates.dt.to_period("W").astype(str))),
        "Cronologico 80/20": (idx[:n], idx[n:]),
    }
    rows = []
    for name, (tr, te) in splits.items():
        model = RandomForestRegressor(random_state=SEED, n_jobs=-1, **rf_params).fit(X[tr], y[tr])
        pred = model.predict(X[te])
        rows.append({"Split": name, "R2": r2_score(y[te], pred), "MAE": mean_absolute_error(y[te], pred)})
    return pd.DataFrame(rows)
