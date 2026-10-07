"""Caricamento, pulizia e selezione delle colonne del dataset orario."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from .config import DATA_ZIP, DATA_ZIP_URL, DATASET_CSV, DATETIME

NUMERIC_TEXT_COLUMNS = ["funding_rate", "fear_gread_index", "google_trends_buy_crypto", "google_trends_bitcoin"]
SENTIMENT_COLUMNS = ["funding_rate", "fear_gread_index", "google_trends_buy_crypto", "google_trends_bitcoin"]

# Gruppi tematici (prezzo di chiusura, nomi brevi dopo `rename_close_columns`)
CRYPTOS = ["BTC", "BNB", "DOGE", "ETH", "SOL", "XRP"]
COMMODITIES = ["cattle", "corn", "crude", "gold", "silver", "soybeans", "wheat"]
INDICES = ["CAC", "DAX", "Dow", "EURO", "FTSE", "IBOVESPA", "IPC", "NASDAQ", "Russell", "SP", "SP_TSE"]
MACRO_TRENDS = ["funding rt", "fear greed", "G_Trends crypto", "G_Trends BTC", "VIX"]
GROUPS = {"Cryptos": CRYPTOS, "Commodities": COMMODITIES, "Indices": INDICES, "Macro e Trends": MACRO_TRENDS}

# Sottostringa della colonna originale -> nome breve (l'ordine conta: GSPTSE prima di GSPC)
_SHORT_NAMES = [
    ("BTC_USDT", "BTC"), ("BNB_USDT", "BNB"), ("DOGE_USDT", "DOGE"), ("ETH_USDT", "ETH"),
    ("SOL_USDT", "SOL"), ("XRP_USDT", "XRP"),
    ("cattle", "cattle"), ("corn", "corn"), ("crude", "crude"), ("gold", "gold"),
    ("silver", "silver"), ("soybeans", "soybeans"), ("wheat", "wheat"),
    ("FCHI", "CAC"), ("GDAXI", "DAX"), ("DJI", "Dow"), ("STOXX50E", "EURO"), ("FTSE", "FTSE"),
    ("BVSP", "IBOVESPA"), ("MXX", "IPC"), ("IXIC", "NASDAQ"), ("RUT", "Russell"),
    ("GSPTSE", "SP_TSE"), ("GSPC", "SP"), ("VIX", "VIX"),
]
_SHORT_SENTIMENT = {
    "funding_rate": "funding rt",
    "fear_gread_index": "fear greed",
    "google_trends_buy_crypto": "G_Trends crypto",
    "google_trends_bitcoin": "G_Trends BTC",
}


def load_raw(zip_path: Path | str = DATA_ZIP, csv_name: str = DATASET_CSV) -> pd.DataFrame:
    """Legge il CSV orario dallo zip locale; se manca lo scarica da GitHub."""
    zip_path = Path(zip_path)
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf, zf.open(csv_name) as f:
            return pd.read_csv(f)

    import requests

    response = requests.get(DATA_ZIP_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as zf, zf.open(csv_name) as f:
        return pd.read_csv(f)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Converte date e colonne numeriche, rimuove duplicati e righe vuote, ordina nel tempo."""
    out = df.copy()
    out[DATETIME] = pd.to_datetime(out[DATETIME])
    for col in NUMERIC_TEXT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.drop_duplicates().dropna().sort_values(DATETIME).reset_index(drop=True)
    return out


def load_dataset() -> pd.DataFrame:
    """Dataset orario pulito, pronto per le analisi."""
    return clean(load_raw())


def _is_ohl(col: str) -> bool:
    """True per le colonne open/high/low (compreso `open_interest`, escluso come nei notebook)."""
    low = col.lower()
    return any(k in low for k in ("open", "high", "low")) and "close" not in low


def market_trend_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Colonne per il market trend: solo prezzi di chiusura + variabili di sentiment (no volumi)."""
    keep = [c for c in df.columns if not (_is_ohl(c) or "volume" in c.lower())]
    return df[keep]


def sentiment_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Colonne per il Fear & Greed: chiusure, volumi e sentiment (no open/high/low)."""
    keep = [c for c in df.columns if not _is_ohl(c)]
    return df[keep]


def short_name(col: str) -> str:
    """Nome breve di una colonna di chiusura o di sentiment (es. 'S&P_Close ^GSPC' -> 'SP')."""
    if col in _SHORT_SENTIMENT:
        return _SHORT_SENTIMENT[col]
    return next((short for key, short in _SHORT_NAMES if key in col), col)


def rename_close_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tiene Datetime, sentiment e prezzi di chiusura con nomi brevi (BTC, gold, SP, ...)."""
    close_cols = [c for c in df.columns if "close" in c.lower()]
    sub = df[[DATETIME, *SENTIMENT_COLUMNS, *close_cols]]
    return sub.rename(columns={c: short_name(c) for c in sub.columns if c != DATETIME})
