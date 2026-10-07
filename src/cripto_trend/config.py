"""Percorsi e costanti del progetto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ZIP = ROOT / "data" / "raw" / "Dataset.zip"
DATASET_CSV = "merged_fix_to_hour.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
NOTEBOOKS_DIR = ROOT / "notebooks"
DOCS_DIR = ROOT / "docs"

# Copia remota del dataset (per Colab o clone senza LFS)
DATA_ZIP_URL = "https://github.com/FedeGambe/Bitcoin_Prediction/raw/main/data/raw/Dataset.zip"

DATETIME = "Datetime"
TARGET_PRICE = "BTC_USDT_1h_close"
TARGET_SENTIMENT = "fear_gread_index"  # nome originale del dataset (refuso "gread")

# Orizzonti in ore (dati orari)
GIORNO = 24
SETTIMANA = 24 * 7
MESE = 24 * 30

SEED = 42
