# Notebook

Notebook originali del project work (Google Colab), lasciati invariati come riferimento. Il codice riutilizzabile è in `src/cripto_trend/` e le pipeline in `scripts/`.

| File | Contenuto |
|---|---|
| `02_market_trend.ipynb` | PCA, LSTM single e multi-step, classificazione LightGBM |
| `03_fear_greed_sentiment.ipynb` | PCA, confronto regressori, Random Forest, LOWESS |
| `projectwork_crif_colab_export.py` | Export completo del notebook, compresa l'analisi descrittiva |

I notebook leggono modelli e dataset da percorsi Colab/Drive: per eseguirli in locale usa le funzioni del pacchetto (`cripto_trend.data.load_dataset()`, `models/`).
