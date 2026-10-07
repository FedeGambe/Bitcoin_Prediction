# Cripto Trend & Sentiment
### Machine Learning per l'analisi predittiva di Bitcoin e del Fear & Greed Index

Project work finale dell'academy **Data Science e Generative AI** (Crif, aprile 2025).

- **Pagina dei risultati:** https://fedegambe.github.io/Bitcoin_Prediction/
- Notebook originale su Colab: [![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DZLNtwSHd5MNxl6iJUWaGjxv-o84B_CB?usp=sharing)

## Obiettivo

Partendo da un dataset **orario** (aprile 2023 – aprile 2025, 17.515 righe) con Bitcoin, altre criptovalute, materie prime, indici azionari e indicatori di sentiment, il progetto:

1. **Analisi descrittiva**: confronta andamento, rendimenti e correlazioni di BTC con gli altri asset.
2. **Market trend**: prevede il prezzo di BTC con reti **LSTM** (single-step a 1 ora e multi-step a 336 ore) e la direzione a una settimana con **LightGBM** (walk-forward con purging).
3. **Fear & Greed Index**: lo stima dalle componenti principali del mercato (**PCA** + confronto di 16 regressori, **Random Forest**) e ne interpreta i fattori con **regressione polinomiale e LOWESS**.

## Risultati principali

| Analisi | Modello | Risultato |
|---|---|---|
| Prezzo BTC a 1 ora | LSTM su 9 componenti PCA (senza il prezzo passato di BTC) | MAE 18.057 $, sottostima sistematica (R² −5,2) |
| Prezzo BTC a 2 settimane | LSTM multi-step | MAE 23.768 $ |
| Direzione a 1 settimana | LightGBM walk-forward | accuratezza 0,49, sotto la baseline "sempre su" (0,53) |
| Fear & Greed Index | Random Forest su 6 componenti PCA | R² 0,985 con split casuale, **0,00 con split cronologico** |

I soli dati di mercato non bastano a prevedere né il prezzo né il sentiment. L'R² del notebook sul Fear & Greed nasce dallo split casuale: l'indice ha un valore al giorno ripetuto sulle 24 ore, quindi ore dello stesso giorno finiscono in train e test. Il Fear & Greed resta legato al rendimento di BTC a 30 giorni (correlazione 0,68) e alle ricerche su Google. Limiti e dettagli sono nella [pagina dei risultati](https://fedegambe.github.io/Bitcoin_Prediction/) e in [`reports/`](reports/).

## Struttura

```
├── data/raw/Dataset.zip          dataset orario (merged_fix_to_hour.csv)
├── docs/                         pagina GitHub Pages (index.html, progetto.json, img/)
├── models/                       LSTM addestrate (.keras) e storie di training (.pkl)
├── notebooks/                    notebook originali del corso
├── references/                   testo della consegna
├── reports/                      note metodologiche e metriche (JSON) prodotte dagli script
├── scripts/                      pipeline eseguibili
├── src/cripto_trend/             pacchetto Python
│   ├── config.py                 percorsi e costanti
│   ├── data.py                   caricamento, pulizia, selezione colonne
│   ├── features.py               base 100, rendimenti, VIF, PCA, finestre per LSTM
│   ├── plots.py                  grafici Plotly
│   └── models/                   lstm.py, classification.py, sentiment.py
└── tests/                        test pytest
```

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dl,dev]"        # senza [dl] niente TensorFlow/LightGBM

python scripts/run_descrittiva.py --html    # statistiche + grafici in reports/figures/
python scripts/run_market_trend.py          # LSTM salvate in models/ (--train per riaddestrare)
python scripts/run_fear_greed.py            # --grid per la grid search completa
python scripts/build_pagina.py              # rigenera i dati dei grafici di docs/index.html
pytest
```

Gli script riproducono le metriche dei notebook. Per pubblicare la pagina: *Settings → Pages → branch `main`, cartella `/docs`*.

## Note sulla ristrutturazione

Rispetto al notebook sono stati corretti alcuni errori che impedivano l'esecuzione end-to-end (variabili non definite come `test_clas` e `dates_all`, percorsi Google Drive, URL dello zip rinominato) e il calcolo dell'R² multi-step, che confrontava valori scalati 0–1 con prezzi in dollari.
