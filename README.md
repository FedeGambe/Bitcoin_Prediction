# Cripto Trend & Sentiment
### Machine Learning per l’Analisi Predittiva di Bitcoin e del Fear and Greed Index.

Progetto per Crif dell'accademy Data Science e Generative AI.

- Link Colab: [![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DZLNtwSHd5MNxl6iJUWaGjxv-o84B_CB?usp=sharing)

## Obiettivo del Progetto

L'obiettivo principale di questo progetto è sviluppare un sistema predittivo basato su **modelli di Machine Learning (ML)** per stimare:

- L’**andamento del mercato** di Bitcoin (Market Trend).
- Il **Fear and Greed Index**, indicatore del sentiment generale del mercato.

Per raggiungere questo scopo, è stato utilizzato un dataset contenente **dati orari** relativi a Bitcoin, altre criptovalute, indici azionari e variabili di sentiment di mercato.

L’approccio adottato si articola in due fasi principali:

### ➤ **Analisi Descrittiva**
Analisi esplorativa delle caratteristiche storiche di Bitcoin, confrontato con altri asset (indici finanziari e materie prime), calcolo dei rendimenti, valutazione di volatilità e correlazioni.

### ➤ **Analisi Predittiva**
Utilizzo di algoritmi di ML e Deep Learning (LSTM, LightGBM, Random Forest) per:

- Prevedere l’andamento futuro del prezzo di Bitcoin (previsioni univariate e multivariate).
- Stimare il **Fear and Greed Index** attraverso modelli regressivi e interpretativi.

L’intero workflow si compone delle seguenti **fasi operative**:

1. **Importazione e Pulizia del Dato**
2. **Analisi Descrittiva del Bitcoin e degli altri asset**
3. **Predizione del Market Trend di Bitcoin** con modelli LSTM e classificazione  
   📂 [Vai alla cartella Market Trend](./Bitcoin_Prediction/2_Market_Trend)
4. **Predizione del Fear and Greed Index** e analisi dei fattori di sentiment  
   📂 [Vai alla cartella Fear & Greed Analysis](./Bitcoin_Prediction/3_Fear_Greed_Analysis)

> Questo progetto unisce approcci quantitativi e qualitativi, combinando tecniche predittive e interpretative per comprendere e anticipare le dinamiche dei mercati finanziari in ambito crypto.
