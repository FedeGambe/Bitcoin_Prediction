## 2. Analisi Predittiva del Market Trend

### 2.1 Introduzione

L’obiettivo era sviluppare modelli predittivi per stimare il market trend di BTC. Sono stati utilizzati diversi approcci, tra cui:

- **LSTM (Long Short-Term Memory)** per previsioni:
  - univariate (single step)
  - multivariate (multi-step)
- **Modelli di classificazione** per determinare se il prezzo di chiusura sarebbe stato maggiore o minore del prezzo di apertura.

---

### 2.2 Preprocessing dei Dati

- Il dataset è stato pulito da **valori nulli e duplicati**.
- I dati sono stati **normalizzati** con **MinMax Scaler** per garantire una scala uniforme.
- Per ridurre la **multicollinearità** tra le variabili:
  - È stata applicata la **PCA (Principal Component Analysis)**, che ha ridotto il numero di variabili mantenendo alta la varianza.
  - È stato calcolato il **VIF (Variance Inflation Factor)** per individuare variabili collineari.
- La PCA ha individuato **9 nuove dimensioni** che spiegano il **95% della varianza totale**.

---

### 2.3 Modelli

Le nuove dimensioni ottenute tramite PCA sono state utilizzate per i seguenti modelli:

#### 🔹 Modello LSTM per Previsione del Prezzo

- Ha previsto il prezzo del BTC su base storica.
- Addestrato per **50 epoche** con un train set di **21.569 neuroni**, e una **sequence_length** di **una settimana**.
- La previsione era inizialmente sottostimata ed è stata **corretta di circa 18.056$**, migliorandone l'accuratezza.

#### 🔹 Modello LSTM Multi-Step per Previsione della Tendenza

- Previsione su **due settimane (336 ore)**.
- Necessario poiché il modello single-step non poteva gestire previsioni di lungo periodo.
- Correzione applicata di **23.768$** (scarto medio tra valore predetto e reale).
- La previsione ha indicato un **leggero rialzo** del prezzo per le due settimane successive.

#### 🔹 Modello di Classificazione con LightGBM

- Obiettivo: prevedere se il BTC chiuderà **in rialzo o in ribasso** entro una settimana.
- Utilizza **tecnica del purging**, per simulare riaddestramento settimanale realistico.
- Le **performance** sono state **inefficienti**, simili a quelle di un modello casuale.
- Conclusione: le **informazioni storiche sui prezzi non bastano** per prevedere il comportamento del mercato. Variabili esogene non sono state considerate.

---

### 2.4 Conclusioni sui Modelli

- I modelli LSTM mostrano **buona capacità predittiva a breve termine**, ma **scarsa generalizzazione a lungo termine** senza l’integrazione di dati esterni.
- I modelli di classificazione **non hanno superato le performance di un modello casuale**, confermando l’insufficienza dei soli dati storici per la previsione del market trend.

#### 🔍 Analisi Futura

Per migliorare l'approccio predittivo, si consiglia di:

- Integrare **fattori esogeni** come:
  - tassi d’interesse
  - politiche monetarie
  - eventi geopolitici ed economici globali
  - notizie legate a Bitcoin
- Costruire modelli **scenario-based**, capaci di simulare l’impatto di eventi specifici sul **sentiment di mercato**.

---
