## 3. Analisi del Fear and Greed Index e Fattori di Sentiment di Mercato

### 3.1 Introduzione

L’indice **Fear and Greed** è un indicatore sintetico che misura il sentiment del mercato, oscillando tra due estremi:

- **Paura** (valori bassi): tipica nei momenti di incertezza o ribasso.
- **Avidità** (valori alti): presente nei periodi di euforia e ottimismo.

Per questa fase dell’analisi è stato adottato un approccio metodologico coerente con le sezioni precedenti:

- Dopo la costruzione iniziale del dataset, è stato calcolato il **VIF (Variance Inflation Factor)** per individuare casi di **multicollinearità** tra le variabili.
- A causa dell’elevata correlazione tra alcune feature, è stata applicata una **PCA (Principal Component Analysis)**.
- Sono state estratte **6 componenti principali**, in grado di spiegare circa **il 90% della varianza totale**.

---

### 3.2 Interpretazione delle Componenti Principali

#### 1. Global Market Movement *(S&P, Dow, NASDAQ, BTC)*

- **Motivazione**: Componente influenzata da indici azionari globali e da Bitcoin.
- **Implicazioni**: Rappresenta l’andamento generale dei mercati (tradizionali e crypto). Utile per monitorare l’impatto degli asset principali sul sentiment.

#### 2. Market Sentiment and Funding Rates

- **Motivazione**: Dominata da funding rate, VIX e CPI.
- **Implicazioni**: Misura la percezione del rischio e il sentiment generale nei mercati.

#### 3. Crypto Search Trends and Sentiment

- **Motivazione**: Basata su Google Trends e termini di ricerca legati al mondo crypto.
- **Implicazioni**: Riflette l’interesse del pubblico e il sentiment retail.

#### 4. Commodity and Equity Market Dynamics

- **Motivazione**: Include commodities (grano, mais), indici europei (es. CAC 40) e criptovalute.
- **Implicazioni**: Evidenzia l’interconnessione tra mercati fisici e finanziari.

#### 5. Commodities Price Movements

- **Motivazione**: Dominata da prezzi di materie prime (petrolio, bovini).
- **Implicazioni**: Mostra come i movimenti delle commodities influenzano il sentiment e la stabilità macroeconomica.

#### 6. Emerging Market and Macro Trends

- **Motivazione**: Legata a mercati emergenti (es. IBOVESPA) e all'inflazione (CPI).
- **Implicazioni**: Utile per analizzare i cicli macroeconomici globali e segnali dai mercati emergenti.

---

### 3.3 Modello di Machine Learning

È stata condotta un’**analisi comparativa** tra modelli regressivi. Il **Random Forest Regressor** ha ottenuto le migliori performance:

- **R² = 0.98**
- **MSE = 4.17**

Tuttavia, il modello ha mostrato **bassa interpretabilità**.

---

### 3.4 Analisi Interpretativa: LOWESS e Regressione Polinomiale

Per superare i limiti interpretativi dei modelli regressivi complessi, sono stati utilizzati modelli più esplicativi:

- **Regressione polinomiale di grado 3**
- **LOWESS** (Locally Weighted Scatterplot Smoothing): tecnica non parametrica che evidenzia pattern non lineari in modo flessibile e visivo.

#### 📊 Risultati Qualitativi - Tendenze principali:

- **Global Market Movement**: Relazione crescente → l’aumento degli asset globali tende a far crescere l’avidità.
- **Market Sentiment & Crypto Search Trends**: Correlazione positiva → sentiment ottimistico e interesse retail spingono l’indice verso l’alto.
- **Commodity & Equity Dynamics**: Relazione a U invertita → il sentiment peggiora con volatilità, migliora in stabilità.
- **Commodities Price Movements**: Relazione inversa → prezzi alti delle commodities aumentano la paura (timori di inflazione).
- **Emerging Market & Macro Trends**: Relazione a U piatta → effetto più debole, legato a cicli macro e segnali dai mercati emergenti.

---
