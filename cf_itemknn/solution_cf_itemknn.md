# Collaborative Filtering Item-Based (ItemKNN)

Questa cartella contiene l'implementazione, la sperimentazione e la valutazione dell'approccio **Item-to-Item Collaborative Filtering (ItemKNN)** applicato al dataset MIND-small per la raccomandazione di articoli di news.

## 1. Obiettivo e impostazione del problema

L'obiettivo di questo modulo è generare un segnale di raccomandazione personalizzato basato esclusivamente sui pattern di interazione degli utenti (i click), senza utilizzare alcuna informazione testuale sugli articoli. A differenza degli approcci content-based, questo metodo si basa sulla **logica di co-occorrenza**: se due articoli vengono cliccati frequentemente dagli stessi utenti in una finestra temporale ravvicinata, si assume che siano correlati dal punto di vista comportamentale.

## 2. Metodologia

La soluzione implementa un algoritmo di Collaborative Filtering Item-Item vettorializzato, pensato per l'ambiente ad alto turnover tipico della raccomandazione di news.

### A. Modellazione della similarità (analisi dei co-click)
La similarità tra due articoli $i$ e $j$ è calcolata con una variante della **Cosine Similarity con shrinkage**:
$$sim(i, j) = \frac{C_{ij}}{\sqrt{C_i \cdot C_j} + \text{shrink}}$$
dove:
- $C_{ij}$: numero di utenti che hanno cliccato sia l'articolo $i$ sia l'articolo $j$;
- $C_i, C_j$: numero totale di click sugli articoli $i$ e $j$;
- **shrink**: parametro di regolarizzazione che penalizza gli articoli con pochissime interazioni, filtrando così rumore e correlazioni spurie dovute a co-occorrenze poco significative.

### B. Scoring e recency (decadimento temporale)
Per calcolare lo score di un articolo candidato $c$, il sistema aggrega le similarità con tutti gli articoli $h$ presenti nella cronologia di click dell'utente $H$, pesate in base alla loro recency:
$$\text{Score}(c) = \sum_{h \in H} sim(h, c) \cdot \text{decay}^t$$
- **Fattore di decadimento**: riduce l'influenza delle interazioni più vecchie ($t$ rappresenta la posizione cronologica, dove 0 è l'interazione più recente), in modo da dare priorità agli interessi attuali dell'utente.

### C. Fallback di popolarità (ibridazione leggera)
Per mitigare il problema del **cold-start** — dove gli articoli nuovi non hanno ancora alcun segnale collaborativo — il sistema integra un meccanismo di blending: lo score finale è una combinazione pesata del segnale ItemKNN e di un prior di popolarità globale, controllata da un parametro `alpha`.

## 3. Descrizione dei componenti

Il modulo è composto da quattro implementazioni, che documentano il percorso dalla ricerca esplorativa fino alla versione ottimizzata per la produzione:

1. **`itemknn_pure.ipynb`**: notebook esplorativo che implementa la baseline ItemKNN "pura". Si concentra su co-occorrenze a breve finestra temporale per stabilire un limite inferiore di prestazioni, senza ottimizzazioni di recency o popolarità.
2. **`itemknn_fallback.ipynb`**: notebook di ricerca che introduce il **fallback di popolarità**, mostrando come l'integrazione dei trend globali migliori la copertura nei casi in cui il segnale collaborativo è assente.
3. **`itemknn_recency_pure.py`**: script Python ottimizzato per la produzione che implementa il modello **ItemKNN puro**, con scoring vettorializzato e pesatura per recency, pensato per elaborare in modo efficiente matrici di interazione di grandi dimensioni.
4. **`itemknn_recency_fallback.py`**: l'implementazione più avanzata, che combina **decadimento temporale** e **blending di popolarità**. Rappresenta il modello finale più robusto, usato per una personalizzazione bilanciata.

## 4. Risultati sperimentali (MINDsmall DEV)

La tabella seguente riassume le prestazioni delle diverse configurazioni ItemKNN sul set di validazione:

| Variante | AUC | MRR | nDCG@5 | nDCG@10 | P@5 | R@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Pure ItemKNN | 0.5010 | 0.2496 | 0.2243 | 0.2871 | 0.0983 | 0.3357 |
| **ItemKNN + Popularity FB** | **0.5318** | **0.2746** | **0.2465** | **0.3102** | **0.1063** | **0.3634** |
| Recency Pure | 0.5010 | 0.2490 | 0.2235 | 0.2864 | 0.0985 | 0.3368 |
| Recency + Popularity FB | 0.5313 | 0.2726 | 0.2446 | 0.3086 | 0.1060 | 0.3624 |

## 5. Considerazioni conclusive

- **La sfida del cold-start**: il Collaborative Filtering puro ottiene prestazioni prossime al caso casuale (AUC ≈ 0.50) nel dominio delle news, a causa della rapidissima obsolescenza degli articoli.
- **Il valore dell'ibridazione**: integrare un fallback di popolarità produce sistematicamente i risultati migliori (AUC massimo 0.5318), confermando che i segnali comportamentali devono essere integrati con i trend globali quando la copertura collaborativa è insufficiente.
- **Sensibilità temporale**: introdurre il decadimento per recency è potenzialmente utile per catturare gli spostamenti di interesse infra-giornalieri tipici del consumo di news; tuttavia, da solo (senza fallback) non migliora le prestazioni rispetto alla versione pura, segno che la scarsità strutturale del grafo di co-click resta il fattore limitante principale.
- **Efficienza implementativa**: la soluzione sfrutta le routine sparse di **Pandas** e **Scipy**, garantendo buona efficienza computazionale e piena compatibilità con la pipeline dati del progetto basata su MIND.
