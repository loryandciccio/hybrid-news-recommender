# Soluzione: Hybrid Recommender System

Questa cartella contiene l'implementazione del **Hybrid Recommender**, il modello più avanzato del progetto, che combina tre diversi paradigmi di raccomandazione per superare i limiti dei singoli approcci.

## 1. Architettura del Modello
Il sistema adotta un approccio di **weighted blending** combinando tre segnali indipendenti:

1.  **Baseline (Popolarità Globale)**: Fornisce stabilità suggerendo gli articoli più letti in assoluto.
2.  **Collaborative Filtering (Item-Item KNN)**: Cattura i pattern di comportamento degli utenti basandosi sulle co-occorrenze dei click, pesate per la recency (decadimento temporale).
3.  **Content-Based Filtering (SBERT + Metadata)**: Analizza il contenuto semantico degli articoli tramite embedding SBERT (70%) e la pertinenza categoriale (30%).

### Meccanismo di Scoring
Ogni segnale viene normalizzato (Min-Max) all'interno del set di candidati per evitare che una metrica domini le altre. Lo score finale è una combinazione lineare:
$$Score_{final} = w_b \cdot S_{baseline} + w_{cf} \cdot S_{cf} + w_{cbf} \cdot S_{cbf}$$

## 2. Componenti e File
*   **`hybrid_recommender.py`**: L'implementazione principale del modello. Include la logica di caricamento dati, la costruzione della matrice di similarità ItemKNN e il motore di scoring ibrido.
*   **`grid_search_weights.py`**: Script per la ricerca automatizzata dei pesi ottimali tramite grid search.
*   **`parameter_search.py`**: Script di supporto per il tuning dei parametri di decadimento temporale e dei pesi interni al modulo Content-Based.
*   **`weight_optimization.md`**: Report dettagliato sulle fasi di ottimizzazione dei pesi e sulle performance ottenute durante i test.

## 3. Risultati Sperimentali (MINDsmall DEV)
Attraverso una ricerca sistematica (Grid Search), è stata individuata la configurazione ottimale che massimizza l'AUC:

| Configurazione | AUC | MRR | nDCG@5 | nDCG@10 | P@5 | R@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid (Optimized)** | **0.6624** | **0.3809** | **0.3616** | **0.4196** | **0.1269** | **0.4960** |
| Hybrid (Initial) | 0.6299 | 0.3569 | 0.3412 | 0.3961 | 0.1188 | 0.4731 |

**Configurazione Ottimale individuata:**
*   Baseline: 0% (la popolarità globale introduce rumore nel dataset MIND)
*   Collaborative Filtering: 3% (segnale complementare utile)
*   Content-Based Filtering: 97% (segnale dominante basato sulla semantica)

## 4. Conclusioni
L'approccio ibrido si è dimostrato il più efficace nel gestire il dataset MIND, dove il turnover degli articoli è estremamente alto. La predominanza del modulo Content-Based (SBERT) permette di gestire efficacemente il "Cold Start", mentre il modulo CF aggiunge una sfumatura di personalizzazione basata sul comportamento reale degli utenti.
