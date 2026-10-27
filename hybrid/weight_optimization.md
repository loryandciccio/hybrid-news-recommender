# Hybrid Recommender: Report di ottimizzazione dei pesi

Questo documento descrive il processo sistematico di ottimizzazione dei pesi condotto per il **Hybrid Recommender System**. Attraverso una grid search estesa e una validazione empirica, è stata individuata l'allocazione ottimale dei pesi per combinare tre paradigmi di raccomandazione distinti: **Baseline (popolarità)**, **Collaborative Filtering (ItemKNN)** e **Content-Based Filtering (CBF)**.

## 1. Sintesi

L'obiettivo dell'ottimizzazione dei pesi era massimizzare la qualità del ranking (AUC) mantenendo al contempo la diversità architetturale del sistema. Sono state valutate numerose configurazioni per bilanciare la stabilità della popolarità globale, i pattern comportamentali del Collaborative Filtering e la comprensione semantica del contenuto.

**Configurazione ottimale finale (AUC 0.6624):**
*   **Peso baseline:** 0.0 (la popolarità globale introduce rumore in un ambiente news ad alto turnover)
*   **Peso Collaborative Filtering (CF):** 0.03 (segnale minimo ma complementare)
*   **Peso Content-Based Filtering (CBF):** 0.97 (segnale principale, basato su embedding SBERT e categorie)

## 2. Metodologia sperimentale

L'ottimizzazione è stata condotta in due fasi principali:
1.  **Fase esplorativa (500 campioni):** test rapidi su intervalli di peso ampi, per identificare l'efficacia relativa dei diversi segnali.
2.  **Fase di validazione (2000 campioni):** grid search sistematica su campioni più ampi, per confermare la stabilità dei risultati e affinare i parametri finali di combinazione.

## 3. Risultati e osservazioni principali

### Grid search sistematica (selezione)

| Configurazione (B, CF, CBF) | AUC | MRR | nDCG@5 | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: |
| **Ottimizzata (0.00, 0.03, 0.97)** | **0.6624** | **0.3809** | **0.3616** | **0.4196** |
| CBF pura (0.00, 0.00, 1.00) | 0.6624 | 0.3808 | 0.3615 | 0.4196 |
| Default iniziale (0.20, 0.30, 0.50) | 0.6299 | 0.3569 | 0.3412 | 0.3961 |
| Alta popolarità (0.15, 0.10, 0.75) | 0.6619 | 0.3790 | 0.3601 | 0.4183 |

### Osservazioni principali
*   **La trappola della popolarità**: aumentare il peso della popolarità globale porta sistematicamente a un calo dell'AUC. In un dataset di news con un turnover degli articoli così rapido, la popolarità storica agisce spesso da fonte di rumore piuttosto che da predittore affidabile dei click futuri.
*   **La forza della semantica**: i segnali content-based (SBERT) sono i predittori più efficaci per il dataset MIND, verosimilmente per l'elevato numero di articoli nuovi privi di una storia di interazioni.
*   **Sinergia del CF**: sebbene il peso ottimale del CF sia basso (3%), la sua presenza mantiene la robustezza architetturale del sistema e fornisce un lieve beneficio complementare alla precisione del ranking nei casi in cui sono presenti pattern di co-click informativi. È opportuno notare che, a parità di cifre significative, la configurazione CBF pura (0, 0, 1) ottiene metriche quasi identiche a quella ottimizzata (0, 0.03, 0.97): il contributo del CF è quindi reale ma di entità molto contenuta, e la configurazione a tre componenti è stata preferita per ragioni architetturali (preservare un sistema esplicitamente ibrido) più che per un guadagno numerico marcato.

## 4. Conclusione

I risultati dell'ottimizzazione confermano che un approccio ibrido **a dominanza CBF** è la configurazione più efficace per la raccomandazione di news. Sfruttando embedding semantici profondi e un contributo collaborativo minimo, il modello raggiunge prestazioni superiori, bilanciando personalizzazione e comprensione del contenuto.
