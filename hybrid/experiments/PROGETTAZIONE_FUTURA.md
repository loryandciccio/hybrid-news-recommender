# Progettazione futura (non implementata)

Documento di supporto ad [ANALISI_COMPLEMENTARITA.md](ANALISI_COMPLEMENTARITA.md), sezioni 15-16. La classificazione finale di quel report è **Scenario B — complementarità debole ma presente**, che raccomanda di confrontare successivamente un Rule-Based Adaptive Hybrid, una Neural Gating Hybrid (progettata qui sotto) e il CBF puro — **prima** di considerare un Neural Recommender end-to-end come NRMS (comunque descritto qui sotto per completezza, nel caso le opzioni più semplici si rivelassero insufficienti). Nessuna di queste architetture è stata implementata in questa fase.

## Fase 17 — Progetto (non implementato) di una Neural Gating Network

### Feature di input candidate

| Feature | Definizione | Provenienza | Disponibile al momento della raccomandazione? | Rischio di leakage |
| :--- | :--- | :--- | :--- | :--- |
| `history_length` | Numero di articoli nella cronologia dell'utente | Campo `history` dell'impression corrente | Sì | Nessuno |
| `cf_score_norm` | Score CF normalizzato (Min-Max) del candidato, da `_get_cf_scores` | Matrice ItemKNN costruita sul TRAIN + history dell'impression corrente | Sì | Nessuno (già usato nella pipeline attuale) |
| `cbf_score_norm` | Score CBF normalizzato (Min-Max) del candidato, da `_get_cbf_scores` | Embedding SBERT + categoria + history dell'impression corrente | Sì | Nessuno |
| `score_diff = cf_score_norm - cbf_score_norm` | Differenza tra i due segnali | Derivata dalle due sopra | Sì | Nessuno |
| `cf_score_std_candidate_set` | Deviazione standard del punteggio CF grezzo sull'intero candidate set dell'impression | Calcolabile durante lo scoring dell'impression | Sì | Nessuno |
| `cbf_score_std_candidate_set` | Come sopra, per il CBF | Come sopra | Sì | Nessuno |
| `candidate_train_popularity` | Conteggio di click del candidato nel TRAIN (o trasformata log/sqrt) | `item_popularity` calcolato solo sul TRAIN | Sì | Nessuno, purché mai ricalcolato includendo il DEV |
| `cold_start_flag` | 1 se `candidate_train_popularity == 0`, altrimenti 0 | Come sopra | Sì | Nessuno |
| `candidate_category` (one-hot o embedding) | Categoria editoriale del candidato | `news.tsv` (metadato statico) | Sì | Nessuno |
| `max_semantic_sim_history` | Similarità coseno massima tra il candidato e i singoli articoli della history (non il centroide) | Embedding SBERT, calcolabile dalla history | Sì | Nessuno |
| `mean_semantic_sim_history` | Come sopra, media invece di massimo | Come sopra | Sì | Nessuno |
| `history_recency_weight_sum` | Somma dei pesi di recency applicati alla history (proxy della "freschezza" del profilo) | Calcolabile dalla history con timestamp dell'impression | Sì | Nessuno |

**Feature esplicitamente escluse per rischio di leakage**: qualunque statistica calcolata sul DEV (es. CTR del candidato nel DEV), qualunque informazione derivata dalla label `clicked` dell'impression corrente o di impression future dello stesso utente.

### Architettura proposta (minima, non implementata)

```
Input (feature per candidato, elencate sopra)
        ↓
Dense(32) + ReLU
        ↓
Dense(16) + ReLU
        ↓
Dense(2 o 3)      # 2 se si esclude la baseline (si veda Fase 11), 3 se la si mantiene
        ↓
Softmax
        ↓
pesi dinamici (w_CF, w_CBF[, w_B]) per impression/utente
        ↓
score_finale(c) = w_CF · cf_score_norm(c) + w_CBF · cbf_score_norm(c) [+ w_B · baseline_score_norm(c)]
```

Rete deliberatamente piccola (≈600-700 parametri) per non sovradimensionare rispetto alla dimensione del problema (poche feature scalari, non rappresentazioni ad alta dimensionalità).

### Protocollo di training proposto

- **Loss**: cross-entropy binaria sulla label clicked/non-clicked, oppure una loss listwise (es. softmax cross-entropy sul candidate set dell'impression, coerente con l'impostazione MIND) — da validare empiricamente quale converga meglio.
- **Split**: la gating network andrebbe allenata su una porzione del TRAIN (non del DEV, per evitare di ripetere il problema metodologico già documentato per la grid search statica) oppure, se si accetta di restare nel DEV per mancanza di label sul TRAIN in questo formato, su uno split esplicito e temporalmente coerente del DEV in train/validation/test interni (es. primi 2/3 delle impression per training+validation, ultimo 1/3 mai toccato per il test finale) — dichiarato esplicitamente, senza usare lo stesso DEV per tuning e valutazione finale come accaduto per i pesi statici.
- **Early stopping**: su una metrica di validazione (es. AUC), pazienza di poche epoche data la piccola dimensione del modello.
- **Seed**: fisso e dichiarato, per riproducibilità.
- **Metriche**: le stesse già usate nel progetto (AUC, MRR, nDCG@5/10, Precision@5, Recall@5, Novelty@5/10), per restare comparabile con i risultati esistenti.
- **Baseline di confronto**: Hybrid statico 0/3/97 (configurazione attuale) e CBF puro 0/0/1 — la gating network ha senso solo se supera entrambi in modo consistente, non sporadico.

## Fase 18 — Scheda progettuale (non implementata) per NRMS

**Cosa rappresenterebbe**: non un'ottimizzazione dei pesi dell'Hybrid, ma un **paradigma alternativo**: un modello neurale end-to-end che apprende rappresentazioni di news e di interesse utente direttamente dai dati, da usare come benchmark/estensione rispetto all'attuale pipeline a expert combinati linearmente.

**Input**: titolo (e opzionalmente abstract/categoria) degli articoli in cronologia e dei candidati, in forma di sequenze di token o embedding pre-addestrati.

**News encoder**: tipicamente CNN o self-attention sui token del titolo (nell'architettura originale NRMS: multi-head self-attention + additive attention) per produrre una rappresentazione vettoriale per articolo. Nel contesto di questo progetto, gli embedding SBERT già disponibili potrebbero costituire un punto di partenza o un termine di confronto per l'encoder appreso.

**User encoder**: multi-head self-attention sulla sequenza di rappresentazioni degli articoli in cronologia, seguita da additive attention per aggregare in un unico vettore utente — sostituirebbe il centroide/centroide pesato per recency usato oggi nel CBF.

**Candidate scoring**: prodotto scalare (o simile) tra vettore utente e vettore di ciascun candidato.

**Negative sampling**: nell'NRMS originale, K negativi campionati per ogni positivo all'interno della stessa impression (coerente con il formato nativo di MIND, che fornisce già liste di candidati con etichette).

**Loss**: softmax cross-entropy sul positivo contro i K negativi campionati (o su tutto il candidate set dell'impression).

**Training/inference**: training su mini-batch di impression; inference come forward pass su history + candidati.

**Costo computazionale**: significativamente più alto della pipeline attuale (SBERT + ItemKNN + popolarità), sia in training (richiede backpropagation su un encoder testuale, tipicamente per diverse epoche sull'intero TRAIN) sia in inference. Su MIND-small è comunque il regime "small" per cui NRMS è stato originariamente validato in letteratura, quindi il costo è gestibile ma non trascurabile su CPU.

**Dipendenze necessarie**: un framework di deep learning (PyTorch o TensorFlow — non introdotti in questa fase), più eventualmente `torchtext`/tokenizzatori.

**Necessità di GPU**: non strettamente indispensabile su MIND-small con un encoder leggero, ma fortemente consigliata per tempi di training ragionevoli; su CPU l'addestramento sarebbe possibile ma lento.

**Compatibilità con MIND-small**: piena — NRMS è stato proposto e validato proprio sul dataset MIND.

**Confronto da effettuare con il CBF SBERT esistente**: NRMS apprende end-to-end una rappresentazione ottimizzata per il task di click prediction, mentre il CBF attuale usa embedding SBERT generici (pre-addestrati per similarità semantica generale, non per predizione di click) combinati con un profilo utente non appreso (centroide pesato). Il confronto rilevante è se un encoder appreso end-to-end sul segnale di click superi in modo sostanziale un embedding semantico generico.

**Collocazione rispetto alla classificazione Scenario B**: i risultati di questa fase non giustificano di saltare direttamente a NRMS. Andrebbe considerato solo come passo successivo, qualora un Rule-Based Adaptive Hybrid e una Neural Gating Hybrid (entrambi molto più economici) non riuscissero a recuperare il margine di complementarità osservato (si veda ANALISI_COMPLEMENTARITA.md, sezioni 14-15), oppure come esperimento parallelo per capire se il CBF stesso — non solo la sua combinazione con CF/Baseline — sia migliorabile con rappresentazioni apprese end-to-end.
