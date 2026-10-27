# NRMS — implementazione sperimentale e confronto con il CBF puro

Esperimento controllato, interamente contenuto in `hybrid/experiments/nrms/`. **Nessuna modifica ai recommender di produzione (`hybrid/hybrid_recommender.py`, CBF, CF, Baseline). Nessuna modifica a README.md, PRESENTAZIONE_PROGETTO.md, REVISIONE_PROGETTO.md. Nessuna aggiunta a `requirements.txt` principale** (PyTorch 2.11.0 era già presente nell'ambiente, verificato prima di iniziare — usato anche dalla fase Adaptive Hybrid precedente).

---

## 1. Motivazione

Introdurre NRMS (Wu et al., 2019) come primo modello neurale end-to-end del progetto, per rispondere alla domanda esplicita lasciata aperta dalla fase precedente: un modello che apprende direttamente rappresentazioni di news e interessi utente dal testo riesce a superare il CBF semantico (SBERT) attuale? **Non introdotto perché "è una rete neurale": introdotto per essere confrontato onestamente, con la possibilità esplicita che il risultato sia negativo.**

## 2. Perché la linea Adaptive Hybrid è stata chiusa

Risultati consolidati nella fase precedente (`hybrid/experiments/adaptive_hybrid/ADAPTIVE_HYBRID_REPORT.md`): l'Hybrid statico non offre vantaggio pratico sul CBF puro; il CF arricchito resta sostanzialmente casuale (AUC≈0,50); il Rule-Based Adaptive Hybrid non porta beneficio; la Neural Gating Hybrid **peggiora** il CBF in modo statisticamente robusto su MRR/nDCG@5. Conclusione della fase: il problema non è la strategia di blending, è la debolezza intrinseca del segnale collaborativo disponibile in MIND-small con questa costruzione del CF. Questa fase non ha quindi cercato di migliorare ulteriormente il CF, non ha eseguito grid search sui pesi, e non ha modificato il CBF esistente.

## 3. Protocollo temporale

Riutilizzato invariato il protocollo già congelato in `hybrid/experiments/temporal_split_definition.py`:

| Split | Date | Ruolo |
| :--- | :--- | :--- |
| **TRAIN** | 9-11 nov (61.417 righe) | Addestramento NRMS |
| **VALIDATION** | 12-13 nov (65.278 impression valutabili) | Early stopping, selezione (nessun'altra scelta di iperparametri oltre l'early stopping) |
| **INTERNAL_TEST** | 14 nov (30.270 righe) | Un'unica valutazione finale, a modello congelato |

Lo split DEV (15 nov) non è mai stato usato in questa fase. **Limitazione dichiarata esplicitamente**: INTERNAL_TEST era già stato osservato in fasi precedenti (`VALIDAZIONE_METODOLOGICA.md` sezione 9, `ADAPTIVE_HYBRID_REPORT.md`) per le metriche aggregate di altri modelli (CF/Baseline/CBF/Hybrid/Gating) — non è un campione "mai visto" in senso assoluto per il progetto, anche se non è mai stato usato per scegliere alcuna decisione relativa a NRMS (architettura, iperparametri, preprocessing, negative sampling, epoch).

## 4. Preprocessing

Tokenizzazione deterministica: `lowercase` → estrazione token via regex `[a-z0-9]+` (nessuno stemming/stopword removal, per restare semplice e riproducibile). Titoli troncati/paddati a `MAX_TITLE_LEN=20` token (PAD_IDX=0, UNK_IDX=1).

## 5. Vocabolario

Costruito **esclusivamente** dai titoli di `dataset/MINDsmall_train/news.tsv` (51.282 articoli, il catalogo dell'intero periodo TRAIN 9-14 novembre — mai da `dev/news.tsv`, per evitare esattamente il bias di catalogo di cui al punto 6). Soglia di frequenza minima = 2. Risultato: **31.003 token distinti osservati, 19.033 nel vocabolario finale** (incluso PAD/UNK). Salvato in `results/vocab.json`.

## 6. Catalog coverage

**Scoperta metodologica cruciale, gestita esplicitamente**: le fasi precedenti (`VALIDAZIONE_METODOLOGICA.md` sezione 10.2) avevano scoperto che `dev/news.tsv` copre solo il 38-86% dei candidati TRAIN a seconda del giorno, un bias di catalogo che aveva falsato un confronto CBF cross-giorno in una fase precedente. **NRMS qui usa `train/news.tsv` come catalogo per i titoli**, non `dev/news.tsv`:

| Split | % candidati con titolo (train/news.tsv) | % candidati con titolo (dev/news.tsv) | % clicked con titolo (train/news.tsv) | % clicked con titolo (dev/news.tsv) |
| :--- | ---: | ---: | ---: | ---: |
| TRAIN (9-11 nov) | **100,00%** | 47,18% | **100,00%** | 38,34% |
| VALIDATION (12-13 nov) | **100,00%** | 45,69% | **100,00%** | 36,50% |
| INTERNAL_TEST (14 nov) | **100,00%** | 85,70% | **100,00%** | 85,90% |

NRMS ha quindi **sempre** il titolo di ogni candidato del periodo TRAIN, a differenza del CBF (che dipende dagli embedding SBERT precalcolati, disponibili solo per `dev/news.tsv`). Questo però introduce un'asimmetria che va gestita esplicitamente per un confronto equo — si veda sezione 16.

## 7. News encoder

```
token del titolo (max 20) → embedding appreso (dim 128, PAD mascherato)
    → multi-head self-attention (8 head, mascherata sui token di PAD)
    → additive attention (mascherata) → news vector (dim 128)
```

## 8. User encoder

```
sequenza di news vector della history (max 20 news, left-padding)
    → multi-head self-attention (8 head, mascherata sugli slot vuoti)
    → additive attention (mascherata) → user vector (dim 128)
```

History costruita dal campo `history` della riga stessa (già causalmente valida: statica per utente, precede l'intera finestra TRAIN — verificato in `VALIDAZIONE_METODOLOGICA.md` sezione 9.5), troncata alle 20 news più recenti, **non include mai il candidato corrente**. History vuota → tutti gli slot sono padding, mascherati esplicitamente (nessun crash, `nan_to_num` applicato dopo il softmax mascherato).

## 9. Attention

Sia il news encoder sia lo user encoder usano **multi-head self-attention** (8 head, head_dim=16) seguita da **additive attention** (Bahdanau-style, dim=128) per l'aggregazione — fedele al paradigma NRMS originale, un solo layer per encoder (non è una riduzione: è la profondità standard del paper).

## 10. Loss

**Softmax cross-entropy** su (1 positivo + 4 negativi) per training sample, target sempre in posizione 0 (il modello non ha alcuna nozione di "posizione" appresa: lo scoring è un dot product indipendente per candidato, quindi fissare il positivo in posizione 0 non introduce alcun bias sfruttabile). Nessuna MSE.

## 11. Negative sampling

Per ogni (impression, candidato positivo) del TRAIN: **4 negativi** campionati dai candidati negativi della stessa impression (mai da altre impression, mai dal "futuro"). Seed derivato deterministicamente da `(seed_base=42, epoch, indice_sample)` tramite `hash()` su una tupla di soli interi (deterministico, non influenzato da `PYTHONHASHSEED`). Se l'impression ha meno di 4 negativi disponibili, si ricampiona **con ripetizione** (documentato esplicitamente in `data.py`). Il campionamento viene **ri-eseguito ad ogni epoch** (seed diverso per epoch, stesso comportamento riproducibile a parità di seed base). Dataset di training risultante: **90.030 sample** (da 61.417 righe TRAIN, media 1,47 positivi/riga).

## 12. Training

| Parametro | Valore |
| :--- | :--- |
| Seed | 42 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Gradient clipping | norm 5.0 |
| Max epoch | 15 |
| Patience (early stopping su val AUC) | 3 |
| Parametri totali | **2.601.600** |

Nessun tuning aggressivo: un'unica configurazione di iperparametri (dimensioni dichiarate in `config.py`), nessuna grid search.

## 13. Sanity check (eseguiti PRIMA del training completo, tutti superati)

| Check | Esito |
| :--- | :--- |
| A) Un batch passa nel modello | OK — `history=(8,20,20) candidate=(8,5,20) scores=(8,5)` |
| B) Dimensioni tensor coerenti | OK (verificato via assert espliciti) |
| C) Gli score differenziano candidati diversi (pesi non addestrati) | OK — deviazione standard media tra candidati = 0,0053 > 0 |
| D) La loss diminuisce | OK — 1,6106 → 0,4024 su 200 step, sottoinsieme di 64 sample |
| E) Il modello riesce a overfittare un piccolo campione | OK — training accuracy = **1,0000** su 64 sample |

Il training completo è stato lanciato solo dopo che tutti i check sono stati superati.

## 14. Risultati su VALIDATION

**Native (catalogo train/news.tsv, 100% copertura, tutti i candidati, 65.278 impression)** — early stopping, epoch 0-6, **best epoch=3**:

| Epoch | train_loss | val_AUC | val_MRR | val_nDCG@5 |
| :---: | ---: | ---: | ---: | ---: |
| 0 | 1,4021 | 0,5509 | 0,2914 | 0,2652 |
| 1 | 1,3591 | 0,5677 | 0,3021 | 0,2764 |
| 2 | 1,3196 | 0,5733 | 0,3065 | 0,2824 |
| **3** | **1,3009** | **0,5861** | **0,3130** | **0,2906** |
| 4 | 1,2765 | 0,5808 | 0,3127 | 0,2892 |
| 5 | 1,2602 | 0,5777 | 0,3086 | 0,2857 |
| 6 | 1,2436 | 0,5760 | 0,3073 | 0,2837 |

Early stopping all'epoch 6 (nessun miglioramento per 3 epoch dopo il picco).

## 15. Risultati su INTERNAL_TEST

Valutazione unica, a modello congelato (checkpoint dell'epoch 3). Riportati in dettaglio nella sezione 16 (confronto equo), essendo l'unico confronto metodologicamente valido con il CBF.

## 16. Confronto equo con il CBF

**Problema**: NRMS ha titolo per il 100% dei candidati (catalogo `train/news.tsv`); il CBF ha embedding SBERT solo per `dev/news.tsv` (38-86% di copertura a seconda del giorno). Confrontare le metriche "native" dei due modelli sarebbe un confronto non equo (il CBF perderebbe automaticamente su ogni candidato/cliccato assente dal suo catalogo, esattamente l'artefatto già documentato in `VALIDAZIONE_METODOLOGICA.md` sezione 10.2).

**Soluzione applicata**: per ogni impression, i candidati sono filtrati al sottoinsieme presente **anche** in `dev/news.tsv` (l'unico catalogo per cui il CBF ha una rappresentazione reale); l'impression è scartata se dopo il filtro non resta più almeno un positivo e un negativo. Entrambi i modelli sono poi valutati sugli **stessi identici candidate set ridotti**.

**Quota di dataset esclusa**:

| Split | Impression escluse | Candidati esclusi |
| :--- | ---: | ---: |
| VALIDATION | **60,05%** (26.078 rimaste su 65.278) | 72,21% |
| INTERNAL_TEST | **10,80%** (27.002 rimaste su 30.270) | 20,55% |

(La quota di esclusione riflette esattamente il pattern di copertura di `dev/news.tsv` già documentato: molto peggiore a metà settimana, molto migliore il 14 novembre, il giorno più vicino al 15.)

**Risultati sul common evaluation set:**

| Split | Modello | AUC | MRR | nDCG@5 | nDCG@10 | P@5 | R@5 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| VALIDATION (n=26.078) | **NRMS** | 0,5979 | 0,3279 | 0,3178 | 0,3870 | 0,1105 | 0,4689 |
| VALIDATION | **CBF puro** | **0,6180** | **0,3561** | **0,3486** | **0,4131** | **0,1184** | **0,5015** |
| INTERNAL_TEST (n=27.002) | **NRMS** | 0,5901 | 0,3009 | 0,2813 | 0,3425 | 0,0988 | 0,4067 |
| INTERNAL_TEST | **CBF puro** | **0,6325** | **0,3526** | **0,3336** | **0,3924** | **0,1159** | **0,4657** |

**Il CBF supera NRMS su ogni singola metrica, su entrambi gli split, sul confronto equo.** (Si veda `figures/01_nrms_vs_cbf_auc.png`.)

## 17. Bootstrap paired (2.000 resample, seed 42)

| Split | Metrica | mean diff (NRMS − CBF) | IC 95% | Esclude 0? |
| :--- | :--- | ---: | :--- | :---: |
| VALIDATION | AUC | -0,0201 | [-0,0246, -0,0157] | Sì |
| VALIDATION | MRR | -0,0282 | [-0,0324, -0,0235] | Sì |
| VALIDATION | nDCG@5 | -0,0308 | [-0,0352, -0,0261] | Sì |
| INTERNAL_TEST | AUC | -0,0424 | [-0,0469, -0,0380] | Sì |
| INTERNAL_TEST | MRR | -0,0517 | [-0,0558, -0,0474] | Sì |
| INTERNAL_TEST | nDCG@5 | -0,0523 | [-0,0563, -0,0480] | Sì |

**Distinzione tra significatività statistica e rilevanza pratica**: qui non serve nemmeno la distinzione — a differenza della fase Adaptive Hybrid (dove gli effetti erano nella quarta cifra decimale), questi intervalli sono **sia statisticamente non-casuali sia di ampiezza pratica reale** (2-5 punti di AUC, 3-5 punti di MRR/nDCG@5), consistenti in direzione su entrambi gli split indipendenti. Il CBF non vince per un margine trascurabile: vince in modo sostanziale.

## 18. Segment analysis

| Split | Segmento | n | NRMS AUC | CBF AUC | Δ |
| :--- | :--- | ---: | ---: | ---: | ---: |
| VALIDATION | History corta (≤5) | 4.074 | 0,5770 | 0,5755 | **+0,0015** |
| VALIDATION | History media (6-19) | 8.085 | 0,6028 | 0,6329 | -0,0300 |
| VALIDATION | History lunga (≥20, troncata) | 13.919 | 0,6011 | 0,6218 | -0,0207 |
| INTERNAL_TEST | History corta (≤5) | 4.586 | 0,5581 | 0,5843 | -0,0262 |
| INTERNAL_TEST | History media (6-19) | 8.806 | 0,6002 | 0,6455 | -0,0453 |
| INTERNAL_TEST | History lunga (≥20, troncata) | 13.610 | 0,5944 | 0,6403 | -0,0460 |

**Unico segnale a favore di NRMS**: su VALIDATION, per utenti con history molto corta (≤5), NRMS è marginalmente meglio del CBF (+0,0015 AUC — non testato con bootstrap dedicato, ampiezza compatibile con rumore). **Questo margine non si replica su INTERNAL_TEST** (stesso segmento: CBF meglio di -0,0262): non viene quindi interpretato come un vantaggio reale e riproducibile di NRMS sugli utenti cold-start, ma come un effetto isolato di uno split.

Nota sulla segmentazione: quasi metà delle impression ha `history_length` esattamente uguale al tetto di troncamento (20) — una segmentazione per tercili sarebbe stata degenere (il 66° percentile coincide col tetto); si sono quindi usate soglie fisse e dichiarate (≤5 / 6-19 / ≥20) anziché stimate sui dati.

## 19. Error analysis (contingenza TOP-5, solo impression a singolo positivo)

| Split | n | solo NRMS | solo CBF | entrambi | nessuno | NRMS rank migliore | CBF rank migliore | tie |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VALIDATION | 20.543 | 12,74% | 16,24% | 38,94% | 32,08% | 39,45% | 45,42% | 15,13% |
| INTERNAL_TEST | 19.585 | 10,35% | 16,25% | 37,48% | 35,92% | 36,31% | 46,18% | 17,50% |

Il CBF è "solo lui corretto" più spesso di NRMS su entrambi gli split (16% vs 10-13%), e piazza il cliccato in una posizione migliore nel 45-46% delle impression contro il 36-39% di NRMS. **NRMS non introduce una capacità sistematicamente nuova**: la quota "solo NRMS corretto" (10-13%) non è trascurabile in assoluto, ma è sistematicamente inferiore alla quota equivalente del CBF, in entrambi gli split.

## 20. Efficienza

| Metrica | Valore |
| :--- | :--- |
| Tempo preprocessing (tokenizzazione, dataset TRAIN+VALIDATION) | 4,2 s |
| Tempo training (7 epoch, fino a early stopping) | 5.560,8 s (~92,7 min) |
| Tempo inferenza su VALIDATION (65.278 impression) | 516,0 s (~7,9 ms/impression) |
| Dimensione checkpoint | 9,93 MB |
| Numero parametri | 2.601.600 |

**Confronto qualitativo col CBF**: il CBF non richiede training (solo un calcolo di similarità coseno su embedding SBERT pre-esistenti) e la sua inferenza è dello stesso ordine di grandezza o più rapida per candidato. NRMS richiede ~93 minuti di training su CPU per un'unica configurazione (nessuna grid search) e produce un modello che, oltre a essere più costoso, **è anche meno accurato** sul confronto equo.

## 21. Limitazioni

1. **Nessun embedding pretrained**: gli embedding NRMS sono appresi da zero su un vocabolario di soli titoli TRAIN — un limite dichiarato in partenza; embedding pretrained (es. GloVe) potrebbero migliorare NRMS, ma introdurrebbero una dipendenza esterna non disponibile localmente.
2. **Nessuna grid search**: una sola configurazione di iperparametri testata (dim=128, 8 head, batch=64, lr=1e-3); non si può escludere che una configurazione diversa migliori NRMS, ma questo esula dallo scopo "baseline NRMS pulita" di questa fase.
3. **Solo il titolo come input**: NRMS qui non vede abstract, categoria o entità, mentre il CBF combina similarità semantica (SBERT, addestrato su corpus più ampi) e categoria. Un confronto più ricco (NRMS con più campi testuali) è un'estensione naturale ma esplicitamente rimandata.
4. **Common evaluation set molto più piccolo del dataset originale su VALIDATION** (39,95% rimasto): il confronto equo, per costruzione, esclude la maggioranza delle impression di metà settimana — un limite del catalogo disponibile per il CBF (embedding SBERT solo per `dev/news.tsv`), non di NRMS.
5. **INTERNAL_TEST parzialmente osservato in precedenza** (sezione 3): dichiarato esplicitamente, anche se non è mai stato usato per alcuna decisione relativa a NRMS.
6. **Un solo seed, un solo run**: non è stata eseguita una ripetizione multi-seed per stimare la varianza dovuta all'inizializzazione della rete — il margine osservato (2-5 punti di AUC) è comunque ampio rispetto alla varianza tipica di re-run con seed diversi in problemi di questa scala, ma non è stato verificato direttamente.

## 22. Conclusione

**Classificazione secondo il criterio pre-registrato**:

> **NESSUN MIGLIORAMENTO.** NRMS non supera il CBF puro su alcuna metrica, su nessuno dei due split, sul confronto equo (common evaluation set). Il CBF è sistematicamente e sostanzialmente migliore (2-5 punti di AUC, differenze statisticamente robuste via bootstrap paired e consistenti in direzione su entrambi gli split indipendenti).

Il protocollo non è stato alterato per far vincere NRMS: il risultato è riportato come osservato. Questo è un risultato metodologicamente onesto e non banale: un CBF basato su SBERT (un encoder pre-addestrato su corpus enormi e poi congelato) rappresenta un avversario molto forte per un NRMS addestrato da zero su ~90.000 sample con un vocabolario di soli 19.000 token — le architetture NRMS nella letteratura raggiungono generalmente prestazioni competitive con budget di dati ed epoch training ben superiori, spesso con embedding pretrained. Con le risorse e i vincoli di questa fase (nessun embedding esterno, nessuna grid search, singolo run), **NRMS non ha ancora raggiunto il livello del CBF esistente**.

**Non si raccomanda di sostituire il CBF con NRMS allo stato attuale.** Estensioni naturali per una fase successiva (esplicitamente non implementate qui, per rispettare il vincolo di questa fase): embedding pretrained, più campi testuali (abstract, categoria), più epoch/dati di training, eventualmente un ensemble NRMS+CBF — ma solo dopo aver verificato singolarmente l'effetto di ciascuna estensione sulla baseline qui stabilita.
