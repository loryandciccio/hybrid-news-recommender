# Adaptive Hybrid: Rule-Based e Neural Gating — implementazione sperimentale e valutazione

Esperimento controllato, interamente contenuto in `hybrid/experiments/adaptive_hybrid/`. **Nessuna modifica a `hybrid/hybrid_recommender.py` o ad altri recommender di produzione. Nessuna modifica a README.md, PRESENTAZIONE_PROGETTO.md, REVISIONE_PROGETTO.md. Nessuna implementazione di NRMS. Il CBF esistente non è stato sostituito, solo riutilizzato come componente.**

---

## 1. Motivazione

Le fasi precedenti (`ANALISI_COMPLEMENTARITA.md`, `VALIDAZIONE_METODOLOGICA.md`) avevano stabilito un GO condizionato per un Adaptive Hybrid, sulla base di un segnale diagnostico debole ma statisticamente robusto: quando il CF ha una rappresentazione non degenere, tende a piazzare l'item cliccato meglio del CBF nel 58% dei casi, e un classificatore diagnostico minimo (Logistic Regression) prevedeva questa condizione con ROC-AUC=0,58 (IC 95% sopra 0,50). Questa fase implementa e valuta, per la prima volta, un Rule-Based Adaptive Hybrid e una Neural Gating Network minimale, per verificare se questo segnale diagnostico si traduce in un beneficio di raccomandazione reale — non solo in una previsione corretta di quando il CF "vince" un confronto diagnostico.

## 2. L'artefatto CF degenere

Confermato nelle fasi precedenti: su VALIDATION, circa il 67-76% delle impression (a seconda del filtro di valutabilità) ha punteggio CF identico per tutti i candidati (nessun candidato ha una co-occorrenza osservata in FIT). In questi casi il ranking del CF non contiene alcuna informazione: qualunque differenza rispetto al CBF puro sarebbe un artefatto del tie-break deterministico su `news_id`, non un segnale di raccomandazione. Qualunque combinazione CF+CBF **deve** annullare esplicitamente il contributo del CF in questi casi.

## 3. Definizione di CF_AVAILABLE

```
CF_AVAILABLE = 1  se  max(cf_raw) - min(cf_raw) > EPSILON_CF
CF_AVAILABLE = 0  altrimenti
```

con `EPSILON_CF = 1e-9` (documentato in `common.py`). Gli score CF di questo progetto sono shrinkage Bayesiani non negativi, tipicamente esattamente 0,0 quando non c'è nessun vicino osservato: un ranking degenere produce quasi sempre un range esattamente 0,0, quindi la scelta esatta della soglia ha un impatto trascurabile. Verificato empiricamente: nessuna impression con range in (0, 1e-6].

Tasso di CF_AVAILABLE=1 osservato: **29,94%** su GATING_TRAIN (12 nov), **32,60%** su GATING_VALIDATION (13 nov), **23,58%** su INTERNAL_TEST (14 nov, impression a singolo positivo — leggermente diverso perché qui la CF_enriched è rifittata su FIT+VALIDATION più esteso, si veda sezione 5).

**Sia il Rule-Based sia la Neural Gating impongono `w_CF = 0` per costruzione quando CF_AVAILABLE=0** (non è un comportamento appreso o scelto: è un vincolo strutturale, verificato in sezione 12 — le metriche sul sottoinsieme CF_AVAILABLE=0 sono identiche bit-per-bit per tutti e quattro i metodi confrontati).

## 4. Feature utilizzate

Tabella feature → sorgente → rischio di leakage (tutte calcolabili prima di conoscere il candidato cliccato):

| Feature | Sorgente | Rischio di leakage |
| :--- | :--- | :--- |
| `history_length` | Campo `history` dell'impression | Nessuno |
| `candidate_count` | Numero di candidati dell'impression | Nessuno |
| `CF_AVAILABLE` | `max(cf_raw)-min(cf_raw) > ε`, calcolato sui candidati (non sul cliccato) | Nessuno |
| `cf_score_std`, `cf_score_max` | Statistiche sui punteggi CF grezzi di TUTTI i candidati | Nessuno |
| `cf_gap_top1_top2` | Differenza tra il 1° e 2° punteggio CF più alto tra i candidati | Nessuno |
| `cbf_score_std`, `cbf_score_max` | Come sopra, per il CBF | Nessuno |
| `cbf_gap_top1_top2` | Come sopra | Nessuno |
| `top5_overlap_cf_cbf` | Overlap tra i top-5 CF e top-5 CBF tra i candidati | Nessuno (non usa la label) |
| `spearman_cf_cbf` | Correlazione di Spearman tra i vettori score CF e CBF sui candidati | Nessuno (non usa la label) |
| `warm_candidate_ratio` / `pct_cf_representation` | Quota di candidati con `item_counts>0` in FIT | Nessuno |
| `popularity_mean_fit`, `popularity_median_fit` | Popolarità (history-only) dei candidati, calcolata **solo su FIT** | Nessuno |

**Esplicitamente escluse**: identità/rank/score del candidato cliccato, `CF_WINS` (era solo una label diagnostica), qualunque statistica calcolata usando la label o dati futuri rispetto all'impression.

## 5. Split temporale

Riutilizzato integralmente il protocollo già congelato in `temporal_split_definition.py`:

| Split | Date | Ruolo qui |
| :--- | :--- | :--- |
| FIT | 9-11 nov (61.417 righe) | Costruzione CF_enriched per lo sviluppo (GATING_TRAIN/VALIDATION) |
| **GATING_TRAIN** | 12 nov | Addestramento della gating (loss pairwise) e stima soglie rule-based |
| **GATING_VALIDATION** | 13 nov | Early stopping, selezione architettura/iperparametri, selezione regola |
| FIT+VALIDATION | 9-13 nov (126.695 righe) | Ricostruzione CF_enriched per la valutazione finale (stessa scelta di `run_cf_enriched_experiment.py`) |
| **INTERNAL_TEST** | 14 nov | Un'unica valutazione finale, a configurazione congelata |

Lo split DEV (15 nov) non è mai stato usato in questa fase. **Nota dichiarata esplicitamente**: INTERNAL_TEST era già stato osservato in `VALIDAZIONE_METODOLOGICA.md` (sezione 9, metriche aggregate di CF_original/CF_enriched/CBF/Baseline) — non è quindi un campione "mai visto" in senso assoluto per il progetto, anche se non è mai stato usato per scegliere feature/soglie/architettura di questa fase Adaptive Hybrid.

Numero di impression valutabili per split: GATING_TRAIN=32.878, GATING_VALIDATION=30.893, INTERNAL_TEST=29.498 (tutte le impression con ≥1 positivo e ≥1 negativo).

## 6. Rule-Based Adaptive Hybrid: design

Sei configurazioni (comprese le due di controllo), tutte con `w_CF=0` forzato se CF_AVAILABLE=0:

- **R0/R1** — Hybrid statico attuale (0,03/0,97), riferimento.
- **R2** — se `cbf_score_std` < mediana (stimata su GATING_TRAIN, tra i soli CF_AVAILABLE=1): w_CF=0,10; altrimenti 0,03.
- **R3** — stessa logica su `cbf_score_max`.
- **R4** — come R2 ma con soglia di peso più aggressiva (0,20 invece di 0,10).
- **R5** — combinazione: `cbf_score_std` basso E `cf_gap_top1_top2` positivo → w_CF=0,10.

Soglie stimate **esclusivamente su GATING_TRAIN** (mediana `cbf_score_std`=0,2433, mediana `cbf_score_max`=0,9298 tra i soli CF_AVAILABLE=1), selezione della regola migliore fatta guardando **solo** GATING_VALIDATION.

**Risultato della selezione**: nessuna delle regole R2-R5 supera in modo apprezzabile R0/CBF puro su GATING_VALIDATION (AUC 0,5232-0,5234 per tutte, differenze nella quarta cifra decimale). **R3** (soglia su `cbf_score_max`) è stata scelta come "migliore" solo per un margine di AUC di 0,0001 — un margine non interpretabile come reale vantaggio, riportato per trasparenza sulla procedura di selezione più che come segnale di successo.

## 7. Architettura Neural Gating

```
Input (14 feature standardizzate) → Dense(hidden[0]) → ReLU → [Dense(hidden[1]) → ReLU] → Dense(2) → Softmax → [w_CF, w_CBF]
```

Se `CF_AVAILABLE=0`: `w_CF` forzato a 0 e `w_CBF` a 1 **dopo** il softmax, tramite maschera esplicita e rinormalizzazione (`gating_network.py:GatingNet.forward`) — la rete non può in alcun modo assegnare peso effettivo al CF quando il suo ranking è degenere, indipendentemente da cosa produrrebbe il softmax grezzo.

Score finale per candidato: `score = w_CF * CF_norm + w_CBF * CBF_norm` (stessa normalizzazione min-max per candidate-set di `hybrid_recommender.py`).

Confrontate 4 configurazioni predefinite (nessuna grid search estesa): `hidden ∈ {[8], [16,8]}` × `lr ∈ {1e-3, 1e-2}`. Selezionata `hidden=[8], lr=1e-2` (138 parametri) per il miglior AUC su GATING_VALIDATION (0,5227) — comunque **inferiore** al riferimento CBF puro (0,5234) sulla stessa popolazione.

## 8. Loss

Pairwise ranking loss BPR-style, calcolata per ogni impression su tutte le coppie (candidato positivo, candidato negativo):

```
L = -mean( logsigmoid(score_positivo - score_negativo) )
```

mai addestrata a riprodurre `CF_WINS` (che resta solo una label diagnostica delle fasi precedenti). Nessuna MSE sui pesi.

## 9. Protocollo di training

- Seed fisso = 42.
- Scaler (`StandardScaler`) fittato **esclusivamente su GATING_TRAIN** (12 nov), applicato invariato a GATING_VALIDATION/INTERNAL_TEST.
- Batch size = 512 impression; ottimizzatore Adam.
- Early stopping su AUC di GATING_VALIDATION, pazienza = 5 epoch, massimo 25 epoch.
- Nessuna regolarizzazione aggiuntiva (rete già molto piccola: 138-394 parametri).
- Tutte le epoch, i learning rate, il seed e il conteggio parametri sono registrati in `results/gating_training_log.json`.

Esito delle 4 configurazioni (early stopping tra 6 e 25 epoch, nessuna ha superato 0,523 di AUC su GATING_VALIDATION):

| hidden | lr | best val_AUC | n_param | epoch di stop |
| :--- | ---: | ---: | ---: | ---: |
| [8] | 1e-3 | 0,5223 | 138 | 6 |
| **[8]** | **1e-2** | **0,5227** | **138** | **8** |
| [16,8] | 1e-3 | 0,5226 | 394 | 25 |
| [16,8] | 1e-2 | 0,5225 | 394 | 8 |

## 10. Risultati su GATING_VALIDATION (13 novembre, 30.893 impression)

**Tutte le impression:**

| Metodo | AUC | MRR | nDCG@5 | nDCG@10 | P@5 | R@5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CBF puro | 0,5234 | 0,2901 | 0,2620 | 0,3178 | 0,0919 | 0,3795 |
| Hybrid statico 0,03/0,97 | 0,5234 | 0,2899 | 0,2619 | 0,3177 | 0,0919 | 0,3795 |
| Rule-Based (R3) | 0,5233 | 0,2897 | 0,2618 | 0,3176 | 0,0919 | 0,3795 |
| **Neural Gating** | **0,5227** | **0,2850** | **0,2586** | 0,3148 | 0,0912 | 0,3778 |

## 11. Risultati su INTERNAL_TEST (14 novembre, 29.498 impression, valutazione unica e congelata)

**Tutte le impression:**

| Metodo | AUC | MRR | nDCG@5 | nDCG@10 | P@5 | R@5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CBF puro | 0,6104 | 0,3318 | 0,3054 | 0,3630 | 0,1079 | 0,4231 |
| Hybrid statico 0,03/0,97 | 0,6105 | 0,3320 | 0,3054 | 0,3631 | 0,1078 | 0,4230 |
| Rule-Based (R3) | 0,6105 | 0,3320 | 0,3054 | 0,3631 | 0,1079 | 0,4231 |
| **Neural Gating** | 0,6103 | **0,3276** | **0,3033** | 0,3610 | 0,1072 | 0,4227 |

## 12. Metriche su CF_AVAILABLE=1

| Split | Metodo | AUC | MRR | nDCG@5 |
| :--- | :--- | ---: | ---: | ---: |
| VALIDATION (n=10.071) | CBF puro | 0,5347 | 0,2004 | 0,1426 |
| VALIDATION | Hybrid statico | 0,5347 | 0,1997 | 0,1424 |
| VALIDATION | Rule-Based | 0,5345 | 0,1990 | 0,1420 |
| VALIDATION | **Neural Gating** | **0,5324** | **0,1846** | **0,1324** |
| INTERNAL_TEST (n=6.956) | CBF puro | 0,6214 | 0,2388 | 0,1701 |
| INTERNAL_TEST | Hybrid statico | 0,6216 | 0,2395 | 0,1703 |
| INTERNAL_TEST | Rule-Based | 0,6218 | 0,2395 | 0,1704 |
| INTERNAL_TEST | **Neural Gating** | 0,6208 | **0,2208** | **0,1615** |

**Proprio nel sottoinsieme dove il CF ha una rappresentazione reale — l'unico dove ci si aspettava un beneficio — la Neural Gating è sistematicamente la configurazione peggiore, su entrambi gli split.**

## 13. Metriche su CF_AVAILABLE=0

Su entrambi gli split, **le quattro configurazioni producono metriche identiche fino alla quarta cifra decimale** (es. INTERNAL_TEST: AUC=0,6070 per tutte e quattro). Questo conferma che il vincolo strutturale funziona correttamente: quando CF_AVAILABLE=0, tutti i metodi collassano esattamente su CBF puro, senza eccezioni né contaminazioni dall'algoritmo di gating o dalla regola.

## 14. Delta rispetto al CBF (bootstrap paired, 2.000 resample, seed 42)

| Confronto | Metrica | VALIDATION mean Δ | VALIDATION IC 95% | INTERNAL_TEST mean Δ | INTERNAL_TEST IC 95% |
| :--- | :--- | ---: | :--- | ---: | :--- |
| Hybrid statico − CBF | AUC | +0,00001 | [-0,00007, 0,00010] ns | +0,00004 | [-0,00000, 0,00008] ns |
| Hybrid statico − CBF | MRR | -0,00022 | [-0,00040, -0,00005] * | +0,00015 | [-0,00006, 0,00038] ns |
| Rule-Based − CBF | AUC | -0,00008 | [-0,00017, 0,00003] ns | **+0,00008** | **[0,00001, 0,00016] \*** |
| Rule-Based − CBF | MRR | -0,00044 | [-0,00069, -0,00017] * | +0,00015 | [-0,00013, 0,00047] ns |
| **Neural Gating − CBF** | **AUC** | -0,00077 | **[-0,00099, -0,00053] \*** | -0,00015 | [-0,00040, 0,00011] ns |
| **Neural Gating − CBF** | **MRR** | -0,00513 | **[-0,00618, -0,00407] \*** | -0,00424 | **[-0,00541, -0,00304] \*** |
| **Neural Gating − CBF** | **nDCG@5** | -0,00334 | **[-0,00410, -0,00255] \*** | -0,00203 | **[-0,00289, -0,00114] \*** |

`*` = IC 95% esclude lo 0 (statisticamente non-casuale); `ns` = IC include lo 0.

**Distinzione esplicita tra significatività statistica e rilevanza pratica**: l'unico effetto positivo statisticamente non-casuale (Rule-Based su INTERNAL_TEST, ΔAUC=+0,00008) è di **quattro ordini di grandezza troppo piccolo per essere praticamente rilevante** — non viene presentato come un successo. Al contrario, gli effetti negativi della Neural Gating su MRR e nDCG@5 sono sia statisticamente non-casuali (CI stretti, mai vicini allo 0) sia **coerenti in direzione su entrambi gli split indipendenti** (VALIDATION e INTERNAL_TEST) — un pattern più solido, benché anch'esso di ampiezza assoluta modesta (variazioni nella terza cifra decimale).

## 15. Distribuzione dei pesi appresi (Neural Gating)

| Statistica | VALIDATION | INTERNAL_TEST |
| :--- | ---: | ---: |
| `w_CF` medio (tutte le impression) | 0,167 | 0,130 |
| `w_CF` medio quando CF_AVAILABLE=1 | **0,512** | **0,550** |
| `w_CF` medio quando CF_AVAILABLE=0 | 0,000 (per costruzione) | 0,000 (per costruzione) |
| `w_CF` percentile 50 (mediana, tutte) | 0,000 | 0,000 |
| `w_CF` percentile 95 (tutte) | 0,868 | 0,841 |

**Per quintile di `cbf_score_std`** (solo CF_AVAILABLE=1, quintile 1=dispersione più bassa):

| Quintile | VALIDATION w_CF | INTERNAL_TEST w_CF |
| :---: | ---: | ---: |
| 1 (bassa dispersione CBF) | 0,764 | 0,761 |
| 2 | 0,623 | 0,642 |
| 3 | 0,511 | 0,558 |
| 4 | 0,392 | 0,455 |
| 5 (alta dispersione CBF) | 0,268 | 0,334 |

Si veda anche `figures/02_gating_weight_by_cbf_std.png`.

**Per quintile di `history_length`**: pattern debole ma nella direzione attesa (history più lunga → w_CF leggermente più alto: da ~0,45-0,52 nei primi quattro quintili a 0,69/0,69 nell'ultimo, su entrambi gli split).

**Per quintile di `warm_candidate_ratio`**: nessun pattern chiaro (oscilla tra 0,48 e 0,59 senza monotonia).

## 16. Bootstrap / effect size

Riportato in dettaglio in sezione 14. Riassunto: nessun effetto positivo di rilevanza pratica; un effetto negativo piccolo ma statisticamente robusto e riproducibile su due split indipendenti per la Neural Gating (MRR, nDCG@5).

## 17. Rule-Based vs Neural Gating

| Aspetto | Rule-Based | Neural Gating |
| :--- | :--- | :--- |
| AUC vs CBF (VALIDATION / INTERNAL_TEST) | ~pari / +0,00008 (ns praticamente) | -0,00077* / -0,00015 (ns) |
| MRR vs CBF | -0,00044* / +0,00015 (ns) | **-0,00513\* / -0,00424\*** |
| nDCG@5 vs CBF | -0,00021 (ns) / +0,00008 (ns) | **-0,00334\* / -0,00203\*** |
| Interpretabilità | Totale (soglia su una feature) | Parziale (pesi ispezionabili, ma rete non trasparente) |
| Comportamento appreso sensato? | Per costruzione | Sì (monotono su `cbf_score_std`), ma sovrastima il peso ottimale del CF |
| Costo computazionale | Trascurabile | Training + inferenza di una rete (comunque piccola) |

**Applicando il principio di parsimonia**: la Neural Gating non solo non supera il Rule-Based, ma è **sistematicamente peggiore su MRR e nDCG@5, in modo statisticamente robusto su entrambi gli split**. Il Rule-Based, a sua volta, non supera in modo pratico il CBF puro. **Nessuna delle due strategie è preferibile al CBF puro così com'è.**

## 18. Limiti

1. **Il segnale diagnostico (CF_WINS prevedibile con ROC-AUC 0,58) non si è tradotto in un beneficio di raccomandazione**: un conto è prevedere quando il CF "vincerebbe" un confronto di rank in isolamento, un altro è che aggiungere quel peso al punteggio finale migliori le metriche standard — il CBF resta un segnale forte anche nelle impression "vinte" dal CF secondo la label diagnostica, e mescolarlo con un CF non ottimamente calibrato introduce più rumore che segnale netto.
2. **La Neural Gating ha imparato un pattern qualitativamente corretto ma quantitativamente sovradimensionato**: `w_CF` medio di 0,51-0,55 quando disponibile è molto più alto del peso ottimale storicamente individuato per l'Hybrid statico (0,03) — un segno che la loss pairwise, ottimizzata sull'insieme delle impression di GATING_TRAIN, non converge verso un compromesso realmente vantaggioso in aggregato, anche se il gradiente locale (per le impression con CF forte) lo spinge in quella direzione.
3. **Numero di configurazioni testate deliberatamente limitato** (5-6 regole, 4 configurazioni di rete): per costruzione, secondo le istruzioni, non è stata eseguita alcuna ricerca estesa di iperparametri; non si può escludere che un'architettura/loss/regolarizzazione diversa faccia meglio, ma questo esula dallo scopo "minimale e controllato" di questa fase.
5. **INTERNAL_TEST parzialmente osservato in precedenza** (sezione 5): la valutazione finale qui è comunque la prima volta che questi 4 modelli specifici vengono confrontati su quel giorno, ma il giorno stesso non è "vergine" per il progetto nel suo complesso.
6. Il training della gating usa un solo giorno di train (12 nov) e un solo giorno di validazione (13 nov) per l'early stopping — nessun rolling-origin multi-giorno, per limiti di tempo/calcolo.

## 19. Conclusione

**Classificazione secondo il criterio pre-registrato**:

> **NESSUN BENEFICIO.** Né il Rule-Based Adaptive Hybrid né la Neural Gating Network migliorano il CBF puro in modo pratico su alcuna metrica, su nessuno dei due split (GATING_VALIDATION, INTERNAL_TEST). La Neural Gating mostra anzi un peggioramento piccolo ma statisticamente robusto e riproducibile su MRR e nDCG@5.

Non si tratta di un fallimento implementativo: il vincolo CF_AVAILABLE funziona esattamente come previsto (verificato bit-per-bit in sezione 13), la rete converge in modo stabile e impara un pattern interpretabile e qualitativamente sensato (sezione 15/17.2) — semplicemente, il segnale CF disponibile in questo progetto (ItemKNN su MIND-small, con o senza arricchimento dei click delle impression, si veda `VALIDAZIONE_METODOLOGICA.md` sezione 9) è troppo debole e troppo raro (23-33% delle impression) perché qualunque meccanismo di combinazione — statico, a regole o appreso — ne tragga un vantaggio netto misurabile.

**Decisione finale, secondo il principio dichiarato dall'utente**: si chiude la linea di sviluppo Adaptive Hybrid / Neural Gating basata su CF+CBF così come costruiti in questo progetto. **Si raccomanda NRMS (o un altro paradigma neurale end-to-end per il ranking dei candidati) come prossima estensione**, poiché il problema non è la strategia di combinazione dei segnali ma la debolezza intrinseca del segnale collaborativo disponibile — un limite che nessuna gating, per quanto ben progettata, può risolvere.
