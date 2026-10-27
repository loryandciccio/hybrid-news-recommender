# Analisi della complementarità dei segnali

Questo documento è il report della fase sperimentale dedicata a capire se il Collaborative Filtering (CF) e la Baseline di popolarità offrono informazione complementare al Content-Based Filtering (CBF) all'interno della pipeline esistente di `hybrid/hybrid_recommender.py`. **Nessun algoritmo del progetto principale è stato modificato**: tutti gli esperimenti riusano direttamente i metodi `_get_baseline_scores`, `_get_cf_scores`, `_get_cbf_scores`, `_norm`, `get_item_novelty` e `ndcg_at_k` importati da `hybrid_recommender.py`.

Tutti gli script, i dati grezzi e i grafici che supportano questo report si trovano in questa stessa cartella (`hybrid/experiments/`).

## 1. Motivazione

La weight sweep statica (48 valutazioni: 16 configurazioni di pesi × 3 porzioni del DEV — si veda sezione 2) aveva già mostrato che:

- il CBF puro (0/0/100) ottiene l'AUC media più alta;
- il CF è marginale a bassissimo peso (regione 0-5% quasi piatta rispetto a CF=0);
- aumentando ulteriormente il peso del CF, l'AUC degrada in modo monotono e progressivamente più marcato;
- la baseline di popolarità peggiora sistematicamente ogni configurazione testata.

Da qui nasce la domanda sperimentale di questa fase: **il Collaborative Filtering è globalmente poco utile, oppure esistono sottogruppi specifici di utenti/impression nei quali diventa informativo?** Questa domanda è la base per decidere se un futuro sviluppo del progetto giustifichi un Adaptive Hybrid / Neural Gating Network (se esiste complementarità sfruttabile) oppure un vero Neural News Recommender end-to-end come NRMS (se il CBF domina indipendentemente dal segmento).

## 2. Static Weight Sweep — riepilogo

Dati completi in `static_weight_sweep.csv`/`.json` e `static_weight_summary.csv`, prodotti da `run_static_weight_sweep.py` (script riproducibile, dati identici a quelli già raccolti nella fase precedente).

| Configurazione (B/CF/CBF) | AUC media | AUC std | MRR media | Novelty@5 media |
| :--- | :---: | :---: | :---: | :---: |
| 0/0/100 | 0.65380 | 0.00772 | 0.36411 | 21.980 |
| 0/3/97 (attuale) | 0.65375 | 0.00774 | 0.36410 | 21.972 |
| 0/10/90 | 0.65360 | 0.00773 | 0.36337 | 21.943 |
| 0/50/50 | 0.65166 | 0.00797 | 0.35274 | 21.478 |
| 10/20/70 | 0.65128 | 0.00733 | 0.36208 | 21.665 |

Nessun massimo locale: l'AUC media decresce monotonicamente da CF=0 in poi. La regione 0-3% è indistinguibile da CF=0 nei limiti della variabilità osservata tra le tre porzioni; oltre il 5-10% il degrado diventa via via più marcato. La baseline peggiora sistematicamente tutte le metriche testate, in ogni combinazione con CF provata.

## 3. Distribuzione della user history

Calcolata sull'intero DEV set (73.152 impression, di cui 70.938 valide per la valutazione — si veda sezione 12 per i criteri di esclusione). Dati completi in `history_length_distribution.csv`.

| Percentile | Lunghezza history |
| :--- | :---: |
| 10% | 4 |
| 25% | 8 |
| 50% (mediana) | 20 |
| 75% | 43 |
| 90% | 80 |
| 95% | 110 |
| 99% | 189 |
| massimo | 444 |

I bucket adottati sono **0-2, 3-5, 6-10, 11-20, >20**: nessuno risulta troppo piccolo per un'analisi interpretabile (il più piccolo, 0-2, conta 2.824 impression, 4.0% del totale). Tuttavia il bucket ">20" da solo concentra il 49.0% delle impression ed è molto eterogeneo (da 21 a 444 click); per questo motivo è stata aggiunta, **in aggiunta e non in sostituzione**, una suddivisione estesa della coda lunga (21-50, 51-100, >100), documentata esplicitamente e riportata in `results_by_history_length_extended.csv`.

| Bucket | N impression | % |
| :--- | :---: | :---: |
| 0-2 | 2.824 | 4.0% |
| 3-5 | 7.944 | 11.2% |
| 6-10 | 11.139 | 15.7% |
| 11-20 | 14.296 | 20.2% |
| >20 | 34.735 | 49.0% |

## 4. Performance per history length

Tabella completa in `results_by_history_length.csv`. Configurazioni testate: Baseline pura (1/0/0), CF puro (0/1/0), CBF puro (0/0/1), Hybrid attuale (0/0.03/0.97), Hybrid CF 10% (0/0.10/0.90) — tutte valutate esattamente sulle stesse impression di ciascun bucket. Grafico: `figures/02_auc_vs_history_length.png`.

| History | N | Baseline AUC | CF AUC | CBF AUC | Hybrid 0/3/97 AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 0-2 | 2.824 | 0.4963 | 0.5006 | 0.5757 | 0.5757 |
| 3-5 | 7.944 | 0.4973 | 0.5010 | 0.6147 | 0.6147 |
| 6-10 | 11.139 | 0.4913 | 0.5010 | 0.6373 | 0.6373 |
| 11-20 | 14.296 | 0.4887 | 0.5011 | 0.6495 | 0.6494 |
| >20 | 34.735 | 0.4832 | 0.4999 | 0.6575 | 0.6574 |

**Interpretazione**: il CF resta ancorato a un AUC ≈0.50 (indistinguibile dal caso casuale) in **ogni** bucket, indipendentemente dalla lunghezza della cronologia — anche nel bucket con cronologie molto lunghe (>100 click: AUC CF = 0.4973, si veda tabella estesa). Il CBF, al contrario, **migliora** con cronologie più lunghe fino al bucket 21-50 (AUC 0.662), poi si stabilizza/degrada leggermente su cronologie estreme (51-100: 0.657; >100: 0.637, su un campione più piccolo di 4.311 impression). La Baseline **peggiora** monotonicamente con cronologie più lunghe.

## 5. Cold-start

**Categorie di cold-start identificabili con i dati disponibili**, senza inventare informazioni assenti dal dataset:
1. **Candidato mai osservato nel TRAIN** (popolarità di training = 0) — misurabile esattamente da `item_popularity`.
2. **Candidato con pochissime interazioni nel TRAIN** — misurabile come popolarità bassa ma non nulla (si veda sezione 6).
3. **Cronologia utente molto corta** — già coperta in sezione 4 (bucket 0-2).

L'analisi si concentra sul punto 1, applicato specificamente all'articolo effettivamente cliccato (per le 50.525 impression con un solo positivo, il 71.2% del totale valido — si veda sezione 12 per la motivazione di questa restrizione). Dati in `results_by_coldstart.csv`.

**Risultato sorprendente**: il **94.6%** degli articoli cliccati nel DEV non è mai stato osservato nel TRAIN (popolarità di training = 0) — la stragrande maggioranza dei click reali riguarda contenuti che il modello non ha mai visto prima.

| Item cliccato | N impression | Baseline AUC | CF AUC | CBF AUC | Hybrid 0/3/97 AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Cold (mai visto nel TRAIN) | 47.807 (94.6%) | 0.4619 | 0.4976 | 0.6497 | 0.6496 |
| Warm (visto nel TRAIN) | 2.718 (5.4%) | 0.9784 | 0.5574 | 0.5166 | 0.5182 |

**Interpretazione**: nel segmento cold-start (la stragrande maggioranza del traffico), la Baseline è **peggio del caso casuale** (0.4619) e il CF è indistinguibile dal caso casuale (0.4976), mentre il CBF resta forte (0.6497). Questo conferma quantitativamente che il vantaggio del CBF deriva in larga parte dalla sua capacità di gestire articoli nuovi. Nel piccolo segmento "warm" (5.4% dei casi), il quadro si ribalta nettamente: la Baseline raggiunge un AUC di 0.9784, il CF sale a 0.5574, mentre il CBF scende a 0.5166 (appena sopra il caso casuale).

**Avvertenza metodologica importante**: l'AUC di 0.9784 della Baseline nel segmento "warm" va interpretato con cautela. Il segmento è definito condizionando sul fatto che l'item *effettivamente cliccato* abbia popolarità di training positiva — una condizione nota solo *a posteriori*, non disponibile al momento di generare la raccomandazione (non sappiamo in anticipo se l'utente cliccherà un item "caldo" o "freddo"). Il numero non implica quindi che "pesare di più la popolarità" garantisca quell'AUC nella pratica: è un'osservazione post-hoc sul sottoinsieme di click che riguardano contenuti popolari, utile per capire *dove* la popolarità è correlata al comportamento osservato, non una ricetta operativa direttamente applicabile senza un meccanismo (es. una feature calcolabile pre-click, come "il candidate set contiene un item ad altissima popolarità") che permetta di anticipare l'appartenenza a questo segmento.

## 6. Popolarità degli articoli

Popolarità calcolata **esclusivamente sul TRAIN** (si veda sezione 12). Quantili (33°/66°) calcolati tra gli articoli cliccati con popolarità di training > 0: 8 e 59 click. Bucket: `unseen` (pop=0), `bassa` (1-8), `media` (9-59), `alta` (>59). Dati in `results_by_candidate_popularity.csv`.

| Bucket | N impression | Popolarità media TRAIN | Baseline AUC | CF AUC | CBF AUC | Hybrid 0/3/97 AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| unseen | 47.807 | 0 | 0.4619 | 0.4976 | 0.6497 | 0.6496 |
| bassa | 1.274 | 7.4 | 0.9706 | 0.5001 | 0.4427 | 0.4426 |
| media | 702 | 50.2 | 0.9826 | 0.5041 | 0.5059 | 0.5060 |
| alta | 742 | 2.137 | 0.9879 | **0.7063** | 0.6537 | 0.6595 |

**Interpretazione**: questo è l'**unico segmento identificato in tutta l'analisi in cui il CF puro supera in media il CBF puro** (0.7063 vs 0.6537, +0.0526 di AUC), sul bucket degli articoli cliccati più popolari nel TRAIN (742 impression, 1.5% del totale a singolo positivo). Nel bucket "bassa popolarità" il CBF scende sotto il caso casuale (0.4427) — un risultato isolato su un campione più piccolo (1.274 impression), da non sovra-interpretare senza ulteriori verifiche. Il pattern generale conferma l'ipotesi della sezione 5: il vantaggio complessivo del CBF è concentrato quasi interamente sulla long tail (unseen + bassa popolarità = 96.5% dei casi a singolo positivo), mentre CF e Baseline recuperano competitività (CF) o dominano nettamente (Baseline, con l'avvertenza sopra) man mano che ci si sposta verso articoli con più storia di interazioni.

## 7. Complementarità CF vs CBF

Definizione di "corretto": **hit@1** (l'item cliccato è il candidato con punteggio più alto) e **hit@5** (l'item cliccato è tra i primi 5 per punteggio). Calcolate su tutte le 70.938 impression valide (non solo a singolo positivo). Matrice completa in `complementarity_matrix.json`.

**hit@1**:

| | CF corretto: sì | CF corretto: no |
| :--- | :---: | :---: |
| **CBF corretto: sì** | 3.56% (entrambi) | 16.03% (solo CBF) |
| **CBF corretto: no** | 7.70% (solo CF) | 72.71% (nessuno) |

**hit@5**:

| | CF corretto: sì | CF corretto: no |
| :--- | :---: | :---: |
| **CBF corretto: sì** | 28.67% (entrambi) | 27.08% (solo CBF) |
| **CBF corretto: no** | 9.27% (solo CF) | 34.98% (nessuno) |

**Interpretazione**: "solo CF corretto" **non è trascurabile** (7.70% a hit@1, 9.27% a hit@5) — circa un'impression su 13 (hit@1) o su 11 (hit@5) viene risolta correttamente dal CF e sbagliata dal CBF. È però sistematicamente più raro di "solo CBF corretto" (16.03%/27.08%, circa il doppio). Un ipotetico "oracolo" capace di scegliere sempre il modello giusto impression per impression otterrebbe hit@1 = 3.56+7.70+16.03 = 27.29% (contro il 19.59% del solo CBF, +7.70 punti percentuali) e hit@5 = 65.02% (contro il 55.75% del solo CBF, +9.27 punti percentuali). Si tratta di un **limite superiore teorico**, raggiungibile solo con previsione perfetta di quale segnale fidarsi per ogni impression — un gating reale, allenato su feature osservabili, catturerebbe verosimilmente solo una parte di questo margine.

Scomponendo "solo CF corretto" (hit@5) per stato cold-start dell'item cliccato (solo impression a singolo positivo, `complementarity_by_history_length.csv` per il dettaglio per history length):

| Stato item cliccato | N | Solo CF corretto | Solo CBF corretto |
| :--- | :---: | :---: | :---: |
| Warm (visto nel TRAIN) | 2.718 | **12.44%** | 6.14% |
| Cold (mai visto nel TRAIN) | 47.807 | 9.00% | 25.21% |

Nel segmento "warm", il CF risolve correttamente più casi di quanti ne risolva da solo il CBF (12.44% contro 6.14%) — l'unico segmento, oltre a quello di sezione 6, in cui il CF risulta relativamente più utile del CBF anche a livello di singola impression, non solo di AUC media.

Per bucket di history length (`complementarity_by_history_length.csv`, grafico `figures/05_complementarita_per_history_length.png`), la quota "solo CF corretto" (hit@5) varia poco (8.8%-11.1%) tra i bucket, con un valore leggermente più alto nelle cronologie corte (0-2: 11.08%) che in quelle lunghe (>20: 9.16%) — nessun pattern di crescita del vantaggio del CF con cronologie più lunghe, anzi una lieve tendenza opposta.

## 8. Analisi dei ranking

Correlazione di Spearman tra gli score CF e CBF (normalizzati) sui candidati della stessa impression, e overlap/Jaccard dei rispettivi Top-5, calcolati per ciascuna impression e poi aggregati (`ranking_correlation.csv`, riepilogo in `ranking_correlation_summary.csv`).

| Metrica | Media | Mediana | Deviazione standard |
| :--- | :---: | :---: | :---: |
| Correlazione di Spearman (CF, CBF) | 0.028 | 0.015 | 0.149 |
| Overlap Top-5 (frazione) | 0.343 | 0.200 | 0.336 |
| Jaccard Top-5 | 0.272 | 0.111 | 0.325 |

(La correlazione di Spearman è calcolabile solo quando entrambi gli score hanno varianza non nulla sui candidati: disponibile per 16.669 impression su 70.938; la causa più comune di score costante è un CF completamente privo di segnale, tutto a zero, coerente con l'assenza di co-click per quella cronologia.)

**Interpretazione**: la correlazione media è **praticamente nulla** (0.028) — i ranking prodotti da CF e CBF sono, in media, **statisticamente indipendenti**, non ridondanti. L'overlap dei Top-5 è parziale (mediana 0.2 = un solo elemento in comune su 5) confermando che i due modelli propongono liste di raccomandazione sostanzialmente diverse. Questo è coerente con lo scenario "segnali complementari ma la combinazione lineare globale non riesce a sfruttarli pienamente": i ranking non sono ridondanti, ma — come mostrato in sezione 4 — la combinazione lineare statica non traduce questa diversità in un guadagno di AUC misurabile nella maggior parte dei segmenti.

## 9. Baseline

La weight sweep statica mostra che la baseline peggiora sistematicamente la combinazione globale. La segmentazione per stato cold-start (sezione 5) e popolarità (sezione 6) mostra però un segmento chiaramente identificato in cui la popolarità è fortemente correlata al click: il segmento "warm"/popolare (5.4% delle impression a singolo positivo), dove la Baseline raggiunge un AUC osservato di 0.978-0.988. Come discusso in sezione 5, questo numero va interpretato come un'osservazione post-hoc su un sottoinsieme definito dall'esito (il click), non come una garanzia prospettica — ma la sua esistenza, unita al fatto che CF e Baseline concordano più spesso tra loro (33.87% "entrambi corretti" hit@5) che con il CBF, suggerisce che popolarità e comportamento collettivo condividono informazione reale sui contenuti "mainstream", semplicemente minoritaria nel traffico complessivo di MIND-small (dominato da contenuti nuovi). Non è stato identificato alcun segmento in cui la Baseline pura raggiunga un AUC competitivo *nel segmento dominante* (cold-start, 94.6% dei casi): lì resta sotto il caso casuale (0.4619).

## 10. Analisi bootstrap (Fase 13)

Bootstrap paired a livello di impression (10.000 ricampionamenti, seed fisso 42) sulle differenze di AUC/MRR/nDCG@5, sull'intero campione di 70.938 impression valide del DEV. Script: `run_bootstrap_analysis.py`. Risultati completi in `bootstrap_results.json`.

| Confronto | Δ osservata | CI 95% | L'intervallo include lo 0? |
| :--- | :---: | :---: | :---: |
| Hybrid 0/3/97 − CBF puro, ΔAUC | −0.0000375 | [−0.0000599, −0.0000139] | No |
| Hybrid 0/3/97 − CBF puro, ΔMRR | −0.0000241 | [−0.0001043, +0.0000569] | Sì |
| Hybrid 0/3/97 − CBF puro, ΔnDCG@5 | +0.0000... | [−0.00004, +0.00009] | Sì |
| CF puro − CBF puro, ΔAUC | −0.1442 | [−0.1464, −0.1420] | No |
| CF puro − CBF puro, ΔMRR | −0.1143 | [−0.1172, −0.1114] | No |
| CF puro − CBF puro, ΔnDCG@5 | −0.1232 | [−0.1261, −0.1205] | No |
| Baseline pura − CBF puro, ΔAUC | −0.1570 | [−0.1594, −0.1545] | No |

**Interpretazione**: sull'ampio campione di 70.938 impression, l'intervallo di confidenza per ΔAUC (Hybrid − CBF) **non include lo zero**, ma la differenza osservata è estremamente piccola (−0.0000375, circa 0.004 punti percentuali di AUC) — un ordine di grandezza privo di qualunque rilevanza pratica, anche se rilevabile con un campione di questa dimensione. Su MRR e nDCG@5 il confronto Hybrid vs CBF **non emerge come una differenza consistente** nel campione analizzato (l'intervallo include lo zero). Il confronto CF puro vs CBF puro e Baseline pura vs CBF puro, al contrario, mostrano differenze ampie e intervalli di confidenza saldamente lontani dallo zero: su queste due componenti prese singolarmente, il CBF è nettamente e consistentemente più forte nel campione analizzato.

## 11. Implicazioni per un Adaptive Hybrid

**Esistono evidenze che pesi dinamici possano essere migliori di pesi globali?** Le evidenze sono **miste e localizzate**, non generalizzate:

- Su quasi tutti i segmenti testati (bucket di history length, maggioranza cold-start), pesi dinamici non avrebbero probabilmente alcun beneficio misurabile: il CF resta vicino al caso casuale ovunque tranne che negli articoli con popolarità di training elevata.
- Esiste però un segmento chiaramente identificabile — articoli con popolarità di training elevata (sezione 6) e, più in generale, item "warm" (sezione 5/7) — dove CF e Baseline offrono un vantaggio reale rispetto al CBF, sia in AUC media sia in tasso di correttezza a livello di singola impression.
- La quota "solo CF corretto" (7.7-9.3%) non è trascurabile e non è interamente spiegata dalla lunghezza della cronologia (che non mostra un pattern utile), suggerendo che il vantaggio del CF sia legato più alla natura del candidato (popolarità/novità) che al profilo dell'utente.

## 12. Implicazioni per un Neural Recommender

Se il segnale complementare fosse pressoché assente in ogni segmento (Scenario C), la conclusione naturale sarebbe che vale la pena investire in un modello neurale end-to-end (NRMS) capace di superare direttamente il CBF con rappresentazioni apprese, piuttosto che tentare di ricombinare expert deboli. I risultati di questa fase **non supportano pienamente questa conclusione**: esiste un segnale complementare reale, seppur circoscritto a una minoranza di casi (i contenuti popolari/"caldi", ~5% del traffico). Questo suggerisce che, prima di investire nel costo implementativo e computazionale di un NRMS, abbia senso verificare se una combinazione più intelligente (anche solo rule-based) dei tre expert esistenti riesca a catturare questo margine residuo — si veda la classificazione in sezione 13.

## 13. Conclusione

L'evidenza raccolta in questa fase è coerente su più fronti indipendenti (weight sweep statica, segmentazione per history length, cold-start, popolarità, contingenza hit@1/hit@5, correlazione dei ranking, bootstrap):

- Il CBF domina in aggregato e nella stragrande maggioranza dei segmenti testati, in particolare nel segmento dominante del traffico (cold-start, 94.6% delle impression a singolo positivo).
- Il CF, preso da solo, non supera mai il CBF in AUC media nei bucket di history length, ma lo supera chiaramente (+0.053 di AUC) nel segmento degli articoli con popolarità di training elevata, e risolve correttamente più impression del CBF specificamente nel sottoinsieme di item "warm" (12.44% contro 6.14%).
- La quota "solo CF corretto" (7.7-9.3% delle impression) non è trascurabile, e la correlazione dei ranking CF/CBF è vicina allo zero: i due segnali non sono ridondanti.
- La Baseline pura è dannosa nel segmento dominante (cold-start) ma mostra una forte correlazione con il click nel piccolo segmento "warm" — un'osservazione da trattare con cautela per il suo carattere post-hoc (si veda sezione 5).

Questo quadro non descrive né una complementarità forte e diffusa (Scenario A), né una sua assenza pressoché totale (Scenario C): descrive una **complementarità debole ma reale e localizzabile**, concentrata in un segmento minoritario ma identificabile del traffico (contenuti popolari/"caldi").

## 14. Classificazione finale: Scenario B — Complementarità debole ma presente

**Motivazione quantitativa**:
- Il CF non supera generalmente il CBF (in 4 bucket di history length su 4 l'AUC del CF resta ≈0.50, ben sotto il CBF) → esclude lo Scenario A come descrizione generale.
- Il CF risolve correttamente una quota non trascurabile di errori del CBF (7.7-9.3% "solo CF corretto"), con una correlazione dei ranking vicina a zero (0.028) → esclude lo Scenario C, che richiederebbe una quota "rarissima" e ranking sostanzialmente ridondanti.
- Esiste un segmento chiaramente identificato (popolarità di training elevata del candidato) in cui il CF supera il CBF in modo netto (+0.053 di AUC) — non abbastanza esteso (1.5% delle impression a singolo positivo) da giustificare una complementarità "forte", ma sufficiente a escludere che sia "trascurabile".

**Raccomandazione conseguente (come da istruzioni per lo Scenario B)**: confrontare successivamente, come esperimenti separati e non ancora implementati in questa fase:
1. **Rule-Based Adaptive Hybrid**: una regola esplicita e interpretabile che aumenti il peso di CF/Baseline quando il candidate set contiene item con popolarità di training elevata (feature disponibile pre-click), e favorisca il CBF altrimenti — il passo più semplice ed economico, coerente con l'evidenza raccolta.
2. **Neural Gating Hybrid**: si veda la proposta architetturale in sezione 15, da valutare se la regola semplice del punto 1 non si rivelasse sufficiente.
3. **CBF puro**, come termine di paragone imprescindibile per giudicare se le due alternative sopra giustifichino la complessità aggiuntiva.

Nessuna delle tre opzioni è stata implementata in questa fase.

## 15. Se si sviluppasse una Gating Network (progetto, non implementato)

Si veda [PROGETTAZIONE_FUTURA.md](PROGETTAZIONE_FUTURA.md) per il dettaglio completo di feature, architettura e protocollo di training proposti (non implementati in questa fase): tabella delle feature con provenienza e rischio di leakage, architettura minima proposta (Dense 32→16→2/3, softmax), loss, split, early stopping, seed, metriche e baseline di confronto.

## 16. Se si sviluppasse NRMS (scheda progettuale, non implementato)

Si veda [PROGETTAZIONE_FUTURA.md](PROGETTAZIONE_FUTURA.md) per la scheda progettuale completa di NRMS (input, news encoder, user encoder, self-attention, candidate scoring, negative sampling, loss, training/inference, costo computazionale, dipendenze, compatibilità con MIND-small, necessità di GPU, confronto proposto con il CBF SBERT esistente). Nessuna implementazione è stata effettuata in questa fase.
