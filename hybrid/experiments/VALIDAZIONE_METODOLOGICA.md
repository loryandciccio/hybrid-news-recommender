# Validazione metodologica pre-implementazione

Questo documento audita criticamente i risultati della fase di analisi della complementarità (`ANALISI_COMPLEMENTARITA.md`) prima di procedere a qualunque implementazione di Adaptive Hybrid o rete neurale, e progetta (senza implementarli) il protocollo sperimentale, il baseline rule-based e le architetture candidate per la fase successiva.

**Nessun algoritmo del progetto principale è stato modificato in questa fase.** Nessuna rete neurale è stata implementata. README.md, PRESENTAZIONE_PROGETTO.md e REVISIONE_PROGETTO.md non sono stati toccati.

---

## 0. Correzioni preliminari

**0.a — Errore logico nella sintesi discorsiva (non nel documento).** Nella sintesi a chat che riassumeva la sezione 1.3, è stata usata la formulazione errata *"nessun caso A=False con B=True impossibile"*. Questa affermazione è falsa: A=False (presente nel catalogo) e B=True (mai in history) è **il caso più comune del dataset** (70,46% del gruppo B, si veda sezione 1.2). Il testo di questo documento (sezione 1.3, riga corrispondente) non conteneva questo errore: riportava correttamente che il caso strutturalmente impossibile è l'opposto, **A=True (assente dal catalogo) e B=False (presente in una history)** — un articolo che non esiste nel catalogo non può comparire nella history di nessuno. Verificato ora anche empiricamente su tutte le 50.525 righe di `coldstart_category_breakdown.csv`: il conteggio di "A=True & B=False" è **0/50.525**, confermando che l'unica cosa impossibile è quella, non il suo opposto. Nessuna correzione al contenuto tecnico è quindi necessaria; questa nota serve solo a chiarire l'errore di formulazione commesso a voce.

**0.b — Non chiamare più il DEV "FINAL HELD-OUT".** Corretto: il DEV (`MINDsmall_dev/behaviors.tsv`, Nov 15) è stato usato per tuning dei pesi, weight sweep, analisi per segmenti, cold-start, popolarità, CF_WINS, bootstrap e per la scelta preliminare gating/meta-ranker — è a tutti gli effetti un dataset di sviluppo/analisi, non un test set vergine. Ogni occorrenza di "FINAL HELD-OUT" riferita al DEV nella sezione 5 qui sotto va letta come **superata**: la sezione 9 (nuova, in fondo a questo documento) sostituisce quel disegno con uno split puramente interno al TRAIN (FIT/VALIDATION/INTERNAL_TEST), che non tocca mai il DEV. Il DEV resta escluso da qualunque decisione presa in questa fase e nella successiva.

---

## 1. Audit del dato "cold-start 94.6%"

### 1.1 Cosa significava esattamente "94.6%" nel report precedente

Il numero 94.6% in `ANALISI_COMPLEMENTARITA.md` (sezione 5) è definito come: quota di impression DEV a singolo positivo il cui articolo cliccato ha `model.item_popularity.get(clicked_id, 0) == 0`. In `hybrid_recommender.py`, `item_popularity` è costruito **esclusivamente dal campo `history` delle righe di `MINDsmall_train/behaviors.tsv`** (si veda `load_data`, righe 168-182): non usa in alcun modo le etichette di click (`-1`) presenti nel campo `impressions`. Questa è la stessa quantità usata internamente per costruire il grafo di co-click dell'ItemKNN (`_build_itemknn_matrix`), quindi il numero descriveva correttamente **l'assenza di rappresentazione utilizzabile per il Collaborative Filtering** (categoria E sotto), ma la dicitura "mai visto nel TRAIN" usata nel testo del report era **impropria e fuorviante** se interpretata come "articolo sconosciuto/inesistente al momento del training": una larga parte di questi articoli esiste ed è persino stata cliccata nel TRAIN, semplicemente non tramite il campo `history`.

### 1.2 Tabella completa delle definizioni di cold-start

Calcolata sulle 50.525 impression DEV a singolo positivo (stesso insieme usato nel report precedente), verificando per ciascun articolo cliccato tutte e cinque le categorie analizzate, in modo non mutuamente esclusivo (ogni articolo può ricadere in più categorie contemporaneamente):

| Categoria | Definizione | N | % |
| :--- | :--- | :---: | :---: |
| **A** | Assente completamente da `MINDsmall_train/news.tsv` | 12.208 | **24.16%** |
| **B** | Presente in `train/news.tsv` ma mai comparso nel campo `history` di alcuna riga di `TRAIN/behaviors.tsv` | 47.807 | **94.62%** |
| **C** | Presente in `train/news.tsv` ma mai cliccato (etichetta `-1`) nelle impression del TRAIN | 26.082 | **51.62%** |
| **D** | Mai comparso come candidato (click o no) nelle impression del TRAIN | 13.652 | **27.02%** |
| **E** | Nessuna rappresentazione CF utilizzabile (0 occorrenze nel conteggio `item_counts` usato da `_build_itemknn_matrix`) | 47.807 | **94.62%** |

**B ed E coincidono esattamente** (stesso insieme di 47.807 articoli): confermano che il numero "94.6%" del report precedente descrive correttamente la categoria E (il CF non ha alcun segnale per quell'articolo), attraverso la sua base dati B (history del TRAIN). Le altre definizioni raccontano una storia diversa e sostanzialmente più moderata:

- Solo il **24.16%** degli articoli cliccati nel DEV è genuinamente assente dal catalogo `train/news.tsv` (il senso più letterale di "mai visto nel TRAIN").
- Il **51.62%**, non il 94.6%, non è mai stato cliccato nel TRAIN secondo le etichette ufficiali di click (la stessa definizione di popolarità usata in `baseline_model/baseline.ipynb`, che infatti riporta 7.713 articoli "unique clicked items in TRAIN" — coerente con `len(clicked_ids)` qui misurato).
- **Cross-tabulazione B × C**: del 94.62% di articoli "B" (mai in history TRAIN), il **46.56%** (23.522 su 50.525 impression totali, quindi quasi la metà del gruppo B) **è stato comunque cliccato nel TRAIN** secondo le impression ufficiali — semplicemente non tramite il campo `history` dell'utente che ha poi effettuato il click nel DEV. Il **70.46%** del gruppo B è inoltre presente nel catalogo `train/news.tsv`.

### 1.3 Verifica manuale del campione

Campione casuale di 25 righe (`clicked_id`, seed 42) verificato manualmente incrociando titolo (da `MINDsmall_dev/news.tsv`) e appartenenza a A/B/C. Nessuna anomalia di parsing rilevata:
- Gli identificativi degli articoli hanno un prefisso `N` seguito da un numero di cifre variabile; il controllo non ha rilevato problemi di padding o troncamento.
- Nessuno spazio iniziale/finale residuo negli ID (`.strip()` verificato programmaticamente su tutto l'insieme).
- I controlli incrociati su un campione locale hanno restituito titoli distinti e combinazioni di appartenenza ai gruppi coerenti; gli ID e i titoli originali del dataset sono omessi nella copia portfolio.
- Le combinazioni osservate nel campione risultano logicamente coerenti con le definizioni di cold-start; non è emersa alcuna contraddizione nelle condizioni di inclusione dei gruppi.

### 1.4 Formulazione corretta

**Non usare**: "il 94.6% degli articoli cliccati nel DEV non è mai stato visto nel TRAIN" (ambigua, e nella lettura più naturale, falsa: il 75.8% di questi articoli esiste nel catalogo TRAIN).

**Usare invece**: *"Il 94.6% degli articoli cliccati nel DEV non compare nel campo `history` di alcun utente del TRAIN — la stessa base dati usata da `hybrid_recommender.py` per calcolare la popolarità e costruire il grafo di co-click ItemKNN — e quindi non ha alcuna rappresentazione utilizzabile dal componente Collaborative Filtering. Questo non equivale a dire che l'articolo sia sconosciuto: il 70.5% di questi articoli è comunque presente nel catalogo `train/news.tsv`, e il 46.6% è stato persino cliccato nel TRAIN secondo le etichette ufficiali delle impression — semplicemente attraverso un campo diverso (`impressions`, non `history`) da quello usato per costruire la rappresentazione CF."*

Questa correzione **non invalida** la conclusione qualitativa principale (il CBF domina nel segmento in cui CF/Baseline non hanno rappresentazione), ma **ridimensiona** l'entità apparente del fenomeno "cold-start" tout court, e chiarisce che il vero fattore scatenante è una scelta implementativa specifica (popolarità/co-click basati su `history`, non su click ufficiali) più che un limite intrinseco e ineliminabile del dominio. Si noti inoltre — punto già segnalato nell'audit di fase 1 (`REVISIONE_PROGETTO.md`) — che questa è la stessa incoerenza tra la definizione di popolarità di `hybrid_recommender.py` (history-based) e quella di `baseline_model/baseline.ipynb`/`cf_itemknn/*` (click-based, campo `impressions`): un potenziale miglioramento futuro (non implementato qui) sarebbe armonizzare le due definizioni.

Script e dati completi: `hybrid/experiments/run_coldstart_audit.py`, output in `coldstart_category_breakdown.csv` e `coldstart_manual_sample.csv`.

---

## 2. Audit del segmento "high-popularity"

### 2.1 Come è costruito il segmento

Il segmento "alta popolarità" di `ANALISI_COMPLEMENTARITA.md` (sezione 6) è costruito così: per ogni impression DEV a singolo positivo, si osserva la popolarità di training (`model.item_popularity`) dell'articolo **effettivamente cliccato**, e la si confronta con i quantili 33°/66° calcolati sulla stessa popolarità, tra gli articoli cliccati con popolarità > 0. L'impression viene poi assegnata al bucket "alta" se la popolarità dell'articolo cliccato supera il 66° percentile.

**Questo segmento usa quindi lo status dell'item cliccato, cioè richiede di conoscere la label per essere costruito.** Non è una caratteristica dell'intera impression disponibile prima del click: prima di osservare quale candidato viene cliccato, non sappiamo se il "vincitore" sarà un articolo ad alta, media o bassa popolarità — lo sappiamo solo *a posteriori*.

### 2.2 Classificazione esplicita

**Il segmento "high-popularity" e il segmento "warm" di cold-start (sezione 5 di `ANALISI_COMPLEMENTARITA.md`) sono entrambi ANALISI DIAGNOSTICHE POST-HOC, non regole direttamente utilizzabili da un recommender.** Il numero "CF AUC ≈ 0.706 nel segmento alta popolarità" resta un fatto empirico corretto e utile per capire *dove* il segnale CF è relativamente più forte, ma **non implica** che un sistema in produzione possa "sapere in anticipo" di trovarsi in quel segmento e pesare di conseguenza il CF più in alto — a meno di costruire un proxy calcolabile *prima* del click (si veda sezione 3, feature `max_train_pop_candidates`, `pct_warm_candidates`: queste sì sono disponibili pre-click, e sono un tentativo di catturare la stessa informazione in modo legittimo).

Questo punto era già stato segnalato con una nota di cautela in `ANALISI_COMPLEMENTARITA.md` (sezione 5, "Avvertenza metodologica importante"), ma non era stato etichettato esplicitamente come "analisi diagnostica post-hoc" nel titolo/struttura del documento; questa sezione lo rende esplicito e vincolante per l'uso che se ne farà nelle fasi successive.

---

## 3. Feature pre-raccomandazione

Tabella completa delle feature valutate, con provenienza, momento di disponibilità e rischio di leakage. "Utilizzabile" = sì solo se calcolabile interamente da informazioni antecedenti al click, usando esclusivamente il TRAIN per qualunque statistica aggregata (popolarità, ecc.).

### Livello utente

| Feature | Definizione | Sorgente | Disponibile quando | Rischio leakage | Utilizzabile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `history_length` | Numero di articoli nella cronologia dell'utente | Campo `history` dell'impression corrente | Al momento della richiesta di raccomandazione | Nessuno | **Sì** |
| Statistiche aggiuntive sulla history (es. categorie prevalenti, recency dell'ultimo click) | Derivabili dal campo `history` + `news.tsv` | Come sopra | Come sopra | Nessuno, se calcolate solo dagli articoli della history stessa | **Sì**, non ancora implementate in questa fase |

### Livello impression (candidate set)

| Feature | Definizione | Sorgente | Disponibile quando | Rischio leakage | Utilizzabile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `n_candidates` | Numero di candidati nell'impression | Campo `impressions` (solo gli ID, non le etichette) | Al momento della richiesta | Nessuno | **Sì** |
| `pct_unseen_candidates` | Quota di candidati con popolarità TRAIN (history-based) = 0 | `item_popularity`, calcolato solo su TRAIN | Al momento della richiesta | Nessuno (population TRAIN, non DEV) | **Sì** |
| `pct_warm_candidates` | 1 − `pct_unseen_candidates` | Come sopra | Come sopra | Nessuno | **Sì** |
| `mean/median/max_train_pop_candidates` | Statistiche della popolarità TRAIN sui candidati | Come sopra | Come sopra | Nessuno | **Sì** |
| `cf_score_std` / `cbf_score_std` | Deviazione standard degli score grezzi CF/CBF sul candidate set | `_get_cf_scores`/`_get_cbf_scores` (esistenti, non modificati) | Al momento dello scoring, prima di conoscere il click | Nessuno | **Sì** |
| `cf_score_max` / `cbf_score_max` | Punteggio massimo grezzo sul candidate set | Come sopra | Come sopra | Nessuno | **Sì** |
| `cf_gap_top1_top2` / `cbf_gap_top1_top2` | Differenza tra il punteggio più alto e il secondo più alto | Come sopra | Come sopra | Nessuno | **Sì** |
| `topk5_agreement_cf_cbf` | Frazione di sovrapposizione tra i Top-5 CF e CBF | Come sopra | Come sopra | Nessuno | **Sì** |

### Livello candidato (per un eventuale meta-ranker candidate-level)

| Feature | Definizione | Sorgente | Disponibile quando | Rischio leakage | Utilizzabile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `candidate_train_popularity` | Popolarità TRAIN del singolo candidato | `item_popularity` | Al momento dello scoring | Nessuno | **Sì** |
| `cold_start_flag` | 1 se popolarità TRAIN = 0 | Come sopra | Come sopra | Nessuno | **Sì** |
| `cf_score` / `cbf_score` / `baseline_score` (del candidato) | Score grezzi o normalizzati, per candidato | Metodi esistenti | Come sopra | Nessuno | **Sì** |
| `semantic_similarity_features` (es. similarità massima/media con i singoli articoli della history, non solo il centroide) | Embedding SBERT | Come sopra | Come sopra | Nessuno | **Sì**, non ancora calcolate in questa fase |
| `category` / `subcategory` | Metadato editoriale | `news.tsv` | Statico, sempre disponibile | Nessuno | **Sì** |
| `recency` (del candidato, cioè da quanto è pubblicato) | Non presente esplicitamente in MIND-small (nessun timestamp di pubblicazione nel `news.tsv`); solo la data della prima/ultima esposizione come candidato è ricostruibile dai log — e usarla richiederebbe attenzione a non includere esposizioni **future** rispetto all'impression corrente | `behaviors.tsv` (esposizioni) | Solo se calcolata usando esclusivamente esposizioni con timestamp precedente all'impression corrente | **Alto se calcolata scorrettamente** (userebbe implicitamente il fatto che l'articolo è stato mostrato/cliccato *dopo*) | **Condizionatamente sì**, con attenzione — non implementata in questa fase |

**Feature esplicitamente NON utilizzabili** (rischio di leakage diretto, mai calcolate in nessuna fase di questo progetto): qualunque popolarità o CTR calcolati includendo il DEV; la label `clicked` dell'impression corrente o di impression future dello stesso utente; qualunque statistica aggregata a livello di dataset che includa l'esempio da valutare nel proprio stesso calcolo.

---

## 4. Analisi preliminare di CF_WINS

**Non è stato addestrato alcun classificatore.** È stato costruito un dataset diagnostico (`cf_wins_diagnostic.csv`, script `run_cf_wins_diagnostic.py`) con una label descrittiva:

- `rank_CF` / `rank_CBF`: posizione (1-based, tie-break su `news_id` crescente, coerente col resto del progetto) del candidato cliccato nel ranking indotto rispettivamente da CF puro e CBF puro.
- `CF_WINS = 1` se `rank_CF < rank_CBF` (il CF piazza il click più in alto del CBF), altrimenti 0 (pareggi inclusi, trattati conservativamente come "non vince il CF").

Costruito su tutte le impression DEV a singolo positivo (50.525), usando esclusivamente feature pre-click (sezione 3). Script: `run_cf_wins_diagnostic.py`. Output: `cf_wins_diagnostic.csv`, riepilogo in `cf_wins_feature_means.csv` e `cf_wins_correlations.csv`.

**Prevalenza**: `CF_WINS=1` in 16.636 impression su 50.525 (**32.93%**) — il CF piazza il candidato cliccato più in alto del CBF in circa un terzo dei casi a singolo positivo (coerente in ordine di grandezza con la quota "solo CF corretto" più "entrambi corretti" di `ANALISI_COMPLEMENTARITA.md`, con cui questa definizione condivide lo spirito ma non la formula esatta: qui si confrontano i rank, non solo hit@1/hit@5).

### 4.1 Relazione media/correlazione (prima passata)

| Feature | Media se CF_WINS=1 | Media se CF_WINS=0 | Differenza | Cohen's d descrittivo | Correlazione (Pearson) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `cbf_score_std` | 0.2539 | 0.2618 | −0.0080 | **−0.110** | **−0.052** |
| `cbf_score_max` | 0.9295 | 0.9418 | −0.0123 | −0.066 | −0.031 |
| `n_candidates` | 29.24 | 28.43 | +0.82 | 0.025 | 0.012 |
| `topk5_agreement_cf_cbf` | 0.4087 | 0.4151 | −0.0064 | −0.018 | −0.008 |
| `max_train_pop_candidates` | 563.5 | 545.4 | +18.0 | 0.013 | 0.006 |
| `history_length` | 28.65 | 29.09 | −0.45 | −0.013 | −0.006 |
| `pct_warm_candidates` / `pct_unseen_candidates` | ~0.078 / 0.922 | ~0.077 / 0.923 | ~0 | ~0.011 | ~0.005 |
| tutte le altre feature testate | — | — | — | < 0.02 | < 0.01 |

A prima vista, **nessuna feature raggiunge una soglia di effetto convenzionalmente "piccola"** (Cohen's d ≥ 0.2): il massimo osservato è −0.110 per `cbf_score_std`. Una lettura affrettata concluderebbe quindi per un NO-GO netto.

### 4.2 Analisi per decili: un pattern monotono reale, mascherato dalla sintesi lineare

Prima di concludere, è stata condotta un'analisi per decili (necessaria per non scartare un effetto non lineare mascherato da un coefficiente di correlazione lineare basso). Per `cbf_score_std` e `cbf_score_max` emerge un **pattern monotono e consistente**, invisibile nella sola correlazione di Pearson:

| Decile `cbf_score_std` (dal più basso al più alto) | Tasso CF_WINS |
| :--- | :---: |
| 1° (0.103–0.192) | 37.50% |
| 2° (0.192–0.205) | 37.83% |
| 3° (0.205–0.217) | 37.21% |
| 4° (0.217–0.229) | 35.37% |
| 5° (0.229–0.241) | 33.29% |
| 6° (0.241–0.255) | 33.02% |
| 7° (0.255–0.273) | 29.81% |
| 8° (0.273–0.299) | 28.99% |
| 9° (0.299–0.35) | 27.24% |
| 10° (0.35–0.5, più alto) | 29.14% |

Il tasso di `CF_WINS` scende in modo sostanzialmente monotono dal 37.5% (decile più basso di `cbf_score_std`) al 27-29% (decili più alti) — un'escursione di **circa 9-10 punti percentuali**, non spiegabile da rumore casuale su campioni di ~5.000 osservazioni per decile. **Split mediano** (soglia 0.2409): CF_WINS = 36.24% sotto la mediana contro 29.61% sopra (n=25.263 e 25.262) — una differenza di **6.6 punti percentuali**, facile da comunicare e da usare come regola. Lo stesso pattern, nella stessa direzione, si osserva per `cbf_score_max` (40.2% nel decile più basso → 29.9% nel decile più alto).

**Interpretazione**: quando il CBF produce punteggi poco differenziati tra i candidati (bassa varianza, punteggio massimo più contenuto — cioè quando il CBF è "meno sicuro" o trova l'impression meno chiaramente riconducibile a un profilo semantico netto), il CF ha una probabilità relativamente più alta di piazzare il click in una posizione migliore. È un risultato plausibile e interpretabile: non richiede di conoscere l'esito del click, è calcolabile durante lo scoring dell'impression (prima di mostrare le raccomandazioni), e mostra una direzione consistente su tutto l'intervallo dei valori, non un artefatto isolato in un singolo bin.

**Perché la sintesi lineare (sezione 4.1) lo sottostimava**: un Cohen's d/una correlazione di Pearson calcolati su due soli gruppi (CF_WINS=1 vs 0) catturano solo la differenza tra le medie complessive, che può essere modesta anche in presenza di un gradiente monotono reale se la relazione non è ben approssimata da una separazione netta a due gruppi. La lezione metodologica è di non fermarsi al solo confronto di medie quando si cerca un segnale predittivo debole.

### 4.3 Conclusione della sezione 4

Al contrario di quanto una prima lettura frettolosa della sola tabella 4.1 avrebbe suggerito, **esiste un segnale descrittivo plausibile e non trascurabile**: la dispersione (`cbf_score_std`) e il valore massimo (`cbf_score_max`) degli score CBF grezzi sul candidate set — entrambi calcolabili prima di conoscere il click — mostrano una relazione monotona e di ampiezza comunicabile (6-10 punti percentuali) con la probabilità che il CF collochi il click cliccato meglio del CBF. Nessuna delle altre feature testate (popolarità dei candidati, lunghezza della cronologia, numero di candidati, accordo Top-5) mostra un pattern altrettanto chiaro, né in forma lineare né per decili.

---

## 5. Nuovo protocollo sperimentale (proposta, non implementata)

### 5.1 Il problema da risolvere

Il DEV set (`MINDsmall_dev/behaviors.tsv`, 73.152 impression, Nov 15 2019) è stato usato estensivamente in questo progetto per: tuning dei pesi statici dell'Hybrid (grid search a più stadi), analisi per segmenti (history length, cold-start, popolarità), analisi di complementarità, bootstrap. Usarlo di nuovo come "test set indipendente" per valutare una Neural Gating Network o un Adaptive Hybrid violerebbe il principio che il set di valutazione finale non deve mai aver influenzato alcuna decisione di design del modello che si sta valutando.

### 5.2 Cosa rende possibile un vero split temporale

Il campo `time` di `behaviors.tsv` fornisce timestamp completi (data e ora) per ogni impression. La copertura temporale osservata è:

- **TRAIN**: 2019-11-09 00:00:19 → 2019-11-14 23:59:13 (156.965 impression, 6 giorni)
- **DEV**: 2019-11-15 00:00:01 → 2019-11-15 23:58:03 (73.152 impression, 1 giorno)

Questo rende **fattibile** uno split temporale interno al TRAIN, mantenendo l'ordine cronologico (nessuna impression di FIT successiva a impression di VALIDATION), senza toccare il DEV per alcuna decisione di design.

### 5.3 Protocollo proposto

| Split | Intervallo temporale | Impression (stima) | Cosa si addestra/calcola qui |
| :--- | :--- | :--- | :--- |
| **FIT** | 2019-11-09 00:00 → 2019-11-12 (≈primi 4 giorni, ≈70% delle impression TRAIN per volume orario) | ≈110.000 | Baseline (popolarità), matrice di co-click ItemKNN, e — se necessario — qualunque altro artefatto "allenato" sui dati, **ricalcolati usando solo questa porzione**. Gli embedding SBERT non necessitano ri-training (sono pre-addestrati e indipendenti dalle interazioni), ma il profilo utente CBF per-impression in fase di FIT userebbe comunque solo la history disponibile fino a quel momento. |
| **VALIDATION** | 2019-11-13 → 2019-11-14 (≈ultimi 2 giorni di TRAIN) | ≈47.000 | Calcolo delle feature pre-click e della label diagnostica `CF_WINS` **usando i modelli fittati solo su FIT**; selezione delle soglie del Rule-Based Adaptive Hybrid (sezione 6); tuning degli iperparametri e early stopping di un'eventuale gating network. **Nessuna soglia o iperparametro deve essere scelto guardando il FINAL HELD-OUT.** |
| **FINAL HELD-OUT** | DEV, 2019-11-15 (73.152 impression) | 73.152 | Un'**unica valutazione finale**, a modello/regola congelati, di CBF puro, Hybrid statico attuale, Rule-Based Adaptive Hybrid ed eventuale Neural Gating — per il confronto conclusivo. Per questa valutazione finale, Baseline/CF/CBF possono essere ri-fittati sull'intero TRAIN originale (FIT+VALIDATION), esattamente come nella pipeline attuale, dato che la loro logica non cambia: cambia solo cosa viene deciso guardando DEV (nulla, tranne il numero finale). |

### 5.4 Punti di attenzione dichiarati esplicitamente

- **Necessità di ri-addestrare Baseline/CF su FIT per la fase di VALIDATION**: sì, è necessario. `item_popularity` e la matrice ItemKNN andrebbero ricostruiti chiamando `load_data` con `train_behav_path` puntato a un file contenente solo le impression di FIT (stesso formato TSV, filtrato per timestamp) — nessuna modifica al codice esistente, solo un input diverso. Questo è l'unico modo per evitare che le feature/label calcolate su VALIDATION beneficino di informazione (popolarità, co-click) derivata dalle stesse impression che si stanno valutando.
- **Rischio residuo**: gli embedding SBERT (`content_based_text/embeddings/news_embeddings.npy`) sono calcolati una volta sola sul testo degli articoli e non dipendono dalle interazioni: non richiedono ri-fitting per split, e non introducono leakage (il testo di un articolo non cambia nel tempo).
- **Gestione delle news**: gli articoli che compaiono per la prima volta in VALIDATION (non presenti nella storia/popolarità di FIT) sono naturalmente "cold" per Baseline/CF costruiti su FIT — è una condizione realistica e voluta, coerente con la natura del problema, non un difetto del protocollo.
- **Cold-start nel nuovo protocollo**: la definizione di cold-start (sezione 1) andrebbe applicata in modo coerente — un articolo è "freddo" per il modello fittato su FIT se assente dalla sua base di popolarità/co-click, indipendentemente da cosa succede dopo in VALIDATION o DEV.
- **Rischio di leakage residuo tra FIT e VALIDATION**: nessuno strutturale, a patto di rispettare rigorosamente l'ordine temporale nello split e di non ricalcolare mai popolarità/co-click includendo VALIDATION.
- **Se il vero split temporale risultasse impraticabile** (ad esempio per vincoli di tempo/calcolo nel dover rieseguire `_build_itemknn_matrix` su un sottoinsieme di TRAIN): l'alternativa meno preferibile ma dichiarata è usare **cross-validation temporale a blocchi** (es. 3 fold, ciascuno con un confine temporale crescente) solo all'interno del TRAIN per la selezione delle soglie/iperparametri, mantenendo comunque il DEV come unico held-out finale. Il limite di questa alternativa è una minore quantità di dati "puri" per ciascun fold e un costo computazionale triplicato per la fase di validazione.

---

## 6. Rule-Based Adaptive Hybrid (proposta, non implementata)

Tre regole candidate, basate esclusivamente su feature pre-raccomandazione (sezione 3), da valutare (non implementare) sul protocollo di sezione 5, con soglie selezionate **solo su VALIDATION**. **Aggiornate dopo i risultati di sezione 4**: la Regola 1, inizialmente ipotizzata sulla popolarità dei candidati (motivata dall'evidenza post-hoc di `ANALISI_COMPLEMENTARITA.md`), è stata **sostituita** perché l'analisi diagnostica di sezione 4 mostra che la popolarità dei candidati non ha relazione utile con `CF_WINS` se usata come feature pre-click — a differenza della dispersione dello score CBF, che invece la mostra.

**Regola 1 — soglia sulla dispersione/confidenza del CBF (motivata dai risultati di sezione 4)**
```
SE cbf_score_std < soglia_bassa_dispersione   (il CBF e' poco "sicuro": punteggi poco differenziati)
ALLORA aumenta il peso del CF (es. verso 0/0.10/0.90 o oltre)
ALTRIMENTI usa la configurazione CBF-dominante attuale (0/0.03/0.97)
```
Motivazione diretta: sezione 4.2 mostra un tasso di `CF_WINS` del 36-38% quando `cbf_score_std` è sotto la mediana, contro 27-30% quando è sopra — un margine di 6-10 punti percentuali, il segnale pre-click più solido individuato in questa fase.

**Regola 2 — variante sul punteggio massimo del CBF**
```
SE cbf_score_max < soglia_bassa_confidenza
ALLORA aumenta il peso di CF
ALTRIMENTI CBF-dominante
```
Alternativa a `cbf_score_std` (le due sono correlate, r=0.20, ma non identiche): da confrontare in fase di validazione per scegliere quale delle due generalizzi meglio.

**Regola 3 — combinazione delle prime due con un vincolo di confidenza del CF (esplorativa, non ancora supportata da un effetto isolato in sezione 4)**
```
SE cbf_score_std < soglia_bassa_dispersione
   E cf_gap_top1_top2 > soglia_confidenza_cf
ALLORA dai un peso non trascurabile al CF
ALTRIMENTI CBF-dominante
```
Più sofisticata; `cf_gap_top1_top2` da solo non ha mostrato un effetto chiaro in sezione 4 (Cohen's d 0.011), ma potrebbe agire da filtro complementare in combinazione con la Regola 1 — da verificare empiricamente in fase di validazione, senza assumere a priori che l'aggiunta migliori la Regola 1 da sola.

Questo modello, una volta selezionate le soglie su VALIDATION, servirebbe da **controllo** per stabilire se una rete neurale sia realmente necessaria (si veda il criterio Go/No-Go in sezione 8): se il Rule-Based Adaptive Hybrid già recupera una parte sostanziale del margine teorico di complementarità (sezione 7 di `ANALISI_COMPLEMENTARITA.md`), l'investimento in una gating network appresa avrebbe un ritorno marginale inferiore.

---

## 7. Confronto: Neural Gating Network vs Neural Meta-Ranker

| Aspetto | A. Gating Network (impression-level) | B. Neural Meta-Ranker (candidate-level) |
| :--- | :--- | :--- |
| **Output** | Pesi `w_CF, w_CBF[, w_B]` per l'intera impression, poi combinazione lineare degli score esistenti | Uno score finale per ciascun candidato, funzione non lineare delle feature del candidato |
| **Flessibilità** | Limitata: stessa combinazione lineare per tutti i candidati della stessa impression | Alta: può trattare ogni candidato in modo diverso, anche all'interno della stessa impression |
| **Rischio di overfitting** | Più basso (poche uscite, 2-3 pesi per impression) | Più alto (una predizione per candidato, più gradi di libertà) |
| **Interpretabilità** | Alta: i pesi appresi sono direttamente confrontabili con la configurazione statica attuale | Più bassa: uno score composito è meno immediato da spiegare in sede di esame |
| **Coerenza con l'evidenza raccolta** | Adeguata se la complementarità dipende da caratteristiche **dell'impression nel suo complesso** (es. quota di candidati popolari) | Più adeguata se la complementarità dipende da caratteristiche **del singolo candidato** (es. popolarità del singolo item, non la media dell'impression) |
| **Costo implementativo** | Più basso (rete piccola, input a bassa dimensionalità) | Leggermente più alto (richiede una rappresentazione per candidato, non solo per impression) |

**Valutazione alla luce dell'analisi svolta (rivista dopo i risultati di sezione 4)**: una prima ipotesi, basata sull'evidenza post-hoc di `ANALISI_COMPLEMENTARITA.md` (sezione 6, popolarità del *candidato cliccato*), suggeriva che il Meta-Ranker candidate-level potesse essere più adatto, dato che quel segnale è una proprietà del singolo candidato. I risultati descrittivi di sezione 4 di questo documento **non confermano** questa ipotesi: la popolarità dei candidati (`max_train_pop_candidates`, `mean_train_pop_candidates`) — l'unico proxy pre-click disponibile per l'informazione usata in quell'analisi post-hoc — **non mostra alcuna relazione utile con `CF_WINS`** (Cohen's d ≈ 0.01, nessun pattern nei decili). Al contrario, il segnale pre-click più chiaro trovato (`cbf_score_std`, `cbf_score_max`) è per costruzione una statistica **aggregata sull'intero candidate set dell'impression**, non una proprietà del singolo candidato.

Questo capovolge la conclusione preliminare: **l'evidenza raccolta favorisce una Gating Network a livello di impression rispetto a un Meta-Ranker candidate-level**, perché il segnale predittivo realmente trovato (dispersione/massimo dello score CBF sull'impression) è nativamente un input impression-level, esattamente il tipo di feature che una gating network consuma in ingresso. Un Meta-Ranker resterebbe comunque un'opzione tecnicamente valida (potrebbe comunque includere `cbf_score_std`/`cbf_score_max` come feature broadcast su tutti i candidati della stessa impression), ma non ha, allo stato attuale dell'evidenza, un vantaggio dimostrato rispetto alla gating network più semplice.

---

## 8. Criterio Go/No-Go

Definito **prima** di osservare i risultati completi di sezione 4, per evitare di scegliere il criterio in base al risultato desiderato:

> **GO** (procedere con un Adaptive Hybrid rule-based e/o una rete neurale, gating o meta-ranker) se, sul dataset diagnostico di sezione 4, almeno una delle feature pre-click mostra una relazione descrittiva chiaramente non casuale con `CF_WINS` — operativamente: una differenza nella media della feature tra il gruppo `CF_WINS=1` e `CF_WINS=0` che sia consistente in direzione e non trascurabile in ampiezza rispetto alla variabilità osservata, per almeno una feature interpretabile (non un artefatto di rumore isolato in una singola feature tra le tante testate).
>
> **NO-GO su una gating/rule-based hybrid, e valutazione di NRMS come alternativa** se nessuna feature pre-click mostra una relazione descrittiva plausibile con `CF_WINS` — cioè se la complementarità osservata in `ANALISI_COMPLEMENTARITA.md` risultasse imprevedibile con le informazioni disponibili prima del click, e quindi non sfruttabile da alcun meccanismo di gating, per quanto sofisticato.

Il verdetto viene applicato qui di seguito ai risultati descrittivi di sezione 4, senza modifiche a posteriori del criterio pre-registrato sopra.

### 8.1 Verdetto: GO (condizionato)

I risultati di sezione 4.2 soddisfano il criterio GO pre-registrato: `cbf_score_std` e `cbf_score_max` mostrano una relazione descrittiva **monotona, consistente in direzione su tutto l'intervallo dei valori, e di ampiezza comunicabile (6-10 punti percentuali di differenza nel tasso di `CF_WINS`)** — non un artefatto isolato di rumore in una singola feature tra le tante testate, dato che due feature correlate tra loro (`corr(cbf_score_std, cbf_score_max) = 0.20`) mostrano lo stesso pattern in modo indipendente.

**Il GO è condizionato**, non incondizionato, per tre ragioni da riconoscere esplicitamente:
1. L'ampiezza dell'effetto è **modesta**: 6-10 punti percentuali di differenza nel tasso di `CF_WINS`, non una separazione netta. Un modello che sfruttasse solo questo segnale otterrebbe un miglioramento incrementale, non trasformativo.
2. Delle 14 feature testate, solo 2 (correlate tra loro) mostrano un pattern chiaro: la maggioranza delle feature pre-click ipotizzate (popolarità dei candidati, lunghezza della cronologia, numero di candidati, accordo Top-5) **non mostra alcuna relazione utile**, incluso il proxy pre-click della popolarità che si sarebbe atteso più promettente alla luce dell'analisi post-hoc di `ANALISI_COMPLEMENTARITA.md`.
3. Questa è un'analisi descrittiva univariata: non è stata verificata la capacità predittiva di un modello multivariato (che potrebbe fare leggermente meglio combinando più feature deboli) né la sua tenuta su un campione indipendente (si veda il protocollo di sezione 5).

### 8.2 Raccomandazione finale sulla fase successiva

Alla luce di questo GO condizionato, la raccomandazione **non è** procedere direttamente a una rete neurale complessa, bensì una progressione incrementale e a basso rischio, in questo ordine, con verifica ad ogni passo prima di proseguire al successivo:

1. **Rule-Based Adaptive Hybrid** (sezione 6), usando esplicitamente `cbf_score_std`/`cbf_score_max` come feature di condizionamento (non la popolarità, che l'evidenza di sezione 4 sconsiglia), con soglie selezionate sul FIT/VALIDATION di sezione 5, mai sul DEV. Se questa regola semplice recupera già una parte sostanziale del margine (si veda il limite superiore teorico dell'8-9% di `ANALISI_COMPLEMENTARITA.md` sezione 7), una rete neurale aggiuntiva avrebbe un ritorno marginale probabilmente basso.
2. Solo se il passo 1 risultasse insufficiente: **Neural Gating Network** a livello di impression (non un Meta-Ranker candidate-level, per le ragioni di sezione 7 rivista), con le feature di sezione 3 come input, allenata e validata secondo il protocollo di sezione 5.
3. **NRMS non è raccomandato come prossimo passo immediato**: l'evidenza di questa fase mostra una complementarità reale ma modesta e riconducibile a una manciata di feature interpretabili, uno scenario che non giustifica ancora il salto diretto a un paradigma neurale end-to-end sostitutivo del CBF esistente. Resterebbe un'opzione di ricerca valida in una fase più avanzata, se i passi 1-2 esaurissero il margine di miglioramento disponibile senza avvicinarsi a un salto di paradigma.

---

## 9. Audit della costruzione del training set del Collaborative Filtering

**STOP temporaneo esplicito**: nessun Rule-Based Adaptive Hybrid né alcuna rete neurale è stato implementato in questa sezione. Nessun modello di produzione è stato sostituito. Questa sezione risolve una questione strutturale a monte, sollevata prima di investire ulteriormente nel gating: *il CF usato dall'Hybrid sta usando correttamente tutte le interazioni positive disponibili nel TRAIN?*

### 9.1 Fonti di interazione usate oggi, per modello (verificato leggendo il codice)

| Modello | `history` | Click positivi impression (label=1) | Negativi (label=0) | Timestamp/ordine | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `hybrid_recommender.py` → `item_popularity` (Baseline dell'Hybrid) | **Sì** (unica fonte) | **No** | No | No | `load_data`, righe 174-182: `Counter` sul solo campo `history` esploso |
| `hybrid_recommender.py` → `_build_itemknn_matrix` (CF dell'Hybrid) | **Sì** (unica fonte) | **No** | No | No (solo l'ordine interno della history per il decay di recency) | righe 195-265: `local` = solo `history` per riga, filtrato sull'universo `news_id_to_idx` |
| `cf_latent_item/create_interaction_matrix.py` (matrice per ALS) | **Sì** (unica fonte) | **No** | No | No | righe 21-32: `dropna(subset=['history'])`, poi split della sola history |
| `baseline_model/baseline.ipynb` → `get_popularity_scores` | **No** | **Sì** (unica fonte, `endswith("-1")`) | No | No | Popolarità = click ufficiali nelle impression, non history |
| `cf_itemknn/itemknn_recency_pure.py` e `itemknn_recency_fallback.py` → `build_coclick_counts`/`build_popularity` | **Sì** | **Sì** (`local = hist + clicked`, riga 124/151 di ciascun file) | No | No (solo per il decay di recency nello scoring, non nel training) | Questi due moduli standalone **già combinano** history e click positivi — sono gli unici due file del repository che lo fanno |

**Conclusione del punto 1**: il CF e la popolarità usati **dentro l'Hybrid** (il modello finale, oggetto della presentazione) sono costruiti **solo dal campo `history`**, la fonte più povera tra quelle disponibili nel progetto. Due moduli standalone del progetto (`cf_itemknn/itemknn_recency_pure.py` e `_fallback.py`) già implementano una costruzione più ricca (history + click), ma **non sono quelli usati dall'Hybrid**, che reimplementa la propria versione di ItemKNN internamente (`_build_itemknn_matrix`), più semplice e più povera. Nessun modello del progetto usa i negativi delle impression o l'ordine cronologico assoluto tra righe diverse per costruire il CF (l'unico uso del tempo è il decay di recency *all'interno* della history di un singolo utente).

### 9.2 Quanta informazione positiva viene ignorata (TRAIN completo, 156.965 righe, 50.000 utenti)

Calcolato leggendo direttamente `dataset/MINDsmall_train/behaviors.tsv` (script `run_cf_training_source_audit.py`, nessuna modifica al codice esistente):

| Quantità | Valore |
| :--- | ---: |
| Interazioni totali da `history` | 5.107.639 |
| Articoli distinti in `history` | 33.195 |
| Click positivi nelle impression (label=1) | 236.344 |
| Articoli distinti cliccati nelle impression | 7.713 |
| Candidati distinti nelle impression (click o no) | 20.288 |
| Articoli in **entrambi** gli insiemi (history ∩ cliccati) | 1.043 (3,14% degli articoli di history) |
| Articoli **solo** in history (mai cliccati nelle impression TRAIN) | 32.152 |
| Articoli **solo** cliccati (mai in alcuna history) | 6.670 (86,48% degli articoli cliccati) |
| Click positivi totali su articoli assenti da qualunque history | 202.528 (**85,7% di tutti i click positivi del TRAIN**) |
| Copertura CF attuale (history-only) | 33.195 articoli |
| Copertura potenziale (history ∪ click positivi) | 39.865 articoli (**+20,1%**) |

**L'osservazione più rilevante**: solo il 3,14% degli articoli di `history` compare anche tra i click ufficiali delle impression TRAIN. Questo **non è un'anomalia**: `history` è un log fisso per utente (verificato empiricamente in sezione 9.5 — stesso valore per lo stesso utente su tutte le sue righe TRAIN, indipendentemente dal giorno), quindi rappresenta la lettura *pregressa* dell'utente, presumibilmente raccolta prima dell'inizio della finestra di log delle impression (9-14 novembre) — mentre gli `impressions` registrano i click sulle notizie *trending in quella specifica finestra*. Il turnover editoriale delle news fa sì che i due insiemi si sovrappongano pochissimo per costruzione, non per un errore di join.

Il **85,7% dei click positivi del TRAIN non entra mai** nella costruzione di popolarità/co-click dell'Hybrid: sono eventi di interazione realmente osservati (l'utente ha cliccato quell'articolo), semplicemente registrati nel campo "sbagliato" (`impressions` invece di `history`) rispetto a quello che `hybrid_recommender.py` consulta.

### 9.3 Scelta progettuale o limitazione?

**Non è una scelta motivata esplicitamente nel codice**: non c'è un commento, un test o una nota di design in `hybrid_recommender.py` che giustifichi l'uso esclusivo di `history`. È verosimilmente una **semplificazione ereditata dal modo più naturale di leggere `behaviors.tsv`** (la colonna `history` è letteralmente etichettata come "storico dell'utente", quindi la scelta più ovvia per rappresentare le sue preferenze) più che una decisione deliberata di escludere i click delle impression. Il fatto che i due moduli standalone `cf_itemknn/itemknn_recency_pure.py`/`_fallback.py` — scritti nello stesso progetto, presumibilmente in una fase di sperimentazione diversa — **includano correttamente entrambe le fonti** rafforza l'ipotesi che si tratti di un'**incoerenza tra moduli del progetto**, non di una decisione metodologica intenzionale mantenuta con cognizione di causa nell'Hybrid finale.

Dal punto di vista tecnico, per un CF supervisionato sul TRAIN, **ha senso includere anche i click positivi delle impression**: sono interazioni positive osservate esattamente come quelle di `history`, semplicemente con un timestamp diverso (contestuale alla riga, non pregresso). Ignorarle non è conservativo, è una perdita di segnale — a patto di gestire correttamente la componente temporale (sezione 9.4).

**Non lo classifico come "bug"**: il codice fa esattamente quello che è scritto per fare, e produce risultati riproducibili e internamente coerenti con se stesso (nessun crash, nessun comportamento indefinito). È una **limitazione di copertura non necessaria**, introdotta da un uso incompleto dei dati disponibili, distinta da un errore di programmazione.

### 9.4 Controllo temporale: perché l'arricchimento va fatto con cautela

Verificato empiricamente (sezione 9.5): `history` è statica per utente durante tutta la finestra TRAIN, quindi non introduce leakage se riusata "com'è" in qualunque split. Il campo `impressions`, invece, è specifico della singola riga/timestamp: includere il click positivo di una riga per costruire un segnale usato per valutare **quella stessa riga** sarebbe una tautologia (si userebbe l'etichetta per predire l'etichetta). Includerlo per valutare righe di **altri utenti in un momento successivo** è invece legittimo, esattamente come history stessa. Il protocollo adottato (sezione 9.5) rispetta questo vincolo: il grafo di co-click "arricchito" per una fase di valutazione è sempre costruito usando **solo le righe di un periodo temporale strettamente precedente**, mai le righe della fase valutata.

### 9.5 Split temporale congelato e verifica di staticità della history

Verifica empirica (5 utenti campione con >1 riga TRAIN, tutte le righe del TRAIN): per ogni utente con più righe, la lunghezza (e il contenuto) di `history` è **identica su tutte le righe**, indipendentemente dalla data — confermando che `history` precede l'intera finestra di log e può essere trattata come "sicura" rispetto a qualunque split temporale interno al TRAIN.

Split **congelato prima di osservare i risultati** (`temporal_split_definition.py`), basato sulla distribuzione reale dei timestamp (Nov 9: 13.570 righe, Nov 10: 15.048, Nov 11: 32.799, Nov 12: 33.654, Nov 13: 31.624, Nov 14: 30.270 — verificato direttamente su `behaviors.tsv`, non assunto):

| Split | Giorni | Righe | Uso |
| :--- | :--- | ---: | :--- |
| **FIT** | 9-11 novembre | 61.417 | Costruzione di CF_original (solo history) e CF_enriched (history + click positivi di queste sole righe) |
| **VALIDATION** | 12-13 novembre | 65.278 | Valutazione (nessuna scelta di soglie/iperparametri fatta guardando oltre) |
| **INTERNAL_TEST** | 14 novembre | 30.270 | Valutazione **unica**, a configurazione congelata, ricostruendo CF da FIT+VALIDATION (9-13 nov, 126.695 righe) |

Il **DEV reale (15 novembre) resta escluso** da questo intero esperimento.

### 9.6 Risultati: CF_original vs CF_enriched vs CBF puro

Stesso algoritmo ItemKNN (shrinkage Bayesiano, k=200, shrinkage=50, decay=0.8) usato da `hybrid_recommender.py`, applicato identico in entrambe le varianti: **l'unica variabile è la fonte delle interazioni per costruire il grafo di co-click e la popolarità**. Script: `run_cf_enriched_experiment.py`.

**VALIDATION (fit=Nov 9-11, eval=Nov 12-13, 63.771 impression valutabili)**

| Config | AUC | nDCG@5 | MRR | P@5 | Copertura articoli | % candidati "warm" |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CF_original | 0,5015 | 0,2339 | 0,2558 | 0,0823 | 24.235 | 45,68% |
| CF_enriched | 0,5020 | 0,2346 | 0,2566 | 0,0826 | 24.495 (+1,07%) | 45,68% |
| CBF puro | **0,5157** | **0,2629** | **0,2848** | **0,0922** | — | — |
| Baseline_original | 0,4927 | 0,2258 | 0,2437 | 0,0787 | — | — |
| Baseline_enriched | 0,4988 | 0,2399 | 0,2625 | 0,0826 | — | — |

**INTERNAL_TEST (fit=Nov 9-13, eval=Nov 14, configurazione congelata, 29.498 impression valutabili, valutato una sola volta)**

| Config | AUC | nDCG@5 | MRR | P@5 | Copertura articoli | % candidati "warm" |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CF_original | 0,5033 | 0,2109 | 0,2427 | 0,0777 | 25.769 | 85,71% |
| CF_enriched | 0,5036 | 0,2112 | 0,2429 | 0,0778 | 26.513 (+2,89%) | 85,71% |
| CBF puro | **0,6104** | **0,3054** | **0,3318** | **0,1079** | — | — |
| Baseline_original | 0,4746 | 0,1829 | 0,2072 | 0,0686 | — | — |
| Baseline_enriched | 0,4900 | 0,1979 | 0,2220 | 0,0733 | — | — |

### 9.7 Interpretazione

1. **L'arricchimento del CF non produce un miglioramento rilevante, in nessuno dei due stadi**: ΔAUC = +0,0005 (VALIDATION) e +0,0003 (INTERNAL_TEST) — entrambi ordini di grandezza sotto qualunque soglia interpretabile come reale, e ben dentro il rumore atteso da bootstrap su questa numerosità (per confronto, si veda l'ampiezza degli intervalli di confidenza in `bootstrap_results.json`, fase precedente). **L'ipotesi che il CF fosse debole "perché" ignora i click delle impression non è supportata da questo esperimento.**
2. **Il guadagno di copertura è reale ma piccolo alla scala rilevante**: +20,1% sull'intero TRAIN (sezione 9.2), ma solo **+1,07%/+2,89%** quando l'arricchimento è correttamente limitato al solo periodo di FIT (nessun leakage) — perché molti dei 6.670 articoli "solo click" del TRAIN completo sono stati cliccati in **giorni diversi** da quelli inclusi in un dato FIT, quindi non tutti disponibili contemporaneamente.
3. **Il problema reale è più profondo del data-source**: il 79,82% degli articoli candidati nella finestra VALIDATION (Nov 12-13) **non ha alcuna rappresentazione** né in history né nei click di FIT (Nov 9-11) — verificato con un controllo diretto (non solo dedotto dalle metriche). Il turnover editoriale delle notizie fa sì che gli articoli "caldi" in un dato momento siano in larga parte diversi da quelli osservati anche solo 1-3 giorni prima, indipendentemente da quale campo (`history` o `impressions`) si usi per costruire il segnale. Aggiungere i click delle impression sposta leggermente il confine cold/warm, ma non risolve il problema strutturale.
4. **La popolarità (Baseline) trae un beneficio relativo maggiore dall'arricchimento del CF stesso**: `baseline_enriched` supera sempre `baseline_original` (+0,0061 AUC su VALIDATION, +0,0154 su INTERNAL_TEST) — un effetto piccolo ma sistematico e nella direzione attesa, a differenza del CF il cui ΔAUC è indistinguibile da zero. Questo suggerisce che il canale "click delle impression" porti più informazione utile alla popolarità aggregata che alla struttura fine di co-click sfruttata dall'ItemKNN.
5. **CBF resta nettamente il segnale più informativo in entrambi gli stadi**, e mostra una varianza giorno-per-giorno notevole (AUC 0,5157 su Nov 12-13 vs 0,6104 su Nov 14) pur usando esattamente lo stesso algoritmo e gli stessi embedding — un promemoria che **una finestra di 1-2 giorni è un campione rumoroso**, e che le cifre di AUC riportate altrove nel progetto (calcolate sul DEV, un'unica giornata) vanno lette con la stessa cautela.

### 9.8 Rivalutazione Scenario A/B/C e della Gating Network

Il CF "migliore tra i due" (CF_enriched, per una marginalissima superiorità non significativa) **resta sostanzialmente casuale** (AUC ≈ 0,50 su entrambi gli split interni), non diversamente dal CF_original già caratterizzato in `ANALISI_COMPLEMENTARITA.md`. La domanda posta in apertura di questa fase — *"esiste complementarità tra un CF correttamente alimentato con le interazioni disponibili e il CBF?"* — riceve quindi la stessa risposta di prima: la classificazione **Scenario B** (complementarità debole ma non nulla, concentrata in specifici sotto-segmenti, si veda `ANALISI_COMPLEMENTARITA.md` sezione 13) **resta valida** anche dopo aver arricchito correttamente il CF. Non ci sono elementi per riclassificare a Scenario A (forte complementarità) né per declassare a Scenario C (CF del tutto casuale ovunque, dato che i sotto-segmenti di sezione 4-7 di `ANALISI_COMPLEMENTARITA.md` mostravano comunque zone dove il CF vince).

**Conseguenza pratica**: la raccomandazione di sezione 8.2 (Rule-Based Adaptive Hybrid su `cbf_score_std`/`cbf_score_max`, poi eventualmente Gating Network, NRMS non immediato) **non cambia**. Il problema strutturale della copertura del CF (sezione 9.7 punto 3) non è un problema di costruzione del training set risolvibile aggiungendo i click delle impression: è un limite intrinseco del dominio (turnover editoriale) che nessuna delle due varianti di CF qui testate supera. Un'eventuale Gating Network dovrebbe quindi continuare a trattare il CF come un segnale sistematicamente debole con occasionali finestre di utilità (i sotto-segmenti già identificati), non come un segnale "sistemato" dall'arricchimento.

### 9.9 File creati in questa sezione

- `hybrid/experiments/run_cf_training_source_audit.py` — audit statico delle fonti di interazione (sezioni 9.1-9.2)
- `hybrid/experiments/cf_training_source_audit.json` — output numerico corrispondente
- `hybrid/experiments/temporal_split_definition.py` — definizione congelata dello split FIT/VALIDATION/INTERNAL_TEST
- `hybrid/experiments/run_cf_enriched_experiment.py` — script sperimentale CF_original vs CF_enriched vs CBF (nessuna modifica a `hybrid_recommender.py`)
- `hybrid/experiments/cf_enriched_validation_results.json`, `cf_enriched_internal_test_results.json` — risultati numerici

**Nessun file di produzione è stato modificato**: `hybrid/hybrid_recommender.py` non ha ricevuto alcuna modifica in questa sezione (verificato via `git diff --stat`, invariato rispetto alla fase precedente).

---

## 10. Verifica del salto CBF VALIDATION→INTERNAL_TEST e ricalcolo della complementarità sul protocollo temporale interno

Questa sezione risponde a due domande sollevate dall'analisi di sezione 9: (a) perché il CBF salta da AUC≈0.5157 (VALIDATION) a ≈0.6104 (INTERNAL_TEST); (b) se la complementarità CF/CBF osservata sul protocollo DEV esiste ancora nel protocollo temporale interno. **Nessun modello di produzione modificato, nessuna rete implementata.**

### 10.1 Audit del protocollo temporale, modello per modello (verificato leggendo il codice ed i dati, non assunto)

| Modello | Dati di fitting | VALIDATION usata come input? | Dati futuri usati? | Leakage comportamentale? | Nota |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Baseline_original | `history` delle sole righe FIT (9-11 nov) | No | No | No | `history` è statica per utente (sezione 9.5), quindi innocua rispetto a qualunque confine temporale |
| Baseline_enriched | `history` + click positivi delle sole righe FIT | No | No | No | I click aggiunti sono sempre temporalmente ≤ fine FIT, mai di VALIDATION |
| CF_original | Grafo ItemKNN costruito solo da `history` delle righe FIT | No | No | No | Vedi però bias di catalogo articoli, sezione 10.2 |
| CF_enriched | Grafo ItemKNN da `history` + click positivi delle sole righe FIT | No | No | No | Stesso bias di catalogo di CF_original |
| CBF | Nessun fitting su dati comportamentali: per ogni riga usa solo (a) l'embedding SBERT statico del testo e (b) la `history` della riga stessa | No (nessun aggregato cross-riga) | **Sì, in un senso specifico e non comportamentale** — l'universo di articoli/testo deriva da `MINDsmall_dev/news.tsv`, un catalogo fissato al 15 novembre | No (nessuna label di click entra nel CBF) | **Bias di copertura sistematico, quantificato in 10.2**: il catalogo usato non è "conoscibile" in modo uniforme nei giorni TRAIN, perché è il catalogo di UN giorno specifico (il 15 nov), non quello effettivamente attivo in ciascun giorno TRAIN |
| Hybrid (pesi 0/0.03/0.97) | Non eseguito in questo esperimento | — | — | — | Questa fase confronta solo i segnali puri (Baseline/CF/CBF); l'Hybrid combinato non è stato rivalutato sul nuovo split |

**Incongruenza rilevata**: il confronto tra CF (fittato solo su FIT) e CBF (che non richiede fitting comportamentale, ma verifica ogni candidato contro un catalogo articoli fisso e non temporalmente allineato) **non è omogeneo rispetto al contenuto**, anche se lo è rispetto al comportamento. Non è "leakage" nel senso di informazione di click dal futuro, ma è un disallineamento di scope tra ciò che ciascun modello può effettivamente "vedere" per costruire il proprio segnale. Si veda 10.2 per la quantificazione.

### 10.2 Spiegazione quantitativa del salto CBF (0.5157 → 0.6104): non è (principalmente) drift comportamentale

Verificato (`run_daily_causal_audit.py` + controllo aggiuntivo mirato):

**A. Il catalogo articoli usato da CBF (`MINDsmall_dev/news.tsv`, 42.416 articoli) copre l'intero universo di `MINDsmall_train/news.tsv` (51.282 articoli) solo parzialmente e in modo MOLTO diseguale per giorno TRAIN:**

| Giorno | % candidati presenti in `dev/news.tsv` | % candidati presenti in `train/news.tsv` |
| :--- | ---: | ---: |
| 9 nov | 68,50% | 100% |
| 10 nov | 45,89% | 100% |
| 11 nov | 38,17% | 100% |
| 12 nov | 38,01% | 100% |
| 13 nov | 53,05% | 100% |
| 14 nov | **85,70%** | 100% |

Il catalogo `dev/news.tsv` è per costruzione "il catalogo del 15 novembre": copre meglio gli articoli ancora in circolazione vicino a quella data (14 nov: 85,7%) e molto peggio i giorni più lontani (11-12 nov: ~38%). Gli embedding SBERT precalcolati esistono **solo** per gli articoli di `dev/news.tsv` (`news_embeddings.npy`, shape 42.416): per un candidato assente da questo catalogo, sia il segnale semantico sia quello topical del CBF ricevono **score 0 per costruzione**, non perché l'articolo sia davvero "irrilevante".

**B. Quando il candidato CLICCATO è quello assente dal catalogo, il CBF perde automaticamente quella impression** (l'unico item con score potenzialmente informativo riceve 0, gli altri negativi warm hanno score reali): questo è un artefatto meccanico, non un giudizio di qualità del modello.

Verifica diretta, ristretta alle impression in cui il **clicked item è coperto dal catalogo** (quindi il CBF ha almeno la possibilità concreta di valutarlo):

| Giorno | % clicked item coperti da `dev/news.tsv` | AUC CBF (tutte le impression) | AUC CBF (solo clicked item coperto) |
| :--- | ---: | ---: | ---: |
| 12 nov | 26,58% | 0,5085 | **0,7752** |
| 13 nov | 43,17% | 0,5234 | **0,7516** |
| 14 nov | 86,06% | 0,6104 | **0,6673** |

**Risultato chiave**: una volta condizionato sulla copertura del catalogo (stesso "tipo" di impression, cioè quelle in cui il CBF può davvero vedere l'articolo cliccato), il 12 e il 13 novembre mostrano un AUC CBF **più alto**, non più basso, del 14 novembre (0,775/0,752 vs 0,667). Il salto aggregato osservato (0,5157→0,6104) è quindi spiegato in larghissima parte dalla **quota fortemente crescente di impression "vinte automaticamente per assenza di rappresentazione"** man mano che ci si avvicina al 15 novembre (data del catalogo), non da un miglioramento reale della capacità discriminante del CBF.

**C. Verifica di non-confondimento con altre variabili**: la comparazione stratificata per decili di `history_length` e di `n_candidates` (`daily_stratified_comparison.csv`) mostra il gap VALIDATION→INTERNAL_TEST **presente e di ampiezza simile (~0,08-0,13) in OGNI decile**, il che esclude che il salto grezzo sia spiegabile da una diversa distribuzione di lunghezza history o dimensione del candidate set tra i due periodi — è consistente con un fattore comune a tutte le fasce, cioè il bias di catalogo appena descritto.

### 10.3 Classificazione della differenza

| Causa | Quantificata? | Contributo |
| :--- | :---: | :--- |
| A. Drift temporale comportamentale reale | Sì (per differenza) | **Assente o di segno opposto**: a parità di copertura di catalogo, il segnale CBF è più forte, non più debole, il 12-13 novembre |
| B. Diversa composizione impression (n. positivi, ecc.) | Sì | Minima: `n_positive_mean` varia solo 1,45→1,55 nell'intera settimana |
| C. Diverso cold-start (rispetto a history/popolarità) | Sì | Non rilevante per il CBF (che non dipende da popolarità); il cold-start "causale" è anzi più alto il 14 nov (94,8%) che il 12 nov (94,4%) — nella direzione sbagliata per spiegare un CBF più alto |
| D. Diversa distribuzione del candidate-set (dimensione, history length) | Sì (stratificato) | Esclusa come causa principale: gap persistente in ogni decile |
| **E. Protocollo/fitting non omogeneo (bias di catalogo articoli)** | **Sì, quantificato in 10.2** | **Causa dominante**: la copertura del catalogo `dev/news.tsv` passa da 26,6%/43,2% (12/13 nov, per il clicked item) a 86,1% (14 nov), e l'AUC condizionata sulla copertura si inverte |

**Classificazione finale: prevalentemente (E), con un'indicazione secondaria (non conclusiva) che il segnale CBF "vero" potrebbe essere leggermente più forte il 12-13 novembre che il 14, cioè l'opposto della lettura naive.** Questo secondo punto non viene enfatizzato come conclusione definitiva: il sottoinsieme "clicked item coperto" non è un campione casuale (è esso stesso selezionato in modo diverso per giorno, si veda percentuali di copertura molto diverse), quindi anche questo confronto andrebbe preso con cautela; ma è sufficiente per escludere con sicurezza che il salto grezzo rifletta un vero miglioramento del CBF verso la fine della settimana.

**Implicazione per il resto del progetto**: qualunque cifra di AUC calcolata usando `MINDsmall_dev/news.tsv` come universo di articoli per valutare periodi **diversi dal DEV stesso** (incluse le sezioni 9.6-9.7 di questo documento, che confrontano VALIDATION e INTERNAL_TEST) va letta come influenzata da questo bias quando si confrontano periodi tra loro. Il confronto **CF_original vs CF_enriched all'interno dello stesso stage** (sezione 9.6) resta valido, perché il bias di catalogo è identico per le due varianti nello stesso stage e quindi si cancella nel confronto; non è invece valido usare i numeri assoluti di CBF per confrontare VALIDATION e INTERNAL_TEST come se misurassero la stessa cosa.

### 10.4 Performance CBF giorno per giorno (causale, `run_daily_causal_audit.py`)

| Giorno | N impression valutabili | AUC | MRR | nDCG@5 | nDCG@10 | Cold-start (causale) | History length (media/mediana) | Candidate count (media/mediana) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 9 nov | 13.414 | 0,5717 | 0,3037 | 0,2783 | 0,3368 | 100%* | 33,5 / 19 | 38,6 / 26 |
| 10 nov | 14.815 | 0,5402 | 0,3154 | 0,3046 | 0,3558 | 82,4% | 31,6 / 18 | 34,6 / 22 |
| 11 nov | 32.229 | 0,4908 | 0,2837 | 0,2792 | 0,3308 | 90,8% | 31,8 / 18 | 35,6 / 22 |
| 12 nov | 32.878 | 0,5085 | 0,2798 | 0,2637 | 0,3221 | 94,4% | 33,6 / 20 | 35,0 / 24 |
| 13 nov | 30.893 | 0,5234 | 0,2901 | 0,2620 | 0,3178 | 91,8% | 34,3 / 20 | 39,0 / 26 |
| 14 nov | 29.498 | **0,6104** | **0,3318** | **0,3054** | **0,3630** | 94,8% | 34,0 / 20 | 40,4 / 27 |

\* Nov 9 è il primo giorno TRAIN: non esiste alcun giorno precedente, quindi il cold-start causale è per definizione 100% (nessun dato pregresso disponibile) — non un valore comparabile agli altri giorni.

**Osservazione**: AUC del CBF **non mostra un trend monotono** nei primi 5 giorni (0,57→0,54→0,49→0,51→0,52, cioè oscilla), poi salta a 0,61 il 14 nov. Un vero drift temporale continuo produrrebbe una tendenza more graduale; il pattern osservato (stabile/oscillante poi salto netto proprio nell'ultimo giorno, quello più vicino alla data del catalogo `dev/news.tsv`) è coerente con la spiegazione di 10.2 (bias di catalogo), non con un drift comportamentale progressivo.

### 10.5 Complementarità CF_enriched vs CBF ricalcolata su VALIDATION (protocollo temporale interno, mai sul protocollo DEV)

Script: `run_validation_complementarity_v2.py`. CF_enriched costruito solo da FIT (9-11 nov); valutato sulle 45.877 impression a singolo positivo di VALIDATION (12-13 nov).

**Scoperta metodologica critica, verificata prima di riportare i numeri**: il 76,42% delle impression VALIDATION ha **tutti** i candidati con `cf_score=0` (nessuna rappresentazione CF, coerente con la bassa copertura di catalogo/FIT già discussa in sezione 9.7). In questo sottoinsieme degenere, il ranking del CF non contiene alcuna informazione: tutti i candidati sono a pari merito (score 0) e il tie-break deterministico del progetto (`np.lexsort` su `news_id` crescente) decide l'ordine in modo puramente alfabetico. Verificato empiricamente: **anche in questo sottoinsieme puramente degenere, "CF_WINS" (rank_CF < rank_CBF) si verifica nel 41,37% dei casi** — un numero enorme che non riflette alcun segnale reale, ma solo la posizione alfabetica casuale dell'ID dell'articolo cliccato. Il numero grezzo di "CF_WINS" complessivo (45,38%) è quindi **fortemente gonfiato da questo artefatto** e non può essere usato così com'è per una decisione GO/NO-GO.

Tutte le analisi seguenti sono perciò riportate **sia sul campione intero (con l'artefatto dichiarato) sia, come misura primaria, sul sottoinsieme informativo** (`cf_score_max > 0`, cioè almeno un candidato ha una rappresentazione CF reale — 10.817 impression, 23,58% del totale).

**Contingenza TOP-1/5/10 (sottoinsieme informativo, n=10.817)**

| K | solo CF | solo CBF | entrambi | nessuno |
| :---: | ---: | ---: | ---: | ---: |
| 1 | 2,40% | 4,88% | 2,21% | 90,51% |
| 5 | 7,97% | 13,56% | 8,12% | 70,35% |
| 10 | 11,77% | 15,94% | 18,19% | 54,10% |

Per confronto, sullo stesso identico sottoinsieme il numero di riferimento ottenuto sul protocollo DEV era "solo CF hit@5≈9,27%, solo CBF≈27,08%": la proporzione relativa (CBF>CF) è confermata anche qui, ma con margini più stretti (13,56% vs 7,97% a K=5, non 27% vs 9%) — **il protocollo DEV sovrastimava il vantaggio assoluto del CBF**, verosimilmente anch'esso influenzato in parte dallo stesso tipo di effetti di copertura discussi in questo documento.

**Delta rank (sottoinsieme informativo)**: CF migliore nel 58,39% delle impression, CBF migliore nel 33,72%, pareggio 7,89%; mediana Δrank = -2 (il CF piazza il cliccato, mediamente, 2 posizioni più in alto del CBF quando ha una rappresentazione). **Questo è il dato più solido a favore di una complementarità reale**: quando il CF non è degenere, tende sistematicamente a piazzare l'item cliccato meglio del CBF, anche se in valore assoluto la copertura di questi casi è limitata (23,58% delle impression).

### 10.6 Complementarità per segmento (solo feature pre-click)

| Segmento | n | CF_WINS% | CBF_WINS% | hit@5 CF | hit@5 CBF |
| :--- | ---: | ---: | ---: | ---: | ---: |
| History corta | 15.571 | 40,81% | 29,73% | 45,46% | 48,97% |
| History media | 14.801 | 45,74% | 27,96% | 41,61% | 45,05% |
| History lunga | 15.505 | 49,63% | 26,34% | 39,74% | 42,52% |
| Candidate set prevalentemente cold | 15.171 | 30,55% | 12,90% | 66,22% | 66,19% |
| Candidate set quota warm elevata | 15.593 | 47,80% | 40,33% | 31,84% | 37,54% |
| Bassa `cbf_score_std` | 22.939 | 42,00% | 25,27% | 38,88% | 40,99% |
| Alta `cbf_score_std` | 22.938 | 48,76% | 30,76% | 45,68% | 50,07% |

(Nota: questi CF_WINS% includono ancora il campione intero, non solo l'informativo — riportati per coerenza con la segmentazione adottata; il pattern per history length conferma la sezione 4 di `ANALISI_COMPLEMENTARITA.md`: il vantaggio relativo del CF cresce con history più lunga.)

### 10.7 Nuovo dataset diagnostico CF_WINS e test predittivo minimo

Costruito **esclusivamente su VALIDATION** (mai su INTERNAL_TEST), e per il test predittivo **ristretto al sottoinsieme informativo** (cf_score_max>0), per evitare che il classificatore impari semplicemente a riconoscere l'artefatto di tie-break di sezione 10.5 invece di un segnale reale.

Split diagnostico interno, temporale: **TRAIN_DIAGNOSTIC = 12 novembre** (n=5.281), **VALIDATION_DIAGNOSTIC = 13 novembre** (n=5.536). Prevalenza CF_WINS: 61,47% (train) / 55,46% (val) — nota: nel sottoinsieme informativo il CF vince la maggioranza dei confronti diretti, coerente con la sezione 10.5.

| Modello | ROC-AUC | PR-AUC | Balanced accuracy | Precision | Recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline (majority/prevalenza) | 0,500 | 0,6147 | 0,500 | 0,000 | 0,000 |
| **Logistic Regression** | **0,5836** | 0,6194 | 0,5447 | 0,6404 | 0,2971 |
| Decision Tree (depth=3) | 0,5214 | 0,5736 | 0,5173 | 0,5917 | 0,2459 |

**Verifica di robustezza**: bootstrap non parametrico (2.000 resample, seed 42) sul ROC-AUC della Logistic Regression: media 0,5838, IC 95% = **[0,5687, 0,5985]**, interamente sopra 0,50 (0/2000 resample ≤ 0,50). Il segnale non è quindi un artefatto di un singolo giorno campionato per caso: è debole ma statisticamente riproducibile su questo split.

I coefficienti della Logistic Regression (standardizzati) indicano `cf_score_max` (+0,320, il predittore più forte), `warm_candidate_ratio` (-0,223) e `cf_gap_top1_top2` (-0,202) come i contributi principali — cioè: il CF vince più spesso quando ha un punteggio massimo alto ma un candidate set relativamente meno "denso" di articoli warm, un pattern plausibile (il CF è più utile quando ha un segnale forte e concentrato, non quando è genericamente disponibile per molti item). Il Decision Tree, molto più debole (ROC-AUC 0,52), non riesce a recuperare un pattern nonlineare comparabile con soli 3 livelli di profondità — indicazione che il segnale è prevalentemente lineare/monotono, coerente con l'analisi per decili di `ANALISI_COMPLEMENTARITA.md`.

### 10.8 Classificazione Caso 1/2/3 e decisione

Applicando il criterio pre-registrato (senza modifiche a posteriori):

- Esiste una quota non trascurabile di casi CF-only/CF-migliore: **sì** (7,97-11,77% hit@K "solo CF" nel sottoinsieme informativo; 58,39% "CF migliore in rank" nello stesso sottoinsieme).
- Un classificatore semplice individua questi casi con segnale superiore alla baseline: **sì, ma debole**: ROC-AUC 0,5836 (IC 95% interamente sopra 0,50, quindi statisticamente non-casuale), non "chiaramente superiore" nel senso di una separazione forte (soglie convenzionali: 0,5-0,6 = discriminazione debole).

**Classificazione: CASO 1 (debole/condizionato)**, non un CASO 2 puro (non è vero che "le feature pre-click non permettono affatto di identificare i casi": lo fanno, con un margine reale ma modesto sopra il caso) né un CASO 3 (la quota di complementarità non è affatto trascurabile nel sottoinsieme informativo).

### 10.9 Decisione GO/NO-GO (aggiornata)

**GO condizionato per un Adaptive Hybrid rule-based**, con le stesse tre cautele già dichiarate in sezione 8.1 (effetto modesto, poche feature realmente informative, analisi non ancora multivariata ottimizzata) più una quarta, nuova: **la complementarità reale è confinata al 23,58% delle impression in cui il CF ha una rappresentazione non degenere** — qualunque regola o gating dovrebbe quindi prima instradare le impression tra "CF potenzialmente utile" (`cf_score_max>0`) e "CF strutturalmente cieco" (`cf_score_max==0`, la maggioranza), e solo nel primo gruppo vale la pena applicare la logica fine di sezione 6 di `VALIDAZIONE_METODOLOGICA.md` (soglie su `cbf_score_std`/`cbf_score_max`). Questo è in linea con la Regola 1/2 già proposte (si applicano naturalmente solo quando serve differenziare, e nel sottoinsieme degenere il CF non ha comunque nulla da offrire indipendentemente dalla soglia).

**Non emergono elementi per passare a NRMS in questa fase**: il segnale predittivo, per quanto modesto, è statisticamente robusto (bootstrap) e non "imprevedibile" secondo il criterio NO-GO.

### 10.10 Verifica finale su INTERNAL_TEST: limitazione dichiarata esplicitamente

**INTERNAL_TEST (14 novembre) NON è stato utilizzato per scegliere feature, soglie, classificatore o architettura in questa sezione** (10.5-10.9 usano esclusivamente FIT e VALIDATION). **Va però dichiarato onestamente che INTERNAL_TEST era già stato osservato nella sezione 9** di questo stesso documento per le metriche aggregate di CF_original/CF_enriched/CBF/Baseline (sezione 9.6, tabella INTERNAL_TEST) — quindi non è più, in senso stretto, un campione "mai visto" per l'insieme del progetto, anche se non è mai stato usato per decisioni relative al Rule-Based Adaptive Hybrid o alla Gating Network specificamente. Non viene eseguita in questa fase alcuna verifica aggiuntiva del pattern di complementarità su INTERNAL_TEST, proprio per non consumare ulteriormente questo campione prima di una eventuale implementazione reale del Rule-Based Adaptive Hybrid, che dovrebbe essere il primo (e per ora unico) momento in cui INTERNAL_TEST viene riletto, una sola volta, a configurazione congelata.

### 10.11 Limitazioni metodologiche di questa sezione

1. **Bias di catalogo articoli** (sezione 10.2): tutte le metriche assolute di CBF (e la quota di copertura "warm" del CF) calcolate con `MINDsmall_dev/news.tsv` come universo sono valide per confronti *entro* lo stesso stage/giorno, ma non per confronti *tra* giorni/stage diversi, che vanno sempre accompagnati da un controllo di copertura come quello di 10.2.
2. **Artefatto di tie-break** (sezione 10.5): qualunque cifra di complementarità CF/CBF basata su rank o hit@K va sempre accompagnata dalla percentuale di impression "degeneri" (cf_score_max=0); il numero grezzo non condizionato è fuorviante per eccesso di CF_WINS.
3. **Split diagnostico a singolo giorno**: la validazione del classificatore usa un solo giorno di train (12 nov) e un solo giorno di test (13 nov) — n adeguato (>5.000) ma un solo confine temporale; una validazione più solida richiederebbe più confini (es. rolling-origin su più giorni), non eseguita qui per limiti di tempo/calcolo.
4. **INTERNAL_TEST parzialmente "consumato"** per le metriche aggregate di sezione 9, dichiarato esplicitamente in 10.10.
5. Il Decision Tree usato è deliberatamente poco profondo (max_depth=3) per mantenere una funzione diagnostica: non esclude che un modello più espressivo (fuori scope qui, sarebbe già un passo verso una rete) catturi più segnale.

### 10.12 File creati in questa sezione

- `hybrid/experiments/run_daily_causal_audit.py`, `daily_cbf_audit.csv`, `daily_stratified_comparison.csv`
- `hybrid/experiments/run_validation_complementarity_v2.py`, `v2_cf_wins_diagnostic.csv`, `v2_topk_contingency.json`, `v2_topk_contingency_corrected.json`, `v2_rank_delta.csv`, `v2_delta_rank_summary.json`, `v2_segment_complementarity.csv`, `v2_classifier_results.json`, `v2_classifier_results_informative_subset.json`

**Nessun file di produzione modificato in questa sezione** (`hybrid/hybrid_recommender.py` invariato, verificato via `git diff --stat`).

Questa raccomandazione **non implementa nulla**: propone solo l'ordine e le condizioni con cui procedere.
