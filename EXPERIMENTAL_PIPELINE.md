# Guida alla pipeline sperimentale

Questo documento mappa l'intero percorso sperimentale del progetto — dalla prima
analisi esplorativa fino al confronto finale NRMS/CBF — sugli script del
repository e sui capitoli della relazione finale
[`Progettazione__analisi_e_validazione_sperimentaledi_sistemi_di_raccomandazione_per_notizie.pdf`](Progettazione__analisi_e_validazione_sperimentaledi_sistemi_di_raccomandazione_per_notizie.pdf),
che è la fonte canonica del progetto. Ogni fase riporta la domanda a cui
risponde, i file di input, gli script principali, l'output prodotto, il
risultato e la domanda che apre la fase successiva. L'obiettivo è permettere
di seguire il progetto dall'inizio alla fine senza dover ricostruire la
cronologia dei file.

Nessun algoritmo, dataset o risultato è stato modificato per produrre questa
guida: è un documento di sola documentazione, redatto leggendo il codice e i
risultati esistenti.

---

## Fase 01 — Analisi esplorativa dei dati

- **Domanda**: quali caratteristiche strutturali del dataset MIND-small
  motivano le scelte progettuali della pipeline?
- **Input**: `dataset/MINDsmall_train/behaviors.tsv`, `news.tsv`;
  `dataset/MINDsmall_dev/behaviors.tsv`, `news.tsv`
- **Codice**: `data_analysis.ipynb`
- **Output**: grafici in `images/`
- **Risultato**: sparsità estrema (densità ≈ 0.000091), CTR 4.04%, vita
  mediana di un articolo ≈ 5.8h, cold-start utente/articolo, pattern
  circadiani.
- **Collegamento alla relazione**: Cap. 2 (News recommendation e dataset
  MIND), Cap. 3 (Analisi esplorativa dei dati)
- **Fase successiva**: quali famiglie di recommender funzionano meglio in
  queste condizioni?

## Fase 02 — Modelli di riferimento (Popularity, ItemKNN, ALS)

- **Domanda**: come si comportano popolarità e Collaborative Filtering su
  MIND-small?
- **Input**: `dataset/MINDsmall_train/behaviors.tsv` (history e click),
  `dataset/MINDsmall_dev/behaviors.tsv` (candidate set di valutazione)
- **Codice**: `baseline_model/baseline.ipynb`;
  `cf_itemknn/itemknn_recency_pure.py`, `itemknn_recency_fallback.py`,
  `itemknn_pure.ipynb`, `itemknn_fallback.ipynb`;
  `cf_latent_item/create_interaction_matrix.py`, `latent_item_cf.py`,
  `run_latent_item_cf.py`
- **Output**: `baseline_model/baseline_prediction.txt`
- **Risultato**: AUC 0.50–0.53, vicino al caso casuale.
- **Collegamento alla relazione**: Cap. 5, §§ Baseline di popolarità,
  Collaborative Filtering ItemKNN, Collaborative Filtering a fattori latenti
- **Fase successiva**: un segnale semantico del contenuto fa meglio?

## Fase 03 — Content-Based Filtering

- **Domanda**: la similarità semantica del testo supera i segnali
  comportamentali?
- **Input**: `dataset/MINDsmall_train/news.tsv`, `dataset/MINDsmall_dev/news.tsv`
  (titolo, abstract, categoria)
- **Codice**: `content_based_text/SD_textBasedSimilarity.ipynb` (SBERT puro);
  `content_based_clicked/Hybrid_Content_Profile.ipynb` (SBERT + metadata +
  recency)
- **Output**: `content_based_text/embeddings/news_embeddings.npy`,
  `content_based_clicked/Embeddings/newsembeddings.npy`
- **Risultato**: AUC 0.6418 / 0.6307 — il salto prestazionale più rilevante tra
  i modelli di riferimento.
- **Collegamento alla relazione**: Cap. 5, §§ Content-Based Filtering
  (similarità testuale pura, ibrido testo+metadata+recency), § Prima
  valutazione comparativa dei modelli di riferimento
- **Fase successiva**: popolarità e CF possono ancora dare informazione
  complementare, se combinati con il CBF?

## Fase 04 — Hybrid Recommender (architettura)

- **Domanda**: come combinare i tre segnali in un unico ranking?
- **Input**: score prodotti dai modelli delle Fasi 02–03 sullo stesso
  candidate set DEV
- **Codice**: `hybrid/hybrid_recommender.py`
- **Documentazione tecnica**: `hybrid/solution_hybrid.md`
- **Risultato**: architettura a normalizzazione min-max per-candidate-set e
  combinazione lineare pesata.
- **Collegamento alla relazione**: Cap. 6, § Architettura, § Normalizzazione
  e combinazione
- **Fase successiva**: quali pesi massimizzano il ranking?

## Fase 05 — Ricerca iniziale dei pesi

- **Domanda**: quale combinazione POP/CF/CBF massimizza il ranking?
- **Input**: `dataset/MINDsmall_dev/behaviors.tsv` (prime 500, poi prime 2000
  impression)
- **Codice**: `hybrid/grid_search_weights.py`, `hybrid/parameter_search.py`
- **Output**: `hybrid/grid_search_results.csv`, `grid_search_2000_results.csv`,
  `tuning_results.csv`
- **Documentazione**: `hybrid/weight_optimization.md`
- **Risultato**: configurazione selezionata `0% / 3% / 97%` (Pop/CF/CBF), AUC
  0.6624 sul campione di tuning.
- **Collegamento alla relazione**: Cap. 6, §§ Ricerca iniziale dei pesi,
  Risultato della configurazione selezionata, Confronto controllato sulle
  stesse impression
- **Fase successiva**: quel 3% di CF produce un beneficio robusto, o è
  trascurabile?

## Fase 06 — Protocollo sperimentale e strategia di validazione

- **Domanda**: qual è il ruolo esatto di TRAIN e DEV nella pipeline? Il
  riutilizzo ripetuto del DEV per la selezione dei pesi introduce un bias di
  model selection? Cosa rappresentano esattamente le 500/2000/70.938
  impression citate nei capitoli precedenti e successivi, e a quali
  condizioni due risultati sono davvero comparabili?
- **Input**: codice sorgente di `hybrid/hybrid_recommender.py`; le tre
  porzioni da 2000 impression del DEV (offset 0, 2000, 40000)
- **Codice**: nessuno script diagnostico dedicato — verifica diretta di
  `hybrid/hybrid_recommender.py` e del parametro `offset` nello script di
  valutazione
- **Documentazione**: `hybrid/experiments/VALIDAZIONE_METODOLOGICA.md`
- **Risultato**: nessuna evidenza di label leakage diretto; il DEV, riusato
  per tuning e per il report della configurazione selezionata, non
  costituisce una stima finale indipendente (model-selection bias, distinto
  dal leakage); tre porzioni del DEV confermano lo stesso pattern
  qualitativo, pur non bastando per un test statistico formale di stabilità;
  definiti i criteri di comparabilità tra risultati e le diverse numerosità
  campionarie usate nel resto della relazione (500/2000 per il tuning, tre
  porzioni da 2000 per il controllo di robustezza, 70.938 per i confronti
  statistici su campione ampio).
- **Collegamento alla relazione**: Cap. 7, "Protocollo sperimentale e
  strategia di validazione" (7 sottosezioni: separazione training/valutazione,
  ruolo del DEV nella model selection, valutazione su porzioni differenti del
  DEV, comparabilità dei risultati, dimensione del campione di valutazione,
  protocollo per gli esperimenti successivi, principio metodologico adottato)
- **Fase successiva**: il risultato regge esplorando sistematicamente l'intero
  spazio dei pesi, con un test statistico rigoroso?

## Fase 07 — Sensibilità ai pesi (weight sweep + bootstrap)

- **Domanda**: il risultato regge esplorando sistematicamente lo spazio dei
  pesi, con un test statistico rigoroso?
- **Input**: le tre porzioni da 2000 impression del DEV (Fase 06); l'intero
  campione DEV con history non vuota (n=70.938 delle 73.152 impression)
- **Codice**: `hybrid/experiments/run_static_weight_sweep.py`,
  `hybrid/experiments/run_bootstrap_analysis.py`
- **Output**: `static_weight_sweep.csv/json`, `static_weight_summary.csv`,
  `bootstrap_results.json`
- **Risultato**: 16 configurazioni × 3 porzioni DEV (48 valutazioni); il CBF
  puro (0/0/100) ottiene l'AUC medio più alto; paired bootstrap (n=70.938)
  conferma l'assenza di un vantaggio pratico del 3% di CF (differenza
  statisticamente non nulla ma di ampiezza irrilevante, 3.75×10⁻⁵ AUC).
- **Collegamento alla relazione**: Cap. 8 (Ricerca estesa dei pesi e paired
  bootstrap)
- **Fase successiva**: se un peso fisso non aiuta, il problema è nella
  combinazione lineare o nel segnale CF stesso?

## Fase 08 — Analisi del segnale collaborativo

- **Domanda**: perché il CF è strutturalmente debole in questo dataset?
- **Input**: `dataset/MINDsmall_train/behaviors.tsv` (campo `history` e click
  ufficiali delle impression)
- **Codice**: `hybrid/experiments/run_coldstart_audit.py`,
  `hybrid/experiments/run_cf_training_source_audit.py`,
  `hybrid/experiments/run_segment_analysis.py`,
  `hybrid/experiments/temporal_split_definition.py`
- **Output**: `coldstart_category_breakdown.csv`, `cf_training_source_audit.json`,
  `results_by_history_length.csv`, `history_length_distribution.csv`
- **Risultato**: 5 definizioni di cold-start non equivalenti (24%–94.6%);
  l'85.7% dei click ufficiali del TRAIN non compare in alcuna `history`; il
  CF ignora la maggior parte del segnale comportamentale disponibile.
- **Collegamento alla relazione**: Cap. 9, §§ Analisi per segmento, Audit del
  cold-start, Audit delle sorgenti del Collaborative Filtering
- **Fase successiva**: arricchire le fonti di training risolve la debolezza?

## Fase 09 — Arricchimento del segnale collaborativo (CF enriched)

- **Domanda**: includere anche i click ufficiali (oltre alla `history`)
  migliora il ranking del CF?
- **Input**: split temporale interno FIT/VALIDATION (Fase 11), history e
  click ufficiali del periodo FIT
- **Codice**: `hybrid/experiments/run_cf_enriched_experiment.py`
- **Output**: `cf_enriched_validation_results.json`,
  `cf_enriched_internal_test_results.json`
- **Risultato**: la copertura migliora (+1–3%), ma l'AUC resta vicino al caso
  casuale (~0.50): il turnover editoriale, non la scelta della fonte, è il
  vincolo dominante.
- **Collegamento alla relazione**: Cap. 9, § CF_original contro CF_enriched
- **Fase successiva**: il CF, pur debole in media, è complementare al CBF in
  specifiche impression?

## Fase 10 — Analisi di complementarità

- **Domanda**: il CF piazza meglio il candidato cliccato in un sottoinsieme
  specifico di impression?
- **Input**: score CF e CBF sull'intero DEV (prima analisi) e su VALIDATION
  del protocollo temporale interno (analisi corretta)
- **Codice**: `hybrid/experiments/run_complementarity_analysis.py`,
  `hybrid/experiments/run_cf_wins_diagnostic.py`,
  `hybrid/experiments/run_validation_complementarity_v2.py`
- **Output**: `complementarity_matrix.json`, `v2_topk_contingency*.json`,
  `v2_classifier_results*.json`, `cf_wins_diagnostic.csv`
- **Documentazione**: `hybrid/experiments/ANALISI_COMPLEMENTARITA.md`
- **Risultato**: individuato un artefatto di tie-breaking sui punteggi CF
  degeneri (76% delle impression); definita la condizione CF_AVAILABLE; sul
  sottoinsieme informativo (23.58%) il CF vince nel 58.39% dei casi —
  complementarità reale ma limitata e condizionale.
- **Collegamento alla relazione**: Cap. 9, §§ Complementarità CF/CBF (prima
  analisi e artefatto di tie-breaking), Complementarità corretta,
  classificatore diagnostico per CF_WINS
- **Fase successiva**: questa complementarità regge anche verificando
  l'omogeneità del protocollo temporale e del catalogo?

## Fase 11 — Verifica del protocollo temporale e del catalogo

- **Domanda**: il DEV, ormai esaurito dal riutilizzo ripetuto nelle fasi
  precedenti, va sostituito da uno split indipendente? La copertura del
  catalogo è omogenea tra periodi?
- **Input**: `dataset/MINDsmall_train/behaviors.tsv` (timestamp),
  `dataset/MINDsmall_train/news.tsv` e `dataset/MINDsmall_dev/news.tsv`
  (cataloghi a confronto)
- **Codice**: `hybrid/experiments/temporal_split_definition.py`,
  `hybrid/experiments/run_daily_causal_audit.py`
- **Output**: `daily_cbf_audit.csv`, `daily_stratified_comparison.csv`
- **Risultato**: split interno FIT/VALIDATION/INTERNAL_TEST; individuato un
  artefatto di catalog coverage (38%–86% a seconda del giorno) che aveva
  simulato un falso concept drift comportamentale.
- **Collegamento alla relazione**: Cap. 10 (Verifica del protocollo temporale
  e della copertura del catalogo)
- **Fase successiva**: una strategia di combinazione dipendente
  dall'impression può sfruttare la complementarità condizionale osservata?

## Fase 12 — Adaptive Hybrid (rule-based)

- **Domanda**: un peso CF condizionato allo stato dell'impression, anziché
  fisso, produce un beneficio?
- **Input**: feature diagnostiche (dispersione/massimo score CBF, CF_AVAILABLE)
  calcolate su GATING_TRAIN (12 novembre), valutate su GATING_VALIDATION
  (13 novembre) e INTERNAL_TEST (14 novembre)
- **Codice**: `hybrid/experiments/adaptive_hybrid/rule_based_hybrid.py`,
  `build_features.py`, `common.py`, `evaluate_adaptive.py`
- **Output**: `results/rule_based_selection.json`,
  `results/final_comparison_validation.json`,
  `results/final_comparison_internal_test.json`
- **Documentazione**: `hybrid/experiments/adaptive_hybrid/ADAPTIVE_HYBRID_REPORT.md`
- **Risultato**: sei regole testate, nessun beneficio misurabile rispetto al
  CBF puro.
- **Collegamento alla relazione**: Cap. 11 (Adaptive Hybrid)
- **Fase successiva**: una rete che apprenda la funzione di peso, invece di
  regole discrete, recupera il margine mancante?

## Fase 13 — Neural Gating

- **Domanda**: una rete di gating neurale, addestrata sui dati, fa meglio delle
  regole?
- **Input**: le stesse 14 feature diagnostiche della Fase 12, su
  GATING_TRAIN/GATING_VALIDATION/INTERNAL_TEST
- **Codice**: `hybrid/experiments/adaptive_hybrid/gating_network.py`,
  `train_gating.py`
- **Output**: `results/gating_best_arch.json`,
  `results/gating_model_best.pt`, `results/gating_training_log.json`
- **Risultato**: la rete (138 parametri) apprende una politica interpretabile
  (riduce il peso CF quando il CBF è più "sicuro"), ma sovrappesa un segnale
  CF debole e peggiora le metriche in modo statisticamente robusto.
- **Collegamento alla relazione**: Cap. 12 (Neural Gating)
- **Fase successiva**: un modello neurale end-to-end, che apprenda
  direttamente da testo e cronologia, supera il CBF semantico?

## Fase 14 — NRMS

- **Domanda**: un'architettura neurale specifica per news recommendation
  (Neural News Recommendation with Multi-Head Self-Attention) supera SBERT?
- **Input**: `dataset/MINDsmall_train/news.tsv` (titoli, catalogo TRAIN
  completo), split FIT/VALIDATION per training ed early stopping
- **Codice**: `hybrid/experiments/nrms/model.py`, `data.py`, `train.py`,
  `evaluate.py`, `tokenizer.py`, `run_nrms_experiment.py`
- **Output**: `results/training_log.json`,
  `results/nrms_native_validation_metrics.json`; checkpoint
  `checkpoints/nrms_best.pt`
- **Documentazione**: `hybrid/experiments/nrms/NRMS_REPORT.md`
- **Risultato**: ~2.6M parametri, addestrato da zero (~93 min CPU), AUC nativo
  0.5861 su VALIDATION (non ancora comparabile col CBF: cataloghi diversi).
- **Collegamento alla relazione**: Cap. 13 (NRMS)
- **Fase successiva**: su un insieme di valutazione comune, NRMS supera il
  CBF semantico?

## Fase 15 — Confronto controllato NRMS vs CBF

- **Domanda**: su un insieme di valutazione comune (stesso catalogo, stesse
  impression), NRMS supera il CBF semantico?
- **Input**: score NRMS (Fase 14) e score CBF, filtrati al common evaluation
  set su VALIDATION e INTERNAL_TEST
- **Codice**: `hybrid/experiments/nrms/analyze_comparison.py`
- **Output**: `results/comparison_validation.json`,
  `comparison_internal_test.json`, `analysis_validation.json`,
  `efficiency.json`
- **Risultato**: il CBF supera NRMS su ogni metrica (2–5 punti AUC),
  differenza sia statisticamente sia praticamente significativa (bootstrap).
- **Collegamento alla relazione**: Cap. 14 (Confronto controllato CBF contro
  NRMS)
- **Fase successiva**: quale lezione metodologica complessiva emerge
  dall'intero percorso, e quali sono i suoi limiti?

## Fase 16 — Discussione, limiti e conclusioni

- **Input**: risultati aggregati di tutte le fasi precedenti
- **Codice**: nessuno — sintesi dei risultati precedenti
- **Risultato**: il contenuto semantico (CBF/SBERT) resta il segnale più
  robusto nel protocollo utilizzato; nessun tentativo successivo (CF
  arricchito, Adaptive Hybrid, Neural Gating, NRMS) produce un miglioramento
  misurabile; la validità del protocollo (omogeneità dei campioni, isolamento
  degli artefatti, distinzione statistica/pratica) è tanto importante quanto
  la metrica finale.
- **Collegamento alla relazione**: Capp. 15–17 (Discussione complessiva,
  Limiti metodologici, Conclusioni); Appendice A (Riproducibilità)

---

## Mappa capitolo della relazione ↔ fase ↔ file principali

| Capitolo relazione | Fase pipeline | File/script principali |
|---|---|---|
| Cap. 2–3 Dataset, EDA | Fase 01 | `data_analysis.ipynb` |
| Cap. 5 Modelli di riferimento | Fasi 02–03 | `baseline_model/baseline.ipynb`, `cf_itemknn/itemknn_recency_*.py`, `cf_latent_item/latent_item_cf.py`, `content_based_text/SD_textBasedSimilarity.ipynb`, `content_based_clicked/Hybrid_Content_Profile.ipynb` |
| Cap. 6 Hybrid Recommender | Fasi 04–05 | `hybrid/hybrid_recommender.py`, `hybrid/grid_search_weights.py`, `hybrid/parameter_search.py` |
| Cap. 7 Protocollo sperimentale e strategia di validazione | Fase 06 | `hybrid/hybrid_recommender.py` (parametro `offset`), `hybrid/experiments/VALIDAZIONE_METODOLOGICA.md` |
| Cap. 8 Ricerca estesa dei pesi e paired bootstrap | Fase 07 | `hybrid/experiments/run_static_weight_sweep.py`, `run_bootstrap_analysis.py` |
| Cap. 9 Analisi del segnale collaborativo | Fasi 08–10 | `run_coldstart_audit.py`, `run_cf_training_source_audit.py`, `run_cf_enriched_experiment.py`, `run_complementarity_analysis.py`, `run_validation_complementarity_v2.py` |
| Cap. 10 Verifica del protocollo temporale e della copertura del catalogo | Fase 11 | `temporal_split_definition.py`, `run_daily_causal_audit.py` |
| Cap. 11 Adaptive Hybrid | Fase 12 | `adaptive_hybrid/rule_based_hybrid.py` |
| Cap. 12 Neural Gating | Fase 13 | `adaptive_hybrid/gating_network.py`, `train_gating.py` |
| Cap. 13 NRMS | Fase 14 | `nrms/model.py`, `nrms/train.py`, `nrms/run_nrms_experiment.py` |
| Cap. 14 Confronto controllato CBF contro NRMS | Fase 15 | `nrms/analyze_comparison.py` |
| Capp. 15–17 + Appendice A: Discussione, limiti, conclusioni, riproducibilità | Fase 16 | — |

## Note

- La relazione finale, in formato PDF, è
  [`Progettazione__analisi_e_validazione_sperimentaledi_sistemi_di_raccomandazione_per_notizie.pdf`](Progettazione__analisi_e_validazione_sperimentaledi_sistemi_di_raccomandazione_per_notizie.pdf)
  nella root del repository — è la fonte canonica per capitoli, tabelle e
  formule citate in questa guida. I sorgenti LaTeX in
  [`relazione/`](relazione/) sono allineati a questo PDF: una compilazione
  pulita (`pdflatex` → `bibtex` → `pdflatex` ×2) riproduce lo stesso
  contenuto, stessa numerazione di capitoli e tabelle.
- `hybrid/experiments/VALIDAZIONE_METODOLOGICA.md`,
  `hybrid/experiments/ANALISI_COMPLEMENTARITA.md`,
  `hybrid/experiments/adaptive_hybrid/ADAPTIVE_HYBRID_REPORT.md` e
  `hybrid/experiments/nrms/NRMS_REPORT.md` sono i report tecnici dettagliati
  delle singole fasi diagnostiche: questa guida ne è una sintesi orientata
  alla navigazione, non un sostituto.
- `PRESENTAZIONE_PROGETTO.md` resta la guida operativa per l'esposizione
  orale (pitch, domande d'esame); `materiali_esame/` contiene la
  presentazione PowerPoint e il PDF di studio.
