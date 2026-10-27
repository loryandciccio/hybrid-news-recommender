#!/usr/bin/env python3
"""
Definizione CONGELATA dello split temporale interno usato dall'esperimento
CF_original vs CF_enriched (si veda run_cf_enriched_experiment.py).

Motivazione: il DEV set (2019-11-15) e' stato usato estesamente in fasi
precedenti per weight sweep, analisi per segmenti, cold-start, popolarita',
CF_WINS e bootstrap. Non puo' quindi fungere da "test set" vergine per
QUESTA decisione (CF_original vs CF_enriched). Si costruisce percio' una
partizione temporale interna al solo TRAIN (2019-11-09 - 2019-11-14),
mai usata finora per prendere alcuna decisione.

Il campo `history` di behaviors.tsv e' STATICO per utente durante tutta la
finestra TRAIN (verificato empiricamente: stesso utente, righe in giorni
diversi, stessa lunghezza di history) - precede quindi l'intera finestra di
log delle impression e non introduce leakage temporale tra le partizioni
sotto definite. Il campo `impressions` (e le sue label di click) e' invece
specifico di ciascuna riga/timestamp, e va trattato con attenzione temporale.

Split (basato sulla data, dal campo `time`):
    FIT            : 2019-11-09, 2019-11-10, 2019-11-11  (~61.417 righe)
    VALIDATION     : 2019-11-12, 2019-11-13              (~65.278 righe)
    INTERNAL_TEST  : 2019-11-14                           (~30.270 righe)

Uso previsto (si veda run_cf_enriched_experiment.py):
- CF_enriched per la fase VALIDATION e' costruito da: history (statica,
  quindi condivisa) + click positivi delle SOLE impression in FIT.
  Si valuta poi su VALIDATION (temporalmente successivo a FIT).
- Eventuali iperparametri/soglie del CF_enriched vanno scelti guardando
  SOLO le metriche su VALIDATION.
- Una volta congelata la configurazione, il CF_enriched per l'INTERNAL_TEST
  e' ricostruito da: history + click positivi di FIT UNION VALIDATION,
  e valutato UNA SOLA VOLTA su INTERNAL_TEST (2019-11-14), senza ulteriore
  tuning successivo.
- Il DEV set reale (2019-11-15) resta escluso da questo intero esperimento.
"""

FIT_DATES = ["2019-11-09", "2019-11-10", "2019-11-11"]
VALIDATION_DATES = ["2019-11-12", "2019-11-13"]
INTERNAL_TEST_DATES = ["2019-11-14"]

# Conteggi riga osservati (vedi run_cf_training_source_audit.py e verifica
# diretta di behaviors.tsv["time"].dt.date.value_counts()):
EXPECTED_ROW_COUNTS = {
    "2019-11-09": 13570,
    "2019-11-10": 15048,
    "2019-11-11": 32799,
    "2019-11-12": 33654,
    "2019-11-13": 31624,
    "2019-11-14": 30270,
}


def split_name_for_date(date_str: str) -> str:
    if date_str in FIT_DATES:
        return "FIT"
    if date_str in VALIDATION_DATES:
        return "VALIDATION"
    if date_str in INTERNAL_TEST_DATES:
        return "INTERNAL_TEST"
    raise ValueError(f"Data fuori dal range TRAIN atteso: {date_str}")
