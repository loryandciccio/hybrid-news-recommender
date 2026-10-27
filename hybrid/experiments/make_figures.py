#!/usr/bin/env python3
"""
Genera i 5 grafici per l'analisi di complementarita' (Fase 14).
Legge i CSV gia' prodotti da run_static_weight_sweep.py, run_complementarity_analysis.py
e run_segment_analysis.py; non esegue alcun calcolo nuovo e non modifica alcun algoritmo.

Uso (dalla root del progetto):
    python3 hybrid/experiments/make_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIG_DIR = "hybrid/experiments/figures"

# --- 1. AUC vs peso CF (static weight sweep) ---
summ = pd.read_csv("hybrid/experiments/static_weight_summary.csv")
b0 = summ[summ['baseline_weight']==0].sort_values('cf_weight')
plt.figure(figsize=(8,5))
plt.errorbar(b0['cf_weight'], b0['AUC_mean'], yerr=b0['AUC_std'], marker='o', capsize=3, color='#2c5f8a')
plt.xlabel("Peso Collaborative Filtering (%)")
plt.ylabel("AUC media (3 porzioni DEV)")
plt.title("AUC media in funzione del peso CF (baseline = 0)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_auc_vs_peso_cf.png", dpi=200)
plt.close()

# --- 2. AUC per modello vs history length ---
hist_tab = pd.read_csv("hybrid/experiments/results_by_history_length.csv")
order = ['0-2','3-5','6-10','11-20','>20']
hist_tab['bucket'] = pd.Categorical(hist_tab['bucket'], categories=order, ordered=True)
hist_tab = hist_tab.sort_values('bucket')
plt.figure(figsize=(8,5))
plt.plot(hist_tab['bucket'], hist_tab['baseline_pure__AUC'], marker='s', label='Baseline pura', color='#999999')
plt.plot(hist_tab['bucket'], hist_tab['cf_pure__AUC'], marker='^', label='CF puro', color='#d62728')
plt.plot(hist_tab['bucket'], hist_tab['cbf_pure__AUC'], marker='o', label='CBF puro', color='#2ca02c')
plt.plot(hist_tab['bucket'], hist_tab['hybrid_0_03_97__AUC'], marker='d', label='Hybrid 0/3/97', color='#1f77b4', linestyle='--')
plt.xlabel("Lunghezza della cronologia utente (bucket)")
plt.ylabel("AUC medio")
plt.title("AUC per modello in funzione della lunghezza della cronologia")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_auc_vs_history_length.png", dpi=200)
plt.close()

# --- 3. Delta AUC CF-CBF e Hybrid-CBF vs history length ---
plt.figure(figsize=(8,5))
plt.axhline(0, color='black', linewidth=0.8)
plt.plot(hist_tab['bucket'], hist_tab['delta_AUC_CF_CBF'], marker='^', label='ΔAUC = CF − CBF', color='#d62728')
plt.plot(hist_tab['bucket'], hist_tab['delta_AUC_Hybrid_CBF']*10, marker='d', label='ΔAUC = Hybrid − CBF (×10, per leggibilità)', color='#1f77b4')
plt.xlabel("Lunghezza della cronologia utente (bucket)")
plt.ylabel("Differenza di AUC medio")
plt.title("Differenza di AUC (CF/Hybrid vs CBF) per lunghezza della cronologia")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_delta_auc_vs_history_length.png", dpi=200)
plt.close()

# --- 4. nDCG@5 per modello vs history length ---
plt.figure(figsize=(8,5))
plt.plot(hist_tab['bucket'], hist_tab['baseline_pure__nDCG5'], marker='s', label='Baseline pura', color='#999999')
plt.plot(hist_tab['bucket'], hist_tab['cf_pure__nDCG5'], marker='^', label='CF puro', color='#d62728')
plt.plot(hist_tab['bucket'], hist_tab['cbf_pure__nDCG5'], marker='o', label='CBF puro', color='#2ca02c')
plt.plot(hist_tab['bucket'], hist_tab['hybrid_0_03_97__nDCG5'], marker='d', label='Hybrid 0/3/97', color='#1f77b4', linestyle='--')
plt.xlabel("Lunghezza della cronologia utente (bucket)")
plt.ylabel("nDCG@5 medio")
plt.title("nDCG@5 per modello in funzione della lunghezza della cronologia")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_ndcg5_vs_history_length.png", dpi=200)
plt.close()

# --- 5. Percentuale "solo CF"/"solo CBF" per history bucket ---
comp = pd.read_csv("hybrid/experiments/complementarity_by_history_length.csv")
comp['bucket'] = pd.Categorical(comp['bucket'], categories=order, ordered=True)
comp = comp.sort_values('bucket')
x = np.arange(len(comp))
width = 0.35
plt.figure(figsize=(8,5))
plt.bar(x - width/2, comp['only_CF_%'], width, label='Solo CF corretto (hit@5)', color='#d62728')
plt.bar(x + width/2, comp['only_CBF_%'], width, label='Solo CBF corretto (hit@5)', color='#2ca02c')
plt.xticks(x, comp['bucket'])
plt.xlabel("Lunghezza della cronologia utente (bucket)")
plt.ylabel("Percentuale di impression (%)")
plt.title("Complementarità CF/CBF (hit@5) per lunghezza della cronologia")
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_complementarita_per_history_length.png", dpi=200)
plt.close()

print("5 grafici salvati in", FIG_DIR)
