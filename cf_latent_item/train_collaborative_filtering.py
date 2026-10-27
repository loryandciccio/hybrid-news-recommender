"""
Script dimostrativo: allena un modello ALS di esempio sulla matrice di interazione
utente-item e stampa alcune raccomandazioni e articoli simili a titolo illustrativo.

NOTA: questo script NON fa parte della pipeline di valutazione del progetto e NON
salva i fattori latenti su disco. La pipeline reale (usata per produrre le metriche
riportate nel README/relazione) alza il modello ALS a runtime in
`cf_latent_item/run_latent_item_cf.py`, a partire dalla matrice prodotta da
`create_interaction_matrix.py`.
"""

import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from pathlib import Path

# Risolve i percorsi in modo da leggere gli artefatti dalla cartella cf_latent_item
BASE_DIR = Path(__file__).resolve().parent
matrix_path = BASE_DIR / "sparse_user_item_interaction.npz"
user_ids_path = BASE_DIR / "user_ids.npy"
news_ids_path = BASE_DIR / "news_ids.npy"

# Carica i dati prodotti da create_interaction_matrix.py
sparse_interaction_matrix_coo = sparse.load_npz(matrix_path)
user_ids = np.load(user_ids_path, allow_pickle=True)
news_ids = np.load(news_ids_path, allow_pickle=True)

# Converte in matrice CSR (più efficiente per le operazioni riga usate da ALS)
sparse_interaction_matrix_csr = sparse_interaction_matrix_coo.tocsr()

# Stampa shape e lunghezze a scopo diagnostico
print("Shape di sparse_interaction_matrix_csr:", sparse_interaction_matrix_csr.shape)
print("Numero di utenti (user_ids):", len(user_ids))
print("Numero di articoli (news_ids):", len(news_ids))

# Inizializza il modello (random_state fissato per riproducibilità)
model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20, random_state=42)

# Allena il modello.
# La libreria implicit si aspetta la trasposta della matrice utente-item per il training.
model.fit(sparse_interaction_matrix_csr.T)

# --- Genera raccomandazioni di esempio per un utente campione ---
sample_user_index = 0  # indice utente arbitrario, modificabile
sample_user_id = user_ids[sample_user_index]

print(f"Raccomandazioni per l'utente: {sample_user_id} (indice: {sample_user_index})")

# Estrae la riga dell'utente campione dalla matrice di interazione
user_items = sparse_interaction_matrix_csr[sample_user_index]

# Ottiene le raccomandazioni (esclude gli articoli già letti dall'utente)
recommendations = model.recommend(sample_user_index, user_items, N=5, filter_already_liked_items=True)

print("Raccomandazioni (indici articolo, score):", recommendations)
for item_index, score in zip(*recommendations):
    print(f"  Indice articolo: {item_index}, Score: {score:.4f}")

# --- Trova articoli simili a un articolo campione ---
sample_news_index = 10  # indice articolo arbitrario, modificabile
sample_news_id = news_ids[sample_news_index]

print(f"\nArticoli simili a: {sample_news_id} (indice: {sample_news_index})")

# Trova gli articoli più simili nello spazio latente
similar_items = model.similar_items(sample_news_index, N=5)

for item_index, score in zip(*similar_items):
    print(f"  Indice articolo: {item_index}, Score: {score:.4f}")