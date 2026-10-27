import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import save_npz
from pathlib import Path

# I percorsi sono risolti relativamente a questo file, in modo che tutti gli
# artefatti prodotti (matrice + mapping degli ID) restino dentro cf_latent_item/.
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset" / "MINDsmall_train"

# Carica i dati di comportamento (behaviors) dalla cartella dataset condivisa nella root del progetto
behaviors_path = DATASET_DIR / "behaviors.tsv"
behaviors_df = pd.read_csv(
    behaviors_path,
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"],
)

# Scarta le righe senza cronologia (history)
behaviors_df = behaviors_df.dropna(subset=['history'])

# Costruisce la lista delle interazioni utente-articolo a partire dalla colonna history.
# Nota: questo loop riga per riga è accettabile alla scala del TRAIN di MIND-small
# (~157k righe), ma sarebbe vettorizzabile con uno split+explode di pandas per dataset più grandi.
interactions = []
for index, row in behaviors_df.iterrows():
    user_id = row['user_id']
    history = row['history'].split()
    for news_id in history:
        interactions.append((user_id, news_id))

# Costruisce un DataFrame dalla lista di interazioni
interactions_df = pd.DataFrame(interactions, columns=['user_id', 'news_id'])

# Crea una mappatura da user_id/news_id (stringhe) a indici interi
user_ids = interactions_df['user_id'].astype('category').cat
news_ids = interactions_df['news_id'].astype('category').cat

# Ottiene i codici interi per utenti e articoli
row = user_ids.codes
col = news_ids.codes
data = np.ones(len(row))

# Costruisce la matrice sparsa utente-item (implicita, binaria: 1 = almeno un click)
sparse_interaction_matrix = coo_matrix((data, (row, col)), shape=(len(user_ids.categories), len(news_ids.categories)))

# Mostra shape e un riepilogo della matrice di interazione
print("Shape della matrice sparsa utente-item:", sparse_interaction_matrix.shape)
print("\nNumero di elementi non nulli:", sparse_interaction_matrix.nnz)

# Salva la matrice sparsa e i mapping degli ID dentro cf_latent_item/
matrix_path = BASE_DIR / "sparse_user_item_interaction.npz"
user_ids_path = BASE_DIR / "user_ids.npy"
news_ids_path = BASE_DIR / "news_ids.npy"

save_npz(matrix_path, sparse_interaction_matrix)
print(f"\nMatrice sparsa utente-item salvata in {matrix_path}")

np.save(user_ids_path, user_ids.categories.to_numpy())
np.save(news_ids_path, news_ids.categories.to_numpy())
print(f"Mapping degli ID utente e articolo salvati in {user_ids_path} e {news_ids_path}")