import json
import os
import pickle

import numpy as np
from data_loader import load_atp_data, chunk_matches_to_text

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")

model = None
chunks = []
embeddings_cache = None
bm25 = None
df_global = None
ready = False

def _get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def _index_files_exist():
    files = ["chunks.json", "embeddings.npy", "bm25.pkl"]
    return all(os.path.exists(os.path.join(INDEX_DIR, f)) for f in files)

def build_index():
    global chunks, embeddings_cache, bm25, df_global, ready
    df_global = load_atp_data()

    if _index_files_exist():
        print("Loading pre-built index from disk...")
        with open(os.path.join(INDEX_DIR, "chunks.json")) as f:
            chunks = json.load(f)
        embeddings_cache = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
        with open(os.path.join(INDEX_DIR, "bm25.pkl"), "rb") as f:
            bm25 = pickle.load(f)
        ready = True
        print(f"Loaded index with {len(chunks)} match records.")
        return

    from rank_bm25 import BM25Okapi

    print("No pre-built index found. Building from scratch...")
    chunks = chunk_matches_to_text(df_global)
    print(f"Encoding {len(chunks)} match records...")

    embeddings_cache = _get_model().encode(chunks, show_progress_bar=True)
    embeddings_cache = np.array(embeddings_cache).astype("float32")

    tokenized = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    save_index()
    ready = True
    print("Index built and saved.")

def save_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w") as f:
        json.dump(chunks, f)
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings_cache)
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    print(f"Index saved to {INDEX_DIR}")

def reciprocal_rank_fusion(dense_indices, sparse_indices, k=60):
    """Combine dense and sparse results using RRF scoring."""
    scores = {}
    for rank, idx in enumerate(dense_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(sparse_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

def is_aggregation_query(query):
    keywords = ["most wins", "most matches", "who won the most", "how many",
                "total wins", "win count", "most victories", "best record"]
    return any(k in query.lower() for k in keywords)

TOURNAMENT_MAP = {
    "acapulco": "Acapulco",
    "australian open": "Australian Open",
    "roland garros": "Roland Garros",
    "french open": "Roland Garros",
    "wimbledon": "Wimbledon",
    "us open": "Us Open",
    "indian wells": "Indian Wells Masters",
    "miami": "Miami Masters",
    "monte carlo": "Monte Carlo Masters",
    "madrid": "Madrid Masters",
    "rome": "Rome Masters",
    "canada": "Canada Masters",
    "cincinnati": "Cincinnati Masters",
    "shanghai": "Shanghai Masters",
    "paris": "Paris Masters",
    "toronto": "Canada Masters",
    "halle": "Halle",
    "queens": "Queen's Club",
    "barcelona": "Barcelona",
    "hamburg": "Hamburg",
    "vienna": "Vienna",
    "basel": "Basel",
    "tokyo": "Tokyo",
}

def get_tournament_name(query):
    query_lower = query.lower()
    for keyword, name in TOURNAMENT_MAP.items():
        if keyword in query_lower:
            return name
    return None

def analytical_query(query):
    tourney = get_tournament_name(query)
    df = df_global.copy()
    if tourney:
        df = df[df["tourney_name"].str.contains(tourney, case=False, na=False)]
    if df.empty:
        return None, []
    win_counts = df["winner_name"].value_counts()
    top_player = win_counts.index[0]
    top_count = win_counts.iloc[0]
    summary = f"{top_player} won the most matches"
    if tourney:
        summary += f" at {tourney}"
    summary += f" with {top_count} wins."
    top_matches = df[df["winner_name"] == top_player][["tourney_name", "round", "loser_name", "score"]].to_dict("records")
    context = [f"{top_player} defeated {m['loser_name']} in the {m['round']} with score {m['score']}" for m in top_matches]
    return summary, context

def _bm25_only(filtered_indices, query, top_k):
    """Fallback retrieval using only BM25 (no dense embeddings needed)."""
    from rank_bm25 import BM25Okapi

    tokenized_query = query.lower().split()
    if len(filtered_indices) < len(chunks):
        filtered_tokenized = [chunks[i].lower().split() for i in filtered_indices]
        filtered_bm25 = BM25Okapi(filtered_tokenized)
        scores = filtered_bm25.get_scores(tokenized_query)
    else:
        scores = bm25.get_scores(tokenized_query)
    top = np.argsort(scores)[::-1][:top_k]
    return [chunks[filtered_indices[p]] for p in top]

def retrieve(query, top_k=15):
    tourney = get_tournament_name(query)

    if tourney:
        filtered_indices = [i for i, c in enumerate(chunks) if tourney in c]
    else:
        filtered_indices = list(range(len(chunks)))

    if not filtered_indices:
        filtered_indices = list(range(len(chunks)))

    if len(filtered_indices) <= 50:
        return [chunks[i] for i in filtered_indices]

    # Try hybrid retrieval (dense + sparse), fall back to BM25-only
    try:
        import faiss
        from rank_bm25 import BM25Okapi

        query_embedding = _get_model().encode([query]).astype("float32")

        filtered_embeddings = embeddings_cache[filtered_indices]
        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)
        _, dense_positions = temp_index.search(query_embedding, min(top_k, len(filtered_indices)))
        dense_indices = [filtered_indices[p] for p in dense_positions[0]]

        tokenized_query = query.lower().split()
        if tourney:
            filtered_tokenized = [chunks[i].lower().split() for i in filtered_indices]
            filtered_bm25 = BM25Okapi(filtered_tokenized)
            sparse_scores = filtered_bm25.get_scores(tokenized_query)
        else:
            sparse_scores = bm25.get_scores(tokenized_query)
        sparse_top = np.argsort(sparse_scores)[::-1][:top_k]
        sparse_indices = [filtered_indices[p] for p in sparse_top]

        combined = reciprocal_rank_fusion(dense_indices, sparse_indices)
        return [chunks[i] for i in combined[:top_k]]
    except Exception:
        return _bm25_only(filtered_indices, query, top_k)