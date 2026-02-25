import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from data_loader import load_atp_data, chunk_matches_to_text

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []
embeddings_cache = None
bm25 = None
df_global = None

def build_index():
    global chunks, embeddings_cache, bm25, df_global
    df_global = load_atp_data()
    chunks = chunk_matches_to_text(df_global)
    print(f"Building index over {len(chunks)} match records...")

    embeddings_cache = model.encode(chunks, show_progress_bar=True)
    embeddings_cache = np.array(embeddings_cache).astype("float32")

    tokenized = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    print("Index built successfully.")

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

    query_embedding = model.encode([query]).astype("float32")

    # Dense retrieval using cached embeddings
    filtered_embeddings = embeddings_cache[filtered_indices]
    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)
    _, dense_positions = temp_index.search(query_embedding, min(top_k, len(filtered_indices)))
    dense_indices = [filtered_indices[p] for p in dense_positions[0]]

    # Sparse BM25 retrieval
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