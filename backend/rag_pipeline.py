import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from data_loader import load_atp_data, chunk_matches_to_text

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []
index = None
df_global = None

def build_index():
    global chunks, index, df_global
    df_global = load_atp_data()
    chunks = chunk_matches_to_text(df_global)
    print(f"Building index over {len(chunks)} match records...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print("Index built successfully.")

def is_aggregation_query(query):
    keywords = ["most wins", "most matches", "who won the most", "how many", 
                "total wins", "win count", "most victories", "best record"]
    return any(k in query.lower() for k in keywords)

def get_tournament_name(query):
    tournaments = {
        "wimbledon": "Wimbledon",
        "us open": "US Open",
        "french open": "Roland Garros",
        "roland garros": "Roland Garros",
        "australian open": "Australian Open",
    }
    for keyword, name in tournaments.items():
        if keyword in query.lower():
            return name
    return None

def analytical_query(query):
    """Handle aggregation queries directly with Pandas."""
    tourney = get_tournament_name(query)
    df = df_global.copy()
    
    if tourney:
        df = df[df["tourney_name"].str.contains(tourney, case=False, na=False)]
    
    if df.empty:
        return None, []
    
    win_counts = df["winner_name"].value_counts()
    top_player = win_counts.index[0]
    top_count = win_counts.iloc[0]
    
    summary = f"In the dataset, {top_player} won the most matches"
    if tourney:
        summary += f" at {tourney}"
    summary += f" with {top_count} wins."
    
    top_matches = df[df["winner_name"] == top_player][["tourney_name", "round", "loser_name", "score"]].to_dict("records")
    context = [f"{top_player} defeated {m['loser_name']} in the {m['round']} round with score {m['score']}" for m in top_matches]
    
    return summary, context

def retrieve(query, top_k=15):
    query_embedding = model.encode([query]).astype("float32")
    tourney = get_tournament_name(query)
    
    if tourney:
        filtered = [c for c in chunks if tourney in c]
    else:
        filtered = chunks
    
    if not filtered:
        filtered = chunks
    
    if len(filtered) <= 50:
        return filtered
    
    filtered_embeddings = model.encode(filtered, show_progress_bar=False).astype("float32")
    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)
    distances, indices = temp_index.search(query_embedding, min(top_k, len(filtered)))
    return [filtered[i] for i in indices[0]]