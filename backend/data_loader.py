import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data/atp_matches.csv")

def load_atp_data():
    df = pd.read_csv(DATA_PATH)
    
    # Keep only the most useful columns
    cols = [
        "tourney_name", "surface", "tourney_date", "round",
        "winner_name", "loser_name", "score",
        "winner_rank", "loser_rank",
        "winner_age", "loser_age",
        "w_ace", "l_ace",
        "w_svpt", "l_svpt",
        "w_1stWon", "l_1stWon",
        "minutes"
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.dropna(subset=["winner_name", "loser_name", "score"])
    return df

def chunk_matches_to_text(df):
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"{row['winner_name']} defeated {row['loser_name']} "
            f"at {row.get('tourney_name', 'Unknown Tournament')} "
            f"on {row.get('surface', 'Unknown')} surface "
            f"in the {row.get('round', 'Unknown')} round "
            f"with score {row.get('score', 'N/A')}. "
            f"Winner rank: {row.get('winner_rank', 'N/A')}, "
            f"Loser rank: {row.get('loser_rank', 'N/A')}. "
            f"Match duration: {row.get('minutes', 'N/A')} minutes."
        )
        chunks.append(text)
    return chunks