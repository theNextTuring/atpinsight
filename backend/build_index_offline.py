"""Run this once locally to pre-build the search index.

The generated files in data/index/ should be committed to the repo
so the server can load them instantly on startup instead of rebuilding.
"""
from rag_pipeline import build_index

if __name__ == "__main__":
    build_index()
