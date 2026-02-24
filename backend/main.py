from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os
from dotenv import load_dotenv
from rag_pipeline import build_index, retrieve

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@app.on_event("startup")
def startup_event():
    build_index()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QueryRequest):
    from rag_pipeline import is_aggregation_query, analytical_query, retrieve
    
    if is_aggregation_query(request.question):
        # Use direct Pandas query for aggregation
        summary, context = analytical_query(request.question)
        if summary:
            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=256,
                system="""You are ATPInsight, a tennis analytics assistant. 
Answer in ONE concise sentence. No tables, no bullet points, no explanation unless asked.""",
                messages=[{
                    "role": "user",
                    "content": f"Data summary: {summary}\n\nUser question: {request.question}"
                }]
            )
            return {"answer": message.content[0].text, "sources": context}
    
    # Fall back to RAG for non-aggregation questions
    context_chunks = retrieve(request.question)
    context = "\n".join(context_chunks)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        system="""You are ATPInsight, a tennis analytics assistant.
Answer concisely using only the provided context. Be direct and specific.
If the context lacks sufficient information, say so briefly.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {request.question}"
        }]
    )
    return {"answer": message.content[0].text, "sources": context_chunks}

@app.get("/health")
def health():
    return {"status": "ok"}