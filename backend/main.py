from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os
from dotenv import load_dotenv
from rag_pipeline import build_index, retrieve, is_aggregation_query, analytical_query

load_dotenv()

@asynccontextmanager
async def lifespan(app):
    build_index()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://atpinsight.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
ACCESS_CODE = os.getenv("ACCESS_CODE")

class QueryRequest(BaseModel):
    question: str
    access_code: str

@app.post("/ask")
def ask(request: QueryRequest):
    if request.access_code != ACCESS_CODE:
        raise HTTPException(status_code=401, detail="Invalid access code")

    if is_aggregation_query(request.question):
        summary, context = analytical_query(request.question)
        if summary:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system="""You are ATPInsight, a tennis analytics assistant. 
Answer in ONE concise sentence. No tables, no bullet points, no explanation unless asked.""",
                messages=[{
                    "role": "user",
                    "content": f"Data summary: {summary}\n\nUser question: {request.question}"
                }]
            )
            return {"answer": message.content[0].text, "sources": context}

    context_chunks = retrieve(request.question)
    context = "\n".join(context_chunks)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
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