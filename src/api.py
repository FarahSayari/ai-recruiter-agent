# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any, Dict
import os
import logging
from .storage import CVStorage
from qdrant_client import QdrantClient
try:
    # Relative import since we are inside the src package
    from .agentGraph import run_agent, get_bert_embeddings
except ImportError:
    # Fallback: add this directory to path then plain import
    import sys
    sys.path.append(os.path.dirname(__file__))
    from agentGraph import run_agent

cv_storage = CVStorage(host="localhost", port=6333, collection_name="cvs")
qdrant = QdrantClient(url="http://localhost:6333")

app = FastAPI(title="AI Recruiter Agent API")
# Allow local Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),  # e.g. "http://localhost:8501"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRequest(BaseModel):
    job_description: str
    top_k: int = 5
    with_explain: bool = True

class CandidateResult(BaseModel):
    candidate_id: Any
    score: float
    skills: List[str] | None = None
    experience: Any | None = None
    explanation: str | None = None

class AgentResponse(BaseModel):
    results: List[CandidateResult]

# --- Chat models and agent wiring ---
try:
    from .llm_agent import RecruiterAgent
except ImportError:
    from llm_agent import RecruiterAgent

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

agent = RecruiterAgent()
# --- end chat wiring ---

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        reply = agent.handle_message(request.message)
        return {"reply": reply}
    except Exception as e:
        logging.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=AgentResponse)
def query_agent(request: JobRequest):
    try:
        # Prefer calling with with_explain; fall back if older signature
        try:
            results: List[Dict[str, Any]] = run_agent(
                request.job_description,
                top_k=request.top_k,
                with_explain=request.with_explain
            )
        except TypeError:
            results = run_agent(
                request.job_description,
                top_k=request.top_k
            )

        # Coerce any numpy types for JSON
        for r in results:
            r["score"] = float(r.get("score", 0.0))

        return {"results": results}
    except Exception as e:
        logging.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(e))
