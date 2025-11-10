# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any, Dict
import os
import PyPDF2
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

# --- new: imports for PDF upload ---
from fastapi import UploadFile, File, Form, Request
from io import BytesIO
try:
    from .preprocessing import preprocess_text
except Exception:
    # Fallback minimal cleaner
    def preprocess_text(t: str) -> str:
        return " ".join((t or "").split())

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

# --- new: simple PDF text extractor using PyPDF2 ---
def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing error: {e}")

# --- new: query from uploaded PDF (e.g., job description PDF) ---
@app.post("/query_from_pdf", response_model=AgentResponse)
def query_from_pdf(file: UploadFile = File(...), top_k: int = 5, with_explain: bool = True):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        raw = file.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file.")
        text = _extract_text_from_pdf_bytes(raw)
        if not text:
            raise HTTPException(status_code=400, detail="No text found in PDF.")
        job_desc = preprocess_text(text)

        # Prefer calling with with_explain; fall back if older signature
        try:
            results: List[Dict[str, Any]] = run_agent(job_desc=job_desc, top_k=top_k, with_explain=with_explain)
        except TypeError:
            try:
                results = run_agent(job_desc, top_k=top_k)
            except TypeError:
                results = run_agent(job_desc)

        for r in results:
            r["score"] = float(r.get("score", 0.0))
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("PDF query error")
        raise HTTPException(status_code=500, detail=str(e))

# --- updated: chat supports optional PDF and infers top_k from message ---
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    file: UploadFile | None = File(None),
    message: str | None = Form(None),
):
    try:
        # If JSON request, read message from body
        ctype = request.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                data = await request.json()
            except Exception:
                data = {}
            if isinstance(data, dict):
                message = data.get("message", message)

        user_msg = (message or "").strip()

        # If a PDF file is provided, use it as job description (optionally combined with message)
        if file is not None:
            raw = await file.read()
            if not raw:
                raise HTTPException(status_code=400, detail="Empty file.")
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are supported.")
            text = _extract_text_from_pdf_bytes(raw)
            if not text:
                raise HTTPException(status_code=400, detail="No text found in PDF.")
            job_desc = preprocess_text(text)
            # Optional free-text notes from the message
            if user_msg:
                job_desc = f"{job_desc}\n\nAdditional notes: {user_msg}"
            # Infer desired number of candidates from the message (default handled in agent)
            k = agent.infer_top_k_from_text(user_msg)
            reply = agent.handle_job_description(job_desc, top_k=k)
            return {"reply": reply}

        # No file: normal chat flow (intent detection + RAG if needed), with internal top_k inference
        reply = agent.handle_message(user_msg)
        return {"reply": reply}
    except HTTPException:
        raise
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
