# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any, Dict
import os
import PyPDF2
import logging
import time
from .storage import CVStorage
from qdrant_client import QdrantClient
try:
    # Relative import since we are inside the src package
    from .agentGraph import run_agent, get_bert_embeddings, run_agent_email, run_agent_schedule
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

# Simple timing middleware to surface backend processing time per request
@app.middleware("http")
async def timing_middleware(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dur_ms = (time.perf_counter() - t0) * 1000
    try:
        response.headers["X-Process-Time-ms"] = f"{dur_ms:.1f}"
    except Exception:
        pass
    print(f"[TIMING] {request.method} {request.url.path} -> {dur_ms:.1f} ms")
    return response

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

class EmailRequest(BaseModel):
    job_description: str
    top_k: int = 5
    subject: str | None = None
    role: str | None = None
    reply_to: str | None = None  # optional Reply-To header

# --- Chat models and agent wiring ---
try:
    from .llm_agent import RecruiterAgent, parse_requested_top_k
except ImportError:
    from llm_agent import RecruiterAgent, parse_requested_top_k

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

agent = RecruiterAgent()
# --- end chat wiring ---

# --- email sender wiring ---
try:
    from .mailer import send_email, render_invite_subject, render_invite_body
except ImportError:
    from mailer import send_email, render_invite_subject, render_invite_body

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

        # Attempt to extract explicit candidate count from message
        requested_k = None
        try:
            requested_k = parse_requested_top_k(user_msg)
        except Exception:
            requested_k = None

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
            # Use explicitly requested top_k if present else infer
            k = requested_k or agent.infer_top_k_from_text(user_msg)
            reply = agent.handle_job_description(job_desc, top_k=k)
            return {"reply": reply}

        # No file: normal chat flow (intent detection + RAG if needed), with internal top_k inference
        reply = agent.handle_message(user_msg, forced_top_k=requested_k)
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
@app.post("/email")
def email():
    summary = run_agent_email()
    return {"summary": summary}


@app.post("/send_emails")
def send_emails(req: EmailRequest):
    """Run candidate search then send (or dry-run) emails to top matches."""
    try:
        # Run retrieval (prefer with_explain if available)
        try:
            results: List[Dict[str, Any]] = run_agent(
                req.job_description, top_k=req.top_k, with_explain=True
            )
        except TypeError:
            results = run_agent(req.job_description, top_k=req.top_k)

        storage = CVStorage()
        subject = req.subject or render_invite_subject(req.role)
        sent = 0
        attempted = 0
        failures: list[dict] = []

        for r in results[: req.top_k]:
            cid = r.get("candidate_id")
            if cid is None:
                continue
            email_addr = r.get("email") or storage.get_email_by_candidate_id(cid)
            if not email_addr:
                failures.append({"candidate_id": cid, "reason": "no email"})
                continue
            body = render_invite_body(r.get("name"), req.job_description, req.reply_to)
            attempted += 1
            try:
                if send_email(email_addr, subject, body, reply_to=req.reply_to):
                    sent += 1
                else:
                    failures.append({"candidate_id": cid, "reason": "send failed"})
            except Exception as e:
                failures.append({"candidate_id": cid, "reason": str(e)})

        return {
            "attempted": attempted,
            "sent": sent,
            "failures": failures,
            "from_email": os.getenv("FROM_EMAIL", os.getenv("SMTP_USER", "")),
            "dry_run": os.getenv("EMAIL_DRY_RUN", "1") == "1",
        }
    except Exception as e:
        logging.exception("Send emails error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/schedule_interviews")
def schedule_interviews():
    """Generate calendar appointments for last results and email candidates with details."""
    try:
        summary = run_agent_schedule()
        return {"summary": summary}
    except Exception as e:
        logging.exception("Schedule interviews error")
        raise HTTPException(status_code=500, detail=str(e))