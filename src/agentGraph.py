from typing import Dict, Any, List
import numpy as np
import re
import os

# Robust import across langgraph versions
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    try:
        from langgraph.graph.graph import Graph as StateGraph
        from langgraph.graph import END
    except Exception:
        StateGraph = None
        END = None

from storage import CVStorage

# Attempt to import real email sender utilities
try:
    from mailer import send_email, render_invite_subject, render_invite_body
except Exception:
    try:
        from .mailer import send_email, render_invite_subject, render_invite_body  # type: ignore
    except Exception:
        send_email = None  # fallback to simulation if mailer unavailable
        def render_invite_subject(role: str | None = None) -> str:
            return "Interview Invitation"
        def render_invite_body(candidate_name: str | None, job_desc_snippet: str, reply_to: str | None = None) -> str:
            return (
                f"Hello {candidate_name or 'Candidate'},\n\nWe'd like to invite you to an interview.\n\n"
                f"Role details: {job_desc_snippet[:500]}\n\nBest regards,\nRecruitment Team"
            )

# Try imports from agentNodes; provide safe fallbacks
try:
    from agentNodes import analyze_cv, rank_candidates, explain_rankingLLM, _sanitize_vec
except Exception:
    def _sanitize_vec(vec):
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def analyze_cv(t: str):
        return {"skills": [], "experience": None, "full_text": t or ""}

    def explain_rankingLLM(info: Dict[str, Any], job: str) -> str:
        return ""

    def rank_candidates(cv_embeddings, job_embedding, candidate_ids):
        from sklearn.metrics.pairwise import cosine_similarity
        scores = []
        for cid, emb in zip(candidate_ids, cv_embeddings):
            if emb is None or job_embedding is None:
                scores.append((cid, 0.0))
                continue
            a = _sanitize_vec(emb).reshape(1, -1)
            b = _sanitize_vec(job_embedding).reshape(1, -1)
            if a.shape[1] != b.shape[1]:
                scores.append((cid, 0.0))
                continue
            s = float(cosine_similarity(a, b)[0][0])
            scores.append((cid, s))
        return sorted(scores, key=lambda x: x[1], reverse=True)

# Fallback preprocess_text
try:
    from preprocessing import preprocess_text
except Exception:
    def preprocess_text(t: str) -> str:
        return " ".join((t or "").lower().split())

# Embedder import with fallback
try:
    from embeddings import get_bert_embeddings
except ImportError:
    from embeddings import generate_embeddings
    import pandas as pd

    def get_bert_embeddings(texts: List[str]):
        df_tmp = pd.DataFrame({"_x": texts})
        embs = generate_embeddings(df_tmp, ["_x"])
        return embs["_x"]

# Qdrant storage
cv_storage = CVStorage(host="localhost", port=6333, collection_name="cvs")

def _extract_vector_from_point(p):
    # Single unnamed vector
    if getattr(p, "vector", None) is not None:
        return p.vector
    # Named vectors
    vs = getattr(p, "vectors", None)
    if vs is None:
        return None
    if isinstance(vs, dict):
        return vs.get("default") or (next(iter(vs.values())) if vs else None)
    data = getattr(vs, "data", None)
    if isinstance(data, dict):
        return data.get("default") or (next(iter(data.values())) if data else None)
    return None

# Skill keywords (simple demo list; extend as needed)
SKILL_KEYWORDS = {
    "python","java","c++","c","go","rust","sql","nosql","mongodb","postgres","docker",
    "kubernetes","aws","azure","gcp","tensorflow","pytorch","react","node","django",
    "flask","fastapi","git","linux","nlp","machine","learning","ml","deep","data",
    "analysis","pandas","numpy","javascript","typescript","html","css","spark","hadoop",
    "excel","powerbi","tableau"
}

def _extract_skills_from_text(text: str) -> List[str]:
    if not text:
        return []
    tokens = {t.strip(".,:;()[]{}").lower() for t in text.split()}
    found = sorted({kw for kw in SKILL_KEYWORDS if kw in tokens})
    return found

def _embed_job_desc(job_desc: str):
    job_desc_clean = preprocess_text(job_desc or "")
    job_vec = _sanitize_vec(get_bert_embeddings([job_desc_clean])[0])
    return job_desc_clean, job_vec

def _vector_search(job_vec, top_k: int):
    try:
        hits = cv_storage.client.search(
            collection_name=cv_storage.collection_name,
            query_vector=job_vec.tolist(),
            limit=max(top_k, 10),
            with_vectors=True,
            with_payload=True,
        )
    except Exception as e:
        print("[search] primary error:", e)
        hits = []
    if not hits:
        hits, _ = cv_storage.client.scroll(
            collection_name=cv_storage.collection_name,
            limit=max(top_k, 50),
            with_vectors=True,
            with_payload=True,
        )
        print("[search] fallback scroll hits:", len(hits))
    else:
        print("[search] search hits:", len(hits))
    return hits

def _prepare_cvs(hits, job_vec):
    cvs = []
    for h in hits:
        payload = h.payload or {}
        text = payload.get("resume_text") or ""  # SHOULD be original or contain email
        vec_raw = _extract_vector_from_point(h)
        if vec_raw is None and text:
            try:
                vec_raw = get_bert_embeddings([text])[0]
            except Exception as e:
                print("[prepare_cvs] embed resume error:", e)
                vec_raw = None
        vec = _sanitize_vec(vec_raw) if vec_raw is not None else None
        if vec is None or vec.shape[0] != job_vec.shape[0]:
            print("[prepare_cvs] drop id:", h.id)
            continue
        # Prefer stored email in payload; fallback to extracting from raw text
        email = _email_from_payload(payload) or _extract_email(text)
        cvs.append({
            "candidate_id": payload.get("candidate_id", h.id),
            "resume_text": text,
            "vector": vec,
            "email": email,  # may be None if truly absent
        })
    print("[prepare_cvs] kept cvs:", len(cvs))
    return cvs

def _normalize_whitespace(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

EMAIL_FALLBACK_DOMAIN = os.getenv("EMAIL_FALLBACK_DOMAIN", "example.com")

# --- email extraction (real emails only; no synthetic fabrication) ---
EMAIL_FIELD_NAMES = ("email", "Email", "candidate_email", "contact_email")

def _extract_email(text: str) -> str | None:
    """
    Extract first raw email from ORIGINAL resume text.
    NOTE: If preprocessing removed emails, you must store the original email
    in Qdrant payload during ingestion (e.g. storage.add_cv_embeddings) under
    a key listed in EMAIL_FIELD_NAMES so we can retrieve it here.
    """
    if not text:
        return None
    # Straight regex only (no obfuscation normalization to avoid false positives)
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0).strip() if m else None

def _email_from_payload(payload: Dict[str, Any]) -> str | None:
    if not payload:
        return None
    for k in EMAIL_FIELD_NAMES:
        val = payload.get(k)
        if isinstance(val, str) and "@" in val:
            return val.strip()
    return None

def _candidate_name_from_text(text: str) -> str | None:
    if not text:
        return None
    # heuristic: first two capitalized words at start
    lines = text.strip().splitlines()
    for line in lines[:5]:
        tokens = [t for t in re.split(r"[\s,]+", line) if t]
        caps = [t for t in tokens if re.match(r"^[A-Z][a-zA-Z\-]+$", t)]
        if 1 <= len(caps) <= 3:
            return " ".join(caps[:2])
    return None

def analyze_node(state: Dict[str, Any]) -> Dict[str, Any]:
    cvs = state.get("cvs", [])
    analyzed = {}
    for cv in cvs:
        analyzed[cv["candidate_id"]] = analyze_cv(cv["resume_text"])
    new_state = dict(state)
    new_state["analyzed"] = analyzed
    return new_state

def skill_extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts skills from resume text when analyze_cv provided none; keeps existing if present."""
    analyzed = state.get("analyzed", {})
    cvs = state.get("cvs", [])
    skill_map = {}
    for cv in cvs:
        cid = cv["candidate_id"]
        info = analyzed.get(cid, {})
        skills = info.get("skills") or []
        if not skills:
            # Fallback extraction
            skills = _extract_skills_from_text(cv.get("resume_text", ""))
            # Update analyzed entry non-invasively
            if cid in analyzed:
                analyzed[cid]["skills"] = skills
        skill_map[cid] = skills
    new_state = dict(state)
    new_state["skills_extracted"] = skill_map
    new_state["analyzed"] = analyzed  # ensure updates persist
    return new_state

def rank_node(state: Dict[str, Any]) -> Dict[str, Any]:
    cvs = state.get("cvs", [])
    job_emb = state.get("job_embedding")
    ids = [c["candidate_id"] for c in cvs]
    embs = [c["vector"] for c in cvs]
    ranking = rank_candidates(embs, job_emb, ids)
    score_map = {cid: s for cid, s in ranking}
    ranked = [{**c, "score": float(score_map.get(c["candidate_id"], 0.0))} for c in cvs]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    new_state = dict(state)
    new_state["ranked"] = ranked
    return new_state

def explain_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ranked = state.get("ranked", [])
    analyzed = state.get("analyzed", {})
    skill_map = state.get("skills_extracted", {})
    job_desc = state.get("job_desc_clean", "")
    top_k = int(state.get("top_k", 5))
    results = []
    for cv in ranked[:top_k]:
        cid = cv["candidate_id"]
        info = analyzed.get(cid, {"full_text": cv.get("resume_text", "")})
        if "full_text" not in info:
            info["full_text"] = cv.get("resume_text", "")
        skills = info.get("skills") or skill_map.get(cid, [])
        try:
            explanation = explain_rankingLLM(info, job_desc)
        except Exception as e:
            explanation = f"LLM error: {e}"
        # Only real email (payload or raw text). No synthetic fallback.
        email = cv.get("email")
        if not email:
            email = _extract_email(info.get("full_text", ""))  # attempt once
        results.append({
            "candidate_id": cid,
            "score": cv["score"],
            "skills": skills,
            "experience": info.get("experience"),
            "explanation": explanation,
            "email": email,  # None if not found
        })
    new_state = dict(state)
    new_state["results"] = results
    return new_state

# ---------------- New: email sender node and persistence helpers ----------------
_last_state: Dict[str, Any] | None = None  # module-level fallback cache


def email_sender_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Send (or simulate) interview invitation emails to top candidates.

    Uses real SMTP via mailer.send_email when available; respects EMAIL_DRY_RUN.
    Falls back to prior behavior (logging only) if mailer not importable.
    """
    results = state.get("results") or []
    ranked = state.get("ranked") or []
    top_k = int(state.get("top_k", 5))
    job_desc_clean = state.get("job_desc_clean", state.get("job_desc", ""))

    if not results and ranked:
        # Build lightweight surrogate list
        results = []
        for cv in ranked[:top_k]:
            email_guess = cv.get("email") or _extract_email(cv.get("resume_text", ""))
            results.append({
                "candidate_id": cv.get("candidate_id"),
                "score": cv.get("score", 0.0),
                "explanation": cv.get("explanation", ""),
                "email": email_guess,
                "resume_text": cv.get("resume_text", ""),
            })

    if not results:
        summary = "No candidate list available to email. Run a search first."
        print("[email_sender_node]", summary)
        return {"email_summary": summary, "emailed": [], "emailed_entries": []}

    emailed_ids: List[Any] = []
    emailed_entries: List[Dict[str, Any]] = []
    by_id = {cv.get("candidate_id"): cv for cv in ranked} if ranked else {}
    subject = render_invite_subject(None)
    reply_to = os.getenv("REPLY_TO") or os.getenv("FROM_EMAIL") or os.getenv("SMTP_USER")

    for r in results[:top_k]:
        cid = r.get("candidate_id")
        if cid is None:
            continue
        email = r.get("email")
        if not email and cid in by_id:
            email = by_id[cid].get("email")
        if not email:
            analyzed = state.get("analyzed", {})
            info = analyzed.get(cid, {})
            email = _extract_email(info.get("full_text", "")) or _extract_email(r.get("explanation", ""))
        if not email:
            try:
                email = cv_storage.get_email_by_candidate_id(cid)
            except Exception:
                email = None

        # Prepare body & attempt send (or skip if mailer missing / no email)
        if email and send_email is not None:
            name_source = r.get("resume_text") or by_id.get(cid, {}).get("resume_text", "")
            name = _candidate_name_from_text(name_source)
            body = render_invite_body(name, job_desc_clean, reply_to)
            try:
                ok = send_email(email, subject, body, reply_to=reply_to)
            except Exception as e:
                print(f"[email_sender_node] send error {cid} <{email}>: {e}")
                ok = False
            status = "sent" if ok else "failed"
        else:
            status = "skipped" if email else "no_email"

        if not email:
            email = "unknown"
        print(f"Email {status} to {cid} <{email}>")
        emailed_ids.append(cid)
        emailed_entries.append({"candidate_id": cid, "email": email, "status": status})

    summary = f"Attempted interview invitations to {len(emailed_ids)} candidates."
    return {"email_summary": summary, "emailed": emailed_ids, "emailed_entries": emailed_entries}

def _save_last_state(state: Dict[str, Any]):
    global _last_state
    _last_state = state
    if _compiled_graph is not None and hasattr(_compiled_graph, "save_state"):
        try:
            _compiled_graph.save_state("last_run", state)
            print("[state] saved via graph.save_state('last_run')")
        except Exception as e:
            print("[state] graph.save_state failed:", e)
    else:
        print("[state] saved in _last_state (no graph persistence available)")


def _load_last_state() -> Dict[str, Any] | None:
    if _compiled_graph is not None and hasattr(_compiled_graph, "load_state"):
        try:
            st = _compiled_graph.load_state("last_run")
            if isinstance(st, dict):
                print("[state] loaded via graph.load_state('last_run')")
                return st
        except Exception as e:
            print("[state] graph.load_state failed:", e)
    return _last_state


# ---------------- Build/compile graph; single robust run_agent ----------------
def build_agent_graph():
    if StateGraph is None or END is None:
        raise RuntimeError("LangGraph not available; use run_agent_simple instead.")
    graph = StateGraph(dict)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("skillExtractor", skill_extractor_node)
    graph.add_node("rank", rank_node)
    graph.add_node("explain", explain_node)
    # New node registered (not on default path to avoid auto emails)
    graph.add_node("emailSender", email_sender_node)
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "skillExtractor")
    graph.add_edge("skillExtractor", "rank")
    graph.add_edge("rank", "explain")
    graph.add_edge("explain", END)
    # (emailSender is intentionally not linked; invoked separately when needed)
    graph.set_entry_point("retrieve")
    return graph.compile()

try:
    _compiled_graph = build_agent_graph()
except Exception:
    _compiled_graph = None

def run_agent(job_desc: str, top_k: int = 5, with_explain: bool = False) -> List[Dict[str, Any]]:
    if _compiled_graph is None:
        return run_agent_simple(job_desc, top_k=top_k, with_explain=with_explain)

    out = _compiled_graph.invoke({"job_desc": job_desc, "top_k": top_k})
    if not isinstance(out, dict):
        print(f"[run_agent] unexpected graph output type: {type(out)}")
        return run_agent_simple(job_desc, top_k=top_k, with_explain=with_explain)

    res = out.get("results")
    if res:
        _save_last_state(out)  # persist full state with emails if present
        return res
    # Fallback: build from ranked if present
    ranked = out.get("ranked", [])
    analyzed = out.get("analyzed", {})
    job_desc_clean = out.get("job_desc_clean", "")
    if ranked:
        results = []
        for cv in ranked[:top_k]:
            info = analyzed.get(cv["candidate_id"], {"full_text": cv.get("resume_text", "")})
            if "full_text" not in info:
                info["full_text"] = cv.get("resume_text", "")
            try:
                explanation = explain_rankingLLM(info, job_desc_clean)
            except Exception as e:
                explanation = f"LLM error: {e}"
            email = cv.get("email")
            if not email:
                email = _extract_email(info.get("full_text", "")) or _extract_email(explanation)
            results.append({
                "candidate_id": cv["candidate_id"],
                "score": cv["score"],
                "skills": info.get("skills", []),
                "experience": info.get("experience"),
                "explanation": explanation,
                "email": email,
            })
        # persist reconstructed state with emails
        new_state = dict(out)
        new_state["results"] = results
        _save_last_state(new_state)
        print(f"[run_agent] fallback built results from ranked: {len(results)}")
        return results

    print("[run_agent] graph returned no results; using simple fallback")
    return run_agent_simple(job_desc, top_k=top_k, with_explain=False)

# Optionally persist simple path too (when graph unavailable)
def run_agent_simple(job_desc: str, top_k: int = 5, with_explain: bool = False) -> List[Dict[str, Any]]:
    """
    Simple agent run without graph; for fallback or direct use.
    """
    job_desc_clean, job_vec = _embed_job_desc(job_desc)
    top_k = min(max(top_k, 1), 100)
    hits = _vector_search(job_vec, top_k)
    cvs = _prepare_cvs(hits, job_vec)
    results = []
    for cv in cvs:
        info = analyze_cv(cv["resume_text"])
        try:
            explanation = explain_rankingLLM(info, job_desc_clean)
        except Exception as e:
            explanation = f"LLM error: {e}"
        email = cv.get("email")
        if not email:
            email = _extract_email(cv.get("resume_text", "")) or _extract_email(explanation)
        results.append({
            "candidate_id": cv["candidate_id"],
            "score": cv.get("score", 0.0),
            "skills": info.get("skills", []),
            "experience": info.get("experience"),
            "explanation": explanation,
            "email": email,
        })
    _save_last_state({"results": results, "top_k": top_k, "job_desc_clean": job_desc_clean})
    return results

def run_agent_email() -> str:
    """
    Reuse previously saved state and send interview emails to top candidates,
    returning a detailed reply that includes recipient emails.
    """
    state = _load_last_state()
    if not state:
        return "No previous candidate list found. Run a search first."
    email_out = email_sender_node(state)
    # Update persisted state with email summary/details
    merged_state = dict(state)
    merged_state.update(email_out)
    _save_last_state(merged_state)
    # --- new: build a detailed human-readable reply with emails ---
    entries = email_out.get("emailed_entries", [])
    if not entries:
        return email_out.get("email_summary", "Done.")
    lines = [email_out.get("email_summary", "Done.")]
    for e in entries:
        lines.append(f"- {e.get('candidate_id')} <{e.get('email')}>")
    return "\n".join(lines)