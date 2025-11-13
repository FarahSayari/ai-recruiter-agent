import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from ollama import Client

# Initialize Ollama once (respect host if provided)
try:
    _ollama_host = os.getenv("OLLAMA_HOST")
    ollama_client = Client(host=_ollama_host) if _ollama_host else Client()
except Exception:
    # Defer import/initialization errors to call site
    ollama_client = None

# -----------------------------
# CV Analyzer Node
# -----------------------------

def _sanitize_vec(vec):
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if np.isnan(arr).all():
        return None
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    return arr

def analyze_cv(cv_text_clean: str) -> dict:
    """
    Very lightweight analyzer: extract frequent tokens as "skills" and a naive years-of-experience.
    """
    text = (cv_text_clean or "").strip()
    if not text:
        return {"skills": [], "experience": None, "full_text": ""}
    tokens = [t for t in text.split() if t.isalpha() and len(t) > 1]
    counts = Counter(tokens)
    skills = [w for w, _ in counts.most_common(20)]
    experience = None
    for tok in text.split():
        if tok.isdigit():
            v = int(tok)
            if 0 < v < 50:
                experience = v
                break
    return {"skills": skills, "experience": experience, "full_text": text}


# -----------------------------
# Job Matcher Node
# -----------------------------
def match_job(cv_embedding: np.ndarray, job_embedding: np.ndarray) -> float:
    """
    Cosine similarity with shape and None checks.
    """
    cv_vec = _sanitize_vec(cv_embedding)
    job_vec = _sanitize_vec(job_embedding)
    if cv_vec is None or job_vec is None:
        return 0.0
    if cv_vec.shape[0] != job_vec.shape[0]:
        return 0.0
    return float(cosine_similarity(cv_vec.reshape(1, -1), job_vec.reshape(1, -1))[0][0])



# -----------------------------
# Ranker Node
# -----------------------------
def rank_candidates(cv_embeddings: list, job_embedding: np.ndarray, candidate_ids: list) -> list:
    """
    Returns list of tuples (candidate_id, score) sorted desc by score.
    """
    if not cv_embeddings or job_embedding is None:
        return []
    n = min(len(cv_embeddings), len(candidate_ids))
    scores = []
    for cid, emb in zip(candidate_ids[:n], cv_embeddings[:n]):
        scores.append((cid, match_job(emb, job_embedding)))
    return sorted(scores, key=lambda x: x[1], reverse=True)

# -----------------------------
# Explainer Node (Phi3-based)
# -----------------------------
def explain_rankingLLM(cv_info: dict, job_desc_clean: str) -> str:
    """Explain candidate fit with optional fast path and prompt capping.

    Behavior:
    - If EXPLAIN_WITH_LLM=0, returns a concise template using extracted skills only.
    - Otherwise, calls Ollama with shortened resume/job snippets and limited output tokens.
    """
    full_text = (cv_info or {}).get("full_text", "") or ""
    skills = (cv_info or {}).get("skills", []) or []
    job_text = job_desc_clean or ""

    # Fast path: disable LLM explanations entirely for speed
    if os.getenv("EXPLAIN_WITH_LLM", "1") == "0":
        lead_sk = ", ".join(skills[:5]) if skills else "relevant skills"
        return f"Skills match: {lead_sk}. Aligned to the role requirements with applicable experience."

    if not full_text or not job_text:
        # Minimal fallback
        lead_sk = ", ".join(skills[:5]) if skills else "relevant skills"
        return f"Skills match: {lead_sk}."

    # Cap prompt sizes to reduce latency
    max_resume = int(os.getenv("MAX_RESUME_CHARS", "1200"))
    max_job = int(os.getenv("MAX_JOB_CHARS", "800"))
    resume_snip = full_text[:max(100, max_resume)]
    job_snip = job_text[:max(80, max_job)]

    prompt = f"""
You are an HR assistant. In 2-3 concise sentences, explain why this candidate fits the job.

Resume:
{resume_snip}

Job Description:
{job_snip}

Focus on concrete skills, relevant experience, and clear alignment. Avoid fluff.
"""
    try:
        if ollama_client is None:
            return "Explanation temporarily unavailable."
        model_name = os.getenv("OLLAMA_MODEL", "phi3")
        options = {"num_predict": int(os.getenv("LLM_NUM_PREDICT", "128"))}
        try:
            resp = ollama_client.generate(model=model_name, prompt=prompt, options=options)
        except TypeError:
            # Older client without options support
            resp = ollama_client.generate(model=model_name, prompt=prompt)
        return (resp or {}).get("response", "").strip()
    except Exception as e:
        return f"LLM error: {e}"