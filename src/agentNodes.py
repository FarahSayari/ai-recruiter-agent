import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from ollama import Client

# Initialize Ollama once
ollama_client = Client()

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
    """
    Uses Ollama (phi3) to explain why candidate fits the job.
    """
    full_text = (cv_info or {}).get("full_text", "") or ""
    job_text = job_desc_clean or ""
    if not full_text or not job_text:
        return "Insufficient data to generate explanation."
    prompt = f"""
You are an HR assistant. Explain why this candidate is suitable for the job.

Candidate Resume Text:
{full_text}

Job Description:
{job_text}

Highlight concrete skills, relevant experience, and alignment with job requirements. Be concise.
"""
    try:
        resp = ollama_client.generate(model="phi3", prompt=prompt)
        return (resp or {}).get("response", "").strip()
    except Exception as e:
        return f"LLM error: {e}"