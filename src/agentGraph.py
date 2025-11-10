from typing import Dict, Any, List
import numpy as np

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
        text = payload.get("resume_text") or ""
        vec_raw = _extract_vector_from_point(h)
        if vec_raw is None and text:
            try:
                vec_raw = get_bert_embeddings([text])[0]
            except Exception as e:
                print("[prepare_cvs] embed resume error:", e)
                vec_raw = None
        vec = _sanitize_vec(vec_raw) if vec_raw is not None else None
        if vec is None:
            print("[prepare_cvs] drop (vec None) id:", h.id)
            continue
        if vec.shape[0] != job_vec.shape[0]:
            print("[prepare_cvs] drop (dim mismatch) id:", h.id, "vec_dim:", getattr(vec, "shape", None))
            continue
        cvs.append({"candidate_id": payload.get("candidate_id", h.id), "resume_text": text, "vector": vec})
    print("[prepare_cvs] kept cvs:", len(cvs))
    return cvs

# ---------------- Graph nodes (retrieve -> analyze -> rank -> explain) ----------------

def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    job_desc_clean, job_vec = _embed_job_desc(state.get("job_desc", ""))
    top_k = int(state.get("top_k", 5))
    print("[retrieve] job_vec shape:", getattr(job_vec, "shape", None))
    hits = _vector_search(job_vec, top_k)
    cvs = _prepare_cvs(hits, job_vec)
    return {
        "job_desc_clean": job_desc_clean,
        "job_embedding": job_vec,
        "cvs": cvs,
        "top_k": top_k,
        "with_explain": state.get("with_explain", False),
    }

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
        # Merge skills (prefer analyzed; fallback to skill_map)
        skills = info.get("skills") or skill_map.get(cid, [])
        try:
            explanation = explain_rankingLLM(info, job_desc)
        except Exception as e:
            explanation = f"LLM error: {e}"
        results.append({
            "candidate_id": cid,
            "score": cv["score"],
            "skills": skills,
            "experience": info.get("experience"),
            "explanation": explanation,
        })
    new_state = dict(state)
    new_state["results"] = results
    return new_state

# ---------------- Non-graph simple runner (debug/fallback) ----------------

def run_agent_simple(job_desc: str, top_k: int = 5, with_explain: bool = False) -> List[Dict[str, Any]]:
    job_desc_clean, job_vec = _embed_job_desc(job_desc)
    hits = _vector_search(job_vec, top_k)
    cvs = _prepare_cvs(hits, job_vec)
    if not cvs:
        return []
    from sklearn.metrics.pairwise import cosine_similarity
    scores = []
    for cv in cvs:
        s = float(cosine_similarity(cv["vector"].reshape(1, -1), job_vec.reshape(1, -1))[0][0])
        scores.append({**cv, "score": s})
    scores.sort(key=lambda x: x["score"], reverse=True)
    top = scores[:top_k]
    results = []
    for cv in top:
        info = analyze_cv(cv["resume_text"])
        if not info.get("skills"):
            info["skills"] = _extract_skills_from_text(cv["resume_text"])
        expl = ""
        if with_explain:
            try:
                expl = explain_rankingLLM(info, job_desc_clean)
            except Exception as e:
                expl = f"LLM error: {e}"
        results.append({
            "candidate_id": cv["candidate_id"],
            "score": cv["score"],
            "skills": info.get("skills", []),
            "experience": info.get("experience"),
            "explanation": expl,
        })
    return results

# ---------------- Build/compile graph; single robust run_agent ----------------

def build_agent_graph():
    if StateGraph is None or END is None:
        raise RuntimeError("LangGraph not available; use run_agent_simple instead.")
    graph = StateGraph(dict)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("skillExtractor", skill_extractor_node)  # new node
    graph.add_node("rank", rank_node)
    graph.add_node("explain", explain_node)
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "skillExtractor")          # new edge
    graph.add_edge("skillExtractor", "rank")             # new edge
    graph.add_edge("rank", "explain")
    graph.add_edge("explain", END)
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
            results.append({
                "candidate_id": cv["candidate_id"],
                "score": cv["score"],
                "skills": info.get("skills", []),
                "experience": info.get("experience"),
                "explanation": explanation,
            })
        print(f"[run_agent] fallback built results from ranked: {len(results)}")
        return results

    print("[run_agent] graph returned no results; using simple fallback")
    return run_agent_simple(job_desc, top_k=top_k, with_explain=False)