# retriever.py

import numpy as np
from typing import List, Dict
from ollama import Client
from embeddings import get_bert_embeddings
from storage import CVStorage  # your existing storage class

# -----------------------------
# Initialize Ollama client
# -----------------------------
ollama_client = Client()

def explain_rankingLLM(cv_info: dict, job_desc_clean: str) -> str:
    """
    Generate explanation why a candidate fits a job.
    """
    prompt = f"""
You are an HR assistant. Explain why this candidate is suitable for the job.

Candidate Resume Text:
{cv_info['full_text']}

Job Description:
{job_desc_clean}

Highlight skills, experience, and relevant points.
"""
    response = ollama_client.generate(model='phi3', prompt=prompt)
    return response['response']


# -----------------------------
# Retriever Class
# -----------------------------
class CVRetriever:
    def __init__(self, storage: CVStorage, top_k: int = 5):
        """
        RAG Retriever using Qdrant storage.

        Args:
            storage: CVStorage instance
            top_k: number of top candidates to return
        """
        self.storage = storage
        self.top_k = top_k

    def retrieve_top_candidates(self, job_description: str) -> List[Dict]:
        """
        Given a job description, retrieve top CVs and generate explanations.
        """
        # Generate embedding for job description
        job_embedding = get_bert_embeddings([job_description])[0]  # shape (768,)

        # Retrieve top candidates from Qdrant
        results = self.storage.client.search(
            collection_name=self.storage.collection_name,
            query_vector=job_embedding.tolist(),
            limit=self.top_k
        )

        # Prepare output with explanations
        candidates = []
        for r in results:
            payload = r.payload or {}
            cv_info = {
                "full_text": payload.get("resume_text"),
                "candidate_id": payload.get("candidate_id"),
            }
            explanation = explain_rankingLLM(cv_info, job_description)
            candidates.append(
                {
                    "id": r.id,
                    "score": r.score,
                    "candidate_id": cv_info["candidate_id"],
                    "cv_text": cv_info["full_text"],
                    "explanation": explanation
                }
            )

        return candidates