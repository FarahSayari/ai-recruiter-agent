# storage.py

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct

class CVStorage:
    """
    Handles storing CV embeddings in Qdrant with metadata.
    """

    def __init__(self, host="localhost", port=6333, collection_name="cvs"):
        """
        Initialize Qdrant client and collection.

        Args:
            host (str): Qdrant host
            port (int): Qdrant port
            collection_name (str): Name of the Qdrant collection
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        # Create collection if it doesn't exist
        existing_collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing_collections:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance="Cosine")  # 768-dim embeddings
            )
            print(f"Created collection '{collection_name}' in Qdrant.")
        else:
            print(f"Using existing collection '{collection_name}' in Qdrant.")

    def add_cv_embeddings(self, embeddings: np.ndarray, df: pd.DataFrame):
        """
        Store CV embeddings in Qdrant with metadata.
    
        Args:
            embeddings (np.ndarray): Shape (num_cvs, embedding_dim)
            df (pd.DataFrame): Must have 'Resume_clean' column
        """
        points = []
    
        for idx, emb in enumerate(embeddings):
            # Ensure numeric and replace NaNs
            emb = np.nan_to_num(emb.astype(float))  # convert to float, replace NaNs
            if emb.shape[0] != 768:
                raise ValueError(f"Embedding at index {idx} has wrong shape: {emb.shape}")
    
            metadata = {
                "candidate_id": int(df.index[idx]),
                "resume_text": df.loc[df.index[idx], "Resume_clean"]
            }
    
            # Add point as dict (recommended for latest qdrant-client)
            points.append({
                "id": int(df.index[idx]),
                "vector": emb.tolist(),
                "payload": metadata
            })
    
        # Upsert to Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Stored {len(points)} CV embeddings in collection '{self.collection_name}'.")

    def add_single_cv(self, candidate_id: str, resume_text: str, embedding, skills: list[str] | None = None):
        from qdrant_client.http import models as qm
        vec = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        payload = {
            "candidate_id": candidate_id,
            "resume_text": resume_text,
        }
        if skills:
            payload["skills"] = skills
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qm.PointStruct(
                    id=None,  # or cast candidate_id to int if you use numeric IDs
                    vector=vec,
                    payload=payload,
                )
            ],
        )