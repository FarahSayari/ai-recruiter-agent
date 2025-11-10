# embeddings.py

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd

# Device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model & tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)
model.eval()

def get_bert_embeddings(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """
    Generate BERT embeddings for a list of texts.

    Args:
        texts (list of str): Input texts to embed.
        batch_size (int): Number of texts per batch.

    Returns:
        np.ndarray: Array of embeddings (num_texts x 768).
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT batches"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over tokens
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def generate_embeddings(df: pd.DataFrame, text_columns: list[str], batch_size: int = 16) -> dict:
    """
    Generate BERT embeddings for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with cleaned text columns.
        text_columns (list of str): Columns to embed.
        batch_size (int): Batch size for BERT.

    Returns:
        dict: {column_name: np.ndarray of embeddings}
    """
    bert_embeddings = {}
    for col in text_columns:
        if col in df.columns:
            print(f"Generating BERT embeddings for '{col}' ...")
            bert_embeddings[col] = get_bert_embeddings(df[col].tolist(), batch_size=batch_size)
            print(f"'{col}' embeddings done! Shape: {bert_embeddings[col].shape}")
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")
    return bert_embeddings
