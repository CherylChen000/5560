import spacy
import numpy as np
from typing import List, Dict, Any

# Import the spacy library and load the large English model
nlp = spacy.load("en_core_web_lg")

# Return a vector representation for the given input string
# `doc.vector` is a NumPy array. When the input is a single word that exists in the model's vocabulary, it corresponds to that word's pretrained vector.
def calculate_embedding(input_word: str):
    doc = nlp(input_word)
    return doc.vector

# Compute embeddings for a list of texts
def embed_texts(texts: List[str]) -> Dict[str, Any]:
    vectors = [nlp(t).vector for t in texts] # Call `nlp(text).vector` for each input string
    # Convert NumPy arrays to Python lists (float32) so they can be serialized as JSON
    return {
        "model": "en_core_web_lg",
        "dim": int(vectors[0].shape[0]) if vectors else 0,
        "vectors": [v.astype("float32").tolist() for v in vectors],
    }

# Compute the cosine similarity between two embedding vectors.
# Given two real-valued vectors `a` and `b`, the cosine similarity is defined as: cos_sim(a, b) = (a · b) / (||a|| * ||b||)
# To avoid division-by-zero issues, if either vector has zero L2 norm (which can happen if a model returns a zero vector for out-of-vocabulary inputs), this function returns 0.0.
def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype="float32")
    vb = np.array(b, dtype="float32")
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
