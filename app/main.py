from typing import Union
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
import numpy as np
from app.embeddings import embed_texts, cosine_similarity, nlp
from app.schemas import (
    TextGenerationRequest,
    EmbedRequest, EmbedResponse, 
    SimilarityRequest, SimilarityResponse,
    BigramGenRequest, BigramGenResponse,
)


app = FastAPI(title="FastAPI: Bigram + spaCy Embeddings")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.get("/gaussian")
def sample_gaussian(mean: float = 0.0, variance: float = 1.0, size: int = 1) -> list[float]:
    """Sample from a Gaussian distribution with given mean and variance."""
    std_dev = np.sqrt(variance)
    samples = np.random.normal(mean, std_dev, size)
    return samples.tolist()

class EmbedRequest(BaseModel):
    #Request body for embeddings:
    texts: List[str]

class SimilarityRequest(BaseModel):
    # Request body for cosine similarity:
    text_a: str
    text_b: str

@app.get("/health")
def health():
    """
    Simple health check that also reports the loaded spaCy model name (non-invasive; does not affect embeddings behavior).
    """
    model_name = getattr(nlp, "meta", {}).get("name", "en_core_web_lg")
    return {"status": "ok", "spacy_model": model_name}

@app.post("/embeddings")
def embeddings(req: EmbedRequest):
    return embed_texts(req.texts)

@app.post("/embeddings/similarity")
def embeddings_similarity(req: SimilarityRequest):
    emb = embed_texts([req.text_a, req.text_b])
    sim = cosine_similarity(emb["vectors"][0], emb["vectors"][1])
    return {"model": emb["model"], "dim": emb["dim"], "similarity": sim}
