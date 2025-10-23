# app/main.py

from typing import Union, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

from app.bigram_model import BigramModel
from app.embeddings import embed_texts, cosine_similarity, nlp
from app.schemas import (
    TextGenerationRequest as SchemasTextGenReq,  # kept for compatibility if used elsewhere
    EmbedRequest as SchemasEmbedReq, EmbedResponse, 
    SimilarityRequest as SchemasSimReq, SimilarityResponse,
    BigramGenRequest, BigramGenResponse,
)

from app.infer_cifar10 import load_cnn, predict_image_bytes

# --- GAN imports ---
import os, glob, torch
from io import BytesIO
from fastapi.responses import StreamingResponse
from torchvision.utils import make_grid, save_image
from app.gan_model import Generator
from app.schemas import GANSampleRequest
   


app = FastAPI(title="FastAPI: Bigram + spaCy Embeddings + CIFAR10")

# =========================
# Bigram model 
# =========================

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
    # Request body for embeddings:
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

# =========================
# CIFAR10 CNN 
# =========================

_cifar_model = None  # lazy-loaded at startup

@app.on_event("startup")
def _load_cifar10_model_on_startup():
    """
    Try to load the CIFAR-10 CNN weights at API startup.
    If the weights aren't present yet, the /predict/cifar10 route will return 503.
    """
    global _cifar_model
    try:
        _cifar_model = load_cnn("models/cnn_cifar10.pth")
        print("✅ Loaded CIFAR-10 CNN model weights.")
    except Exception as e:
        # It's okay if training hasn't happened yet; you can still start the API.
        print(f"⚠️ Skipping CIFAR-10 CNN load on startup: {e}")

@app.post("/predict/cifar10")
async def predict_cifar10(file: UploadFile = File(...)):
    """
    Accept an image upload, run it through the CNN classifier, and return
    the predicted CIFAR-10 label with confidence.
    """
    if _cifar_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first (app/train_cifar10.py), then restart the API.")

    image_bytes = await file.read()
    try:
        label, confidence = predict_image_bytes(_cifar_model, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    return JSONResponse({"label": label, "confidence": round(confidence, 4)})


# =========================
# GAN: sampling endpoint
# =========================

GAN_ARTIFACT_DIR = "./artifacts/gan"

def _latest_ckpt(dirpath: str) -> str | None:
    paths = sorted(glob.glob(os.path.join(dirpath, "ckpt_e*.pt")))
    return paths[-1] if paths else None

@app.post("/gan/sample", summary="Generate MNIST-like images with the GAN")
def gan_sample(req: GANSampleRequest):
    ckpt = req.ckpt_path or _latest_ckpt(GAN_ARTIFACT_DIR)
    if not ckpt or not os.path.exists(ckpt):
        raise HTTPException(status_code=400, detail="Checkpoint not found. Train first or provide ckpt_path.")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    state = torch.load(ckpt, map_location=device)
    z_dim = state.get("z_dim", req.z_dim)

    G = Generator(z_dim).to(device)
    G.load_state_dict(state["G"])
    G.eval()

    if req.seed is not None:
        torch.manual_seed(req.seed)

    with torch.no_grad():
        z = torch.randn(req.n, z_dim, device=device)
        imgs = G(z).cpu()
        grid = make_grid(imgs, nrow=int(req.n ** 0.5), normalize=True, value_range=(-1, 1))
        buf = BytesIO()
        save_image(grid, buf)
        buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
