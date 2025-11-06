from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import numpy as np
import os, glob, torch
from io import BytesIO
from torchvision.utils import make_grid, save_image

from app.bigram_model import BigramModel
from app.embeddings import embed_texts, cosine_similarity, nlp
from app.schemas import (
    TextGenerationRequest,
    EmbedRequest, EmbedResponse,
    SimilarityRequest, SimilarityResponse,
    BigramGenRequest, BigramGenResponse,
    GANSampleRequest,
)
from app.infer_cifar10 import load_cnn, predict_image_bytes
from app.gan_model import Generator

from io import BytesIO
from app.energy_model import EnergyNet, langevin_sample
from app.diffusion_model import UNetSmall, Diffusion
from app.schemas import EnergySampleRequest, DiffusionSampleRequest


app = FastAPI(title="FastAPI: Bigram + spaCy Embeddings + CIFAR10 + GAN")

# keep the small corpus for bigram demo
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)

# local request model used by /generate to keep old behavior
class TextGenReqLocal(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenReqLocal):
    text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": text}

@app.get("/gaussian")
def sample_gaussian(mean: float = 0.0, variance: float = 1.0, size: int = 1) -> list[float]:
    std = np.sqrt(variance)
    return np.random.normal(mean, std, size).tolist()

@app.get("/health")
def health():
    model_name = getattr(nlp, "meta", {}).get("name", "en_core_web_lg")
    return {"status": "ok", "spacy_model": model_name}

@app.post("/embeddings")
def embeddings(req: EmbedRequest) -> EmbedResponse:
    return embed_texts(req.texts)  # type: ignore[return-value]

@app.post("/embeddings/similarity")
def embeddings_similarity(req: SimilarityRequest) -> SimilarityResponse:
    emb = embed_texts([req.text_a, req.text_b])
    sim = cosine_similarity(emb["vectors"][0], emb["vectors"][1])
    return {"model": emb["model"], "dim": emb["dim"], "similarity": sim}

# CIFAR10 classifier endpoint (lazy load on startup; ok if weights missing)
_cifar_model = None

@app.on_event("startup")
def _load_cifar10_model_on_startup():
    global _cifar_model
    try:
        _cifar_model = load_cnn("models/cnn_cifar10.pth")
        print("Loaded CIFAR-10 CNN model weights.")
    except Exception as e:
        print(f"Skipping CIFAR-10 CNN load on startup: {e}")

@app.post("/predict/cifar10")
async def predict_cifar10(file: UploadFile = File(...)):
    if _cifar_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first (app/train_cifar10.py), then restart the API.")
    image_bytes = await file.read()
    try:
        label, confidence = predict_image_bytes(_cifar_model, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    return JSONResponse({"label": label, "confidence": round(confidence, 4)})

# GAN sampling: load latest checkpoint if user didn't pass a path
GAN_ARTIFACT_DIR = "./artifacts/gan"

def _latest_ckpt(dirpath: str) -> str | None:
    paths = sorted(glob.glob(os.path.join(dirpath, "ckpt_e*.pt")))
    return paths[-1] if paths else None

@app.post("/gan/sample")
def gan_sample(req: GANSampleRequest):
    ckpt = req.ckpt_path or _latest_ckpt(GAN_ARTIFACT_DIR)
    if not ckpt or not os.path.exists(ckpt):
        raise HTTPException(status_code=400, detail="Checkpoint not found. Train first or provide ckpt_path.")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # load generator from checkpoint and produce an image grid
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
        save_image(grid, buf, format="PNG")  # add format for buffer saves
        buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@app.post("/energy/sample")
def energy_sample(req: EnergySampleRequest):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = req.ckpt_path or _latest_ckpt("./artifacts/energy")
    if not ckpt or not os.path.exists(ckpt):
        raise HTTPException(status_code=400, detail="Energy checkpoint not found. Train first or provide ckpt_path.")

    state = torch.load(ckpt, map_location=device)
    net = EnergyNet().to(device)
    net.load_state_dict(state["model"])
    net.eval()

    with torch.no_grad():
        imgs = langevin_sample(net, n=req.n, device=device)
        grid = make_grid(imgs.cpu(), nrow=int(req.n ** 0.5), normalize=True, value_range=(-1, 1))
        buf = BytesIO()
        save_image(grid, buf, format="PNG")
        buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/diffusion/sample")
def diffusion_sample(req: DiffusionSampleRequest):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = req.ckpt_path or _latest_ckpt("./artifacts/diffusion")
    if not ckpt or not os.path.exists(ckpt):
        raise HTTPException(status_code=400, detail="Diffusion checkpoint not found. Train first or provide ckpt_path.")

    state = torch.load(ckpt, map_location=device)
    net = UNetSmall().to(device)
    net.load_state_dict(state["model"])
    net.eval()

    diff = Diffusion(net, T=req.T, device=device)
    with torch.no_grad():
        imgs = diff.sample(n=req.n).cpu()
        grid = make_grid(imgs, nrow=int(req.n ** 0.5), normalize=True, value_range=(-1, 1))
        buf = BytesIO()
        save_image(grid, buf, format="PNG")
        buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
