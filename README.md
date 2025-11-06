Project Structure
SPS_GENAI/
│
├── app/
│   ├── __init__.py
│   ├── bigram_model.py          # Text bigram model
│   ├── cnn_model.py             # CNN classifier (CIFAR-10)
│   ├── gan_model.py             # DCGAN generator/discriminator
│   ├── diffusion_model.py       # DDPM model for CIFAR-10
│   ├── energy_model.py          # Energy-Based Model (EBM) for CIFAR-10
│   ├── embeddings.py            # spaCy word-vector embeddings API
│   ├── infer_cifar10.py
│   ├── infer_gan.py
│   ├── infer_energy.py
│   ├── infer_diffusion.py
│   ├── main.py                  # FastAPI app with all endpoints
│   ├── schemas.py               # Pydantic data schemas
│   ├── train_cifar10.py
│   ├── train_gan.py
│   ├── train_energy_cifar10.py
│   ├── train_diffusion_cifar10.py
│
├── helper_lib/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
│
├── artifacts/
│   ├── energy/                  # EBM checkpoints + sample images
│   └── diffusion/               # Diffusion checkpoints + samples
│
├── data/                        # MNIST / CIFAR-10 datasets (auto-downloaded)
├── Dockerfile
├── requirements.txt
├── README.md
├── energy.png                   # latest generated EBM sample
├── diffusion.png                # latest generated diffusion sample
└── .gitignore



Prerequisites
Python 3.11+

(Optional) Docker Desktop
docker build -t fastapi-genai .
docker run -p 8000:8000 fastapi-genai
# visit http://localhost:8000/docs


Requirements
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.5
torch>=2.2
torchvision>=0.17
numpy>=1.26
pillow>=10.0
tqdm>=4.66
spacy>=3.7
requests>=2.31

Environment Setup
# 1️⃣ Clone the repo
git clone https://github.com/CherylChen000/5560.git
cd 5560

# 2️⃣ Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Download spaCy model (for embeddings)
python -m spacy download en_core_web_lg

# 5️⃣ Run API locally
uvicorn app.main:app --reload

# Then open http://127.0.0.1:8000/docs



Module Overview

Module 1 – Bigram Text Model
Generates short text sequences using bigram probabilities.

curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word":"the","length":10}'


Module 2 – spaCy Embeddings & Similarity

Returns 300-dimensional vectors from en_core_web_lg.

curl -X POST http://127.0.0.1:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{"texts":["king","queen"]}'

Computes similarity:

curl -X POST http://127.0.0.1:8000/embeddings/similarity \
  -H "Content-Type: application/json" \
  -d '{"text_a":"king","text_b":"queen"}'


Module 6 – GAN on MNIST

Implements a DCGAN for handwritten-digit generation.

python -m app.train_gan

Generates samples:

curl -X POST http://127.0.0.1:8000/gan/sample \
  -H "Content-Type: application/json" \
  -d '{"n":64}' \
  --output gan.png

Module 8 – Energy-Based Model (EBM)

Learns an energy landscape on CIFAR-10.

Trains quickly (~3 epochs demo):

python -m app.train_energy_cifar10

Generates samples via API:

curl -X POST http://127.0.0.1:8000/energy/sample \
  -H "Content-Type: application/json" \
  -d '{"n":64}' \
  --output energy.png
open energy.png


Module 8 – Diffusion Model (DDPM)

Lightweight U-Net diffusion model for CIFAR-10.

Train 1 epoch demo:

python -m app.train_diffusion_cifar10

Sample via API:

curl -X POST http://127.0.0.1:8000/diffusion/sample \
  -H "Content-Type: application/json" \
  -d '{"n":64,"T":400}' \
  --output diffusion.png
open diffusion.png

Expected Results

Energy Model: fuzzy colorful blocks that gradually shape into object-like forms.

Diffusion Model: noisy → structured images with clearer texture.
Both return valid PNG grids confirming correct deployment.