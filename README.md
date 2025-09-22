Project Structure
app/
  __init__.py
  bigram_model.py       # your original logic (only a one-line constructor call was fixed)
  embeddings.py         # loads en_core_web_lg; uses nlp(text).vector
  main.py               # original routes + new embeddings + health
  schemas.py            # Pydantic models (with explanations)
requirements.txt
Dockerfile              # optional (builds a runnable image)
README.md

Prerequisites

Python 3.11+

(Optional) Docker Desktop

Quickstart (no Docker)
# 0) clone the repo
git clone https://github.com/<YOUR-USER>/<YOUR-REPO>.git
cd <YOUR-REPO>

# 1) create & activate a virtualenv
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1

# 2) install dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# 3) download the spaCy model used by the notebook
python -m spacy download en_core_web_lg

# 4) run the API
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs

API Reference
Original Endpoints (Module 1)

GET /
Returns {"Hello": "World"}.

POST /generate
Body

{ "start_word": "the", "length": 20 }


Response

{ "generated_text": "..." }


GET /gaussian?mean=0&variance=1&size=1
Returns an array of samples from Normal(mean, variance).

New Endpoints (Module 2)

GET /health
Shows service status and the loaded spaCy model (should be en_core_web_lg).

POST /embeddings
Body

{ "texts": ["king", "queen"] }


Response

{
  "model": "en_core_web_lg",
  "dim": 300,
  "vectors": [[...],[...]]
}


POST /embeddings/similarity
Body

{ "text_a": "king", "text_b": "queen" }


Response

{ "model": "en_core_web_lg", "dim": 300, "similarity": 0.78 }

Example cURL
# health
curl -s http://127.0.0.1:8000/health | jq

# bigram generate (original)
curl -s -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 10}' | jq

# embeddings
curl -s -X POST http://127.0.0.1:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{"texts": ["king", "queen"]}' | jq

# similarity
curl -s -X POST http://127.0.0.1:8000/embeddings/similarity \
  -H "Content-Type: application/json" \
  -d '{"text_a": "king", "text_b": "queen"}' | jq

Screenshots (optional but helpful)

Create docs/screenshots/ and add PNGs named like:

docs/
  screenshots/
    swagger.png                # /docs page
    embeddings_response.png    # POST /embeddings response
    docker_desktop.png         # container running (optional)


Reference them here:

Swagger UI


Embeddings Response


Docker Desktop (optional)


Optional: Docker
# build (from the project root)
docker build -t fastapi-embeddings .

# run (host:8000 -> container:8000)
docker run -p 8000:8000 fastapi-embeddings

# then open http://localhost:8000/docs


What Docker does: builds a self-contained image with Python 3.11, requirements, and the spaCy en_core_web_lg model baked in, then starts Uvicorn in a container.

Notes on Implementation

Embeddings logic mirrors the notebook exactly:

import spacy
nlp = spacy.load("en_core_web_lg")
doc = nlp(text)
vec = doc.vector


Bigram model retains your original functions and probability logic.
Only a tiny constructor fix was applied so it doesn’t crash:

self.vocab, self.bigram_probs = self.analyze_bigrams(" ".join(corpus))

Troubleshooting

ModuleNotFoundError: spacy
Ensure deps and model are installed:

python -m pip install -r requirements.txt
python -m spacy download en_core_web_lg


ImportError: cannot import name 'BigramModel'
Make sure app/__init__.py exists (even empty) and imports use:

from app.bigram_model import BigramModel


Docker pull/build errors on macOS (credential helper)
Edit ~/.docker/config.json and remove "credsStore": "desktop" if pulls fail.

Submission

Verify endpoints at http://127.0.0.1:8000/docs.

Commit & push:

git add .
git commit -m "Module 2: add spaCy embeddings; preserve original bigram API"
git push


Submit the repo URL: https://github.com/CherylChen000/5560.
