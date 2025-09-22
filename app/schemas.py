"""
schemas.py
-----------
Data models (schemas) for request/response bodies used by the FastAPI app.

Why this file exists
--------------------
FastAPI integrates tightly with Pydantic models to:
  1) Validate incoming JSON payloads and coerce them to the correct Python types.
  2) Produce consistent, well-typed responses via `response_model=...` in route decorators.
  3) Auto-generate OpenAPI/Swagger documentation (what you see at /docs).
  
This file contains **only type definitions** (no business logic). Keeping these
together acts as an API contract: every endpoint in `app/main.py` declares
exactly which schema it accepts/returns.
"""

"""
schemas.py
-----------
Data models (schemas) for request/response bodies used by the FastAPI app.

Why this file exists
--------------------
FastAPI integrates tightly with Pydantic models to:
  1) Validate incoming JSON payloads and coerce them to the correct Python types.
  2) Produce consistent, well-typed responses via `response_model=...` in route decorators.
  3) Auto-generate OpenAPI/Swagger documentation (what you see at /docs).
  
This file contains **only type definitions** (no business logic). Keeping these
together acts as an API contract: every endpoint in `app/main.py` declares
exactly which schema it accepts/returns.
"""

from pydantic import BaseModel
from typing import List


class EmbedRequest(BaseModel):
    """
    Request body for `POST /embeddings`.

    Attributes
    ----------
    texts : List[str]
        A list of strings to embed using spaCy. The API will return a vector
        for each input string, preserving order.
    """
    texts: List[str]


class EmbedResponse(BaseModel):
    """
    Response body for `POST /embeddings`.

    Attributes
    ----------
    model : str
        The name of the spaCy model used (e.g., "en_core_web_lg").
    dim : int
        Dimensionality of the returned vectors.
    vectors : List[List[float]]
        List of embedding vectors (one per input), each serialized as a list of floats.
    """
    model: str
    dim: int
    vectors: List[List[float]]


class SimilarityRequest(BaseModel):
    """
    Request body for `POST /embeddings/similarity`.

    Attributes
    ----------
    text_a : str
        First input string to compare.
    text_b : str
        Second input string to compare.
    """
    text_a: str
    text_b: str


class SimilarityResponse(BaseModel):
    """
    Response body for `POST /embeddings/similarity`.

    Attributes
    ----------
    model : str
        The spaCy model used to produce the embeddings.
    dim : int
        Dimensionality of the embeddings used for the similarity calculation.
    similarity : float
        Cosine similarity in the range [-1, 1], where 1.0 indicates identical
        direction, 0.0 indicates orthogonality, and -1.0 indicates opposite direction.
    """
    model: str
    dim: int
    similarity: float


class TextGenerationRequest(BaseModel):
    """
    Request body for `POST /generate`.

    Attributes
    ----------
    start_word : str
        The starting word (seed) for bigram-based text generation.
    length : int, default=20
        The total number of tokens (including the start word) to generate.
        Defaults to 20 if omitted.
    """
    start_word: str
    length: int = 20

class BigramGenRequest(TextGenerationRequest):
    """Backward-compatible alias for older imports."""
    pass

from pydantic import BaseModel

class BigramGenResponse(BaseModel):
    """Alias response model for /generate."""
    generated_text: str
