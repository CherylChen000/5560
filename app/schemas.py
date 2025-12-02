from typing import List
from pydantic import BaseModel

# text embedding inputs/outputs
class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    model: str
    dim: int
    vectors: List[List[float]]

# cosine similarity inputs/outputs
class SimilarityRequest(BaseModel):
    text_a: str
    text_b: str

class SimilarityResponse(BaseModel):
    model: str
    dim: int
    similarity: float

# bigram text generation
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = 20

class BigramGenRequest(TextGenerationRequest):
    pass

class BigramGenResponse(BaseModel):
    generated_text: str

# GAN sampling body
class GANSampleRequest(BaseModel):
    n: int = 64                 # number of images (square numbers work best for grid)
    ckpt_path: str | None = None  # path to a checkpoint; if None, use latest
    z_dim: int = 100
    seed: int | None = None     # optional for reproducibility

class EnergySampleRequest(BaseModel):
    ckpt_path: str | None = None
    n: int = 64

class DiffusionSampleRequest(BaseModel):
    ckpt_path: str | None = None
    n: int = 64
    T: int = 400


class LLMGenerationRequest(BaseModel):
    question: str
    max_new_tokens: int = 64

class LLMRLGenerationRequest(BaseModel):
    question: str
    max_new_tokens: int = 64

