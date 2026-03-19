
from pydantic import BaseModel, Field

class Chunk(BaseModel):
  text: str
  score: float = Field(ge=0.0, le=1.0)
  source: str


def filter_chunks(chunks: list[Chunk], threshold: float = 0.7) -> list[Chunk]:
  return sorted(
    [chunk for chunk in chunks if chunk['score'] >= threshold],
    key=lambda chunk: chunk['score'],
    reverse=True
  )

chunks = [
    {"text": "Fraud detection requires...", "score": 0.92, "source": "doc_1.pdf"},
    {"text": "Trust signals include...", "score": 0.65, "source": "doc_2.pdf"},
    {"text": "Anomaly patterns show...", "score": 0.81, "source": "doc_1.pdf"},
]
print(filter_chunks(chunks))