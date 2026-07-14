from pydantic import BaseModel, Field

class Chunk(BaseModel):
    text: str
    score: float = Field(ge=0.0, le=1.0)
    source: str