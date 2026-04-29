from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str  # plain text or a URL


class PredictResponse(BaseModel):
    label: str
    score: float
    reasoning: str
    reflected: bool   # True when reflection step was triggered
    source: str       # "text" | "url"
