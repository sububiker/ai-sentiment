from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str  # plain text, a URL, or a research question


class PredictResponse(BaseModel):
    label: str
    score: float
    reasoning: str
    source: str            # "text" | "url" | "web_search" | "combined"
    tools_called: list[str]  # names of tools Claude invoked, in order
