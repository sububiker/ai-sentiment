from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="Sample AI App (Sentiment)")

# Tiny toy training set (demo)
TRAIN_X = [
    "i love this", "this is amazing", "so good", "fantastic experience",
    "i hate this", "this is terrible", "so bad", "awful experience",
    "very happy", "really enjoyed it", "very sad", "really disliked it"
]
TRAIN_Y = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]

model: Pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=200)),
])
model.fit(TRAIN_X, TRAIN_Y)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    proba_pos = float(model.predict_proba([req.text])[0][1])  # P(positive)
    label = "positive" if proba_pos >= 0.5 else "negative"
    return {"label": label, "score": proba_pos}