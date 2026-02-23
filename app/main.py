from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def ui():
        return """
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <title>AI Sentiment — Test UI</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 700px; margin: 2rem auto; }
                    textarea { width: 100%; height: 120px; }
                    button { padding: 8px 16px; }
                    .result { margin-top: 1rem; padding: 1rem; border: 1px solid #ddd; background:#f9f9f9 }
                </style>
            </head>
            <body>
                <h1>AI Sentiment — Test UI</h1>
                <p>Enter text below and click <strong>Predict</strong>.</p>
                <textarea id="text" placeholder="Type a sentence..."></textarea>
                <div style="margin-top:8px">
                    <button id="btn">Predict</button>
                </div>
                <div id="out" class="result" aria-live="polite"></div>

                <script>
                const btn = document.getElementById('btn');
                const out = document.getElementById('out');
                btn.addEventListener('click', async () => {
                    const text = document.getElementById('text').value;
                    out.textContent = 'Requesting...';
                    try {
                        const res = await fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ text })
                        });
                        if (!res.ok) throw new Error(res.statusText);
                        const data = await res.json();
                        out.innerHTML = `<strong>Label:</strong> ${data.label} <br/><strong>Score:</strong> ${data.score.toFixed(3)}`;
                    } catch (err) {
                        out.textContent = 'Error: ' + err.message;
                    }
                });
                </script>
            </body>
        </html>
        """

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    proba_pos = float(model.predict_proba([req.text])[0][1])  # P(positive)
    label = "positive" if proba_pos >= 0.5 else "negative"
    return {"label": label, "score": proba_pos}