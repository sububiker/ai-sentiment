import json
import os

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Sample AI App (Sentiment)")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. "
    "Classify the sentiment of the given text as either 'positive' or 'negative'. "
    "Respond with a JSON object containing exactly two fields:\n"
    "- \"label\": either \"positive\" or \"negative\"\n"
    "- \"score\": a float between 0.0 and 1.0 representing your confidence "
    "(values above 0.5 indicate positive sentiment, below 0.5 indicate negative)\n"
    "Respond with only the JSON object and no other text."
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {"model": "claude-opus-4-7", "backend": "anthropic", "version": "1.0"}


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
    try:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=256,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": req.text}],
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e}")

    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        raise HTTPException(status_code=502, detail="No text in Claude response")

    try:
        result = json.loads(text_block.text)
        return {"label": result["label"], "score": float(result["score"])}
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected Claude response format: {e}")
