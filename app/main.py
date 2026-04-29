import logging

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from config import settings
from models.schemas import PredictRequest, PredictResponse
from agents.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Sentiment Agent", version="2.0.0")
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model": settings.model,
        "backend": "anthropic",
        "version": "2.0.0",
        "pipeline": ["fetch", "analyze", "reflect"],
        "reflection_band": [settings.reflection_low, settings.reflection_high],
    }


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <!doctype html>
    <html>
        <head>
            <meta charset="utf-8" />
            <title>AI Sentiment Agent</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 750px; margin: 2rem auto; }
                textarea { width: 100%; height: 100px; }
                button { padding: 8px 20px; margin-top: 8px; }
                .result { margin-top: 1rem; padding: 1rem; border: 1px solid #ddd; background: #f9f9f9; }
                .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
                .positive { background: #d4edda; color: #155724; }
                .negative { background: #f8d7da; color: #721c24; }
                .tag { background: #e2e3e5; color: #383d41; margin-left: 6px; }
            </style>
        </head>
        <body>
            <h1>AI Sentiment Agent</h1>
            <p>Enter plain text <strong>or paste a URL</strong> — the agent will fetch and analyze it.</p>
            <textarea id="text" placeholder="Type a sentence, or paste https://..."></textarea>
            <div><button id="btn">Analyze</button></div>
            <div id="out" class="result" style="display:none"></div>

            <script>
            const btn = document.getElementById('btn');
            const out = document.getElementById('out');

            btn.addEventListener('click', async () => {
                const text = document.getElementById('text').value.trim();
                if (!text) return;
                out.style.display = 'block';
                out.innerHTML = '<em>Analyzing...</em>';
                btn.disabled = true;

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text })
                    });
                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || res.statusText);
                    }
                    const d = await res.json();
                    const labelClass = d.label === 'positive' ? 'positive' : 'negative';
                    out.innerHTML = `
                        <strong>Label:</strong>
                        <span class="badge ${labelClass}">${d.label}</span>
                        ${d.reflected ? '<span class="badge tag">reflected</span>' : ''}
                        ${d.source === 'url' ? '<span class="badge tag">from URL</span>' : ''}
                        <br/><br/>
                        <strong>Score:</strong> ${d.score.toFixed(3)}<br/>
                        <strong>Reasoning:</strong> ${d.reasoning}
                    `;
                } catch (err) {
                    out.innerHTML = '<span style="color:red">Error: ' + err.message + '</span>';
                } finally {
                    btn.disabled = false;
                }
            });
            </script>
        </body>
    </html>
    """


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return run_pipeline(_client, req.text)
    except httpx.HTTPStatusError as exc:
        logger.warning("URL fetch failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch URL (HTTP {exc.response.status_code}): {exc.request.url}",
        )
    except httpx.RequestError as exc:
        logger.warning("URL request error: %s", exc)
        raise HTTPException(status_code=400, detail=f"Could not reach URL: {exc.request.url}")
    except anthropic.APIError as exc:
        logger.error("Claude API error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Claude API error: {exc}")
    except ValueError as exc:
        logger.error("Pipeline value error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
