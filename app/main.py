import logging

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from config import settings
from models.schemas import PredictRequest, PredictResponse
from agents.runner import run_agent
from tools.history import init_db, store_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Sentiment Agent", version="3.0.0")
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model": settings.model,
        "backend": "anthropic",
        "version": "3.0.0",
        "mode": "fully-agentic",
        "tools": ["fetch_url", "search_web", "analyze_history"],
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
                .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; margin-left: 4px; }
                .positive { background: #d4edda; color: #155724; }
                .negative { background: #f8d7da; color: #721c24; }
                .tool  { background: #cce5ff; color: #004085; }
                .meta  { color: #666; font-size: 0.85rem; margin-top: 8px; }
            </style>
        </head>
        <body>
            <h1>AI Sentiment Agent</h1>
            <p>Enter plain text, a <strong>URL</strong>, or a research question like
               <em>"Has sentiment about Tesla improved this month?"</em></p>
            <textarea id="text" placeholder="Type text, paste a URL, or ask a question..."></textarea>
            <div><button id="btn">Analyze</button></div>
            <div id="out" class="result" style="display:none"></div>

            <script>
            const btn = document.getElementById('btn');
            const out = document.getElementById('out');

            btn.addEventListener('click', async () => {
                const text = document.getElementById('text').value.trim();
                if (!text) return;
                out.style.display = 'block';
                out.innerHTML = '<em>Agent is thinking — may use tools...</em>';
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
                    const toolBadges = d.tools_called.map(t =>
                        `<span class="badge tool">${t}</span>`).join('');
                    out.innerHTML = `
                        <strong>Label:</strong>
                        <span class="badge ${labelClass}">${d.label}</span>
                        ${toolBadges}
                        <br/><br/>
                        <strong>Score:</strong> ${d.score.toFixed(3)}<br/>
                        <strong>Reasoning:</strong> ${d.reasoning}
                        <div class="meta">Source: ${d.source} &nbsp;|&nbsp;
                        Tools used: ${d.tools_called.length > 0 ? d.tools_called.join(' → ') : 'none (direct answer)'}</div>
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
        result = run_agent(_client, req.text)
        store_result(
            text=req.text,
            label=result.label,
            score=result.score,
            reasoning=result.reasoning,
            source=result.source,
        )
        return result
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
        logger.error("Agent error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
