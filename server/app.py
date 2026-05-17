"""FastAPI server for the g4h live demo.

Orchestrates: URL → URL-only indicators → (optional) DOM fetch enrichment →
combined indicator string → Gemma 4 classifier → label + scores back to the UI.

Loads the model once at startup (via FastAPI's lifespan), keeps it in memory
across requests. Single worker — model is too big to fork.

Endpoints:
    GET  /                  -> static/index.html (UI)
    GET  /static/<path>     -> static assets
    GET  /health            -> {"status": "ok", "model_loaded": bool}
    POST /predict           -> {label, scores, indicators, fetch_meta}
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.extract import extract_indicators
from src.infer import InferenceBundle, load_for_inference, predict_one
from server.fetch import fetch_enrich

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("g4h.server")

ARTIFACT_DIR = os.environ.get("G4H_ARTIFACT_DIR", "runs/gemma4-e4b-cls")
STATIC_DIR = Path(__file__).parent / "static"

# Module-level state — the loaded inference bundle. Populated by lifespan.
_bundle: InferenceBundle | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once at startup, hold it across all requests."""
    global _bundle
    log.info("loading model from %s", ARTIFACT_DIR)
    if not os.path.isdir(ARTIFACT_DIR):
        log.error("artifact dir %s not found", ARTIFACT_DIR)
        _bundle = None
    else:
        _bundle = load_for_inference(ARTIFACT_DIR)
        log.info("model ready (labels=%s, max_length=%s)", _bundle.id2label, _bundle.max_length)
    yield
    log.info("shutting down")


app = FastAPI(title="g4h", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Request / response models ---

class PredictRequest(BaseModel):
    url: str = Field(..., description="URL to classify")
    fetch_dom: bool = Field(False, description="Run a server-side Playwright fetch for richer indicators")


class PredictResponse(BaseModel):
    label: str
    scores: dict[str, float]
    indicators: list[str]
    indicator_text: str
    fetch_meta: dict[str, Any] | None = None


# --- Routes ---

@app.get("/")
async def root():
    """Serve the static UI."""
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if _bundle is None:
        raise HTTPException(503, "model not loaded — check G4H_ARTIFACT_DIR")

    url = (req.url or "").strip()
    if not url:
        raise HTTPException(400, "url is required")

    # 1. URL-only extraction (instant)
    indicators = extract_indicators(url)
    fetch_meta: dict[str, Any] | None = None

    # 2. Optional DOM enrichment (slow)
    if req.fetch_dom:
        log.info("fetch_dom=True for %s", url[:80])
        try:
            result = await fetch_enrich(url)
            fetch_meta = {
                "final_url": result.final_url,
                "status": result.status,
                "response_headers": result.response_headers,
                "title": result.title,
                "error": result.error,
            }
            # Drop the meta-only no_indicators sentinel if we added real ones
            if result.indicators and indicators == ["meta:no_indicators:{}"]:
                indicators = []
            indicators.extend(result.indicators)
        except Exception as e:
            log.exception("fetch_enrich failed")
            fetch_meta = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    indicator_text = " ".join(indicators)

    # 3. Predict
    label, scores = predict_one(_bundle, indicator_text)
    log.info("predict url=%r label=%s top_score=%.3f n_indicators=%d",
             url[:80], label, max(scores.values()), len(indicators))

    return PredictResponse(
        label=label,
        scores=scores,
        indicators=indicators,
        indicator_text=indicator_text,
        fetch_meta=fetch_meta,
    )


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    log.exception("unhandled exception")
    return JSONResponse(500, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
