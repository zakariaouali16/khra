"""
FastAPI wrapper around the existing LLMRouterV2.

This intentionally does NOT modify llm_router_phase5_1.py. It imports the
class as-is, builds one router instance at startup, and repoints its Ollama
URL / model via environment variables so the same code runs locally and on GCP.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8080

Environment variables:
    OLLAMA_URL    default http://localhost:11434/api/generate
    MODEL_NAME    default llama3
    TAXONOMY_PATH default ./taxonomy_phase5.json (next to this file)
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from llm_router_phase5_1 import LLMRouterV2

BASE_DIR = Path(__file__).resolve().parent

TAXONOMY_PATH = os.environ.get("TAXONOMY_PATH", str(BASE_DIR / "taxonomy_phase5.json"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3")

app = FastAPI(title="LLM Router Phase 5", version="5.3")

# Allow the page to be served from anywhere. Tighten allow_origins to your
# own domain once you know where the frontend lives.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Built once at startup: loads the taxonomy and the valid-label set.
router = LLMRouterV2(model_name=MODEL_NAME, taxonomy_path=TAXONOMY_PATH)
# The router hardcodes localhost in __init__; repoint it without editing the file.
router.api_url = OLLAMA_URL


class RouteRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    """Serve the query page."""
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "ollama_url": OLLAMA_URL,
        "label_count": len(router.valid_labels),
    }


# Defined as a plain `def` (not async) on purpose: route_request makes two
# blocking Ollama calls, so FastAPI runs it in a threadpool and the event
# loop stays responsive to other requests.
@app.post("/route")
def route(req: RouteRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    try:
        return router.route_request(prompt)
    except Exception as e:  # noqa: BLE001 - surface anything the router didn't catch
        raise HTTPException(status_code=502, detail=f"router failure: {e}")
