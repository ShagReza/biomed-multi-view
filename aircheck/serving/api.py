"""Web API for MMELON predictions (FastAPI).

Endpoints
- GET  /health   is the server up?
- GET  /targets  the targets it can score, with each model's type and view
- POST /predict  {"target": "WDR91", "smiles": ["CCO", ...]}  ->  a score for each SMILES, in the same order

Settings (environment variables, see .env.example)
- MODELS_CONFIG  path to a JSON file with one entry per target (see models.example.json)
- MAX_SMILES     most SMILES per request (default 50)
- DEVICE         "cpu" or "cuda" (default: a GPU if there is one)

Run locally, from this folder:  MODELS_CONFIG=models.json uvicorn api:app --port 8080
then open http://localhost:8080/docs to try it.
"""

import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import inference_embedding_lgbm
import inference_multiview_singleview

MAX_SMILES = int(os.environ.get("MAX_SMILES", "50"))
DEVICE = os.environ.get("DEVICE") or None
TARGETS = {}  # target name -> (loaded model, predict function, public model info, lock)


def load_targets(config_path: str):
    # Load every target's model once, at startup
    config = json.loads(Path(config_path).read_text())
    if not config:
        raise ValueError(f"{config_path} lists no targets")
    for name, spec in config.items():
        kind, models = spec.get("type"), spec.get("models", "full")
        if kind == "multiview_singleview":
            model = inference_multiview_singleview.load_model(
                spec["path"], view=spec.get("view"), device=DEVICE, models=models)
            info = {"type": kind, "view": model.view, "models": models, "n_models": len(model.modules)}
            TARGETS[name] = (model, inference_multiview_singleview.predict, info, threading.Lock())
        elif kind == "embedding_lgbm":
            model = inference_embedding_lgbm.load_model(
                spec["path"], encoder_path=spec.get("encoder_path"), device=DEVICE, models=models)
            info = {"type": kind, "view": model.view, "encoder_view": model.encoder_view, "models": models,
                    "n_models": len(model.boosters)}
            TARGETS[name] = (model, inference_embedding_lgbm.predict, info, threading.Lock())
        else:
            raise ValueError(f"Target {name!r}: type must be 'multiview_singleview' or 'embedding_lgbm', not {kind!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("MODELS_CONFIG")
    if not config_path:
        raise RuntimeError("Set MODELS_CONFIG to the JSON file that lists the targets (see models.example.json)")
    load_targets(config_path)
    yield


app = FastAPI(title="AIRCHECK MMELON prediction API", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    target: str = Field(..., description="A target from GET /targets", examples=["WDR91"])
    smiles: list[str] = Field(..., min_length=1, description=f"SMILES to score (at most {MAX_SMILES})",
                              examples=[["CCO", "c1ccccc1"]])


class MoleculeResult(BaseModel):
    smiles: str
    score: Optional[float] = Field(None, description="Predicted probability of being a hit (0 to 1); null if invalid")
    valid: bool = Field(..., description="False if RDKit cannot read the SMILES")
    rank: Optional[int] = Field(None, description="1 = highest score in this request; null if invalid")


class PredictResponse(BaseModel):
    target: str
    model: dict
    results: list[MoleculeResult]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/targets")
def targets():
    return {name: info for name, (_, _, info, _) in TARGETS.items()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if request.target not in TARGETS:
        raise HTTPException(404, f"Unknown target {request.target!r}; known targets: {sorted(TARGETS)}")
    if len(request.smiles) > MAX_SMILES:
        raise HTTPException(422, f"At most {MAX_SMILES} SMILES per request; got {len(request.smiles)}")

    model, predict_fn, info, lock = TARGETS[request.target]
    with lock:  # one request at a time per model
        results = predict_fn(model, request.smiles)

    # Rank the valid molecules by score (1 = highest)
    order = sorted((i for i, r in enumerate(results) if r["valid"]), key=lambda i: -results[i]["score"])
    rank = {i: k + 1 for k, i in enumerate(order)}
    return PredictResponse(
        target=request.target, model=info,
        results=[MoleculeResult(**r, rank=rank.get(i)) for i, r in enumerate(results)],
    )
