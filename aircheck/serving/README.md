# MMELON prediction API

A small web API that scores molecules (SMILES) with a fine-tuned MMELON model, so a website can request
predictions. It reuses the inference logic of `../notebooks/MMELON_inference.ipynb`; no training code is included.

| File | Purpose |
|---|---|
| `inference.py` | Loads a fine-tuned model and scores SMILES. Plain Python, usable without the API. |
| `api.py` | The web API (FastAPI): endpoints, request and response formats, errors. |
| `requirements-serving.txt` | Extra packages for the API only (FastAPI, Uvicorn); not needed for the notebooks. |
| `.env.example` | The settings the API reads (names with placeholder values). Copy to `.env` for local use. |

Status: in progress.
