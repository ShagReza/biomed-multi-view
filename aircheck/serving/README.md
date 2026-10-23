# MMELON prediction API

A small web API that scores molecules (SMILES) with a fine-tuned MMELON model, so a website can request
predictions. It reuses the inference logic of `../notebooks/MMELON_inference.ipynb`; no training code is included.

| File | Purpose |
|---|---|
| `inference_multiview_singleview.py` | Scores SMILES with a fine-tuned MMELON model (multi-view or single view): the final model, or the CV fold models averaged. |
| `inference_embedding_lgbm.py` | Scores SMILES with LightGBM on MMELON embeddings: the final model (`full.txt`) or the fold models. |
| `api.py` | The web API (FastAPI): endpoints, request and response formats, errors. |
| `requirements-serving.txt` | Extra packages for the API only (FastAPI, Uvicorn); not needed for the notebooks. |
| `.env.example` | The settings the API reads (names with placeholder values). Copy to `.env` for local use. |
| `models.example.json` | Example `MODELS_CONFIG`: one entry per target, with its model type and path. |

## Inference files

Both files work the same way: `load_model(...)` once, then `predict(model, smiles)`, which returns one result per
SMILES in the same order: `{"smiles": ..., "score": 0-1 or None, "valid": True/False}` (invalid SMILES get no score).
Run them from this folder.

| | Final model (default, `models="full"`) | CV fold models, averaged (`models="folds"`) |
|---|---|---|
| `inference_multiview_singleview.py` | the final model's folder (or its `best_model.ckpt`), e.g. `output/<DATASET>/finetuned/mmelon-final_<time>` | the CV run folder, e.g. `output/<DATASET>/mmelon-cv` |
| `inference_embedding_lgbm.py` | the LightGBM run folder (uses `models/full.txt`) | the same folder (uses `models/fold_*.txt`) |

```python
from inference_multiview_singleview import load_model, predict
model = load_model("../output/sample-data/finetuned/mmelon-final_<time>")             # final model
model = load_model("../output/sample-data/singleview-cv_text", models="folds")        # fold ensemble
predict(model, ["CCO", "c1ccccc1"])

from inference_embedding_lgbm import load_model, predict
model = load_model("../output/sample-data/lgbm/lgbm-text-finetuned",
                   encoder_path="../output/sample-data/finetuned/mmelon-final_<time>")  # the MMELON model behind the embeddings
predict(model, ["CCO", "c1ccccc1"])
```

- The view (multi-view, text, image or graph; for LightGBM, fused, text, image or graph) is read from the saved model.
  `inference_multiview_singleview.load_model(..., view="text")` additionally checks it.
- Folds are more accurate but K times slower and use K times the memory; the API uses the final model.
- LightGBM runs made before the final model was added have only fold models: use `models="folds"`.

## API

All models in `MODELS_CONFIG` are loaded once at startup.

Run it locally:

```bash
cd aircheck/serving
cp models.example.json models.json        # then edit the paths to your models
MODELS_CONFIG=models.json ~/envs/biomed-multiview/bin/uvicorn api:app --port 8090
```

- Use the `biomed-multiview` env's `uvicorn` (or `micromamba activate ~/envs/biomed-multiview` first); another
  Python fails with `No module named 'lightgbm'`.
- Wait for `Application startup complete` (about 30-60 s), then open http://localhost:8090/docs to try the endpoints
  (`/` itself shows "Not Found"; that is normal).
- Port 8080 is used by Jupyter on the VM, so use another port such as 8090. Stop the server with Ctrl+C.
- Optional settings: `MAX_SMILES` (default 50) and `DEVICE` (`cpu` or `cuda`), see `.env.example`.

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /targets` | each target with its model type, view and number of models |
| `POST /predict` | body `{"target": "WDR91", "smiles": ["CCO", ...]}` (at most `MAX_SMILES`, default 50) → `{"target", "model", "results": [{"smiles", "score", "valid", "rank"}]}`, same order as sent; rank 1 = highest score |

Errors: 404 for an unknown target, 422 for an empty or too long list. The website reads the user's file and sends
the SMILES as a JSON list.

Status: API done and tested locally; Docker image next.
