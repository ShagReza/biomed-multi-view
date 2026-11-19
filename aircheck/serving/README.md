# MMELON prediction API

A small web API that scores molecules (SMILES) with a fine-tuned MMELON model, so a website can request
predictions. No training code is included.

The two inference files here are the one place MMELON scoring is written down: the benchmark imports them
(see [../benchmark/methods/common.py](../benchmark/methods/common.py)), so a benchmark score and an API score
for the same molecule and model come from exactly the same code.

| File | Purpose |
|---|---|
| `inference_multiview_singleview.py` | Scores SMILES with a fine-tuned MMELON model (multi-view or single view): the final model, or the CV fold models averaged. |
| `inference_embedding_lgbm.py` | Scores SMILES with LightGBM on MMELON embeddings: the final model (`full.txt`) or the fold models. |
| `api.py` | The web API (FastAPI): endpoints, request and response formats, errors. |
| `requirements-serving.txt` | Extra packages for the API only (FastAPI, Uvicorn); not needed to run the benchmark. |
| `.env.example` | The settings the API reads (names with placeholder values). Copy to `.env` for local use. |
| `models.example.json` | Example `MODELS_CONFIG`: one entry per target, with its model type and path. |
| `Dockerfile` | Builds the CPU image that runs the API. |
| `Dockerfile.dockerignore` | What goes into the image: only the code, never models, data or settings. |

## Inference files

Both files work the same way: `load_model(...)` once, then `predict(model, smiles)`, which returns one result per
SMILES in the same order: `{"smiles": ..., "score": 0-1 or None, "valid": True/False}` (invalid SMILES get no score).
Run them from this folder.

| | Final model (default, `models="full"`) | CV fold models, averaged (`models="folds"`) |
|---|---|---|
| `inference_multiview_singleview.py` | the folder holding `best_model.ckpt`, e.g. a benchmark run's `models/final` | the folder holding `fold_<i>/best_model.ckpt`, e.g. a benchmark run's `models` |
| `inference_embedding_lgbm.py` | the LightGBM run folder (uses `models/full.txt`) | the same folder (uses `models/fold_*.txt`) |

Paths below are benchmark runs, `output/<experiment>/<run name>/` (see
[../benchmark/README.md](../benchmark/README.md)); a run made by the archived notebooks works the same way.

```python
from inference_multiview_singleview import load_model, predict
model = load_model("../output/wdr91-cluster/mmelon/finetune_multiview/models/final")           # final model
model = load_model("../output/wdr91-cluster/mmelon/finetune_text/models", models="folds")      # fold ensemble
predict(model, ["CCO", "c1ccccc1"])

from inference_embedding_lgbm import load_model, predict
model = load_model("../output/wdr91-cluster/mmelon/lgbm_pretrained_fused")                     # pretrained embeddings
model = load_model("../output/wdr91-cluster/mmelon/lgbm_finetuned_fused",                      # fine-tuned embeddings
                   encoder_path="../output/wdr91-cluster/mmelon/finetune_multiview/models/final")
predict(model, ["CCO", "c1ccccc1"])
```

- The view (multi-view, text, image or graph; for LightGBM, fused, text, image or graph) is read from the saved model.
  `inference_multiview_singleview.load_model(..., view="text")` additionally checks it.
- For LightGBM on **pretrained** embeddings, `encoder_path` can be left out: the base model is found in
  `output/base_models/`. For **fine-tuned** embeddings, give the model the embeddings came from, or let it use
  the path recorded in the run's `config.json`.
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

## Docker

The image holds the code and its packages (CPU-only PyTorch, about 3.5 GB). **Models are not in it**: they are
mounted when the container runs, so a model can change without rebuilding.

**Build**, from the repository root (the first build takes 10-20 minutes, mostly compiling
`pytorch-fast-transformers`):

```bash
cd <repository root>
docker build -f aircheck/serving/Dockerfile -t mmelon-api .
```

If docker says "permission denied", add yourself to the docker group once (`sudo usermod -aG docker $USER`), then
run `newgrp docker` or open a new terminal.

**Run**, giving it the models folder and the config:

```bash
docker run -p 8090:8080 \
  -v /path/to/models:/models:ro \
  -v /path/to/models-docker.json:/config/models.json:ro \
  -e MODELS_CONFIG=/config/models.json \
  mmelon-api
# then open http://localhost:8090/docs
```

- The paths inside that `models.json` must be the paths **inside** the container (`/models/...`), not the host's,
  so keep a separate copy from the one used without Docker.
- The container serves port **8080** (what Cloud Run expects); `-p 8090:8080` maps it to 8090 on your machine.
  It reads `PORT`, `MODELS_CONFIG`, `MAX_SMILES` and `DEVICE`.
- `:ro` mounts the models read-only. Add `--network none` to check that it needs no internet.

**Push to Artifact Registry** (fill in your own region, project and repository):

```bash
gcloud auth configure-docker <REGION>-docker.pkg.dev
docker tag mmelon-api <REGION>-docker.pkg.dev/<PROJECT>/<REPOSITORY>/mmelon-api:v1
docker push <REGION>-docker.pkg.dev/<PROJECT>/<REPOSITORY>/mmelon-api:v1
```

**To deploy it** (Cloud Run or any other host), three things are needed: the image above, a bucket holding the
models and a `models.json` with `/models/...` paths, and the bucket mounted at `/models` with
`MODELS_CONFIG=/models/models.json`. The website then calls `POST /predict`.

Status: API and Docker image done and tested locally; the image is in Artifact Registry.
