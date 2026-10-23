"""Score molecules with LightGBM on MMELON embeddings (from MMELON_embedding_lgbm_cv.ipynb).

Each SMILES is turned into the embedding view the LightGBM run was trained on (fused, text, image or graph), the same
way as ../notebooks/MMELON_embedding_extraction.ipynb, and scored with the run's final LightGBM model
(models/full.txt, trained on all training rows) or, with models="folds", the average of its CV fold models
(as in MMELON_embedding_lgbm_inference.ipynb).

    from inference_embedding_lgbm import load_model, predict

    model = load_model("path/to/lgbm/<RUN_NAME>", encoder_path="path/to/best_model.ckpt")   # load once
    results = predict(model, ["CCO", "not_a_smiles"])
    # [{"smiles": "CCO", "score": 0.12, "valid": True}, {"smiles": "not_a_smiles", "score": None, "valid": False}]

encoder_path is the MMELON model the embeddings came from: a fine-tuned best_model.ckpt (or its run folder), or the
pretrained base model folder (the one with model.safetensors). None takes it from the LightGBM run's config.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy
from bmfm_sm.predictive.modules.finetune_lightning_module import FineTuneLightningModule
from inference_multiview_singleview import VIEWS, batches, clean_smiles, fuse_each

EMBEDDING_KEYS = {"text": "TextModel", "image": "ImageModel", "graph": "Graph2dModel"}  # "fused" is computed
BASE_MODEL_NAME = "biomed.sm.mv-te-84m"


@dataclass
class LoadedLgbmModel:
    encoder: torch.nn.Module   # the MMELON model that makes the embeddings
    encoder_view: str          # "multiview" or the single view it was trained on
    view: str                  # the embedding view LightGBM uses: "fused", "text", "image" or "graph"
    boosters: list             # the final model, or one LightGBM model per CV fold (their scores are averaged)
    path: Path
    device: str


def _load_encoder(encoder_path: Path, device: str):
    # A pretrained base model folder, or a fine-tuned checkpoint (multi-view or single view)
    if (encoder_path / "model.safetensors").exists():
        model = SmallMoleculeMultiViewModel.from_pretrained(
            LateFusionStrategy.ATTENTIONAL, model_path=str(encoder_path), huggingface=True)
        return model.to(device).eval(), "multiview"
    ckpt = encoder_path / "best_model.ckpt" if encoder_path.is_dir() else encoder_path
    if not ckpt.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")
    model_class = torch.load(ckpt, map_location="cpu")["hyper_parameters"]["base_model_class"]
    view = next((v for v, (cls, _, _) in VIEWS.items() if cls == model_class), None)
    if view is None:
        raise ValueError(f"{ckpt} is not a fine-tuned MMELON model ({model_class})")
    module = FineTuneLightningModule.load_from_checkpoint(ckpt, map_location=device)
    return module.model.to(device).eval(), view


def load_model(lgbm_run_dir, encoder_path=None, device=None, models="full") -> LoadedLgbmModel:
    """Load a LightGBM run and the MMELON encoder its embeddings came from.

    models: "full" (default) uses the final model models/full.txt; "folds" averages the CV fold models."""
    run = Path(lgbm_run_dir).expanduser()
    config_path = run / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Not a LightGBM run folder (no config.json): {run}")
    config = json.loads(config_path.read_text())
    if models == "full":
        paths = [run / "models" / "full.txt"]
        if not paths[0].exists():
            raise FileNotFoundError(f"No final model {paths[0]} (older runs have only fold models: use models='folds')")
    elif models == "folds":
        paths = sorted((run / "models").glob("fold_*.txt"), key=lambda p: int(p.stem.split("_")[1]))
        if not paths:
            raise FileNotFoundError(f"No models/fold_*.txt in {run}")
    else:
        raise ValueError(f"models must be 'full' or 'folds', not {models!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if encoder_path is None:  # from the run's config: a checkpoint path, or "base: <Hugging Face ID>"
        recorded = config["embedding_model"]
        encoder_path = (run.parents[2] / "base_models" / BASE_MODEL_NAME if recorded.startswith("base:")
                        else recorded)
    encoder, encoder_view = _load_encoder(Path(encoder_path).expanduser(), device)

    view = config["view"]
    if encoder_view != "multiview" and view != encoder_view:
        raise ValueError(f"The LightGBM run uses {view!r} embeddings, but the encoder is a {encoder_view!r} model")
    boosters = [lgb.Booster(model_file=str(p)) for p in paths]
    return LoadedLgbmModel(encoder=encoder, encoder_view=encoder_view, view=view, boosters=boosters,
                           path=run, device=device)


def _embeddings(model: LoadedLgbmModel, smiles: list, batch_size: int) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for batch in batches(model.encoder_view, smiles, batch_size, model.device):
            if model.encoder_view != "multiview":
                emb = model.encoder.forward0(batch)  # a single-view encoder gives only its own view
            else:
                separate = model.encoder.get_separate_outputs(batch)
                emb = fuse_each(model.encoder, separate) if model.view == "fused" else separate[EMBEDDING_KEYS[model.view]]
            parts.append(emb.float().cpu().numpy())
    X = np.concatenate(parts)
    if len(X) != len(smiles):
        raise RuntimeError(f"Got {len(X)} embeddings for {len(smiles)} molecules")
    return X


def predict(model: LoadedLgbmModel, smiles: list, batch_size: int = 32) -> list:
    """Score a list of SMILES. Returns one result per input, in the same order:
    {"smiles": ..., "score": float between 0 and 1, "valid": True}, or score None and valid False for a SMILES
    that RDKit cannot read."""
    cleaned, valid = clean_smiles(smiles)
    kept = [s for s, ok in zip(cleaned, valid) if ok]
    scores = iter([])
    if kept:
        X = _embeddings(model, kept, batch_size)
        scores = iter(np.mean([b.predict(X) for b in model.boosters], axis=0).tolist())  # one model, or fold average
    return [{"smiles": original, "score": next(scores) if ok else None, "valid": ok}
            for original, ok in zip(smiles, valid)]
