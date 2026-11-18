"""Shared by the MMELON benchmark methods: paths, the base model, data, logging, scoring.

Scoring goes through ../../serving/ (the inference code behind the API), so benchmark scores, the inference
notebooks and the API all use the same steps.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AIRCHECK_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = AIRCHECK_DIR.parent
sys.path.insert(0, str(AIRCHECK_DIR / "serving"))  # the serving modules import each other by plain name
import inference_embedding_lgbm as lgbm_inference  # noqa: E402
import inference_multiview_singleview as mv_inference  # noqa: E402

BASE_MODEL = "ibm-research/biomed.sm.mv-te-84m"  # Hugging Face model ID
BASE_MODEL_DIR = AIRCHECK_DIR / "output" / "base_models" / BASE_MODEL.split("/")[-1]  # shared with the notebooks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = min(24, os.cpu_count())  # data-loading processes: all CPUs, up to 24 (run.py --num-workers changes it)


def base_model_dir() -> Path:
    """The pretrained MMELON model, downloaded once into aircheck/output/base_models/."""
    if not (BASE_MODEL_DIR / "model.safetensors").exists():
        from huggingface_hub import snapshot_download

        print(f"Downloading {BASE_MODEL} ...", flush=True)
        snapshot_download(repo_id=BASE_MODEL, local_dir=BASE_MODEL_DIR)
    return BASE_MODEL_DIR


class Log:
    """Print a line and append it to progress.log in the run folder (follow it with `tail -f`)."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def class_weights(labels, balance: bool):
    """Loss weights [[label 0, label 1]]: label 1 gets (number of 0s) / (number of 1s). None = unweighted."""
    if not balance:
        return None
    labels = np.asarray(labels)
    n_neg, n_pos = int((labels == 0).sum()), int((labels == 1).sum())
    assert n_neg > 0 and n_pos > 0, "training data needs both labels to balance the classes"
    return [[1.0, round(n_neg / n_pos, 4)]]


def valid_smiles(smiles) -> tuple:
    """Stripped SMILES and a valid flag for each (RDKit can read it), as the serving code checks them."""
    return mv_inference.clean_smiles(list(smiles))


def score_with(modules: list, view: str, smiles, batch_size: int) -> np.ndarray:
    """Scores from fine-tuned models (averaged if several, e.g. the CV fold models); NaN for invalid SMILES."""
    cleaned, valid = valid_smiles(smiles)
    scores = np.full(len(cleaned), np.nan)
    if any(valid):
        model = mv_inference.LoadedModel(modules=modules, view=view, path=None, device=DEVICE)
        scores[np.array(valid)] = mv_inference._scores(model, [s for s, ok in zip(cleaned, valid) if ok], batch_size)
    return scores


def load_modules(ckpts: list) -> list:
    from bmfm_sm.predictive.modules.finetune_lightning_module import FineTuneLightningModule

    return [FineTuneLightningModule.load_from_checkpoint(c, map_location=DEVICE).to(DEVICE).eval() for c in ckpts]


def predictions_frame(split: str, rows, smiles, labels, scores) -> pd.DataFrame:
    """Rows of predictions.csv, in the format aircheck-benchmark-core reads."""
    return pd.DataFrame({"split": split, "row": np.asarray(rows), "SMILES": np.asarray(smiles),
                         "LABEL": np.asarray(labels).astype(int), "score": np.asarray(scores, dtype=float)})


def fork_commit() -> str:
    """This repo's commit, with "-dirty" if tracked files have uncommitted changes."""
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({os.cpu_count()} cores)"
