"""Score molecules with a fine-tuned MMELON model: multi-view or single view (text, image or graph).

Uses the same steps as the inference notebooks (../notebooks/MMELON_inference*.ipynb and
MMELON_singleview_inference*.ipynb), so the scores match them.

    from inference_multiview_singleview import load_model, predict

    model = load_model("path/to/finetuned/<run>", view="multiview")   # load once: the final model (default)
    # or the CV fold models, averaged:  load_model("path/to/<CV run>", models="folds")
    results = predict(model, ["CCO", "c1ccccc1", "not_a_smiles"])
    # [{"smiles": "CCO", "score": 0.41, "valid": True}, ..., {"smiles": "not_a_smiles", "score": None, "valid": False}]

`score` is the model's probability that the molecule is a hit (label 1), between 0 and 1.
"""

import logging
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader

from bmfm_sm.core.data_modules.namespace import TaskType
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.multimodal_finetune_dataset import MultiModalFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline
from bmfm_sm.predictive.modules.finetune_lightning_module import FineTuneLightningModule

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)
RDLogger.DisableLog("rdApp.*")

# Per view: the model class saved in the checkpoint, the data pipeline, and the inputs it needs
VIEWS = {
    "multiview": ("bmfm_sm.predictive.modules.smmv_model.SmallMoleculeMultiView",
                  MultiModalFinetuneDataPipeline, ["TEXT_MODEL", "IMAGE_MODEL", "GRAPH_2D_MODEL"]),
    "text": ("bmfm_sm.predictive.modules.text_models.TextModel", TextFinetuneDataPipeline, ["TEXT_MODEL"]),
    "image": ("bmfm_sm.predictive.modules.image_models.ImageModel", ImageFinetuneDataPipeline, ["IMAGE_MODEL"]),
    "graph": ("bmfm_sm.predictive.modules.graph_2d_models.Graph2dModel", Graph2dFinetuneDataPipeline, ["GRAPH_2D_MODEL"]),
}


@dataclass
class LoadedModel:
    modules: list   # one model, or one per CV fold (their scores are averaged)
    view: str
    path: Path
    device: str


def _view_of(ckpt: Path) -> str:
    model_class = torch.load(ckpt, map_location="cpu")["hyper_parameters"]["base_model_class"]
    found = next((v for v, (cls, _, _) in VIEWS.items() if cls == model_class), None)
    if found is None:
        raise ValueError(f"{ckpt} is not a fine-tuned MMELON model ({model_class})")
    return found


def load_model(path, view=None, device=None, models="full") -> LoadedModel:
    """Load a fine-tuned MMELON model.

    models: "full" (default): path is the final model, a best_model.ckpt or the run folder that contains it.
            "folds": path is a CV run folder (from MMELON*_finetuning_cv.ipynb); all fold_*/best_model.ckpt
            are loaded and their scores averaged.
    view: "multiview", "text", "image" or "graph". None reads it from the checkpoint; if given, it must match.
    device: "cpu" or "cuda"; None uses a GPU if there is one.
    """
    path = Path(path).expanduser()
    if models == "full":
        ckpts = [path / "best_model.ckpt" if path.is_dir() else path]
        if not ckpts[0].exists():
            hint = " (this looks like a CV run folder: use models='folds')" if list(path.glob("fold_*")) else ""
            raise FileNotFoundError(f"Model not found: {ckpts[0]}{hint}")
    elif models == "folds":
        ckpts = sorted(path.glob("fold_*/best_model.ckpt"), key=lambda p: int(p.parent.name.split("_")[1]))
        if not ckpts:
            raise FileNotFoundError(f"No fold_*/best_model.ckpt in {path} (models='folds' needs a CV run folder)")
    else:
        raise ValueError(f"models must be 'full' or 'folds', not {models!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    views = {_view_of(c) for c in ckpts}
    if len(views) > 1:
        raise ValueError(f"The fold models in {path} are different views: {sorted(views)}")
    found = views.pop()
    if view is not None and view != found:
        raise ValueError(f"{ckpts[0]} is a {found!r} model, not {view!r}")

    modules = [FineTuneLightningModule.load_from_checkpoint(c, map_location=device).to(device).eval() for c in ckpts]
    return LoadedModel(modules=modules, view=found, path=path, device=device)


def _to_device(batch, device):
    # Batches are dicts (multiview, graph) or tuples (text, image)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(v, device) for v in batch)
    return batch.to(device) if torch.is_tensor(batch) else batch


def clean_smiles(smiles: list):
    # Strip spaces and check each SMILES with RDKit; returns the cleaned SMILES and a valid flag for each
    cleaned = [s.strip() if isinstance(s, str) else "" for s in smiles]
    return cleaned, [bool(s) and Chem.MolFromSmiles(s) is not None for s in cleaned]


def batches(view: str, smiles: list, batch_size: int, device: str):
    # Yield the MMELON input batches for `view` (in order); the data pipeline reads the SMILES from a CSV file
    _, pipeline, modalities = VIEWS[view]
    with tempfile.TemporaryDirectory() as tmp:
        pd.DataFrame({"smiles": smiles, "label": 0}).to_csv(Path(tmp) / "data_predict.csv", index=False)
        dataset = pipeline(data_dir=tmp, stage="predict", dataset_args={
            "task_type": TaskType.CLASSIFICATION, "num_tasks": 1, "modalities": modalities,
            "smiles_col": "smiles", "label_cols": ["label"], "split_col": None})
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn):
            yield _to_device(batch, device)


def fuse_each(multiview_model, separate) -> torch.Tensor:
    # The attention fusion averages its view weights over a batch; fuse each molecule on its own
    # so a result does not depend on the other molecules in the request
    per_view = [separate[k] for k in ("Graph2dModel", "ImageModel", "TextModel")]
    return torch.cat([multiview_model.aggregator([v[i:i + 1] for v in per_view])[0] for i in range(len(per_view[0]))])


def _scores(model: LoadedModel, smiles: list, batch_size: int) -> list:
    scores = []
    with torch.no_grad():
        for batch in batches(model.view, smiles, batch_size, model.device):  # each molecule is prepared once
            per_model = []
            for m in model.modules:
                if model.view == "multiview":
                    logits = m.pred_head(fuse_each(m.model, m.model.get_separate_outputs(batch)))
                else:
                    logits = m(batch)
                per_model.append(torch.sigmoid(logits).reshape(-1))
            scores += torch.stack(per_model).mean(0).cpu().tolist()  # one model, or the average over folds
    if len(scores) != len(smiles):
        raise RuntimeError(f"Got {len(scores)} scores for {len(smiles)} molecules")
    return scores


def predict(model: LoadedModel, smiles: list, batch_size: int = 32) -> list:
    """Score a list of SMILES. Returns one result per input, in the same order:
    {"smiles": ..., "score": float between 0 and 1, "valid": True}, or score None and valid False for a SMILES
    that RDKit cannot read."""
    cleaned, valid = clean_smiles(smiles)
    scores = iter(_scores(model, [s for s, ok in zip(cleaned, valid) if ok], batch_size)) if any(valid) else iter(())
    return [{"smiles": original, "score": next(scores) if ok else None, "valid": ok}
            for original, ok in zip(smiles, valid)]
