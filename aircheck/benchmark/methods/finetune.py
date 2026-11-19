"""Fine-tune MMELON (multi-view, or one view: text, image or graph) for binary classification.

The same training as ../../archive-to-be-deleted/notebooks/MMELON_finetuning_cv.ipynb and MMELON_singleview_finetuning_cv.ipynb:
1. Train all folds side by side, one epoch at a time (up to `epochs`), scoring each fold's test rows after every
   epoch; average the metrics over the folds.
2. Pick the epoch by `select_metrics` (first decides, later ones break ties); stop after `patience` epochs
   without improvement. Each fold's model from that epoch is kept.
3. Predictions: each fold's test rows scored by that fold's model (split foldN), the test file scored by the
   fold models averaged (test_ensemble) and, unless final=False, by one final model trained on all training
   rows for the selected number of epochs (test). The final model never sees the test file.
"""

import gc
import json
import logging
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aircheck_benchmark_core.metrics import HIT_KS

from . import common

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

import pytorch_lightning as pl  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel  # noqa: E402
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy, TaskType  # noqa: E402
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline  # noqa: E402
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline  # noqa: E402
from bmfm_sm.predictive.data_modules.multimodal_finetune_dataset import MultiModalFinetuneDataPipeline  # noqa: E402
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline  # noqa: E402
from bmfm_sm.predictive.modules.finetune_lightning_module import FineTuneLightningModule  # noqa: E402

FUSION_STRATEGY = LateFusionStrategy.ATTENTIONAL

DEFAULTS = dict(view="multiview", epochs=50, patience=5, batch_size=16, learning_rate=2e-5, weight_decay=0.01,
                seed=42, balance_classes=True, select_metrics=["hit@200", "hit@100", "hit@50", "rocauc"], final=True)

VIEW_SPECS = {
    "multiview": dict(
        model_class="bmfm_sm.predictive.modules.smmv_model.SmallMoleculeMultiView",
        model_params={"agg_arch": FUSION_STRATEGY.value[0], "agg_gate_input": FUSION_STRATEGY.value[1],
                      "agg_weight_freeze": FUSION_STRATEGY.value[2], "inference_mode": False},
        pipeline=MultiModalFinetuneDataPipeline, modalities=["TEXT_MODEL", "IMAGE_MODEL", "GRAPH_2D_MODEL"],
        weight_prefix=None,
    ),
    "text": dict(model_class="bmfm_sm.predictive.modules.text_models.TextModel", model_params={},
                 pipeline=TextFinetuneDataPipeline, modalities=["TEXT_MODEL"], weight_prefix="model_text."),
    "image": dict(model_class="bmfm_sm.predictive.modules.image_models.ImageModel",
                  model_params={"baseModel": "ResNet18"},
                  pipeline=ImageFinetuneDataPipeline, modalities=["IMAGE_MODEL"], weight_prefix="model_image."),
    "graph": dict(model_class="bmfm_sm.predictive.modules.graph_2d_models.Graph2dModel",
                  model_params={"encoder_embed_dim": 512},
                  pipeline=Graph2dFinetuneDataPipeline, modalities=["GRAPH_2D_MODEL"], weight_prefix="model_graph."),
}


class HitAtKFineTuneModule(FineTuneLightningModule):
    # Adds hit@K and PR AUC to the scores logged after every epoch (as val_hit@K and val_pr_auc)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_scores, self._val_labels = [], []

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self._shared_step(batch, "val")
        self._val_scores.append(y_hat.detach().float().reshape(-1).cpu())
        self._val_labels.append(y.detach().reshape(-1).cpu())
        return loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        scores, labels = torch.cat(self._val_scores), torch.cat(self._val_labels)
        top = torch.argsort(scores, descending=True)
        for k in HIT_KS:
            self.log(f"val_hit@{k}", float((labels[top[:k]] == 1).sum()))
        y = labels.numpy().astype(int)
        pr_auc = average_precision_score(y, scores.numpy()) if len(set(y.tolist())) == 2 else float("nan")
        self.log("val_pr_auc", float(pr_auc))


def pretrained_state(view: str) -> dict:
    """Pretrained weights: the whole multi-view model, or only the chosen view's encoder."""
    state = SmallMoleculeMultiViewModel.from_pretrained(
        FUSION_STRATEGY, model_path=str(common.base_model_dir()), huggingface=True
    ).state_dict()
    prefix = VIEW_SPECS[view]["weight_prefix"]
    return state if prefix is None else {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def make_model(view: str, state: dict, s: dict, weights) -> HitAtKFineTuneModule:
    spec = VIEW_SPECS[view]
    module = HitAtKFineTuneModule(
        base_model_class=spec["model_class"], model_params=spec["model_params"],
        task_type="classification", num_tasks=1, checkpoint_path=None,
        lr=s["learning_rate"], weight_decay=s["weight_decay"],
        finetuning_args={"weight_freeze": "unfrozen", "initialization": "default", "head_arch": "mlp",
                         "use_norm": True, "head_dropout": 0.2, "class_weights": weights},
    )
    result = module.model.load_state_dict(state, strict=False)
    # strict=False would silently skip mismatched weights, so check explicitly
    assert not result.missing_keys and not result.unexpected_keys, (
        f"Pretrained weights did not match: missing={result.missing_keys[:5]}, unexpected={result.unexpected_keys[:5]}"
    )
    return module


def make_loader(view: str, data_dir: Path, stage: str, shuffle: bool, batch_size: int) -> DataLoader:
    dataset = VIEW_SPECS[view]["pipeline"](data_dir=str(data_dir), stage=stage, dataset_args={
        "task_type": TaskType.CLASSIFICATION, "num_tasks": 1, "modalities": VIEW_SPECS[view]["modalities"],
        "smiles_col": "smiles", "label_cols": ["label"], "split_col": None})
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn,
                      num_workers=common.NUM_WORKERS)


def save_weights(state_path: Path, out_path: Path):
    # The full training state without the optimizer: loadable with FineTuneLightningModule.load_from_checkpoint
    ckpt = torch.load(state_path, map_location="cpu")
    for key in ("optimizer_states", "lr_schedulers"):
        ckpt.pop(key, None)
    torch.save(ckpt, out_path)


def round_key(means, select_metrics) -> tuple:
    # Larger is better; loss is negated so a lower loss ranks higher. Ties keep the earlier epoch.
    return tuple(float(means[m]) * (-1 if m == "loss" else 1) for m in select_metrics)


def train_cv(data: pd.DataFrame, folds: list, s: dict, models_dir: Path, log) -> tuple:
    """Train the folds side by side; returns (selected epoch, epochs trained, per fold-epoch metrics)."""
    view = s["view"]
    allowed = [f"hit@{k}" for k in HIT_KS] + ["rocauc", "pr_auc", "precision", "recall", "specificity", "loss"]
    assert set(s["select_metrics"]) <= set(allowed), f"select_metrics must come from {allowed}"

    weights = []
    for i, (train_idx, test_idx) in enumerate(folds):
        data_dir = models_dir / f"fold_{i}" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        data.iloc[train_idx].to_csv(data_dir / "data_train.csv", index=False)
        data.iloc[test_idx].to_csv(data_dir / "data_val.csv", index=False)  # "val" stage = this fold's test rows
        weights.append(common.class_weights(data["label"].iloc[train_idx], s["balance_classes"]))

    state = pretrained_state(view)
    loaders = {i: (make_loader(view, models_dir / f"fold_{i}" / "data", "train", True, s["batch_size"]),
                   make_loader(view, models_dir / f"fold_{i}" / "data", "val", False, s["batch_size"]))
               for i in range(len(folds))}
    rows, best_key, best_epoch, wait = [], None, None, 0
    log(f"Training {len(folds)} folds side by side, up to {s['epochs']} epochs (patience {s['patience']})")
    for epoch in range(s["epochs"]):
        for fold in range(len(folds)):
            fold_dir = models_dir / f"fold_{fold}"
            state_path = fold_dir / "training_state.ckpt"
            pl.seed_everything(s["seed"] + 1000 * fold + epoch, verbose=False)
            trainer = pl.Trainer(max_epochs=epoch + 1, logger=False, enable_checkpointing=False,
                                 default_root_dir=fold_dir, accelerator="auto", devices=1, num_sanity_val_steps=0,
                                 enable_model_summary=False, enable_progress_bar=False)
            module = make_model(view, state, s, weights[fold])
            # Continue this fold from its previous epoch (weights and optimizer state are restored)
            trainer.fit(module, *loaders[fold], ckpt_path=state_path if epoch > 0 else None)
            trainer.save_checkpoint(state_path)
            rows.append(dict(fold=fold, epoch=epoch, **{k.removeprefix("val_"): float(v)
                                                       for k, v in trainer.callback_metrics.items()
                                                       if k.startswith("val_")}))
            log(f"epoch {epoch}, fold {fold + 1}/{len(folds)} done")
            del trainer, module
            gc.collect()

        metrics = pd.DataFrame(rows)
        means = metrics[metrics["epoch"] == epoch].mean(numeric_only=True)
        key = round_key(means, s["select_metrics"])
        if best_key is None or key > best_key:
            best_key, best_epoch, wait = key, epoch, 0
            for fold in range(len(folds)):
                save_weights(models_dir / f"fold_{fold}" / "training_state.ckpt",
                             models_dir / f"fold_{fold}" / "best_model.ckpt")
            status = "best so far, models saved"
        else:
            wait += 1
            status = f"no improvement ({wait}/{s['patience']})"
        log(f"epoch {epoch:3d} mean over folds:  " + "  ".join(f"{m} {means[m]:.4f}" for m in s["select_metrics"])
            + f"  -> {status}")
        metrics.to_csv(models_dir / "cv_metrics_per_epoch.csv", index=False)
        if wait >= s["patience"]:
            log(f"Stopping early: no improvement of the fold average for {s['patience']} epochs")
            break

    for fold in range(len(folds)):
        (models_dir / f"fold_{fold}" / "training_state.ckpt").unlink()
    log(f"Selected epoch {best_epoch} after {epoch + 1} epochs")
    return best_epoch, epoch + 1, pd.DataFrame(rows)


def train_final(data: pd.DataFrame, epochs: int, s: dict, models_dir: Path, log) -> Path:
    """One model on all training rows for a fixed number of epochs (no test file involved)."""
    view, final_dir = s["view"], models_dir / "final"
    (final_dir / "data").mkdir(parents=True, exist_ok=True)
    data.to_csv(final_dir / "data" / "data_train.csv", index=False)
    pl.seed_everything(s["seed"], verbose=False)
    module = make_model(view, pretrained_state(view), s, common.class_weights(data["label"], s["balance_classes"]))
    trainer = pl.Trainer(max_epochs=epochs, logger=False, enable_checkpointing=False, default_root_dir=final_dir,
                         accelerator="auto", devices=1, limit_val_batches=0, num_sanity_val_steps=0,
                         enable_model_summary=False, enable_progress_bar=False)
    log(f"Training the final model on all {len(data):,} training rows for {epochs} epoch(s)")
    trainer.fit(module, make_loader(view, final_dir / "data", "train", True, s["batch_size"]))
    trainer.save_checkpoint(final_dir / "training_state.ckpt")
    save_weights(final_dir / "training_state.ckpt", final_dir / "best_model.ckpt")
    (final_dir / "training_state.ckpt").unlink()
    del trainer, module
    gc.collect()
    return final_dir / "best_model.ckpt"


def run(data: pd.DataFrame, folds: list, test: pd.DataFrame, s: dict, out_dir: Path, log) -> tuple:
    """Returns (predictions, report extras, settings used). `data` and `test` have columns smiles, label."""
    s = {**DEFAULTS, **{k: v for k, v in s.items() if v is not None}}
    assert s["view"] in VIEW_SPECS, f"view must be one of {list(VIEW_SPECS)}"
    bad = data["smiles"].isna() | (data["smiles"].astype(str).str.strip() == "")
    assert not bad.any(), f"{bad.sum()} training rows have no SMILES, e.g. rows {data.index[bad][:5].tolist()}"
    models_dir = out_dir / "models"
    needed_gb = 1.4 * len(folds) + 0.7
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    assert free_gb > needed_gb + 2, f"Need ~{needed_gb:.0f} GB free for checkpoints, only {free_gb:.0f} GB available"

    best_epoch, epochs_run, cv_metrics = train_cv(data, folds, s, models_dir, log)
    fold_ckpts = [models_dir / f"fold_{i}" / "best_model.ckpt" for i in range(len(folds))]

    parts = []
    log("Scoring each fold's test rows with that fold's model")
    for i, (_, test_idx) in enumerate(folds):
        rows = data.iloc[test_idx]
        modules = common.load_modules([fold_ckpts[i]])
        parts.append(common.predictions_frame(f"fold{i}", test_idx, rows["smiles"], rows["label"],
                                              common.score_with(modules, s["view"], rows["smiles"], s["batch_size"])))
        del modules
        gc.collect()

    log("Scoring the test file with the fold models averaged")
    modules = common.load_modules(fold_ckpts)
    parts.append(common.predictions_frame("test_ensemble", test.index, test["smiles"], test["label"],
                                          common.score_with(modules, s["view"], test["smiles"], s["batch_size"])))
    del modules
    gc.collect()

    if s["final"]:
        final_ckpt = train_final(data, best_epoch + 1, s, models_dir, log)
        log("Scoring the test file with the final model")
        modules = common.load_modules([final_ckpt])
        parts.append(common.predictions_frame("test", test.index, test["smiles"], test["label"],
                                              common.score_with(modules, s["view"], test["smiles"], s["batch_size"])))
        del modules

    (models_dir / "selection.json").write_text(json.dumps(
        {"select_metrics": s["select_metrics"], "best_epoch": best_epoch, "epochs_trained": epochs_run,
         "models": [str(p) for p in fold_ckpts]}, indent=2))
    extras = {"selected epoch": best_epoch, "epochs trained": epochs_run}
    return pd.concat(parts, ignore_index=True), extras, s
