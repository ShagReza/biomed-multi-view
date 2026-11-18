"""LightGBM on MMELON embeddings (fused, text, image or graph).

The same steps as ../../notebooks/MMELON_embedding_extraction.ipynb, MMELON_embedding_lgbm_cv.ipynb and
MMELON_embedding_lgbm_inference.ipynb:
1. Embed the training and test molecules with an MMELON encoder: the pretrained base model by default, or a
   fine-tuned checkpoint (`encoder`). Embeddings are cached in <out>/embeddings/ and reused by a rerun.
2. One LightGBM model per fold (split foldN = that fold's test rows), plus one on all training rows.
3. The test file scored by the fold models averaged (test_ensemble) and by the model on all rows (test).

With a fine-tuned encoder that has seen the training labels, the CV numbers are over-optimistic; only the test
numbers are valid then. The pretrained base model (the default) gives a clean CV.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import common

DEFAULTS = dict(view="fused", encoder=None, lgbm_params="suggested", balance_classes=True, seed=42, batch_size=32)

LGBM_PARAMS = {
    "default": {},
    "suggested": dict(scale_pos_weight=8, n_estimators=3000, learning_rate=0.02, num_leaves=31, min_child_samples=100,
                      subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=5.0),
}
VIEWS = ("fused", "text", "image", "graph")


def load_encoder(encoder):
    """(encoder module, the view it was trained on): the pretrained base model, or a fine-tuned checkpoint."""
    path = common.base_model_dir() if encoder is None else Path(encoder).expanduser()
    return common.lgbm_inference._load_encoder(path, common.DEVICE)


def embed(encoder, encoder_view: str, view: str, smiles, batch_size: int, cache: Path, source: str) -> tuple:
    """(embeddings of the valid molecules, valid flag per input row). Cached in `cache` (.npz); the cache is
    reused only for the same molecules, view and encoder (`source`)."""
    cleaned, valid = common.valid_smiles(smiles)
    valid = np.array(valid, dtype=bool)
    kept = np.array(cleaned, dtype=object)[valid]
    key = np.array([view, source])
    if cache.exists():
        saved = np.load(cache, allow_pickle=True)
        if np.array_equal(saved["key"], key) and np.array_equal(saved["smiles"], kept):
            return saved["X"], valid
    model = common.lgbm_inference.LoadedLgbmModel(encoder=encoder, encoder_view=encoder_view, view=view,
                                                  boosters=[], path=None, device=common.DEVICE)
    X = common.lgbm_inference._embeddings(model, list(kept), batch_size).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, smiles=kept, key=key)
    return X, valid


def lgbm_params(labels, s: dict) -> dict:
    params = dict(LGBM_PARAMS[s["lgbm_params"]])
    if s["balance_classes"]:
        n_pos = int(np.sum(labels))
        n_neg = len(labels) - n_pos
        assert n_neg > 0 and n_pos > 0, "training rows need both labels to balance the classes"
        params["scale_pos_weight"] = round(n_neg / n_pos, 4)
    return params


def fit(X, y, s: dict) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(random_state=s["seed"], verbose=-1, **lgbm_params(y, s))
    model.fit(X, y)
    return model


def train_and_score(X_train, valid_train, labels, folds, X_test, valid_test, s: dict, models_dir: Path) -> dict:
    """Fold models, the model on all rows, and their scores (NaN where a molecule has no embedding).

    Returns {"fold<i>": scores of that fold's test rows, "test_ensemble": ..., "test": ...}."""
    models_dir.mkdir(parents=True, exist_ok=True)
    position = np.full(len(valid_train), -1)
    position[valid_train] = np.arange(valid_train.sum())  # training row -> row of X_train
    labels = np.asarray(labels).astype(int)

    scores, fold_test_scores = {}, []
    for i, (train_idx, test_idx) in enumerate(folds):
        train_idx = train_idx[valid_train[train_idx]]  # rows without an embedding (invalid SMILES) are skipped
        model = fit(X_train[position[train_idx]], labels[train_idx], s)
        model.booster_.save_model(models_dir / f"fold_{i}.txt")
        fold_scores = np.full(len(test_idx), np.nan)
        has = valid_train[test_idx]
        fold_scores[has] = model.predict_proba(X_train[position[test_idx[has]]])[:, 1]
        scores[f"fold{i}"] = fold_scores
        fold_test_scores.append(model.predict_proba(X_test)[:, 1] if len(X_test) else np.array([]))

    all_rows = np.where(valid_train)[0]
    full = fit(X_train[position[all_rows]], labels[all_rows], s)
    full.booster_.save_model(models_dir / "full.txt")
    for split, values in (("test_ensemble", np.mean(fold_test_scores, axis=0)),
                          ("test", full.predict_proba(X_test)[:, 1] if len(X_test) else np.array([]))):
        scores[split] = np.full(len(valid_test), np.nan)
        scores[split][valid_test] = values
    return scores


def run(data: pd.DataFrame, folds: list, test: pd.DataFrame, s: dict, out_dir: Path, log) -> tuple:
    """Returns (predictions, report extras, settings used). `data` and `test` have columns smiles, label."""
    s = {**DEFAULTS, **{k: v for k, v in s.items() if v is not None}}
    assert s["view"] in VIEWS, f"view must be one of {VIEWS} for embed_lgbm"
    assert s["lgbm_params"] in LGBM_PARAMS, f"lgbm_params must be one of {list(LGBM_PARAMS)}"
    encoder, encoder_view = load_encoder(s["encoder"])
    if encoder_view != "multiview" and s["view"] != encoder_view:
        raise ValueError(f"A {encoder_view!r} encoder gives only {encoder_view!r} embeddings, not {s['view']!r}")

    cache, source = out_dir / "embeddings", str(s["encoder"] or common.BASE_MODEL)
    log(f"Embedding {len(data):,} training molecules ({s['view']}, encoder: {s['encoder'] or 'pretrained base model'})")
    X_train, valid_train = embed(encoder, encoder_view, s["view"], data["smiles"], s["batch_size"],
                                 cache / "train.npz", source)
    log(f"Embedding {len(test):,} test molecules")
    X_test, valid_test = embed(encoder, encoder_view, s["view"], test["smiles"], s["batch_size"],
                               cache / "test.npz", source)
    for name, valid in (("training", valid_train), ("test", valid_test)):
        if not valid.all():
            log(f"{(~valid).sum():,} {name} molecules have invalid SMILES: no embedding, empty score")

    log(f"Training LightGBM ({s['lgbm_params']} parameters) on {len(folds)} folds and on all rows")
    scores = train_and_score(X_train, valid_train, data["label"], folds, X_test, valid_test, s, out_dir / "models")

    parts = [common.predictions_frame(f"fold{i}", test_idx, data["smiles"].iloc[test_idx],
                                      data["label"].iloc[test_idx], scores[f"fold{i}"])
             for i, (_, test_idx) in enumerate(folds)]
    parts += [common.predictions_frame(split, test.index, test["smiles"], test["label"], scores[split])
              for split in ("test_ensemble", "test")]
    (out_dir / "models" / "lgbm_params.json").write_text(json.dumps(LGBM_PARAMS[s["lgbm_params"]], indent=2))
    return pd.concat(parts, ignore_index=True), {}, s
