"""MMELON embeddings, saved as parquet and shared between runs.

The same steps and file layout as ../../archive-to-be-deleted/notebooks/MMELON_embedding_extraction.ipynb: all views come from one
pass of the encoder, and each view is written as parquet parts of `CHUNK_SIZE` molecules:

    <dir>/config.json                     encoder, data file, views, chunk size
    <dir>/<view>/part-00000.parquet       row_index (row in the input), smiles, embedding
    <dir>/invalid_smiles.csv              rows RDKit cannot read (skipped, no embedding)

Finished parts are reused, so an interrupted extraction resumes and a second run with another `--view` costs
nothing. The embeddings of a fine-tuned encoder go in their own folder, never mixed with the pretrained ones.
"""

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import common

CHUNK_SIZE = 1000  # molecules per part file, as in the extraction notebook
VIEWS = ("fused", "text", "image", "graph")
VIEW_KEYS = {"text": "TextModel", "image": "ImageModel", "graph": "Graph2dModel"}  # "fused" is computed


def load_encoder(encoder):
    """(encoder module, the view it was trained on): the pretrained base model, or a fine-tuned checkpoint."""
    path = common.base_model_dir() if encoder is None else Path(encoder).expanduser()
    return common.lgbm_inference._load_encoder(path, common.DEVICE)


def available_views(encoder_view: str) -> tuple:
    """A multi-view encoder gives every view; a single-view one only its own."""
    return VIEWS if encoder_view == "multiview" else (encoder_view,)


def _part(out_dir: Path, view: str, chunk: int) -> Path:
    return out_dir / view / f"part-{chunk:05d}.parquet"


def _batch_embeddings(encoder, encoder_view: str, views, batch) -> dict:
    """Every wanted view of one batch, from a single pass of the encoder."""
    if encoder_view != "multiview":
        return {encoder_view: encoder.forward0(batch)}  # a single-view encoder gives only its own view
    separate = encoder.get_separate_outputs(batch)
    out = {v: separate[VIEW_KEYS[v]] for v in views if v != "fused"}
    if "fused" in views:
        # The aggregator averages its view weights over the batch, so a batched fused embedding would depend
        # on which molecules share the batch. Fuse each molecule on its own instead.
        out["fused"] = common.mv_inference.fuse_each(encoder, separate)
    return out


def extract(encoder, encoder_view: str, smiles, out_dir: Path, source: str, data_path=None, views=None,
            batch_size=32, log=print) -> Path:
    """Write the embeddings of `smiles` to `out_dir` (reusing finished parts). `source` names the encoder."""
    views = tuple(views) if views else available_views(encoder_view)
    unknown = set(views) - set(available_views(encoder_view))
    assert not unknown, f"A {encoder_view!r} encoder cannot give the view(s) {sorted(unknown)}"

    cleaned, valid = common.valid_smiles(smiles)
    molecules = pd.DataFrame({"row_index": np.where(valid)[0], "smiles": np.array(cleaned, dtype=object)[valid]})
    config = {"encoder": source, "data_path": str(data_path) if data_path else None,
              "n_rows": len(cleaned), "n_valid": len(molecules), "views": sorted(views), "chunk_size": CHUNK_SIZE}
    config_path = out_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        same = {k: previous.get(k) for k in ("encoder", "n_rows", "n_valid")}
        assert same == {k: config[k] for k in same}, (
            f"{out_dir} holds embeddings of other molecules or another encoder:\n{previous}\nUse another folder.")
        config["views"] = sorted(set(config["views"]) | set(previous.get("views", [])))
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    for view in views:
        (out_dir / view).mkdir(exist_ok=True)
    if not valid.count(True):
        return out_dir
    if not all(valid):
        pd.DataFrame({"row_index": np.where(~np.array(valid))[0],
                      "smiles": np.array(smiles, dtype=object)[~np.array(valid)]}).to_csv(
            out_dir / "invalid_smiles.csv", index=False)

    n_chunks = (len(molecules) + CHUNK_SIZE - 1) // CHUNK_SIZE
    todo = [c for c in range(n_chunks) if not all(_part(out_dir, v, c).exists() for v in views)]
    if not todo:
        log(f"Embeddings ready in {out_dir} ({len(molecules):,} molecules, views {sorted(views)})")
        return out_dir
    log(f"Embedding {len(molecules):,} molecules in {len(todo)} of {n_chunks} chunk(s), views {sorted(views)}")

    for done, chunk in enumerate(todo, start=1):
        rows = molecules.iloc[chunk * CHUNK_SIZE:(chunk + 1) * CHUNK_SIZE]
        parts = {v: [] for v in views}
        with torch.no_grad():
            for batch in common.mv_inference.batches(encoder_view, rows["smiles"].tolist(), batch_size,
                                                     common.DEVICE):
                for view, emb in _batch_embeddings(encoder, encoder_view, views, batch).items():
                    parts[view].append(emb.float().cpu().numpy())
        for view in views:
            matrix = np.concatenate(parts[view])
            assert len(matrix) == len(rows), f"chunk {chunk}: {len(matrix)} embeddings for {len(rows)} molecules"
            frame = rows.assign(embedding=list(matrix))
            # Write to a hidden temp file first, so an interrupted write never looks like a finished part
            final = _part(out_dir, view, chunk)
            tmp = final.with_name(f".{final.name}.tmp")
            frame.to_parquet(tmp, index=False)
            tmp.rename(final)
        del parts
        gc.collect()
        log(f"  chunk {done}/{len(todo)} done ({len(rows):,} molecules)")
    return out_dir


def load(out_dir: Path, view: str, smiles) -> tuple:
    """(embeddings of the valid molecules, valid flag per input row), checked against `smiles` row by row."""
    folder = Path(out_dir) / view
    assert folder.is_dir() and any(folder.glob("part-*.parquet")), f"No {view!r} embeddings in {out_dir}"
    frame = pd.read_parquet(folder).sort_values("row_index").reset_index(drop=True)
    assert frame["row_index"].is_unique, f"Duplicate row_index in {folder}"

    cleaned, valid = common.valid_smiles(smiles)
    valid = np.array(valid, dtype=bool)
    expected = np.where(valid)[0]
    assert np.array_equal(frame["row_index"].to_numpy(), expected), (
        f"{folder} holds {len(frame)} embeddings for rows that are not the {len(expected)} valid molecules here")
    mismatch = int((frame["smiles"].to_numpy() != np.array(cleaned, dtype=object)[valid]).sum())
    assert mismatch == 0, f"{mismatch} rows have different SMILES in {folder} and the input"
    return np.stack(frame["embedding"].to_numpy()).astype(np.float32), valid
