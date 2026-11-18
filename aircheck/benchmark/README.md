# MMELON benchmark runs

`run.py` runs one MMELON method on one dataset from the command line and writes its predictions in the
standard format of [aircheck-benchmark-core](https://github.com/ShagReza/aircheck-benchmark-core). Core then
computes the metrics and the report, the same way for every model in the benchmark.

This folder is plain Python. The notebooks in [../notebooks/](../notebooks/) are unchanged and still work on their own.

## Setup

1. Set up the MMELON environment with [../INSTALL.md](../INSTALL.md).
2. **Install aircheck-benchmark-core into that environment** (`run.py` imports it):
   ```bash
   git clone git@github.com:ShagReza/aircheck-benchmark-core.git ~/repos/aircheck-benchmark-core
   conda activate ~/envs/biomed-multiview
   pip install -e ~/repos/aircheck-benchmark-core
   aircheck-bench --version   # check
   ```

## Run

```bash
# 1. Folds, made once per dataset (by core; the same files for every model)
aircheck-bench make-folds --train ~/Datasets/wdr91/train.parquet --out ~/Datasets/wdr91/folds --method cluster

# 2. One command per method, from the repo root
python aircheck/benchmark/run.py --method finetune --view multiview \
    --folds ~/Datasets/wdr91/folds/train_cluster_cv_folds.npz --test ~/Datasets/wdr91/test.parquet \
    --out results/wdr91/mmelon/finetune_multiview
python aircheck/benchmark/run.py --method embed_lgbm --view fused \
    --folds ~/Datasets/wdr91/folds/train_cluster_cv_folds.npz --test ~/Datasets/wdr91/test.parquet \
    --out results/wdr91/mmelon/embed_lgbm_fused

# 3. Report on every run in the folder (by core)
aircheck-bench report --results results/wdr91/ --folds ~/Datasets/wdr91/folds/train_cluster_cv_folds.npz
```
The training molecules are the folds CSV next to `--folds` (`<name>_cv_data.csv`), so the folds and the rows they
refer to always match. Long runs: start them with `nohup ... &` or in `tmux`, and follow `<out>/progress.log`.

## Methods

| `--method` | `--view` | What it does | Same steps as |
|---|---|---|---|
| `finetune` | `multiview` (default), `text`, `image`, `graph` | Fine-tunes MMELON on every fold side by side, picks the epoch by the fold average, then trains a final model on all training rows for that many epochs | `MMELON_finetuning_cv`, `MMELON_singleview_finetuning_cv`, `MMELON*_inference*` |
| `embed_lgbm` | `fused` (default), `text`, `image`, `graph` | Embeds the molecules with MMELON (pretrained base model by default, or `--encoder <best_model.ckpt>`), then LightGBM per fold and on all rows | `MMELON_embedding_extraction`, `MMELON_embedding_lgbm_cv`, `MMELON_embedding_lgbm_inference` |

Settings (left out = the default in `methods/<method>.py`, the same as the notebooks):

| Option | Method | Default |
|---|---|---|
| `--epochs`, `--patience` | finetune | 50, 5 |
| `--batch-size` | both | 16 (finetune), 32 (embed_lgbm) |
| `--learning-rate`, `--weight-decay` | finetune | 2e-5, 0.01 |
| `--select-metrics` | finetune | hit@200 hit@100 hit@50 rocauc |
| `--no-final` | finetune | final model on (skip it to save time; then only `test_ensemble`) |
| `--encoder` | embed_lgbm | pretrained base model |
| `--lgbm-params` | embed_lgbm | `suggested` (or `default`) |
| `--no-balance` | both | classes balanced |
| `--seed` | both | 42 |
| `--num-workers` | finetune | all CPUs, up to 24. Each worker holds its own data, so on a small VM use 0–2 (multi-view with 8 workers ran out of memory on 8 GB) |
| `--name` | both | display name in the report |

## Output

In `--out`:
```
predictions.csv   split, row, SMILES, LABEL, score: one row per scored molecule
                  split = fold0, fold1, ... (each fold's test rows), test_ensemble (test file, fold models
                  averaged), test (test file, final model / model on all rows)
run.json          model, method, display name, settings, selected epoch, hardware, this repo's commit, core version
progress.log      what happened, with times
models/           the trained models (fold_<i>/, final/ for finetune; fold_<i>.txt, full.txt for embed_lgbm)
embeddings/       embed_lgbm only: cached embeddings, reused when the same run is repeated
```
`run.py` prints the scores at the end (computed by core, which also checks that every fold's predictions are
exactly that fold's test rows). Molecules with a SMILES that RDKit cannot read get an empty score; core ranks
them last.

## Differences from the notebooks

- **Final model:** `MMELON_finetuning.ipynb` keeps the best epoch as measured on its test file. Here the final
  model trains for a fixed number of epochs (the CV-selected epoch + 1), so the test file is never used to
  choose anything.
- **Fold predictions:** the notebooks keep only fold metrics. Here each fold's test rows are scored with that
  fold's model and saved, using the inference code in [../serving/](../serving/) (one molecule at a time
  through the fusion, as the inference notebooks and the API do). The epoch is still chosen from the metrics
  logged during training, as in the notebooks.
- **Folds and metrics** come from aircheck-benchmark-core instead of `cluster_cv_folds.ipynb` and the
  notebooks' `compute_metrics`; both give the same results.
