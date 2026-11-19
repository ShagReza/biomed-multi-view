# MMELON benchmark runs

`run.py` runs one MMELON method on one dataset from the command line and writes its predictions in the
standard format of [aircheck-benchmark-core](https://github.com/ShagReza/aircheck-benchmark-core). Core then
computes the metrics and the report, the same way for every model in the benchmark.

This folder is plain Python: `run_all.sh` → `run.py` → `methods/`. The notebooks in [notebooks/](notebooks/)
do the same steps one at a time for exploring, and define no code of their own. The old notebooks in
[../notebooks/](../notebooks/) are untouched and still work on their own.

## Setup

1. Set up the MMELON environment with [../INSTALL.md](../INSTALL.md).
2. **Install aircheck-benchmark-core into that environment** (`run.py` imports it):
   ```bash
   git clone git@github.com:ShagReza/aircheck-benchmark-core.git ~/repos/aircheck-benchmark-core
   conda activate ~/envs/biomed-multiview
   pip install -e ~/repos/aircheck-benchmark-core
   aircheck-bench --version   # check
   ```

## An experiment

An **experiment** is one dataset with one set of conditions, and everything it produces stays in its own
folder under `aircheck/output/`, so this repo is self-contained: whoever is interested in MMELON finds the
folds, the models, the predictions and the report in one place.

```
aircheck/output/<experiment>/
├── experiment.json      the train/test files and fold settings this experiment was made with
├── folds/               the fold files (made here the first time, then reused)
├── <run name>/          predictions.csv, run.json, progress.log, models/
├── embeddings/          shared by the LightGBM runs of the same encoder
├── report.md            every run in this experiment, with report_summary.csv next to it
└── run_all.log
```
Use a new name for other conditions, for example `wdr91-cluster` and `wdr91-stratified`. `experiment.json` is
checked on every run, so one name cannot end up holding results from two different datasets.

## Run everything (the recipe)

```bash
aircheck/benchmark/run_all.sh --experiment wdr91-cluster \
    --train ~/Datasets/wdr91/train.parquet --test ~/Datasets/wdr91/test.parquet
```
That one command makes the folds (if they are not there yet), does the 12 runs below one after another, and
ends with `aircheck-bench report`, which writes **`aircheck/output/wdr91-cluster/report.md`** and
`report_summary.csv` next to it. `aircheck/benchmark/run_all.sh --help` lists every option.

| Run | Method |
|---|---|
| `finetune_multiview`, `finetune_text`, `finetune_image`, `finetune_graph` | fine-tuning, one per view |
| `lgbm_pretrained_fused`, `..._text`, `..._image`, `..._graph` | LightGBM on pretrained embeddings |
| `lgbm_finetuned_fused`, `..._text`, `..._image`, `..._graph` | LightGBM on embeddings from the `finetune_multiview` final model |

- **Resumable:** a run that already has `run.json` is skipped, so starting the script again after an
  interruption (or after the VM shut down) continues where it stopped. Delete a run's folder to redo it.
- **`--only a,b` / `--skip a,b`** pick runs (`--list` prints the names), and any further options go to every
  `run.py` call, e.g. `--num-workers 2` or `--epochs 3` for a quick try. `--only none` just rewrites the report.
- **Fold method:** `--cv-method cluster` (default), `stratified` or `random`, with `--n-folds`; or `--folds
  <npz>` to use fold files from somewhere else. The folds are made once and then reused by every run.
- **Disk:** `aircheck/output/` is gitignored, and a full experiment is tens of GB (one fine-tuned model per
  fold, per view). Put it on a bigger disk with `--root /home/jupyter/benchmark` if the boot disk is tight.
- **Long runs:** `nohup aircheck/benchmark/run_all.sh --experiment wdr91-cluster ... > /dev/null 2>&1 &`,
  then follow `aircheck/output/wdr91-cluster/run_all.log`.

## Run one method

```bash
python aircheck/benchmark/run.py --method finetune --view multiview \
    --folds aircheck/output/wdr91-cluster/folds/train_cluster_cv_folds.npz \
    --test ~/Datasets/wdr91/test.parquet --out aircheck/output/wdr91-cluster/finetune_multiview
```
The training molecules are the folds CSV next to `--folds` (`<name>_cv_data.csv`), so the folds and the rows they
refer to always match. Follow a run in `<out>/progress.log`. The notebooks in [notebooks/](notebooks/) do the
same through `run_one` in `run.py`, one step at a time.

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
| `--embeddings-dir` | embed_lgbm | `<out>/../embeddings/<encoder>/` |
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
config.json       embed_lgbm only: what ../serving/ needs to score new molecules with this run's models
```
`embed_lgbm` writes the embeddings one folder up, so every view and every rerun of the same encoder shares them:
```
<out>/../embeddings/<pretrained | finetuned-<run name>>/<train | test>/
├── config.json                    encoder, number of molecules, views
├── <view>/part-00000.parquet      row_index, smiles, embedding — the same layout as the extraction notebook
└── invalid_smiles.csv             rows RDKit cannot read (no embedding, empty score)
```
All four views come from one pass of the encoder, so the first LightGBM run of an encoder pays for the
embeddings and the next three are nearly free.
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
- **One folder per experiment** instead of `output/<dataset>/<run name>/` plus `report/report-<dataset>.md`.
- **Embeddings:** the same parquet files, but all four views come from one pass and are shared by the runs of
  an encoder, where the extraction notebook was run once per model and dataset.

## A note on reproducing numbers

Embeddings computed on a GPU and on a CPU differ by about 1e-6 per value. That is enough to move LightGBM's
splits and change hit@50 by a molecule or two, so the metrics are **not** identical across machines. With the
same embedding files the numbers match the old notebooks exactly (checked on the sample data). So:

- Keep the embeddings an experiment produced; do not delete them and recompute on other hardware mid-way.
- Compare models within one experiment, which is what the report does.
- `run.json` records the hardware, the fork commit and the core version of every run.
