# Usage recipe

How to go from a train and a test file to evaluated models, and which notebook to run at each step. All notebooks are in `aircheck/notebooks/`; run them from that folder. Each notebook's settings are in its first cells.

## Before you start

- Set up the environment once with [INSTALL.md](INSTALL.md).
- Put your data in its own folder outside the repository, for example `~/Datasets/<project>/`, so it is never committed:
  - `train.parquet`: used for cross-validation and training.
  - `test.parquet`: the **final test set**. Use it only in the inference steps, never to choose a model.
- Both files need a SMILES column and a 0/1 label column; `train.parquet` also needs an `ECFP4` column for step 1. CSV works too.
- Steps 2 to 5 are independent of each other: run the ones you need.
- Set the same **`DATASET`** name (first setting in every notebook) for all steps on one dataset. Every output then goes to
  `aircheck/output/<DATASET>/`, and the default paths between notebooks line up.

## 1. Make the folds

**`cluster_cv_folds.ipynb`**: set `TRAIN_PATH` and the column names.

It writes two files to `aircheck/output/<DATASET>/cv_folds/`, and every k-fold notebook below needs both:
- `train_cluster_cv_data.csv`: the training molecules. Use it as `DATA_PATH`.
- `train_cluster_cv_folds.npz`: the folds. Use it as `CV_INDICES_PATH`. The number of folds (up to 5) depends on the
  clusters, and the CV notebooks read it from this file.

## 2. MMELON (multi-view)

1. **`MMELON_finetuning_cv.ipynb`**: cross-validation. Set `DATA_PATH`, `CV_INDICES_PATH` and `RUN_NAME`.
   - The CV result is in `cv_summary_per_epoch.csv`.
   - The chosen epoch is `best_epoch` in `selection.json`.
2. **`MMELON_inference_cv.ipynb`**: scores `test.parquet` with the fold models averaged together. Set `CV_RUN_DIR` to the run folder.
3. *Optional, one final model:* **`MMELON_finetuning.ipynb`** on all of `train.parquet`, then **`MMELON_inference.ipynb`** on `test.parquet`.
   - Set `EPOCHS = best_epoch + 1`, because epochs are counted from 0 in `selection.json`.

> **Warning:** `MMELON_finetuning.ipynb` keeps the best of its epochs as measured on `TEST_PATH`. If that is the final test set, the test result is slightly optimistic.

## 3. Single view (text, image or graph)

Same as step 2 with the `MMELON_singleview_*` notebooks. Set `VIEW = "text"`, `"image"` or `"graph"`, and repeat for each view you want to compare.

Single-view CV run folders are named `<RUN_NAME>_<VIEW>`.

## 4. LightGBM on MMELON embeddings

1. **`MMELON_embedding_extraction.ipynb`**, run twice: once on `train_cluster_cv_data.csv` and once on `test.parquet`, each with its own `EMBEDDING_NAME`.
   - Use **the same model** both times. `MODEL_PATH = None` uses the pretrained base model.
2. **`MMELON_embedding_lgbm_cv.ipynb`**: CV with the train embeddings and the same folds as step 1. Set `VIEW`.
3. **`MMELON_embedding_lgbm_inference.ipynb`**: scores the test embeddings with the fold models.

> **Warning:** if the embeddings come from a model fine-tuned on `train.parquet`, that model has already seen every training molecule's label. The LightGBM **CV numbers are then over-optimistic**; only the test result is valid. For a clean CV comparison, use the base model (`MODEL_PATH = None`).

## 5. Report

Collect the results into the report (see [report/report-sampledata.md](report/report-sampledata.md)):

| What | File |
|---|---|
| CV result, MMELON or single view | `aircheck/output/<DATASET>/<RUN_NAME>/cv_summary_per_epoch.csv` (the `best_epoch` row) and `selection.json` |
| CV result, LightGBM | `aircheck/output/<DATASET>/lgbm/<RUN_NAME>/cv_summary.csv` |
| Test result | `.../predictions/<PREDICTION_NAME>/metrics.csv` (or `metrics.json` for a single model) |
| Scored test molecules | `.../predictions/<PREDICTION_NAME>/predictions.csv` |

All metrics list hit@200, hit@100 and hit@50 first (the number of actives among the top K scored molecules), then AUROC, PR AUC, precision, recall, specificity and accuracy.
