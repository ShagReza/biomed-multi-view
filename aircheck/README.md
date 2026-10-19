# AIRCHECK notebooks for MMELON

This folder is the AIRCHECK team's addition to IBM's [biomed-multi-view](../README.md) (MMELON) repository. Everything outside `aircheck/` is the original repository, unchanged.

The notebooks here fine-tune and evaluate MMELON for **hit identification**: binary classification of molecules from SMILES, ranked and scored with **hit@K** (how many of the top K predicted molecules are actives). They cover:

- fine-tuning the full multi-view model, or a single view (text, image or graph) on its own
- k-fold cross-validation, including the cluster-held-out folds used in `ml-pipeline-aircheck`
- inference on a final test set, with a single model or an ensemble of fold models
- extracting MMELON embeddings and training LightGBM on them

## Folder layout

```
aircheck/
├── README.md                    # this file
├── INSTALL.md                   # setup, and a fix for the pytorch-fast-transformers build
├── requirements-aircheck.txt    # extra packages on top of the repository's own
├── sample_data/
│   ├── train.parquet            # 600 molecules: ID, SMILES, LABEL, ECFP4
│   └── test.parquet             # 200 molecules: the final test set for the inference notebooks
├── notebooks/                   # all notebooks and scripts
├── report/                      # usage recipe and result reports
└── output/                      # created by the notebooks; not tracked by git
```

## Getting started

1. Follow [INSTALL.md](INSTALL.md).
2. Open the notebooks in `notebooks/` and run them from that folder. Every notebook's default paths point at `sample_data/` and `output/`, so the steps below run end to end on the sample data without edits.
3. For your own data, change the paths and column names in each notebook's first cell. Input files can be `.csv` or `.parquet`, and only the needed columns are read.

For a short step-by-step guide to which notebook to run when, see the [usage recipe](report/USAGE_RECIPE.md).

## Workflow

**1. Make the folds** from the training file (the test file is never used here):

| Notebook / script | What it does |
|---|---|
| `cluster_cv_folds.ipynb` | Cluster-held-out folds: positives clustered on ECFP4 Tanimoto distance, each fold tested on unseen clusters. Same method as `cv_method: 'cluster'` in `ml-pipeline-aircheck`. Writes the fold `.npz` and the matching CSV to `output/cv_folds/`. |
| `stratified_cv.py` | Stratified k-fold alternative: `python stratified_cv.py ../sample_data/train.parquet 5 --label-col LABEL`. Its indices refer to the parquet's rows, so use `DATA_PATH = Path("../sample_data/train.parquet")` with the `cv_5fold_indices.npz` it writes. |

**2. Train and select a model** with cross-validation. The best epoch is chosen by the fold-averaged `SELECT_METRICS` (hit@200, then hit@100, hit@50 and AUROC to break ties):

| Notebook | Model |
|---|---|
| `MMELON_finetuning_cv.ipynb` | MMELON (text + image + graph with attention fusion) |
| `MMELON_singleview_finetuning_cv.ipynb` | One encoder on its own: `VIEW = "text"`, `"image"` or `"graph"` |
| `MMELON_embedding_extraction.ipynb` → `MMELON_embedding_lgbm_cv.ipynb` | MMELON embeddings (text, image, graph or fused) + LightGBM |

**3. Score the final test set** (`sample_data/test.parquet`) with the selected models:

| Notebook | For models from |
|---|---|
| `MMELON_inference_cv.ipynb` | `MMELON_finetuning_cv.ipynb` (mean of the fold models) |
| `MMELON_singleview_inference_cv.ipynb` | `MMELON_singleview_finetuning_cv.ipynb` |
| `MMELON_embedding_lgbm_inference.ipynb` | `MMELON_embedding_lgbm_cv.ipynb`; first extract test embeddings with `MMELON_embedding_extraction.ipynb` (`DATA_PATH = Path("../sample_data/test.parquet")`, `EMBEDDING_NAME = "sample-test-base"`) |

Each inference notebook writes `predictions.csv` (every molecule with its score and rank) and, when the file has labels, the metrics.

**Single train/test split.** `MMELON_finetuning.ipynb` and `MMELON_singleview_finetuning.ipynb` (with `MMELON_inference.ipynb` and `MMELON_singleview_inference.ipynb`) train on one file and pick the best epoch on another. Because that second file is used for model selection, give them a validation file, not the final test set, when you want an unbiased final score.

**Other tools.** `download_from_gcs.py` copies data from a GCP bucket to `~/Datasets`.

## Notes

- **Speed.** On CPU, MMELON takes roughly 0.25 s per molecule per model to score, and training is slow. A GPU makes it 10 to 50 times faster. The sample data is small so it runs on CPU.
- **Starting weights.** All models start from the pretrained MMELON checkpoint `ibm-research/biomed.sm.mv-te-84m`, downloaded once to `output/base_models/`. Single-view models use that view's encoder from it.
- **Scores do not depend on the batch.** MMELON's attention fusion averages view weights over a batch, so the inference and embedding notebooks fuse each molecule on its own.

## Sample data

`sample_data/` is a stratified random subset of the BACE dataset from MoleculeNet (Subramanian et al., *J. Chem. Inf. Model.* 2016; Wu et al., *Chem. Sci.* 2018): `LABEL` is BACE-1 inhibitor activity (1 = active), `ID` is the BACE compound ID, and `ECFP4` is a 2048-bit Morgan fingerprint (radius 2) from RDKit. Train and test share no molecules.
