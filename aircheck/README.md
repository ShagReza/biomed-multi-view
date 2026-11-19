# AIRCHECK benchmark for MMELON

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
│   ├── train.parquet            # 10,221 molecules (221 actives): ID, SMILES, LABEL, ECFP4
│   └── test.parquet             # 1,031 molecules (31 actives): the final test set for the inference notebooks
├── benchmark/                   # benchmark runs: run_all.sh, run.py, methods/ and thin notebooks/
├── serving/                     # prediction API and its Docker image
├── sample_output/               # mock-up of what an experiment produces (made-up numbers)
├── archive-to-be-deleted/       # the old notebooks, their reports and recipe; replaced by benchmark/
└── output/                      # experiments and runs; not tracked by git
```

## Getting started

1. Follow [INSTALL.md](INSTALL.md), then install
   [aircheck-benchmark-core](https://github.com/ShagReza/aircheck-benchmark-core) into the same environment.
2. Run everything on the sample data with one command:
   ```bash
   aircheck/benchmark/run_all.sh --experiment sample-benchmark \
       --train aircheck/sample_data/train.parquet --test aircheck/sample_data/test.parquet
   ```
   Everything it makes goes to `aircheck/output/sample-benchmark/`: the folds, every run's models and
   predictions, and `report.md`. See [benchmark/README.md](benchmark/README.md) for the methods and options,
   and [benchmark/notebooks/](benchmark/notebooks/) for the same steps one at a time.
3. For your own data, give `--train` and `--test` your files and pick a new `--experiment` name.

> **The old notebooks are in [archive-to-be-deleted/](archive-to-be-deleted/)** and still work on their own,
> but they are no longer the way to run things and will be removed.

## Example output

[sample_output/](sample_output/) is a mock-up of the folder a benchmark experiment produces: the layout, the
file formats and an example report, with made-up numbers and the large files left out. A real experiment goes
to `output/<experiment>/`, which git does not track.

## Class balancing

With `BALANCE_CLASSES = True` (the default in the fine-tuning and LightGBM notebooks), label 1 is weighted by (number of 0s) / (number of 1s) in the loss, so both classes count equally while every negative is still used. The ratio is counted on the training rows only: per fold in CV, after `MAX_PER_CLASS` in the single-split notebooks. Set `False` for an unweighted loss.

- **MMELON** passes it as `finetuning_args["class_weights"] = [[1.0, ratio]]`, an option of MMELON's `FineTuneLightningModule`. **LightGBM** uses `scale_pos_weight = ratio`, which replaces the fixed 8 in the `"suggested"` parameters.
- The weights are printed with the fold summary and saved in `config.json` (and in the MMELON checkpoints).
- Weighting raises the predicted scores. hit@K, AUROC and PR AUC depend only on the ranking, so compare weighted and unweighted runs with these. Precision, recall, specificity and accuracy at a 0.5 threshold, and MMELON's `loss` (also weighted), are not comparable between them.

## Notes

- **Speed.** On CPU, MMELON takes roughly 0.25 s per molecule per model to score, and training is slow. A GPU makes it 10 to 50 times faster. The sample data is small so it runs on CPU.
- **Starting weights.** All models start from the pretrained MMELON checkpoint `ibm-research/biomed.sm.mv-te-84m`, downloaded once to `output/base_models/`. Single-view models use that view's encoder from it.
- **Scores do not depend on the batch.** MMELON's attention fusion averages view weights over a batch, so the inference and embedding notebooks fuse each molecule on its own.

## Sample data

`sample_data/` is a small random subset of an AIRCHECK **WDR91** ASMS screen: `LABEL` is 1 for a hit, `ID` is the row in the source file, and `ECFP4` is the 2048-bit ECFP4 fingerprint from the source (counts; the clustering uses them as bits). Both files come from the same screen and share no molecules. It is meant for checking that the notebooks run, not for benchmarking.

The sample's training actives fall mostly in one chemical cluster, so `cluster_cv_folds.ipynb` builds a single fold from it with the default settings. With one fold, the CV result is that fold's score with no standard deviation, and the fold ensemble is a single model.
