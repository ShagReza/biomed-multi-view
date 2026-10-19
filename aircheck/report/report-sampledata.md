# Report: sample data

*Draft structure. Values are filled in after running the [usage recipe](../USAGE_RECIPE.md).*

## Summary

One or two sentences: which model ranks actives best on the final test set (hit@K), and how the cross-validation and test results compare.

## Data

| | Train | Test |
|---|---|---|
| Source | | |
| Molecules | | |
| Actives (label 1) | | |
| Active rate | | |

- **Folds:** cluster-held-out, from `cluster_cv_folds.ipynb`. Number of folds: —. Test positives per fold: —.
- **Overlap between train and test:** — molecules.

## Setup

| Setting | Value |
|---|---|
| Base model | `ibm-research/biomed.sm.mv-te-84m` |
| Epochs (MMELON and single view) | |
| Batch size, learning rate | |
| Epoch selection | `SELECT_METRICS`: hit@200, then hit@100, hit@50, AUROC |
| LightGBM | default hyperparameters; embeddings from: base model or fine-tuned model |
| Hardware | CPU / GPU type |

## Cross-validation results (train, mean ± std over folds)

| Model | Selected epoch | hit@200 | hit@100 | hit@50 | AUROC | PR AUC |
|---|---|---|---|---|---|---|
| MMELON (multi-view) | | | | | | |
| Single view: text | | | | | | |
| Single view: image | | | | | | |
| Single view: graph | | | | | | |
| LightGBM: text embeddings | n/a | | | | | |
| LightGBM: image embeddings | n/a | | | | | |
| LightGBM: graph embeddings | n/a | | | | | |
| LightGBM: fused embeddings | n/a | | | | | |

## Final test results

| Model | Scored with | hit@200 | hit@100 | hit@50 | AUROC | PR AUC |
|---|---|---|---|---|---|---|
| MMELON (multi-view) | fold ensemble | | | | | |
| MMELON (multi-view) | single final model | | | | | |
| Single view: text | fold ensemble | | | | | |
| Single view: image | fold ensemble | | | | | |
| Single view: graph | fold ensemble | | | | | |
| LightGBM: text embeddings | fold ensemble | | | | | |
| LightGBM: image embeddings | fold ensemble | | | | | |
| LightGBM: graph embeddings | fold ensemble | | | | | |
| LightGBM: fused embeddings | fold ensemble | | | | | |

For reference, a random ranking would place about K × (test active rate) actives in the top K.

## Observations

- Best model on the final test set, and by how much.
- Multi-view versus single views: does fusing the views help?
- Fine-tuning versus LightGBM on embeddings.
- How close the CV estimates are to the test results.

## Caveats

- Cluster-held-out CV tests each fold on unseen chemistry, so it is expected to score lower than random splits.
- The selected epoch is chosen on the CV test folds, so the CV numbers are slightly optimistic.
- If LightGBM used embeddings from a fine-tuned model, its CV numbers are over-optimistic (see the recipe); only its test numbers count.

## Run details

| Step | Run folder | Run time |
|---|---|---|
| Folds | `output/cv_folds/` | |
| MMELON CV and test | `output/<RUN_NAME>/` | |
| Single view CV and test | `output/<RUN_NAME>_<VIEW>/` | |
| Embeddings | `output/embeddings/<EMBEDDING_NAME>/` | |
| LightGBM CV and test | `output/lgbm/<RUN_NAME>/` | |
