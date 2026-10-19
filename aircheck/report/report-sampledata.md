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
| LightGBM | hyperparameters: default or suggested (`LGBM_PARAMS_CHOICE`); embeddings from: base model or fine-tuned model |
| Hardware | CPU / GPU type |

## Cross-validation results (train, mean ± std over folds)

| Model | Selected epoch | hit@200 | hit@100 | hit@50 | AUROC | PR AUC |
|---|---|---|---|---|---|---|
| MMELON (multi-view) | | | | | | |
| Single view: text | | | | | | |
| Single view: image | | | | | | |
| Single view: graph | | | | | | |
| LightGBM: text embeddings (pretrained) | n/a | | | | | |
| LightGBM: image embeddings (pretrained) | n/a | | | | | |
| LightGBM: graph embeddings (pretrained) | n/a | | | | | |
| LightGBM: fused embeddings (pretrained) | n/a | | | | | |

## Final test results

| Model | Scored with | hit@200 | hit@100 | hit@50 | AUROC | PR AUC |
|---|---|---|---|---|---|---|
| MMELON (multi-view) | fold ensemble | | | | | |
| MMELON (multi-view) | single final model | | | | | |
| Single view: text | fold ensemble | | | | | |
| Single view: text | single final model | | | | | |
| Single view: image | fold ensemble | | | | | |
| Single view: image | single final model | | | | | |
| Single view: graph | fold ensemble | | | | | |
| Single view: graph | single final model | | | | | |

For reference, a random ranking would place about K × (test active rate) actives in the top K.

## Does fine-tuning help? (final test set)

The same LightGBM classifier (`LGBM_PARAMS_CHOICE`: —) trained on embeddings from the **pretrained** MMELON model (no fine-tuning) and from the **fine-tuned** MMELON model (the single final model above), each scored as a fold ensemble. The only difference between each pair of rows is whether the encoders were fine-tuned on the train set.

| Embedding | Encoder | hit@200 | hit@100 | hit@50 | AUROC | PR AUC |
|---|---|---|---|---|---|---|
| Fused (MMELON) | pretrained | | | | | |
| Fused (MMELON) | fine-tuned | | | | | |
| Text | pretrained | | | | | |
| Text | fine-tuned | | | | | |
| Image | pretrained | | | | | |
| Image | fine-tuned | | | | | |
| Graph | pretrained | | | | | |
| Graph | fine-tuned | | | | | |

Fine-tuning adds value where the fine-tuned row beats the pretrained one. For the end-to-end fine-tuned models themselves (encoder and head trained together), see the final test table above.

## Observations

- Best model on the final test set, and by how much.
- Multi-view versus single views: does fusing the views help?
- Does fine-tuning help: pretrained versus fine-tuned embeddings with the same LightGBM.
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
