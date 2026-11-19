# Benchmark notebooks

The same work as `run_all.sh`, one step at a time, for exploring and explaining. They **define no model code**:
every cell calls [../methods/](../methods/), [../run.py](../run.py) or aircheck-benchmark-core, so a notebook
and a benchmark run can never drift apart.

| Notebook | Step | Replaces, in [../../notebooks/](../../notebooks/) |
|---|---|---|
| [01_folds.ipynb](01_folds.ipynb) | the experiment's cross-validation folds | `cluster_cv_folds.ipynb`, `stratified_cv.py` |
| [02_finetune.ipynb](02_finetune.ipynb) | fine-tune MMELON (any view), score the test set | `MMELON*_finetuning*`, `MMELON*_inference*` |
| [03_embeddings_lgbm.ipynb](03_embeddings_lgbm.ipynb) | embeddings, then LightGBM | `MMELON_embedding_extraction`, `MMELON_embedding_lgbm_*` |
| [04_inference.ipynb](04_inference.ipynb) | score your own molecules with a trained run | the "score my own molecules" part of `*_inference*` |
| [05_report.ipynb](05_report.ipynb) | the report for the whole experiment | `collect_results.ipynb` |

**Before you start:** the `biomed-multiview` kernel with aircheck-benchmark-core installed — see
[../README.md](../README.md).

**How to use them:** every notebook starts with a settings cell; the first setting is `EXPERIMENT`, the folder
under `aircheck/output/` that holds the folds, the runs and the report. Use the same name in all of them, run
`01_folds` first, then any of `02`/`03`, and `05` at the end. Run them from this folder.

Nothing here is needed for a benchmark run: `run_all.sh` does the same steps without opening a notebook.
