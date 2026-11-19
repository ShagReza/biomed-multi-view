# Archive: to be deleted

The AIRCHECK notebooks as they were before the benchmark code. **Nothing here is used any more** and the whole
folder will be removed once a full benchmark run on a GPU has replaced the example results.

It is kept for now so the old way can be compared with the new one, and so nothing is lost by accident.

| Here | Replaced by |
|---|---|
| `notebooks/` (13 notebooks, `run_recipe.py`, `run_notebook.sh`, `stratified_cv.py`) | [../benchmark/](../benchmark/): `run_all.sh`, `run.py`, `methods/` and 5 thin notebooks |
| `report/report-<dataset>.md` | `output/<experiment>/report.md`, written by `aircheck-bench report` |
| `USAGE_RECIPE.md` | [../benchmark/README.md](../benchmark/README.md) |

The notebooks still run if you open them from `notebooks/`. Two things to know:

- They write to `../output/<DATASET>/` in the old layout, next to the benchmark's `output/<experiment>/`.
- Their results are **not** comparable with benchmark results in one report: the benchmark reads `run.json`
  files, which these do not write.

## Run the whole recipe

An alternative to running the notebooks one by one, for when you want every condition (multi-view, each single view,
and LightGBM on pretrained and fine-tuned embeddings). `run_recipe.py` runs every notebook of the recipe in order on
your data (the saved notebooks are not changed), in the background, and writes the report at the end:

```bash
python aircheck/archive-to-be-deleted/notebooks/run_recipe.py --dataset wdr91 --train <train file> --test <test file> --dry-run   # check first
python aircheck/archive-to-be-deleted/notebooks/run_recipe.py --dataset wdr91 --train <train file> --test <test file>
```

- Outputs go to `output/<DATASET>/`; progress is in `output/<DATASET>/recipe_runs/<time>/recipe.log`.
- Columns default to `SMILES` and `LABEL`; change them with `--smiles-col` and `--label-col`.
- If it stops (for example the VM shuts down), continue with `--resume output/<DATASET>/recipe_runs/<time>`.
- With no options it runs on the sample data.

To see what each step produces without running anything, look at `sample_output/sample-data/`: the outputs of the full recipe on the sample data and the resulting [report](report/report-sample-data.md). Model checkpoints, embedding files and the executed notebook copies (`recipe_runs/`) are left out to keep it small.

## Notebooks and scripts

All in `notebooks/`, listed in the order of the [usage recipe](USAGE_RECIPE.md). CV = cross-validation; "test" = the final test set.

| Step | File | What it does |
|---|---|---|
| Folds | `cluster_cv_folds.ipynb` | Cluster-held-out CV folds from the train file (same method as `ml-pipeline-aircheck`) |
| | `stratified_cv.py` | Alternative: stratified k-fold folds |
| MMELON | `MMELON_finetuning_cv.ipynb` | CV fine-tuning of the multi-view model; picks the best epoch |
| | `MMELON_inference_cv.ipynb` | Scores the test set with the fold models (ensemble) |
| | `MMELON_finetuning.ipynb` | One model on a train file, best epoch picked on a second file |
| | `MMELON_inference.ipynb` | Scores a file with that single model |
| Single view | `MMELON_singleview_*.ipynb` (4) | The same four steps for one encoder: `VIEW = "text"`, `"image"` or `"graph"` |
| LightGBM | `MMELON_embedding_extraction.ipynb` | Text, image, graph and fused embeddings (pretrained or fine-tuned model) |
| | `MMELON_embedding_lgbm_cv.ipynb` | CV LightGBM on one embedding view |
| | `MMELON_embedding_lgbm_inference.ipynb` | Scores test embeddings with the LightGBM fold models |
| Report | `collect_results.ipynb` | Gathers every result for one `DATASET` into `report/report-<DATASET>.md` |
| Tools | `run_recipe.py` | Runs the whole recipe in the background and writes the report |
| | `run_notebook.sh` | Runs chosen notebooks in the background (survives a VS Code disconnect) |

The best epoch is chosen by `SELECT_METRICS` (hit@200, then hit@100, hit@50 and AUROC to break ties). Inference
notebooks write `predictions.csv` (every molecule with its score and rank) and, with labels, the metrics.
`MMELON_finetuning.ipynb` picks its epoch on its second file, so give it a validation file, not the final test set,
for an unbiased test score.
