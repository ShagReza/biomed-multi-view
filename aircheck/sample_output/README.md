# sample_output: what a benchmark experiment produces

A **mock-up** of the folder [benchmark/run_all.sh](../benchmark/README.md) writes, so the layout can be seen
without running anything. **Every number in here is made up**, and the large files (models, embeddings, the
folds `.npz`) are replaced by `*.placeholder` notes.

A real experiment lives in `aircheck/output/<experiment>/`, which is not tracked by git.

```
sample_output/wdr91-cluster/
├── experiment.json          the train/test files and fold settings of this experiment
├── folds/                   the fold files, made once and shared by every run and every model
├── finetune_multiview/      one run: predictions.csv, run.json, progress.log, models/
├── lgbm_pretrained_fused/   another run, plus config.json for scoring new molecules
├── embeddings/              shared by the LightGBM runs of one encoder
├── report.md                every run in the experiment, scored by aircheck-benchmark-core
├── report_summary.csv       the same numbers as a table, one row per run and split
└── run_all.log              what the recipe did, with times
```

Only two runs are shown; the other three views (`finetune_text`, `finetune_image`, `finetune_graph`,
`lgbm_pretrained_text` and so on) and the runs on fine-tuned embeddings have the same shape.

**The two files the benchmark reads** are in every run folder:

| File | What it holds |
|---|---|
| `predictions.csv` | one row per scored molecule: `split` (`fold0`…, `test_ensemble`, `test`), `row`, `SMILES`, `LABEL`, `score`. An empty score means the model could not read that SMILES |
| `run.json` | model, method, display name, settings, hardware, the fork's commit and the core version |

Everything else — the metrics, the report, the comparison between models — is computed from those two by
[aircheck-benchmark-core](https://github.com/ShagReza/aircheck-benchmark-core), the same way for every model.
