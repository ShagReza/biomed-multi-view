"""Run one MMELON benchmark method on one dataset and write its predictions for aircheck-benchmark-core.

    python aircheck/benchmark/run.py --method finetune --view multiview \\
        --folds folds/train_cluster_cv_folds.npz --test test.parquet --out results/wdr91/mmelon/finetune_multiview
    python aircheck/benchmark/run.py --method embed_lgbm --view fused --folds ... --test ... --out ...

Methods (see methods/*.py):
  finetune     fine-tune MMELON: --view multiview (default), text, image or graph
  embed_lgbm   LightGBM on MMELON embeddings: --view fused (default), text, image or graph

The training molecules are the folds CSV next to --folds (<name>_cv_data.csv, from `aircheck-bench make-folds`).
Output in --out: predictions.csv and run.json (read by `aircheck-bench score` and `aircheck-bench report`),
progress.log, and models/ (the trained models).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from aircheck_benchmark_core import __version__ as core_version
from aircheck_benchmark_core.folds import load_folds, read_columns
from aircheck_benchmark_core.metrics import score_predictions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import common  # noqa: E402

METHODS = ("finetune", "embed_lgbm")
NAMES = {"finetune": "MMELON fine-tuned ({view})", "embed_lgbm": "LightGBM on MMELON {view} embeddings ({encoder})"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--folds", required=True, help="folds .npz from aircheck-bench make-folds (CSV next to it)")
    p.add_argument("--test", required=True, help="test file (parquet or CSV) with SMILES and labels")
    p.add_argument("--out", required=True, help="run folder (must not hold an earlier run)")
    p.add_argument("--name", help="display name in the report")
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="LABEL")
    p.add_argument("--num-workers", type=int,
                   help=f"data-loading processes for fine-tuning (default {common.NUM_WORKERS}: all CPUs, up to 24); "
                        "fewer use less memory, e.g. 0-2 on a small VM")
    # Method settings: left out -> the method's default (DEFAULTS in methods/<method>.py)
    p.add_argument("--view", help="finetune: multiview|text|image|graph; embed_lgbm: fused|text|image|graph")
    p.add_argument("--seed", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--no-balance", dest="balance_classes", action="store_const", const=False,
                   help="unweighted loss (default: balance the classes)")
    g = p.add_argument_group("finetune")
    g.add_argument("--epochs", type=int, help="maximum epochs (default 50)")
    g.add_argument("--patience", type=int, help="stop after this many epochs without improvement (default 5)")
    g.add_argument("--learning-rate", type=float)
    g.add_argument("--weight-decay", type=float)
    g.add_argument("--select-metrics", nargs="+", help="epoch selection, in priority order")
    g.add_argument("--no-final", dest="final", action="store_const", const=False,
                   help="skip the final model on all training rows (no 'test' split, only 'test_ensemble')")
    g = p.add_argument_group("embed_lgbm")
    g.add_argument("--encoder", help="fine-tuned best_model.ckpt to embed with (default: pretrained base model)")
    g.add_argument("--lgbm-params", choices=("suggested", "default"))
    return p.parse_args(argv)


def load_data(args) -> tuple:
    folds_npz = Path(args.folds).expanduser()
    data_csv = Path(str(folds_npz).replace("_folds.npz", "_data.csv"))
    assert data_csv.exists(), f"No folds CSV next to {folds_npz.name}: expected {data_csv}"
    data = pd.read_csv(data_csv, usecols=[args.smiles_col, args.label_col])
    data = data.rename(columns={args.smiles_col: "smiles", args.label_col: "label"})[["smiles", "label"]]
    data["label"] = data["label"].astype(int)
    folds = load_folds(folds_npz)
    assert max(int(i.max()) for f in folds for i in f) < len(data), f"Fold indices do not fit {data_csv.name}"

    test = read_columns(Path(args.test).expanduser(), [args.smiles_col, args.label_col])
    test = test.rename(columns={args.smiles_col: "smiles", args.label_col: "label"})[["smiles", "label"]]
    assert test["label"].notna().all(), "The test file needs a label for every row"
    test["label"] = test["label"].astype(int)
    return data, folds, test, folds_npz


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    assert not (out / "predictions.csv").exists(), f"{out} already holds a run; choose another --out or delete it"
    out.mkdir(parents=True, exist_ok=True)
    log = common.Log(out / "progress.log")
    if args.num_workers is not None:
        common.NUM_WORKERS = args.num_workers

    data, folds, test, folds_npz = load_data(args)
    log(f"{args.method}: {len(data):,} training rows ({int(data['label'].sum()):,} actives), {len(folds)} folds; "
        f"{len(test):,} test rows; device {common.DEVICE}")

    if args.method == "finetune":
        from methods import finetune as method
    else:
        from methods import embed_lgbm as method
    settings = {k: getattr(args, k) for k in ("view", "seed", "batch_size", "balance_classes", "epochs", "patience",
                                              "learning_rate", "weight_decay", "select_metrics", "final",
                                              "encoder", "lgbm_params")}
    settings = {k: v for k, v in settings.items() if k in method.DEFAULTS}
    predictions, extras, used = method.run(data, folds, test, settings, out, log)

    predictions.to_csv(out / "predictions.csv", index=False)
    encoder = "fine-tuned" if used.get("encoder") else "pretrained"
    run_info = {
        "model": "mmelon",
        "method": args.method,
        "name": args.name or NAMES[args.method].format(view=used["view"], encoder=encoder),
        "settings": used,
        "report_extras": extras,
        "hardware": common.hardware(),
        "fork_commit": common.fork_commit(),
        "core_version": core_version,
        "folds": str(folds_npz.resolve()),
        "test": str(Path(args.test).expanduser().resolve()),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")

    table = score_predictions(predictions, folds_npz=folds_npz)  # also checks the fold rows
    cols = ["split", "n", "n_pos", "hit@200", "hit@100", "hit@50", "rocauc", "pr_auc"]
    log("Done. Scores (computed by aircheck-benchmark-core):\n" + table[cols].round(4).to_string(index=False))
    log(f"Saved {out / 'predictions.csv'} and {out / 'run.json'}")


if __name__ == "__main__":
    main()
