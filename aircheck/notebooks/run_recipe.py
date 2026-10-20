"""Run the whole usage recipe on one dataset, in the background, and write the report at the end.

Each step runs a notebook from this folder with a few settings changed in its first cells (the saved notebooks are not
modified). Steps run one after another; if one fails, the run stops. Executed copies (with all cell outputs) and a log
go to aircheck/output/<DATASET>/recipe_runs/<time>/.

Usage, from any folder, with the notebooks' environment active:
    python aircheck/notebooks/run_recipe.py                                  # sample data, in the background
    python aircheck/notebooks/run_recipe.py --dataset wdr91 \\
        --train ~/Datasets/wdr91/train.parquet --test ~/Datasets/wdr91/test.parquet
    python aircheck/notebooks/run_recipe.py --dry-run                        # show the plan and check it, run nothing

Options: --lgbm-params default|suggested, --smiles-col, --label-col, --resume <recipe_runs folder> (skip the steps
that finished; delete the folder of a step that stopped half-way first), --foreground (do not detach).

It keeps running if VS Code disconnects, but not if the VM shuts down.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent
OUTPUT = NB_DIR.parent / "output"
VIEWS = ["text", "image", "graph"]
LGBM_VIEWS = ["fused", "text", "image", "graph"]


def q(path) -> str:
    return f'Path("{Path(path).expanduser().resolve()}")'


def out(*parts) -> str:
    return 'Path("../output") / DATASET / ' + " / ".join(f'"{p}"' for p in parts)


def build_steps(a):
    """The recipe as (step name, notebook, settings) triples. Settings are Python source; callables are
    evaluated when the step starts (they read results of earlier steps)."""
    base = {"DATASET": f'"{a.dataset}"', "SMILES_COL": f'"{a.smiles_col}"', "LABEL_COL": f'"{a.label_col}"'}
    test = {"DATA_PATH": q(a.test)}
    ds = OUTPUT / a.dataset

    def best_epochs(cv_folder):  # train the final model for as many epochs as CV selected
        n = json.loads((ds / cv_folder / "selection.json").read_text())["best_epoch"] + 1
        return {"EPOCHS": str(n), "PATIENCE": str(n)}

    def latest_final(pattern):
        runs = sorted((ds / "finetuned").glob(pattern))
        assert runs, f"no finished run matching {pattern} in {ds / 'finetuned'}"
        return {"MODEL_PATH": out("finetuned", runs[-1].name)}

    s = [("folds", "cluster_cv_folds.ipynb", {"TRAIN_PATH": q(a.train)}),
         ("mmelon_cv", "MMELON_finetuning_cv.ipynb", {}),
         ("mmelon_cv_test", "MMELON_inference_cv.ipynb", test)]
    for v in VIEWS:
        s += [(f"singleview_cv_{v}", "MMELON_singleview_finetuning_cv.ipynb", {"VIEW": f'"{v}"'}),
              (f"singleview_cv_{v}_test", "MMELON_singleview_inference_cv.ipynb",
               {**test, "CV_RUN_DIR": out(f"singleview-cv_{v}")})]
    split = {"TRAIN_PATH": q(a.train), "TEST_PATH": q(a.test)}
    s += [("mmelon_final", "MMELON_finetuning.ipynb", lambda: {**split, **best_epochs("mmelon-cv")}),
          ("mmelon_final_test", "MMELON_inference.ipynb", lambda: {**test, **latest_final("mmelon-final_[0-9]*")})]
    for v in VIEWS:
        s += [(f"singleview_final_{v}", "MMELON_singleview_finetuning.ipynb",
               lambda v=v: {**split, "VIEW": f'"{v}"', **best_epochs(f"singleview-cv_{v}")}),
              (f"singleview_final_{v}_test", "MMELON_singleview_inference.ipynb",
               lambda v=v: {**test, **latest_final(f"singleview-final_{v}_[0-9]*")})]
    all_views = {"VIEWS": '["text", "image", "graph", "fused"]'}
    s += [("embeddings_train_pretrained", "MMELON_embedding_extraction.ipynb",
           {"EMBEDDING_NAME": '"train-pretrained"', "MODEL_PATH": "None", **all_views}),
          ("embeddings_test_pretrained", "MMELON_embedding_extraction.ipynb",
           {"EMBEDDING_NAME": '"test-pretrained"', "MODEL_PATH": "None", **test, **all_views})]
    lgbm = {"LGBM_PARAMS_CHOICE": f'"{a.lgbm_params}"'}
    for v in LGBM_VIEWS:
        s += [(f"lgbm_cv_{v}_pretrained", "MMELON_embedding_lgbm_cv.ipynb",
               {**lgbm, "VIEW": f'"{v}"', "RUN_NAME": f'"lgbm-{v}-pretrained"',
                "EMBEDDING_DIR": out("embeddings", "train-pretrained")})]
    final_model = lambda: {"MODEL_PATH": latest_final("mmelon-final_[0-9]*")["MODEL_PATH"]}
    s += [("embeddings_train_finetuned", "MMELON_embedding_extraction.ipynb",
           lambda: {"EMBEDDING_NAME": '"train-finetuned"', **final_model(), **all_views}),
          ("embeddings_test_finetuned", "MMELON_embedding_extraction.ipynb",
           lambda: {"EMBEDDING_NAME": '"test-finetuned"', **final_model(), **test, **all_views})]
    for v in LGBM_VIEWS:
        s += [(f"lgbm_cv_{v}_finetuned", "MMELON_embedding_lgbm_cv.ipynb",
               {**lgbm, "VIEW": f'"{v}"', "RUN_NAME": f'"lgbm-{v}-finetuned"',
                "EMBEDDING_DIR": out("embeddings", "train-finetuned")}),
              (f"lgbm_{v}_finetuned_test", "MMELON_embedding_lgbm_inference.ipynb",
               {**test, "LGBM_RUN_DIR": out("lgbm", f"lgbm-{v}-finetuned"),
                "EMBEDDING_DIR": out("embeddings", "test-finetuned")})]
    s += [("report", "collect_results.ipynb", {"TRAIN_PATH": q(a.train), "TEST_PATH": q(a.test)})]
    return [(name, nb, settings, base) for name, nb, settings in s]


def apply_settings(nb, settings: dict, optional: dict = None):
    # Replace the first top-level `NAME = ...` line of each setting (the input cells come first).
    # `settings` must exist in the notebook; `optional` ones (e.g. column names) are set only where they exist.
    for group, required in ((settings, True), (optional or {}, False)):
        for name, value in group.items():
            pattern = re.compile(rf"^{re.escape(name)} = .*$", re.M)
            cell = next((c for c in nb.cells if c.cell_type == "code" and pattern.search(c.source)), None)
            if cell is None:
                if required:
                    raise KeyError(f"setting {name} not found")
                continue
            cell.source = pattern.sub(lambda _: f"{name} = {value}", cell.source, count=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="sample-data")
    p.add_argument("--train", default=str(NB_DIR.parent / "sample_data" / "train.parquet"))
    p.add_argument("--test", default=str(NB_DIR.parent / "sample_data" / "test.parquet"))
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="LABEL")
    p.add_argument("--lgbm-params", default="default", choices=["default", "suggested"])
    p.add_argument("--resume", default=None, help="a previous recipe_runs/<time> folder: skip its finished steps")
    p.add_argument("--dry-run", action="store_true", help="show the plan and check every setting exists")
    p.add_argument("--foreground", action="store_true", help="run in this terminal instead of the background")
    a = p.parse_args()
    for f in (a.train, a.test):
        assert Path(f).expanduser().exists(), f"File not found: {f}"
    steps = build_steps(a)

    if a.dry_run:
        for i, (name, nb_file, settings, base) in enumerate(steps, 1):
            nb = nbf.read(NB_DIR / nb_file, 4)
            static = settings if isinstance(settings, dict) else {}
            apply_settings(nb, {"DATASET": base["DATASET"], **static}, optional=base)  # raises if a setting is missing
            shown = ", ".join(f"{k}={v}" for k, v in static.items()) or ("(settings from earlier steps)" if callable(settings) else "(defaults)")
            print(f"{i:2d}. {name:30s} {nb_file:42s} {shown}")
        print(f"\n{len(steps)} steps; all settings found in their notebooks. Outputs: {OUTPUT / a.dataset}")
        return

    # Absolute paths: the background process runs from the notebooks folder
    run_dir = (Path(a.resume).expanduser().resolve() if a.resume
               else OUTPUT / a.dataset / "recipe_runs" / time.strftime("%Y%m%d_%H%M%S"))
    if a.resume:
        assert (run_dir / "recipe.log").exists(), f"Not a recipe run folder: {run_dir}"
    a.train, a.test = (str(Path(f).expanduser().resolve()) for f in (a.train, a.test))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "recipe.log"
    if not a.foreground:  # detach, so the run survives a closed terminal or VS Code disconnect
        args = ["--dataset", a.dataset, "--train", a.train, "--test", a.test, "--smiles-col", a.smiles_col,
                "--label-col", a.label_col, "--lgbm-params", a.lgbm_params, "--resume", str(run_dir), "--foreground"]
        with open(log_path, "a") as log:
            proc = subprocess.Popen([sys.executable, __file__, *args], stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, cwd=NB_DIR)
        print(f"Started the recipe in the background (process {proc.pid}), {len(steps)} steps.")
        print(f"Follow progress:  tail -f {log_path}")
        print(f"Stop it:          kill {proc.pid}")
        return

    done_file = run_dir / "steps_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    say = lambda msg: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)
    say(f"Recipe on DATASET={a.dataset}: {len(steps)} steps ({len(done)} already done)")
    start_all = time.time()
    for i, (name, nb_file, settings, base) in enumerate(steps, 1):
        if name in done:
            continue
        nb = nbf.read(NB_DIR / nb_file, 4)
        apply_settings(nb, {"DATASET": base["DATASET"], **(settings() if callable(settings) else settings)},
                       optional=base)
        # The notebook must run from this folder (its paths are relative to it): run a temporary copy here,
        # and save the executed copy in the run folder
        tmp = NB_DIR / f".recipe_{i:02d}_{name}.ipynb"
        src = run_dir / f"{i:02d}_{name}.ipynb"
        nbf.write(nb, tmp)
        say(f"[{i}/{len(steps)}] START {name} ({nb_file})")
        t = time.time()
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", tmp.name,
                            "--output-dir", str(run_dir), "--output", src.name, "--ExecutePreprocessor.timeout=-1"],
                           cwd=NB_DIR, capture_output=True, text=True)
        tmp.unlink(missing_ok=True)
        if r.returncode != 0:
            say(f"[{i}/{len(steps)}] FAILED {name}; the executed copy shows the error: {src}")
            print(r.stderr[-3000:], flush=True)
            sys.exit(1)
        with open(done_file, "a") as f:
            f.write(name + "\n")
        say(f"[{i}/{len(steps)}] DONE  {name} in {(time.time() - t) / 60:.1f} min")
    say(f"ALL DONE in {(time.time() - start_all) / 3600:.1f} h. Report: {NB_DIR.parent / 'report' / f'report-{a.dataset}.md'}")


if __name__ == "__main__":
    main()
