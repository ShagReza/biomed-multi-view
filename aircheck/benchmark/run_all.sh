#!/usr/bin/env bash
# Run every MMELON benchmark method on one experiment, then write the report.
#
# An experiment is one dataset with one set of conditions. Everything it produces stays in its own folder:
#
#   aircheck/output/<experiment>/
#   ├── experiment.json      the train/test files and fold settings this experiment was made with
#   ├── folds/               the fold files (made here if they are not there yet)
#   ├── <run name>/          predictions.csv, run.json, progress.log, models/
#   ├── embeddings/          shared by the LightGBM runs of the same encoder
#   ├── report.md            every run in this experiment, with report_summary.csv next to it
#   └── run_all.log
#
# Usage, from the repo root, with the MMELON environment active:
#   aircheck/benchmark/run_all.sh --experiment <name> --train <train file> --test <test file> [options]
#
# Example:
#   aircheck/benchmark/run_all.sh --experiment wdr91-cluster \
#       --train ~/Datasets/wdr91/train.parquet --test ~/Datasets/wdr91/test.parquet
#
# Options:
#   --experiment <name>   folder under --root; use a new name for other conditions (e.g. wdr91-stratified)
#   --train <file>        training data, for the folds (not needed once the folds exist, or with --folds)
#   --test <file>         the final test set
#   --cv-method <name>    cluster (default), stratified or random
#   --n-folds <n>         number of folds (core's default when left out)
#   --folds <npz>         use these fold files instead of making them in the experiment
#   --root <dir>          where experiments live (default: aircheck/output)
#   --report <path>       report file (default: <experiment>/report.md)
#   --smiles-col, --label-col   column names (default SMILES, LABEL)
#   --only a,b / --skip a,b / --list    pick runs; --only none just rewrites the report
#   anything else         goes to every run.py call, e.g. --num-workers 2 or --epochs 3
#
# A run that already has run.json is skipped, so starting this again after an interruption (or after the VM
# shut down) continues where it stopped. Delete a run's folder to redo it.
#
# Long runs: nohup aircheck/benchmark/run_all.sh --experiment ... > /dev/null 2>&1 &
#            tail -f aircheck/output/<experiment>/run_all.log
set -uo pipefail

EXPERIMENT="" ROOT="" TRAIN="" TEST="" FOLDS="" CV_METHOD="cluster" N_FOLDS="" REPORT=""
SMILES_COL="SMILES" LABEL_COL="LABEL" ONLY="" SKIP="" LIST=0 EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --root)       ROOT="$2";       shift 2 ;;
        --train)      TRAIN="$2";      shift 2 ;;
        --test)       TEST="$2";       shift 2 ;;
        --folds)      FOLDS="$2";      shift 2 ;;
        --cv-method)  CV_METHOD="$2";  shift 2 ;;
        --n-folds)    N_FOLDS="$2";    shift 2 ;;
        --report)     REPORT="$2";     shift 2 ;;
        --smiles-col) SMILES_COL="$2"; shift 2 ;;
        --label-col)  LABEL_COL="$2";  shift 2 ;;
        --only)       ONLY=",$2,";     shift 2 ;;
        --skip)       SKIP=",$2,";     shift 2 ;;
        --list)       LIST=1;          shift ;;
        -h|--help) sed -n '2,40p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIRCHECK="$(cd "$HERE/.." && pwd)"

# One line per run: <run name>|<run.py options>. Fine-tuning comes first, so the fine-tuned embeddings can use
# the multi-view final model. Comment out a line to leave that method out.
RUNS=(
    "finetune_multiview|--method finetune --view multiview"
    "finetune_text|--method finetune --view text"
    "finetune_image|--method finetune --view image"
    "finetune_graph|--method finetune --view graph"
    "lgbm_pretrained_fused|--method embed_lgbm --view fused"
    "lgbm_pretrained_text|--method embed_lgbm --view text"
    "lgbm_pretrained_image|--method embed_lgbm --view image"
    "lgbm_pretrained_graph|--method embed_lgbm --view graph"
    "lgbm_finetuned_fused|--method embed_lgbm --view fused --encoder FINAL_CKPT"
    "lgbm_finetuned_text|--method embed_lgbm --view text --encoder FINAL_CKPT"
    "lgbm_finetuned_image|--method embed_lgbm --view image --encoder FINAL_CKPT"
    "lgbm_finetuned_graph|--method embed_lgbm --view graph --encoder FINAL_CKPT"
)
if [[ $LIST -eq 1 ]]; then
    printf '%s\n' "${RUNS[@]%%|*}"
    exit 0
fi

if [[ -z "$EXPERIMENT" || -z "$TEST" ]]; then
    echo "Usage: $0 --experiment <name> --train <train file> --test <test file> [options]" >&2
    echo "       $0 --help" >&2
    exit 2
fi
EXP_DIR="${ROOT:-$AIRCHECK/output}/$EXPERIMENT"
mkdir -p "$EXP_DIR"
LOG="$EXP_DIR/run_all.log"
REPORT="${REPORT:-$EXP_DIR/report.md}"
FINAL_CKPT="$EXP_DIR/finetune_multiview/models/final/best_model.ckpt"  # encoder for the fine-tuned embeddings

say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }
die() { say "$*"; exit 1; }

# The folds: made in the experiment the first time, then reused. --folds uses files from somewhere else.
if [[ -z "$FOLDS" ]]; then
    existing=("$EXP_DIR"/folds/*_cv_folds.npz)
    if [[ -f "${existing[0]}" ]]; then
        FOLDS="${existing[0]}"
        say "Folds: $FOLDS (already made)"
    else
        [[ -n "$TRAIN" ]] || die "This experiment has no folds yet: give --train (or --folds)"
        say "Making $CV_METHOD folds from $TRAIN"
        aircheck-bench make-folds --train "$TRAIN" --out "$EXP_DIR/folds" --method "$CV_METHOD" \
            --smiles-col "$SMILES_COL" --label-col "$LABEL_COL" ${N_FOLDS:+--n-folds "$N_FOLDS"} \
            >> "$LOG" 2>&1 || die "Making the folds failed; see $LOG"
        existing=("$EXP_DIR"/folds/*_cv_folds.npz)
        FOLDS="${existing[0]}"
        say "Folds: $FOLDS"
    fi
fi

# experiment.json: written once, then checked, so one name cannot end up holding results from other data
python - "$EXP_DIR/experiment.json" "$EXPERIMENT" "${TRAIN:-}" "$TEST" "$CV_METHOD" "$FOLDS" <<'PY' || die "Stopping."
import json, sys
from datetime import datetime
from pathlib import Path
path, experiment, train, test, cv_method, folds = Path(sys.argv[1]), *sys.argv[2:]
resolve = lambda p: str(Path(p).expanduser().resolve()) if p else None
now = {"experiment": experiment, "train": resolve(train), "test": resolve(test),
       "cv_method": cv_method, "folds": resolve(folds)}
if path.exists():
    before = json.loads(path.read_text())
    differ = {k: (before.get(k), now[k]) for k in ("train", "test", "folds")
              if now[k] is not None and before.get(k) is not None and before[k] != now[k]}
    if differ:
        print(f"{path} was made with other data; use another --experiment name, or delete that folder:")
        for key, (was, isnow) in differ.items():
            print(f"  {key}: was {was}\n      now {isnow}")
        sys.exit(1)
    now = {**before, **{k: v for k, v in now.items() if v is not None}}
else:
    now["created"] = datetime.now().isoformat(timespec="seconds")
path.write_text(json.dumps(now, indent=2) + "\n")
PY

say "Starting: ${#RUNS[@]} runs in $EXP_DIR (test $TEST)"
started=$(date +%s)
for entry in "${RUNS[@]}"; do
    name="${entry%%|*}"
    read -r -a options <<< "${entry#*|}"
    run_dir="$EXP_DIR/$name"

    if [[ -n "$ONLY" && "$ONLY" != *",$name,"* ]] || [[ -n "$SKIP" && "$SKIP" == *",$name,"* ]]; then
        continue
    fi
    if [[ -f "$run_dir/run.json" ]]; then
        say "$name: already done, skipped"
        continue
    fi
    if [[ "${options[*]}" == *FINAL_CKPT* ]]; then
        if [[ ! -f "$FINAL_CKPT" ]]; then
            say "$name: skipped, no fine-tuned encoder yet ($FINAL_CKPT)"
            continue
        fi
        options=("${options[@]/FINAL_CKPT/$FINAL_CKPT}")
    fi

    say "$name: run.py ${options[*]} ${EXTRA[*]}"
    if ! python "$HERE/run.py" "${options[@]}" --folds "$FOLDS" --test "$TEST" --out "$run_dir" \
            --smiles-col "$SMILES_COL" --label-col "$LABEL_COL" "${EXTRA[@]}" >> "$LOG" 2>&1; then
        say "$name FAILED; see $LOG. Nothing after it was run; fix it and start this script again."
        exit 1
    fi
    say "$name done"
done

if ! compgen -G "$EXP_DIR/*/run.json" > /dev/null; then
    say "No finished run in $EXP_DIR yet, so there is nothing to report."
    exit 0
fi
FOLDS_CSV="${FOLDS/_folds.npz/_data.csv}"
say "Writing the report for every run in $EXP_DIR"
if ! aircheck-bench report --results "$EXP_DIR" --out "$REPORT" --title "$EXPERIMENT" \
        --folds "$FOLDS" --train "$FOLDS_CSV" --test "$TEST" \
        --smiles-col "$SMILES_COL" --label-col "$LABEL_COL" >> "$LOG" 2>&1; then
    die "The report failed; see $LOG"
fi
say "All done in $((($(date +%s) - started) / 60)) min. Report: $REPORT"
say "Table: ${REPORT%.md}_summary.csv"
