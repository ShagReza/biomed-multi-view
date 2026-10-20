#!/usr/bin/env bash
# Run notebooks in the background, one after another, so they keep running if VS Code or the browser disconnects.
# (They still stop if the VM itself shuts down.)
#
# Usage, from any folder, with the notebooks' environment active (e.g. `conda activate biomed-multiview`):
#   aircheck/notebooks/run_notebook.sh MMELON_finetuning_cv.ipynb MMELON_inference_cv.ipynb
#
# Each notebook runs from aircheck/notebooks/ (so its default paths work). If one fails, the later ones are skipped.
# The executed copy of each notebook (with all cell outputs) and a log are saved in aircheck/output/notebook_runs/.
# Follow progress with:  tail -f aircheck/output/notebook_runs/<timestamp>_run.log

set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,10p' "$0"; exit 1; }

NB_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$NB_DIR/../output/notebook_runs"
mkdir -p "$RUNS_DIR"
RUNS_DIR="$(cd "$RUNS_DIR" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUNS_DIR/${STAMP}_run.log"

for nb in "$@"; do
    [ -f "$NB_DIR/$(basename "$nb")" ] || { echo "Notebook not found: $NB_DIR/$(basename "$nb")"; exit 1; }
done
python -c "import jupyter_core, nbconvert" 2>/dev/null || { echo "Activate the notebooks' environment first (jupyter nbconvert not found)"; exit 1; }

nohup bash -c '
    cd "$1"; runs="$2"; stamp="$3"; shift 3
    for nb in "$@"; do
        name="$(basename "$nb" .ipynb)"
        echo "$(date "+%F %T")  START  $name"
        if python -m jupyter nbconvert --to notebook --execute "$name.ipynb" \
               --output "$runs/${stamp}_${name}.ipynb" --ExecutePreprocessor.timeout=-1; then
            echo "$(date "+%F %T")  DONE   $name  -> $runs/${stamp}_${name}.ipynb"
        else
            echo "$(date "+%F %T")  FAILED $name  (later notebooks skipped; see the error above)"
            exit 1
        fi
    done
    echo "$(date "+%F %T")  ALL DONE"
' _ "$NB_DIR" "$RUNS_DIR" "$STAMP" "$@" > "$LOG" 2>&1 &

echo "Started in the background (process $!). Notebooks: $*"
echo "Follow progress:  tail -f $LOG"
echo "Stop it:          kill $!"
