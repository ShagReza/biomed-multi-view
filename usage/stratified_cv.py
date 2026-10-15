"""Generate stratified k-fold train/test indices for a CSV file.

Uses scikit-learn's StratifiedKFold. Indices are 0-based row positions in the
input CSV (header excluded), exactly as returned by StratifiedKFold.split().

Outputs (written to --output-dir, default: the input file's folder):
  cv_<k>fold_indices.npz   arrays fold<i>_train and fold<i>_test for each fold
  cv_<k>fold_assignment.csv  input rows plus a "fold" column: the fold in
                             which each row is in the test set

Example:
  python stratified_cv.py sample.csv 5
  python stratified_cv.py sample.csv 10 --label-col label --seed 0

Load the indices:
  idx = np.load("cv_5fold_indices.npz")
  train_idx, test_idx = idx["fold0_train"], idx["fold0_test"]
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="CSV file to split")
    parser.add_argument("n_folds", type=int, help="number of folds (k)")
    parser.add_argument("--label-col", default="label", help="column to stratify on (default: label)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for shuffling (default: 42)")
    parser.add_argument("--output-dir", default=None, help="where to save outputs (default: input file's folder)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.label_col not in df.columns:
        parser.error(f"label column '{args.label_col}' not found; columns are {list(df.columns)}")

    y = df[args.label_col].to_numpy()
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input_csv))
    os.makedirs(output_dir, exist_ok=True)

    arrays = {}
    fold_of_row = np.full(len(df), -1)
    for i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        arrays[f"fold{i}_train"] = train_idx
        arrays[f"fold{i}_test"] = test_idx
        fold_of_row[test_idx] = i
        print(
            f"fold {i}: train={len(train_idx)} test={len(test_idx)} "
            f"test label counts={pd.Series(y[test_idx]).value_counts().sort_index().to_dict()}"
        )

    npz_path = os.path.join(output_dir, f"cv_{args.n_folds}fold_indices.npz")
    np.savez(npz_path, **arrays)

    assignment_path = os.path.join(output_dir, f"cv_{args.n_folds}fold_assignment.csv")
    df.assign(fold=fold_of_row).to_csv(assignment_path, index=True, index_label="row_index")

    print(f"saved {npz_path}")
    print(f"saved {assignment_path}")


if __name__ == "__main__":
    main()
