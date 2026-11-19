"""Validate folds/folds.parquet for group leakage, class balance, and schema.

Usage: python check_folds.py folds.parquet labels.json
"""
import json
import sys

import pandas as pd


def main(folds_path: str, labels_path: str) -> int:
    labels = json.load(open(labels_path))
    df = pd.read_parquet(folds_path)
    errors: list[str] = []

    # --- Schema ---
    required = {"id", "fold", "group", "model"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")

    if errors:
        print("FAIL (schema):", *errors, sep="\n  ")
        return 1

    n_folds = df["fold"].nunique()
    print(f"rows={len(df)}  folds={n_folds}  groups={df['group'].nunique()}  "
          f"classes={df['model'].nunique()}")

    # --- No group leakage ---
    group_folds = df.groupby("group")["fold"].nunique()
    leaked = group_folds[group_folds > 1]
    if len(leaked) > 0:
        errors.append(f"{len(leaked)} groups appear in multiple folds: "
                      f"{list(leaked.head(10).index)}")

    # --- Per-fold class counts ---
    pivot = df.groupby(["fold", "model"]).size().unstack(fill_value=0)
    print("\nPer-fold class counts:")
    print(pivot.to_string())

    # --- Check all labels present ---
    missing_labels = set(labels) - set(df["model"].unique())
    if missing_labels:
        errors.append(f"labels not in data: {sorted(missing_labels)}")

    extra_labels = set(df["model"].unique()) - set(labels)
    if extra_labels:
        errors.append(f"labels in data but not in labels.json: {sorted(extra_labels)}")

    # --- Per-fold group counts ---
    fold_groups = df.groupby("fold")["group"].nunique()
    print("\nGroups per fold:")
    for f, g in fold_groups.items():
        print(f"  fold {f}: {g} groups")

    if errors:
        print("\nFAIL:", *errors, sep="\n  ")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
