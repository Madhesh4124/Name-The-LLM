"""Validate submission.parquet.

Usage: python validate_submission.py submission.parquet test.csv labels.json
Assumes test.csv has an 'id' column [verify against the competition data].
"""
import json
import sys

import pandas as pd


def main(sub_path: str, test_path: str, labels_path: str) -> int:
    sub = pd.read_parquet(sub_path)
    test = pd.read_csv(test_path)
    labels = set(json.load(open(labels_path)))

    errors = []
    if list(sub.columns) != ["id", "model"]:
        errors.append(f"columns are {list(sub.columns)}, expected ['id', 'model']")
    if len(sub) != len(test):
        errors.append(f"{len(sub)} rows, expected {len(test)}")
    if sub["id"].duplicated().any():
        errors.append("duplicate ids")
    if set(sub["id"]) != set(test["id"]):
        errors.append("ids do not match test.csv")
    if sub["model"].isna().any():
        errors.append("missing predictions")
    unknown = set(sub["model"].dropna()) - labels
    if unknown:
        errors.append(f"unknown labels: {sorted(unknown)}")

    counts = sub["model"].value_counts()
    print(counts.to_string())
    missing = labels - set(counts.index)
    if missing:
        print("WARNING: classes never predicted:", sorted(missing))
    if errors:
        print("FAIL:", *errors, sep="\n  ")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
