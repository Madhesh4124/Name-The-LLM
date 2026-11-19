"""Validate an OOF probability file and print balanced-accuracy diagnostics.

Usage: python validate_oof.py oof.npy folds.parquet labels.json [--save-dir DIR]
folds.parquet must have columns: id, fold, group, model.

When --save-dir is given, writes confusion_matrix.csv and metrics.json there.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score


def main(oof_path: str, folds_path: str, labels_path: str,
         save_dir: str | None = None) -> int:
    labels = json.load(open(labels_path))
    idx = {l: i for i, l in enumerate(labels)}
    oof = np.load(oof_path)
    folds = pd.read_parquet(folds_path)

    errors: list[str] = []
    if oof.shape != (len(folds), len(labels)):
        errors.append(f"shape {oof.shape} != {(len(folds), len(labels))}")
    elif not np.isfinite(oof).all():
        errors.append("non-finite values")
    elif not np.allclose(oof.sum(1), 1.0, atol=1e-3):
        errors.append("rows do not sum to 1 (store probabilities, not logits)")
    if errors:
        print("FAIL:", *errors, sep="\n  ")
        return 1

    y = folds["model"].map(idx).to_numpy()
    prior = np.bincount(y, minlength=len(labels)) / len(y)
    pred_argmax = oof.argmax(1)
    pred_div = (oof / np.clip(prior, 1e-12, None)).argmax(1)

    ba_argmax = balanced_accuracy_score(y, pred_argmax)
    ba_div = balanced_accuracy_score(y, pred_div)
    recalls = recall_score(y, pred_div, average=None, labels=range(len(labels)))
    cm = confusion_matrix(y, pred_div, labels=range(len(labels)))

    print(f"balanced_acc argmax        : {ba_argmax:.4f}")
    print(f"balanced_acc prior-divided : {ba_div:.4f}")
    for name, r in sorted(zip(labels, recalls), key=lambda t: t[1]):
        print(f"  {name:28s} recall={r:.3f}")

    # Confusion matrix (compact)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion matrix (prior-divided preds):")
    print(cm_df.to_string())

    # Optionally persist artifacts
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        cm_df.to_csv(out / "confusion_matrix.csv")
        metrics = {
            "balanced_accuracy_argmax": round(float(ba_argmax), 5),
            "balanced_accuracy_prior_divided": round(float(ba_div), 5),
            "per_class_recall": {n: round(float(r), 5) for n, r in zip(labels, recalls)},
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"\nSaved confusion_matrix.csv and metrics.json to {out}")

    print("PASS")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    save = None
    if "--save-dir" in argv:
        i = argv.index("--save-dir")
        save = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    sys.exit(main(*argv[:3], save_dir=save))
