"""Phase 1: Generate Fixed Grouped 5-Fold Partition.
Splits data/train.csv into 5 folds using StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
grouped on prompt_id. Saves to folds/folds.parquet.
Refuses to overwrite if folds/folds.parquet already exists.
"""
from pathlib import Path
import json
import sys
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def main():
    folds_path = Path("folds/folds.parquet")
    if folds_path.exists():
        print(f"ERROR: {folds_path} already exists! Refusing to overwrite frozen folds.")
        return 1

    train_path = Path("data/train.csv")
    labels_path = Path("configs/labels.json")

    if not train_path.exists():
        print(f"ERROR: {train_path} does not exist.")
        return 1
    if not labels_path.exists():
        print(f"ERROR: {labels_path} does not exist.")
        return 1

    train = pd.read_csv(train_path)
    with open(labels_path, "r") as f:
        labels = json.load(f)

    print(f"Loaded train.csv: {len(train)} rows")
    print(f"Unique prompt_ids: {train['prompt_id'].nunique()}")
    print(f"Unique models: {train['model'].nunique()}")

    # StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    folds = pd.Series(index=train.index, dtype=int)
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(train, train['model'], groups=train['prompt_id'])):
        folds.iloc[val_idx] = fold_idx

    train['fold'] = folds
    train['group'] = train['prompt_id']

    # Final schema: id, fold, group, model
    out_df = train[['id', 'fold', 'group', 'model']].copy()

    # Integrity assertions
    assert len(out_df) == len(train), "Row count mismatch"
    assert out_df['fold'].nunique() == 5, "Folds count mismatch"
    assert set(out_df['model'].unique()) == set(labels), "Model label mismatch"
    
    # Assert zero group leakage
    group_folds = out_df.groupby('group')['fold'].nunique()
    leaked = group_folds[group_folds > 1]
    assert len(leaked) == 0, f"Group leakage detected in {len(leaked)} groups!"

    # Save
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(folds_path, index=False)
    print(f"\nSuccessfully created and saved {folds_path}")

    # Summary diagnostics
    print("\nFolds distribution summary:")
    print(f"Total rows: {len(out_df)}")
    print(f"Rows per fold:\n{out_df['fold'].value_counts().sort_index()}")
    print(f"Groups per fold:\n{out_df.groupby('fold')['group'].nunique()}")
    
    pivot = out_df.groupby(['fold', 'model']).size().unstack(fill_value=0)
    print("\nPer-fold class distribution (rows=fold, cols=model):")
    print(pivot)

    return 0

if __name__ == "__main__":
    sys.exit(main())
