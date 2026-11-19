"""Phase 1: Shuffled-Label Leakage Verification Test.
Permutes target labels randomly across the dataset and trains a classifier
strictly within each fold to verify that CV balanced accuracy collapses to chance (~1/18 = 0.0556).
"""
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

def main():
    print("=" * 60)
    print("PHASE 1: SHUFFLED-LABEL LEAKAGE VERIFICATION TEST")
    print("=" * 60)

    train = pd.read_csv("data/train.csv")
    folds_df = pd.read_parquet("folds/folds.parquet")
    with open("configs/labels.json") as f:
        labels = json.load(f)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Verify alignment between train and folds
    assert (train["id"] == folds_df["id"]).all(), "train and folds row IDs are not aligned!"
    train["fold"] = folds_df["fold"].astype(int)

    # Permute labels randomly across dataset with fixed seed
    np.random.seed(42)
    shuffled_models = np.random.permutation(train["model"].values)
    train["shuffled_model"] = shuffled_models
    y_shuffled = np.array([label_to_idx[m] for m in train["shuffled_model"]])

    print(f"Total samples: {len(train)}")
    print(f"Chance level: 1/18 = {1/18:.4f} (5.56%)")

    oof_preds = np.zeros(len(train), dtype=int)

    # 5-fold loop
    for f in range(5):
        train_mask = train["fold"] != f
        val_mask = train["fold"] == f

        X_tr_text = train.loc[train_mask, "text"]
        y_tr = y_shuffled[train_mask]

        X_va_text = train.loc[val_mask, "text"]
        y_va = y_shuffled[val_mask]

        # Vectorizer fit STRICTLY on training fold
        vec = TfidfVectorizer(max_features=5000, lowercase=True)
        X_tr = vec.fit_transform(X_tr_text)
        X_va = vec.transform(X_va_text)

        clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        clf.fit(X_tr, y_tr)

        preds = clf.predict(X_va)
        oof_preds[val_mask] = preds

        fold_ba = balanced_accuracy_score(y_va, preds)
        print(f"Fold {f}: Shuffled Balanced Accuracy = {fold_ba:.4f} ({fold_ba*100:.2f}%)")

    total_ba = balanced_accuracy_score(y_shuffled, oof_preds)
    print("\n" + "=" * 60)
    print(f"Overall Shuffled OOF Balanced Accuracy: {total_ba:.4f} ({total_ba*100:.2f}%)")
    print(f"Expected Chance Level:                  {1/18:.4f} ({100/18:.2f}%)")
    print(f"Absolute Difference from Chance:        {abs(total_ba - 1/18):.4f}")
    print("=" * 60)

    # Pass condition: must be near chance level (within 2 percentage points)
    assert abs(total_ba - 1/18) < 0.02, f"FAIL: Shuffled BA {total_ba:.4f} deviates significantly from chance {1/18:.4f}!"
    print("\nPASS: No leakage detected. Shuffled balanced accuracy is at chance level.")
    return 0

if __name__ == "__main__":
    main()
