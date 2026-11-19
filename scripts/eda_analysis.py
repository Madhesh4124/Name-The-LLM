"""Phase 0: Exploratory Data Analysis & Data Audit Script.
Analyzes data/train.csv, data/test.csv, and checks formatting, length,
class distributions, prompt structures, and truncation.
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    print("=" * 60)
    print("PHASE 0: EXPLORATORY DATA ANALYSIS & DATA AUDIT")
    print("=" * 60)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    with open("configs/labels.json", "r") as f:
        labels = json.load(f)

    print(f"\n1. DATASET SHAPES & NULL AUDIT")
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"Train columns: {train.columns.tolist()}")
    print(f"Test columns:  {test.columns.tolist()}")
    print(f"Train nulls:\n{train.isnull().sum().to_string()}")
    print(f"Test nulls:\n{test.isnull().sum().to_string()}")

    print(f"\n2. DUPLICATE AUDIT")
    print(f"Train unique IDs: {train['id'].nunique()} / {len(train)}")
    print(f"Train unique texts: {train['text'].nunique()} / {len(train)}")
    exact_dup_texts = train[train.duplicated(subset=['text'], keep=False)]
    print(f"Exact duplicate texts count: {len(exact_dup_texts)}")

    # Test unique
    print(f"Test unique IDs: {test['id'].nunique()} / {len(test)}")
    print(f"Test unique texts: {test['text'].nunique()} / {len(test)}")

    print(f"\n3. TARGET CLASS DISTRIBUTION")
    class_counts = train["model"].value_counts()
    print("Class counts:")
    for l in labels:
        count = class_counts.get(l, 0)
        pct = count / len(train) * 100
        print(f"  {l:25s}: {count:4d} ({pct:.2f}%)")

    # Class balance stats
    min_count = class_counts.min()
    max_count = class_counts.max()
    print(f"Min count: {min_count}, Max count: {max_count}, Ratio: {max_count/min_count:.4f}")

    print(f"\n4. PROMPT GROUPING AUDIT (TRAIN)")
    n_prompts = train["prompt_id"].nunique()
    print(f"Unique prompt_ids: {n_prompts}")
    prompt_counts = train["prompt_id"].value_counts()
    print(f"Prompts with 18 completions: {(prompt_counts == 18).sum()}")
    print(f"Prompts with <18 completions: {(prompt_counts < 18).sum()}")
    for pid, c in prompt_counts[prompt_counts < 18].items():
        present = set(train[train["prompt_id"] == pid]["model"])
        missing = set(labels) - present
        print(f"  Prompt {pid}: {c} completions. Missing: {missing}")

    # Check whether any model repeats within a single prompt
    group_model_counts = train.groupby(["prompt_id", "model"]).size()
    max_model_per_prompt = group_model_counts.max()
    print(f"Max occurrences of any model within a single prompt: {max_model_per_prompt}")
    assert max_model_per_prompt == 1, "A model repeats within a prompt!"

    print(f"\n5. PROMPT GROUPING AUDIT (TEST)")
    print(f"Test columns present: {test.columns.tolist()}")
    print("Does test.csv have prompt_id? ", "prompt_id" in test.columns)
    # Check pairwise character/word similarity in test
    print(f"Test row count: {len(test)}")
    for i in range(len(test)):
        words = set(re.findall(r"\w+", test.loc[i, "text"].lower()))
        print(f"Test row {i} word count: {len(words)}, preview: {test.loc[i, 'text'][:80]}...")

    print(f"\n6. PASSAGE LENGTH DISTRIBUTIONS (CHARS & WORDS)")
    train["char_len"] = train["text"].apply(len)
    train["word_len"] = train["text"].apply(lambda t: len(re.findall(r"\S+", t)))
    train["line_len"] = train["text"].apply(lambda t: len(t.splitlines()))

    length_stats = []
    for l in labels:
        sub = train[train["model"] == l]
        c = sub["char_len"]
        w = sub["word_len"]
        length_stats.append({
            "model": l,
            "char_min": c.min(),
            "char_q25": c.quantile(0.25),
            "char_median": c.median(),
            "char_q75": c.quantile(0.75),
            "char_max": c.max(),
            "char_mean": c.mean(),
            "char_std": c.std(),
            "char_iqr": c.quantile(0.75) - c.quantile(0.25),
            "word_min": w.min(),
            "word_q25": w.quantile(0.25),
            "word_median": w.median(),
            "word_q75": w.quantile(0.75),
            "word_max": w.max(),
            "word_mean": w.mean(),
            "word_std": w.std(),
            "word_iqr": w.quantile(0.75) - w.quantile(0.25),
        })
    df_len = pd.DataFrame(length_stats)
    print("\nLength Stats by Model (Word and Char summary):")
    print(df_len[["model", "word_min", "word_median", "word_mean", "word_max", "char_median", "char_mean"]].to_string())

    print("\nOverall Length Summary:")
    print(f"Chars - Min: {train['char_len'].min()}, Median: {train['char_len'].median()}, Mean: {train['char_len'].mean():.1f}, Max: {train['char_len'].max()}")
    print(f"Words - Min: {train['word_len'].min()}, Median: {train['word_len'].median()}, Mean: {train['word_len'].mean():.1f}, Max: {train['word_len'].max()}")
    print(f"Lines - Min: {train['line_len'].min()}, Median: {train['line_len'].median()}, Mean: {train['line_len'].mean():.1f}, Max: {train['line_len'].max()}")

    print(f"\n7. FORMATTING & TEXT ARTIFACT AUDIT")
    # Patterns to audit
    patterns = {
        "single_newline": r"(?<!\n)\n(?!\n)",
        "double_newline": r"\n\n",
        "tab": r"\t",
        "double_space": r"  ",
        "trailing_whitespace": r"(?m)[ \t]+$",
        "header_h1": r"(?m)^#\s+",
        "header_h2": r"(?m)^##\s+",
        "header_h3": r"(?m)^###\s+",
        "markdown_bold": r"\*\*[^*]+\*\*",
        "markdown_italic": r"(?<!\*)\*(?!\*)[^*]+(?<!\*)\*(?!\*)",
        "blockquote": r"(?m)^>\s+",
        "code_block": r"```",
        "inline_code": r"`[^`\n]+`",
        "bullet_dash": r"(?m)^[ \t]*-\s+",
        "bullet_star": r"(?m)^[ \t]*\*\s+",
        "bullet_dot": r"(?m)^[ \t]*•\s+",
        "numbered_list": r"(?m)^[ \t]*\d+[\.)]\s+",
        "curly_double_quote": r"[“”]",
        "curly_single_quote": r"[‘’]",
        "straight_double_quote": r'"',
        "straight_single_quote": r"'",
        "em_dash": r"—",
        "en_dash": r"–",
        "ellipsis_char": r"…",
        "ellipsis_dots": r"\.\.\.",
        "colon": r":",
        "semicolon": r";",
    }

    # Count pattern occurrences per class
    fmt_results = {p: {} for p in patterns}
    # Also count percentage of passages containing the pattern
    fmt_pct_results = {p: {} for p in patterns}

    for l in labels:
        sub = train[train["model"] == l]
        n_docs = len(sub)
        for p_name, p_regex in patterns.items():
            compiled = re.compile(p_regex)
            counts = sub["text"].apply(lambda t: len(compiled.findall(t)))
            has_p = (counts > 0).sum()
            fmt_results[p_name][l] = counts.sum()
            fmt_pct_results[p_name][l] = (has_p / n_docs) * 100

    df_fmt_counts = pd.DataFrame(fmt_results)
    df_fmt_pcts = pd.DataFrame(fmt_pct_results)

    print("\nPresence (% of passages containing feature) by Model:")
    key_features = ["double_newline", "single_newline", "header_h3", "markdown_bold", "bullet_dash", "bullet_star", "numbered_list", "curly_double_quote", "curly_single_quote", "em_dash", "en_dash", "double_space", "trailing_whitespace"]
    print(df_fmt_pcts[key_features].round(1).to_string())

    print(f"\n8. PASSAGE TERMINATION & TRUNCATION AUDIT")
    # Check what punctuation characters texts end with
    def get_ending_char(text):
        t = text.rstrip()
        return t[-1] if len(t) > 0 else ""

    train["end_char"] = train["text"].apply(get_ending_char)
    standard_terminal_chars = set(['.', '!', '?', '"', '”', "'", "’", ")", "]", "}", "*", "`"])
    train["is_standard_end"] = train["end_char"].apply(lambda c: c in standard_terminal_chars)

    print("\nPassage Termination Rates by Model:")
    term_stats = []
    for l in labels:
        sub = train[train["model"] == l]
        std_pct = sub["is_standard_end"].mean() * 100
        non_std = sub[~sub["is_standard_end"]]
        non_std_examples = non_std["end_char"].value_counts().to_dict()
        term_stats.append({
            "model": l,
            "standard_end_pct": std_pct,
            "non_standard_count": len(non_std),
            "common_non_std_end_chars": str(non_std_examples) if non_std_examples else "None"
        })
    df_term = pd.DataFrame(term_stats)
    print(df_term.to_string())

    print(f"\n9. SKLEARN CHAR ANALYZER COLLAPSE TEST")
    # 2-line test from PLAN.md
    vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 2)).fit(["a\nb"])
    feat_names = vec.get_feature_names_out().tolist()
    print(f"Input: ['a\\nb']")
    print(f"Features produced by analyzer='char': {feat_names}")
    has_newline = any('\n' in f for f in feat_names)
    has_space = any(' ' in f for f in feat_names)
    print(f"Does vocabulary contain '\\n'? {has_newline}")
    print(f"Does vocabulary contain ' '? {has_space}")
    print(f"Conclusion: Newline '\\n' IS collapsed to space: {not has_newline and has_space}")

    # Test with double newline and markdown
    vec2 = TfidfVectorizer(analyzer='char', ngram_range=(1, 3)).fit(["### Header\n\n- Item 1"])
    feat_names2 = vec2.get_feature_names_out().tolist()
    print(f"\nInput: ['### Header\\n\\n- Item 1']")
    print(f"Contains '\\n' in feats? {any(chr(10) in f for f in feat_names2)}")
    print(f"Sample ngrams: {feat_names2[:10]}")

    print(f"\n10. TEST SET FORMATTING AUDIT")
    print(f"Test samples formatting check:")
    for i, r in test.iterrows():
        t = r["text"]
        print(f"Test row {i} (id={r['id']}):")
        print(f"  Length: chars={len(t)}, words={len(t.split())}")
        print(f"  Newlines: \\n={t.count(chr(10))}, \\n\\n={t.count(chr(10)*2)}")
        print(f"  Has curly quotes: bool(re.search(r'[“”‘’]', t)) -> {bool(re.search(r'[“”‘’]', t))}")
        print(f"  Has em-dash: {'—' in t}, en-dash: {'–' in t}")
        print(f"  Ends with: {repr(t[-5:])}")

    # Save summary stats to a temporary json for report compilation
    summary_data = {
        "train_rows": len(train),
        "test_rows": len(test),
        "unique_prompts": n_prompts,
        "n_classes": len(labels),
        "labels": labels,
        "class_counts": class_counts.to_dict(),
        "length_summary": {
            "char_min": int(train["char_len"].min()),
            "char_median": float(train["char_len"].median()),
            "char_mean": float(train["char_len"].mean()),
            "char_max": int(train["char_len"].max()),
            "word_min": int(train["word_len"].min()),
            "word_median": float(train["word_len"].median()),
            "word_mean": float(train["word_len"].mean()),
            "word_max": int(train["word_len"].max()),
            "line_min": int(train["line_len"].min()),
            "line_median": float(train["line_len"].median()),
            "line_mean": float(train["line_len"].mean()),
            "line_max": int(train["line_len"].max()),
        },
        "length_by_model": df_len.to_dict(orient="records"),
        "formatting_pcts": df_fmt_pcts.to_dict(),
        "termination_stats": term_stats,
    }
    with open("scripts/eda_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print("\nSaved scripts/eda_summary.json successfully.")

if __name__ == "__main__":
    main()
