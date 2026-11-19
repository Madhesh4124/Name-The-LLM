import pandas as pd
import json

train = pd.read_csv('data/train.csv')
with open('configs/labels.json') as f:
    labels = json.load(f)

features = {
    'curly_dquote': lambda t: sum(1 for c in t if c in '“”'),
    'curly_squote': lambda t: sum(1 for c in t if c in '‘’'),
    'straight_dquote': lambda t: t.count('"'),
    'straight_squote': lambda t: t.count("'"),
    'em_dash': lambda t: t.count('—'),
    'en_dash': lambda t: t.count('–'),
    'hyphen': lambda t: t.count('-'),
    'colon': lambda t: t.count(':'),
    'semicolon': lambda t: t.count(';'),
    'parens': lambda t: t.count('(') + t.count(')'),
    'latex': lambda t: t.count(r'\('),
    'ellipsis': lambda t: t.count('…') + t.count('...'),
}

results = []
for l in labels:
    sub = train[train['model'] == l]
    row = {'model': l}
    for name, fn in features.items():
        has = sub['text'].apply(lambda t: fn(t) > 0).mean() * 100
        avg = sub['text'].apply(fn).mean()
        row[f'{name}_pct'] = round(has, 1)
        row[f'{name}_avg'] = round(avg, 2)
    results.append(row)

df_feat = pd.DataFrame(results)
pct_cols = ['model'] + [c for c in df_feat.columns if c.endswith('_pct')]
print(df_feat[pct_cols].to_string())
print("\n" + "="*80 + "\n")
avg_cols = ['model'] + [c for c in df_feat.columns if c.endswith('_avg')]
print(df_feat[avg_cols].to_string())
