import pandas as pd

train = pd.read_csv('data/train.csv')
p0_id = train['prompt_id'].iloc[0]
p0 = train[train['prompt_id'] == p0_id]
print(f"Prompt ID: {p0_id}")
print(f"Total completions: {len(p0)}")
for _, r in p0.iterrows():
    print(f"[{r['model']:22s}]: {r['text'][:120]}...")
