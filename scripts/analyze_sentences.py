import pandas as pd
import json
import re
from collections import Counter

train = pd.read_csv('data/train.csv')
with open('configs/labels.json') as f:
    labels = json.load(f)

# Sentence length & count analysis
def analyze_sentences(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0, 0, 0
    words_per_sent = [len(s.split()) for s in sentences]
    return len(sentences), sum(words_per_sent)/len(sentences), max(words_per_sent)

stats = []
for l in labels:
    sub = train[train['model'] == l]
    sent_metrics = sub['text'].apply(analyze_sentences)
    sent_counts = [m[0] for m in sent_metrics]
    avg_sent_lens = [m[1] for m in sent_metrics]
    
    # Lexical diversity (TTR)
    ttrs = []
    for t in sub['text']:
        words = re.findall(r'\b\w+\b', t.lower())
        if words:
            ttrs.append(len(set(words)) / len(words))
        else:
            ttrs.append(0)
            
    stats.append({
        'model': l,
        'avg_sentences': round(sum(sent_counts) / len(sent_counts), 2),
        'avg_words_per_sentence': round(sum(avg_sent_lens) / len(avg_sent_lens), 2),
        'avg_ttr': round(sum(ttrs) / len(ttrs), 4),
    })

df_sent = pd.DataFrame(stats)
print("Sentence & Diversity Stats by Model:")
print(df_sent.to_string())
