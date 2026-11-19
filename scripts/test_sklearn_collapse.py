from sklearn.feature_extraction.text import TfidfVectorizer

print("=== 1. Char vectorizer with 'hello\\nworld' ===")
v_char = TfidfVectorizer(analyzer='char', ngram_range=(1, 2)).fit(["hello\nworld"])
feats_char = v_char.get_feature_names_out().tolist()
print("char ngrams containing '\\n':", [f for f in feats_char if '\n' in f])

print("\n=== 2. Char_wb vectorizer with 'hello\\nworld' ===")
v_wb = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3)).fit(["hello\nworld"])
feats_wb = v_wb.get_feature_names_out().tolist()
print("char_wb ngrams containing '\\n':", [f for f in feats_wb if '\n' in f])
print("Does char_wb strip '\\n' to space boundary?", not any('\n' in f for f in feats_wb))

print("\n=== 3. Char vectorizer with multi-char tokens: '\\n\\n', '###', '**' ===")
text = "### Header\n\n- Item 1\n- Item 2"
v_char3 = TfidfVectorizer(analyzer='char', ngram_range=(2, 4)).fit([text])
feats3 = v_char3.get_feature_names_out().tolist()
print("Any features preserving '\\n\\n' as distinct from '\\n'?:", any('\n\n' in f for f in feats3))
print("Sample feats with '#':", [f for f in feats3 if '#' in f])
print("Sample feats with '-':", [f for f in feats3 if '-' in f])

print("\n=== 4. Char_wb on Markdown text ===")
v_wb4 = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)).fit([text])
feats4 = v_wb4.get_feature_names_out().tolist()
print("Any '\\n' in char_wb features?:", any('\n' in f for f in feats4))
