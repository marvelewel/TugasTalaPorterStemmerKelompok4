import os
import re
import string
import chardet
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi stemmer
ps = PorterStemmer()

# Kamus kata dasar
root_words = set([
    "politik", "ekonomi", "olahraga", "kesehatan", "pendidikan", 
    "sakit", "ajar", "main", "sehat", "kerja", "didik", "guna", "pakai",
    "tahu", "percaya", "ubah", "kaji", "nilai", "teknologi", "finance",
    "government", "education", "sport", "health", "play", "study", "run"
])

def is_root_word(word):
    return word in root_words

def tala_stemmer(word):
    word = re.sub(r'(lah|kah|tah|pun)$', '', word)
    if is_root_word(word): return word
    word = re.sub(r'(ku|mu|nya)$', '', word)
    if is_root_word(word): return word
    word = re.sub(r'^(me|di|ke|se|ber|ter|per)', '', word)
    if is_root_word(word): return word
    word = re.sub(r'^(pe|be|te)', '', word)
    if is_root_word(word): return word
    word = re.sub(r'(kan|an|i)$', '', word)
    if is_root_word(word): return word
    return word

def read_all_files(folder_path):
    texts = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            try:
                texts.append(raw_data.decode(encoding))
            except (UnicodeDecodeError, LookupError):
                texts.append(raw_data.decode('ISO-8859-1', errors='replace'))
    return texts

def preprocess(text):
    text = text.lower()
    text = re.sub(rf"[{string.punctuation}]", "", text)
    return text.split()

def stem_text(text, lang='id'):
    words = preprocess(text)
    if lang == 'id':
        return ' '.join([tala_stemmer(w) for w in words])
    elif lang == 'en':
        return ' '.join([ps.stem(w) for w in words])
    return text

def build_stem_map(texts, lang='id'):
    mapping = {}
    for text in texts:
        words = preprocess(text)
        for word in words:
            root = tala_stemmer(word) if lang == 'id' else ps.stem(word)
            if root not in mapping:
                mapping[root] = set()
            mapping[root].add(word)
    return mapping

def evaluate_stemming_map(stem_map):
    total_conflated_sets = len(stem_map)
    over_stemmed = sum(1 for words in stem_map.values() if len(words) > 5)
    under_stemmed = sum(1 for words in stem_map.values() if len(words) == 1)
    avg_conflation = sum(len(words) for words in stem_map.values()) / total_conflated_sets
    UI = under_stemmed / total_conflated_sets
    OI = over_stemmed / total_conflated_sets
    MWC = avg_conflation
    return UI, OI, MWC

def search(texts, query, lang='id'):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_stemmed = stem_text(query, lang=lang)
    query_vec = vectorizer.transform([query_stemmed])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    return scores

# ==== MAIN ====
path_indo = "C:\\Users\\victus\\OneDrive - Universitas Airlangga\\SEMESTER 6\\TKI\\MINGGU 7\\DOKUMEN TKI\\BIND"
path_eng = "C:\\Users\\victus\\OneDrive - Universitas Airlangga\\SEMESTER 6\\TKI\\MINGGU 7\\DOKUMEN TKI\\BING"
indo_texts = read_all_files(path_indo)
eng_texts = read_all_files(path_eng)

indo_stemmed = [stem_text(t, lang='id') for t in indo_texts]
eng_stemmed = [stem_text(t, lang='en') for t in eng_texts]

indo_map = build_stem_map(indo_texts, lang='id')
eng_map = build_stem_map(eng_texts, lang='en')

ui_id, oi_id, mwc_id = evaluate_stemming_map(indo_map)
ui_en, oi_en, mwc_en = evaluate_stemming_map(eng_map)

print(f"\nIndonesia - UI: {ui_id:.4f}, OI: {oi_id:.4f}, MWC: {mwc_id:.2f}")
print(f"English   - UI: {ui_en:.4f}, OI: {oi_en:.4f}, MWC: {mwc_en:.2f}")

# Grafik evaluasi stemming
labels = ['UI', 'OI', 'MWC']
indo_scores = [ui_id, oi_id, mwc_id]
eng_scores = [ui_en, oi_en, mwc_en]

x = np.arange(len(labels))
width = 0.35
fig1, ax1 = plt.subplots()
bars1 = ax1.bar(x - width/2, indo_scores, width, label='Bahasa Indonesia')
bars2 = ax1.bar(x + width/2, eng_scores, width, label='Bahasa Inggris')
ax1.set_ylabel('Skor')
ax1.set_title('Evaluasi Stemming (UI, OI, MWC)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
for bar in bars1 + bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Pencarian dan grafik
search_queries = [
    "politik", "ekonomi", "olahraga", "kesehatan", "pendidikan",
    "government", "sports", "education", "finance", "technology"
]

top_before = []
top_after = []

print("\nHasil Pencarian Sebelum vs Setelah Stemming (Top Skor)")
for query in search_queries:
    if query in ['government', 'sports', 'education', 'finance', 'technology']:
        original = search(eng_texts, query, lang='en')
        stemmed = search(eng_stemmed, query, lang='en')
    else:
        original = search(indo_texts, query, lang='id')
        stemmed = search(indo_stemmed, query, lang='id')
    top_doc_before = original.argmax()
    top_doc_after = stemmed.argmax()
    top_before.append(top_doc_before)
    top_after.append(top_doc_after)
    print(f"Query: {query}")
    print(f"→ Top doc before stemming: {top_doc_before}")
    print(f"→ Top doc after stemming:  {top_doc_after}\n")

x2 = np.arange(len(search_queries))
fig2, ax2 = plt.subplots(figsize=(10, 5))
bar1 = ax2.bar(x2 - width/2, top_before, width, label='Sebelum Stemming')
bar2 = ax2.bar(x2 + width/2, top_after, width, label='Sesudah Stemming')
ax2.set_ylabel('Index Dokumen Teratas')
ax2.set_title('Perbandingan Pencarian Sebelum vs Sesudah Stemming')
ax2.set_xticks(x2)
ax2.set_xticklabels(search_queries, rotation=45)
ax2.legend()
for bar in bar1 + bar2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{int(yval)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()
