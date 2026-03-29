import os
import re
import time
import math
from flask import Flask, render_template, request, jsonify

from bsbi import BSBIIndex, stemmer, STOPWORDS
from postings_compression import VBEPostings
from lsi import LSIFAISS

app = Flask(__name__)

# ============================================================
# Inisialisasi Search Engine (sekali saat server start)
# ============================================================
print("Memuat index...")
BSBI = BSBIIndex(data_dir='collection', postings_encoding=VBEPostings, output_dir='index')
BSBI.load()

print("Membangun model LSI...")
lsi_model = LSIFAISS(BSBI, k_dimensions=50)
lsi_model.build()

# Bangun Trie dari term_id_map jika belum berbasis Trie (backward compat)
from trie import Trie
term_trie = Trie()
if hasattr(BSBI.term_id_map, 'trie'):
    term_trie = BSBI.term_id_map.trie
    ALL_TERM_STRINGS = BSBI.term_id_map.id_to_str
else:
    # IdMap lama: str_to_id dict, id_to_str list
    ALL_TERM_STRINGS = list(BSBI.term_id_map.id_to_str)
    for i, term in enumerate(ALL_TERM_STRINGS):
        term_trie.insert(term, i)

METHODS = {
    'tfidf':    ('TF-IDF',       lambda q, k: BSBI.retrieve_tfidf(q, k=k)),
    'bm25':     ('BM25',         lambda q, k: BSBI.retrieve_bm25(q, k=k)),
    'wand':     ('WAND BM25',    lambda q, k: BSBI.retrieve_wand(q, k=k, scoring='bm25')),
    'lsi':      ('LSI + FAISS',  lambda q, k: lsi_model.retrieve(q, k=k)),
    'prf':      ('PRF BM25',     lambda q, k: BSBI.retrieve_rocchio(q, k=k)),
}


# ============================================================
# Spell Correction (Levenshtein Distance)
# ============================================================
def levenshtein_distance(s1, s2):
    """Menghitung jarak edit minimum antara dua string."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def spell_correct(query_tokens):
    """
    Untuk setiap token query yang TIDAK ditemukan di dictionary,
    cari kandidat terdekat menggunakan Levenshtein distance.
    Kembalikan saran koreksi jika ada perubahan.
    """
    corrected = []
    changed = False
    for token in query_tokens:
        stemmed = stemmer.stem(token)
        # Cek apakah token sudah ada di dictionary Trie
        if term_trie.search(stemmed) != -1:
            corrected.append(token)
            continue

        # Cari kandidat terdekat
        best_term = token
        best_dist = 3  # Batas maksimal edit distance
        for candidate in ALL_TERM_STRINGS:
            if abs(len(candidate) - len(stemmed)) > 2:
                continue
            dist = levenshtein_distance(stemmed, candidate)
            if dist < best_dist:
                best_dist = dist
                best_term = candidate
        if best_term != token:
            corrected.append(best_term)
            changed = True
        else:
            corrected.append(token)
    if changed:
        return " ".join(corrected)
    return None


# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    method = request.args.get('method', 'bm25')
    k = int(request.args.get('k', 10))

    if not query:
        return jsonify({'results': [], 'time_ms': 0, 'suggestion': None})

    # Spell correction
    raw_tokens = [t for t in re.findall(r'[a-z]+', query.lower()) if t not in STOPWORDS]
    suggestion = spell_correct(raw_tokens)

    # Retrieve
    method_name, retrieve_fn = METHODS.get(method, METHODS['bm25'])
    start = time.time()
    raw_results = retrieve_fn(query, k)
    elapsed_ms = (time.time() - start) * 1000

    # Build results with snippets
    results = []
    for score, doc_path in raw_results:
        snippet = ""
        try:
            with open(doc_path, 'r', encoding='utf8', errors='surrogateescape') as f:
                content = f.read()
                # Ambil ~200 karakter pertama sebagai snippet
                snippet = content[:250].strip()
        except FileNotFoundError:
            snippet = "(File tidak ditemukan)"

        results.append({
            'doc': doc_path,
            'score': round(score, 4),
            'snippet': snippet,
        })

    return jsonify({
        'results': results,
        'time_ms': round(elapsed_ms, 2),
        'method_name': method_name,
        'suggestion': suggestion,
        'total': len(results),
    })


@app.route('/autocomplete')
def autocomplete():
    prefix = request.args.get('q', '').strip().lower()
    if len(prefix) < 2:
        return jsonify([])

    # Stem the last word for Trie lookup
    words = prefix.split()
    last_word = words[-1] if words else ''
    stemmed_prefix = stemmer.stem(last_word) if last_word else ''

    if not stemmed_prefix:
        return jsonify([])

    results = term_trie.get_all_with_prefix(stemmed_prefix)
    suggestions = [term for term, _ in results[:8]]

    return jsonify(suggestions)


if __name__ == '__main__':
    app.run(debug=False, port=5000)
