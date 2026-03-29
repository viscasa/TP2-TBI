import sys
import os
import pickle
import contextlib
import heapq
import time
import math
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from postings_compression import StandardPostings, VBEPostings, EliasGammaPostings

# Inisialisasi stemmer dan stopwords (English)
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = os.path.join(self.data_dir, block_dir_relative)
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = os.path.join(dir, filename)
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                content = f.read().lower()
                # Tokenisasi: ambil hanya kata-kata (huruf saja)
                tokens = re.findall(r'[a-z]+', content)
                for token in tokens:
                    # Buang stopwords
                    if token in STOPWORDS:
                        continue
                    # Stemming dengan Porter Stemmer
                    stemmed = stemmer.stem(token)
                    if stemmed:  # pastikan hasil stem tidak kosong
                        td_pairs.append((self.term_id_map[stemmed], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Preprocess query: tokenize, remove stopwords, stem (sama seperti saat indexing)
        query_tokens = re.findall(r'[a-z]+', query.lower())
        query_tokens = [stemmer.stem(t) for t in query_tokens if t not in STOPWORDS]
        terms = [self.term_id_map[word] for word in query_tokens]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:


            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema BM25 (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Formula BM25:
            score(D, Q) = Σ IDF(t) × [tf(t,D) × (k1 + 1)] / [tf(t,D) + k1 × (1 - b + b × |D|/avdl)]

            IDF(t) = log(N / df(t))

        Parameters
        ----------
        query: str
            Query string
        k: int
            Jumlah dokumen yang dikembalikan (top-K)
        k1: float
            Parameter BM25 untuk term frequency saturation (default 1.2)
        b: float
            Parameter BM25 untuk document length normalization (default 0.75)

        Result
        ------
        List[(float, str)]
            List of (score, doc_name), terurut descending by score
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Preprocess query: tokenize, remove stopwords, stem
        query_tokens = re.findall(r'[a-z]+', query.lower())
        query_tokens = [stemmer.stem(t) for t in query_tokens if t not in STOPWORDS]
        terms = [self.term_id_map[word] for word in query_tokens]

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            # Hitung average document length (avdl)
            N = len(merged_index.doc_length)
            avdl = sum(merged_index.doc_length.values()) / N

            scores = {}
            for term in terms:
                if term not in merged_index.postings_dict:
                    continue  # skip terms yang tidak ada di collection

                df = merged_index.postings_dict[term][1]
                idf = math.log(N / df)
                postings, tf_list = merged_index.get_postings_list(term)

                for i in range(len(postings)):
                    doc_id = postings[i]
                    tf = tf_list[i]
                    dl = merged_index.doc_length[doc_id]

                    # BM25 term weight
                    tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avdl))
                    score = idf * tf_norm

                    if doc_id not in scores:
                        scores[doc_id] = 0
                    scores[doc_id] += score

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def _preprocess_query(self, query):
        """
        Helper: preprocess query menjadi list of term IDs.
        Tokenize, remove stopwords, stem — sama seperti saat indexing.
        """
        query_tokens = re.findall(r'[a-z]+', query.lower())
        query_tokens = [stemmer.stem(t) for t in query_tokens if t not in STOPWORDS]
        return [self.term_id_map[word] for word in query_tokens]

    def retrieve_wand(self, query, k=10, scoring='bm25', k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan algoritma WAND (Weak AND).
        WAND melakukan Document-at-a-Time processing yang bisa men-skip
        dokumen yang tidak mungkin masuk top-K, sehingga lebih efisien.

        Algoritma WAND:
        1. Untuk setiap term di query, hitung upper-bound score (skor maksimal
           yang bisa diberikan term tersebut ke dokumen manapun).
        2. Maintain threshold = skor minimum untuk masuk top-K heap.
        3. Sort term iterators berdasarkan current docID.
        4. Cari pivot: posisi term pertama dimana cumulative upper-bound >= threshold.
        5. Jika semua term sebelum pivot sudah menunjuk ke pivot doc → score doc tersebut.
           Jika tidak → advance term terkecil ke pivot doc (skip!).

        Parameters
        ----------
        query: str
            Query string
        k: int
            Jumlah dokumen top-K yang dikembalikan
        scoring: str
            'tfidf' atau 'bm25'
        k1, b: float
            Parameter BM25 (hanya digunakan jika scoring='bm25')

        Result
        ------
        List[(float, str)]
            List of (score, doc_name), terurut descending by score
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = self._preprocess_query(query)

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            N = len(merged_index.doc_length)
            avdl = sum(merged_index.doc_length.values()) / N if scoring == 'bm25' else 0

            # Load postings dan hitung upper-bound per term
            term_data = []
            for term in terms:
                if term not in merged_index.postings_dict:
                    continue

                df = merged_index.postings_dict[term][1]
                idf = math.log(N / df)
                postings, tf_list = merged_index.get_postings_list(term)

                if not postings:
                    continue

                # Hitung upper-bound score untuk term ini
                if scoring == 'tfidf':
                    max_tf = max(tf_list)
                    upper_bound = idf * (1 + math.log(max_tf)) if max_tf > 0 else 0
                else:  # bm25
                    # Upper-bound BM25: tf tertinggi + dl terpendek
                    max_tf = max(tf_list)
                    # Cari dl terpendek di antara docs yang mengandung term ini
                    min_dl = min(merged_index.doc_length.get(doc_id, avdl) for doc_id in postings)
                    tf_norm_ub = (max_tf * (k1 + 1)) / (max_tf + k1 * (1 - b + b * min_dl / avdl))
                    upper_bound = idf * tf_norm_ub

                term_data.append({
                    'idf': idf,
                    'postings': postings,
                    'tf_list': tf_list,
                    'ptr': 0,
                    'upper_bound': upper_bound,
                })

            if not term_data:
                return []

            # ===== WAND Algorithm =====
            top_k_heap = []  # min-heap of (score, doc_id)
            threshold = 0.0

            while True:
                # Hapus iterators yang sudah habis
                term_data = [td for td in term_data if td['ptr'] < len(td['postings'])]
                if not term_data:
                    break

                # Step 1: Sort terms berdasarkan current docID
                term_data.sort(key=lambda td: td['postings'][td['ptr']])

                # Step 2: Cari pivot — posisi pertama dimana cumulative UB >= threshold
                cumulative_ub = 0.0
                pivot_idx = -1
                for i, td in enumerate(term_data):
                    cumulative_ub += td['upper_bound']
                    if cumulative_ub >= threshold:
                        pivot_idx = i
                        break

                if pivot_idx == -1:
                    break  # Tidak ada lagi yang bisa exceed threshold

                pivot_doc = term_data[pivot_idx]['postings'][term_data[pivot_idx]['ptr']]

                # Step 3: Cek apakah term pertama (smallest doc) juga menunjuk ke pivot_doc
                if term_data[0]['postings'][term_data[0]['ptr']] == pivot_doc:
                    # Semua term dari 0..pivot_idx menunjuk ke pivot_doc
                    # → Fully score dokumen ini

                    score = 0.0
                    for td in term_data:
                        ptr = td['ptr']
                        if ptr < len(td['postings']) and td['postings'][ptr] == pivot_doc:
                            tf = td['tf_list'][ptr]
                            if scoring == 'tfidf':
                                if tf > 0:
                                    score += td['idf'] * (1 + math.log(tf))
                            else:  # bm25
                                dl = merged_index.doc_length.get(pivot_doc, avdl)
                                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avdl))
                                score += td['idf'] * tf_norm
                            td['ptr'] += 1  # advance past pivot_doc

                    # Update top-K heap
                    if len(top_k_heap) < k:
                        heapq.heappush(top_k_heap, (score, pivot_doc))
                        if len(top_k_heap) == k:
                            threshold = top_k_heap[0][0]
                    elif score > top_k_heap[0][0]:
                        heapq.heapreplace(top_k_heap, (score, pivot_doc))
                        threshold = top_k_heap[0][0]

                else:
                    # Step 4: Advance term pertama ke pivot_doc (skip!)
                    td = term_data[0]
                    postings = td['postings']
                    ptr = td['ptr']

                    # Binary search untuk skip ke pivot_doc
                    lo, hi = ptr, len(postings)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if postings[mid] < pivot_doc:
                            lo = mid + 1
                        else:
                            hi = mid
                    td['ptr'] = lo

            # Extract results dari heap
            results = []
            while top_k_heap:
                score, doc_id = heapq.heappop(top_k_heap)
                results.append((score, self.doc_id_map[doc_id]))
            results.reverse()  # descending by score
            return results

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
