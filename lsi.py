import os
import math
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import faiss
from collections import Counter
import time

from bsbi import BSBIIndex
from index import InvertedIndexReader
from postings_compression import VBEPostings, EliasGammaPostings

class LSIFAISS:
    def __init__(self, bsbi_instance, k_dimensions=100):
        """
        Inisialisasi sistem pencarian LSI memanfaatkan backend dari model BSBI.
        Mengonversi Inverted Index menjadi Dense Vector Space dan mengindeks dengan FAISS.
        """
        self.bsbi = bsbi_instance
        self.k_dim = k_dimensions
        
        self.doc_vectors = None
        self.term_proj_matrix = None
        self.faiss_index = None
        self.term_df = {}
        self.N = 0
        self.M = 0
    
    def build(self):
        """Membangun matriks TF-IDF raksasa, melakukan SVD, dan indeks ke FAISS."""
        print("Mempersiapkan LSI + FAISS Vector Indexing...")
        
        if len(self.bsbi.term_id_map) == 0 or len(self.bsbi.doc_id_map) == 0:
            self.bsbi.load()
            
        self.N = len(self.bsbi.doc_id_map)
        self.M = len(self.bsbi.term_id_map)
        
        # Asumsikan BSBI menggunakan EliasGammaPostings seperti tes fase SPIMI kemarin
        # Atau bisa pakai parameter class BSBIInstance
        encoding = self.bsbi.postings_encoding
        
        row_ind = []
        col_ind = []
        data = []
        
        print("Mambaca Inverted Index untuk membangun Matriks Term-Document (Sparse)...")
        with InvertedIndexReader(self.bsbi.index_name, encoding, directory=self.bsbi.output_dir) as index:
            for term_id, postings, tf_list in index:
                df = len(postings)
                self.term_df[term_id] = df
                idf = math.log(self.N / df) if df > 0 else 0
                
                for doc_id, tf in zip(postings, tf_list):
                    if tf > 0:
                        weight = idf * (1 + math.log(tf))
                        # Document-Term matrix (N docs x M terms)
                        row_ind.append(doc_id)
                        col_ind.append(term_id)
                        data.append(weight)
        
        print("Melakukan komputasi SVD (Singular Value Decomposition)...")
        X = sp.csr_matrix((data, (row_ind, col_ind)), shape=(self.N, self.M), dtype=np.float32)
        
        # SVD: X ≈ U * S * V^T
        # U (Doc representations), S (Singular values), Vt (Feature reps)
        # scipy svds mengharapkan num of dim lebih kecil dari ukuran array terkecil
        actual_k = min(self.k_dim, min(X.shape)-1) 
        U, S, Vt = svds(X, k=actual_k)
        
        # Dense document vectors (N x K)
        # Representasi dokumen di sumbu semantic (LSI)
        self.doc_vectors = U * S 
        
        # Term Projection Matrix (M x K)
        # Transformasi vektor term-weight (query) ke LSI space -> Q_dense = Q_raw @ (Vt.T @ S^-1)
        self.term_proj_matrix = Vt.T @ np.diag(1.0 / S)
        
        print("Membangun struktur FAISS untuk kemiripan Cosine Similarity (Inner Product Normalized)...")
        # Normalisasi L2 vector dokumen agar operasi basis 'Inner Product' FAISS bekerja spt Cosine Similarity
        faiss.normalize_L2(self.doc_vectors)
        
        self.faiss_index = faiss.IndexFlatIP(actual_k)
        self.faiss_index.add(self.doc_vectors.astype(np.float32))
        
        print(f"LSI Model telah terbangun sepenuhnya dengan {actual_k} dimensi semantik!")
        
    def retrieve(self, query, k=10):
        """Memproses query string, diubah ke representasi Dense VSM, dicari di FAISS."""
        if not self.faiss_index:
            self.build()
            
        terms = self.bsbi._preprocess_query(query)
        term_counts = Counter(terms)
        
        q_data = []
        q_col = []
        
        for term_id, tf in term_counts.items():
            if term_id in self.term_df: # Kata harus terdaftar di dictionary/corpus
                idf = math.log(self.N / self.term_df[term_id])
                weight = idf * (1 + math.log(tf))
                q_data.append(weight)
                q_col.append(term_id)
                
        if len(q_data) == 0:
            return [] # No match term in query
            
        # Matriks sparsed query [1 baris, M kolom term]
        Q_sparse = sp.csr_matrix((q_data, ([0]*len(q_col), q_col)), shape=(1, self.M), dtype=np.float32)
        
        # Transformasi Query Vector Sparse Dimension menembus matrix Feature Space LSI
        Q_dense = Q_sparse @ self.term_proj_matrix
        
        # Normalisasi skala magnitude Q sebelum dikomparasi secara cosine (Inner Product model FAISS)
        # Konversi ke np float32 krn FAISS pakai library C++ float 32-bit standard
        Q_dense = Q_dense.astype(np.float32)
        faiss.normalize_L2(Q_dense)
        
        # Top-K Search
        scores, doc_ids = self.faiss_index.search(Q_dense, k)
        
        # Ekstrak path name
        results = []
        for score, doc_id in zip(scores[0], doc_ids[0]):
            if doc_id != -1:  # -1 is padded by faiss if K > N elements
                doc_name = self.bsbi.doc_id_map[int(doc_id)]
                results.append((float(score), doc_name))
                
        # FAISS sudah mengurutkan result berdasar similaritas descending layaknya pencarian konvensional
        return results

if __name__ == '__main__':
    import math # karena tadi importnya di dalam loop, pindah di atas
    # Mock / Test Eksekusi LSIFAISS
    bsbi = BSBIIndex(data_dir='collection', postings_encoding=EliasGammaPostings, output_dir='index_spimi')
    
    model_lsi = LSIFAISS(bsbi_instance=bsbi, k_dimensions=50) # Pakai 50 dimensi untuk tes
    
    start_build = time.time()
    model_lsi.build()
    print(f"Waktu Build Model Semantik: {time.time() - start_build:.2f} s\n")
    
    query = "psychodrama for disturbed children"
    print(f"Query LSI (Cari konsep): '{query}'")
    
    start_search = time.time()
    results = model_lsi.retrieve(query, k=5)
    print(f"Waktu Respon Retrieval Query: {(time.time() - start_search)*1000:.3f} ms")
    
    for score, doc in results:
        print(f"  {score:.5f}  -- {doc}")
