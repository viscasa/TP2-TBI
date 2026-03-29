import os
import re
import math
import time
import contextlib
import heapq
from tqdm import tqdm

from index import InvertedIndexReader, InvertedIndexWriter
from bsbi import BSBIIndex, stemmer, STOPWORDS
from util import IdMap, sorted_merge_posts_and_tfs
from postings_compression import VBEPostings, EliasGammaPostings

class SPIMIIndex(BSBIIndex):
    """
    SPIMI (Single-Pass In-Memory Indexing).
    Berbeda dengan BSBI yang mengumpulkan (TermID, DocID) pasangan satu per satu lalu merangkai and sort secara global per block,
    SPIMI langsung membangun dictionary term_id -> {doc_id: tf} secara real-time di memori selama parsing
    dan tidak perlu mensortir seluruh posting sebelum menambahkan.
    Hanya men-sort Dictionary Keys (terms) saat memori penuh.
    """
    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema SPIMI
        """
        # Membaca daftar file
        all_files = []
        for dirpath, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                all_files.append(os.path.join(dirpath, filename))
                
        # Urutkan agar id-id yang diberikan oleh IdMap terbentuk urut
        all_files.sort()

        # Dictionary in-memory: term_id -> {doc_id: tf}
        memory_index = {}
        processed_docs = 0
        block_number = 0
        
        # Batasan memori artifisial. Asumsikan "memori kita penuh" setiap 250 dokumen
        DOC_LIMIT = 250 
        
        for docname in tqdm(all_files, desc="SPIMI Processing"):
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                content = f.read().lower()
                tokens = re.findall(r'[a-z]+', content)
                
                doc_id = self.doc_id_map[docname]
                
                for token in tokens:
                    if token in STOPWORDS:
                        continue
                    stemmed = stemmer.stem(token)
                    if not stemmed:
                        continue
                        
                    term_id = self.term_id_map[stemmed]
                    
                    if term_id not in memory_index:
                        memory_index[term_id] = {}
                    
                    if doc_id not in memory_index[term_id]:
                        memory_index[term_id][doc_id] = 0
                        
                    memory_index[term_id][doc_id] += 1
            
            processed_docs += 1
            
            # Jika memori penuh, flush ke disk (Write Block)
            if processed_docs >= DOC_LIMIT:
                self._write_block_to_disk(memory_index, block_number)
                memory_index = {} # Kosongkan memory
                processed_docs = 0
                block_number += 1
        
        # Flush sisanya jika ada sisa
        if memory_index:
            self._write_block_to_disk(memory_index, block_number)
            
        self.save()
        
        # Lakukan merge
        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

    def _write_block_to_disk(self, memory_index, block_number):
        index_id = f'intermediate_spimi_{block_number}'
        self.intermediate_indices.append(index_id)
        
        with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
            # SPIMI hanya menyortir TermID, tidak perlu menyortir doc_id secara brutal
            # karena parsing sudah dikerjakan terurut dan ditampung ke dictionary
            for term_id in sorted(memory_index.keys()):
                # Extract keys dan TF. Kita sort saja jaga-jaga, walau scara arsitektur 
                # insertion order list dari python dict > 3.6 itu terjamin
                sorted_doc_id = sorted(list(memory_index[term_id].keys()))
                assoc_tf = [memory_index[term_id][doc_id] for doc_id in sorted_doc_id]
                index.append(term_id, sorted_doc_id, assoc_tf)


if __name__ == "__main__":
    import shutil
    
    # Hapus indeks lama agar fresh timing
    if os.path.exists('index_bsbi'): shutil.rmtree('index_bsbi')
    if os.path.exists('index_spimi'): shutil.rmtree('index_spimi')

    os.makedirs('index_bsbi', exist_ok=True)
    os.makedirs('index_spimi', exist_ok=True)

    print("Menguji BSBI vs SPIMI Indexing...")
    
    # 1. Mengukur waktu BSBI
    print("\n--- BSBI ---")
    start = time.time()
    bsbi = BSBIIndex(data_dir='collection', postings_encoding=EliasGammaPostings, output_dir='index_bsbi')
    bsbi.index()
    bsbi_time = time.time() - start
    
    # 2. Mengukur waktu SPIMI
    print("\n--- SPIMI ---")
    start = time.time()
    spimi = SPIMIIndex(data_dir='collection', postings_encoding=EliasGammaPostings, output_dir='index_spimi')
    spimi.index()
    spimi_time = time.time() - start
    
    print("\n=== Perbandingan Waktu Indexing ===")
    print(f"BSBI  : {bsbi_time:.4f} detik")
    print(f"SPIMI : {spimi_time:.4f} detik")
    print(f"Efisiensi SPIMI: {(bsbi_time - spimi_time)/bsbi_time:.2%} lebih cepat/lambat")

    # 3. Uji Correctness
    print("\n=== Uji Correctness Retrieval ===")
    query = "psychodrama for disturbed children"
    print("Query:", query)
    
    res_bsbi = bsbi.retrieve_wand(query, k=5, scoring='bm25')
    res_spimi = spimi.retrieve_wand(query, k=5, scoring='bm25')
    
    print("Hasil WAND BM25 BSBI:")
    for score, docname in res_bsbi:
        print(f"  {docname:30} {score:.4f}")
        
    print("\nHasil WAND BM25 SPIMI:")
    for score, docname in res_spimi:
        print(f"  {docname:30} {score:.4f}")
        
    # extract scores and doc names
    match = res_bsbi == res_spimi
    print(f"\nCorrectness: {'PASSED' if match else 'FAILED'}")

