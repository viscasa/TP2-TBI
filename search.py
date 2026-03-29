import time
from bsbi import BSBIIndex
from postings_compression import VBEPostings
from lsi import LSIFAISS

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')
                          
lsi_model = LSIFAISS(BSBI_instance, k_dimensions=50)

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]

methods = [
    ("TF-IDF",      lambda q, k: BSBI_instance.retrieve_tfidf(q, k=k)),
    ("BM25",        lambda q, k: BSBI_instance.retrieve_bm25(q, k=k)),
    ("WAND TF-IDF", lambda q, k: BSBI_instance.retrieve_wand(q, k=k, scoring='tfidf')),
    ("WAND BM25",   lambda q, k: BSBI_instance.retrieve_wand(q, k=k, scoring='bm25')),
    ("LSI + FAISS", lambda q, k: lsi_model.retrieve(q, k=k)),
    ("PRF BM25",    lambda q, k: BSBI_instance.retrieve_rocchio(q, k=k)),
]

for query in queries:
    print("Query  : ", query)
    print()
    for method_name, retrieve_fn in methods:
        print(f"  {method_name} Results:")
        for (score, doc) in retrieve_fn(query, 10):
            print(f"    {doc:30} {score:>.3f}")
        print()
    print("=" * 60)
    print()