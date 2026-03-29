import re
import math
from bsbi import BSBIIndex
from postings_compression import VBEPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


######## >>>>> DCG (Discounted Cumulative Gain)

def dcg(ranking):
  """ menghitung Discounted Cumulative Gain (DCG)

      Formula: DCG = Σ rel_i / log2(i + 1) untuk i = 1..k

      Parameters
      ----------
      ranking: List[int]
         vektor biner relevansi [1, 0, 1, 1, ...]
        
      Returns
      -------
      Float
        score DCG
  """
  score = 0.
  for i in range(1, len(ranking) + 1):
    rel = ranking[i - 1]
    if rel > 0:
      score += rel / math.log2(i + 1)
  return score


######## >>>>> NDCG (Normalized Discounted Cumulative Gain)

def ndcg(ranking):
  """ menghitung Normalized Discounted Cumulative Gain (NDCG)
  
      NDCG = DCG / IDCG
      dimana IDCG adalah DCG dari ranking ideal (semua dokumen relevan di atas)

      Parameters
      ----------
      ranking: List[int]
         vektor biner relevansi [1, 0, 1, 1, ...]
        
      Returns
      -------
      Float
        score NDCG (0..1). Return 0 jika tidak ada dokumen relevan (IDCG = 0).
  """
  actual_dcg = dcg(ranking)
  
  # IDCG: DCG dari ranking ideal (sort descending relevansi)
  ideal_ranking = sorted(ranking, reverse=True)
  ideal_dcg = dcg(ideal_ranking)
  
  if ideal_dcg == 0:
    return 0.0
  return actual_dcg / ideal_dcg


######## >>>>> AP (Average Precision)

def ap(ranking):
  """ menghitung Average Precision (AP)

      Formula: AP = (1/R) × Σ P@k × rel(k)
      dimana R = jumlah total dokumen relevan dalam ranking,
      dan P@k = precision pada posisi k.

      Parameters
      ----------
      ranking: List[int]
         vektor biner relevansi [1, 0, 1, 1, ...]
        
      Returns
      -------
      Float
        score Average Precision. Return 0 jika tidak ada dokumen relevan.
  """
  R = sum(ranking)  # total dokumen relevan
  if R == 0:
    return 0.0
  
  score = 0.
  relevant_so_far = 0
  for i in range(1, len(ranking) + 1):
    rel = ranking[i - 1]
    if rel == 1:
      relevant_so_far += 1
      precision_at_k = relevant_so_far / i
      score += precision_at_k
  
  return score / R


######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels


######## >>>>> Fungsi helper untuk extract doc ID dari path

def extract_doc_id(doc_path):
  """ Extract numeric doc ID dari path dokumen.
      Contoh: 'collection\\6\\507.txt' → 507
              'collection/6/507.txt'  → 507
  """
  # Ganti backslash ke slash agar cross-platform, lalu ambil angka dari nama file
  doc_path = doc_path.replace('\\', '/')
  match = re.search(r'/(\d+)\.txt$', doc_path)
  if match:
    return int(match.group(1))
  return -1


######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    
    Evaluasi dilakukan untuk KEDUA metode: TF-IDF dan BM25.
    Metric yang dihitung: RBP, DCG, NDCG, AP.
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')
                          
  try:
    from lsi import LSIFAISS
    # Bangun LSI model dengan k=50 semantic concept feature space
    lsi_model = LSIFAISS(BSBI_instance, k_dimensions=50)
  except ImportError:
    lsi_model = None

  # Konfigurasi: (nama metode, fungsi retrieval)
  methods = [
    ("TF-IDF",       BSBI_instance.retrieve_tfidf),
    ("BM25",         BSBI_instance.retrieve_bm25),
    ("WAND TF-IDF",  lambda q, k=k: BSBI_instance.retrieve_wand(q, k=k, scoring='tfidf')),
    ("WAND BM25",    lambda q, k=k: BSBI_instance.retrieve_wand(q, k=k, scoring='bm25')),
    ("PRF BM25",     lambda q, k=k: BSBI_instance.retrieve_rocchio(q, k=k)),
  ]
  if lsi_model:
      methods.append(("LSI (FAISS vec)", lambda q, k=k: lsi_model.retrieve(q, k=k)))

  for method_name, retrieve_fn in methods:
    with open(query_file) as file:
      rbp_scores  = []
      dcg_scores  = []
      ndcg_scores = []
      ap_scores   = []

      for qline in file:
        parts = qline.strip().split()
        qid = parts[0]
        query = " ".join(parts[1:])

        ranking = []
        for (score, doc) in retrieve_fn(query, k=k):
          did = extract_doc_id(doc)
          if did != -1 and did in qrels[qid]:
            ranking.append(qrels[qid][did])

        rbp_scores.append(rbp(ranking))
        dcg_scores.append(dcg(ranking))
        ndcg_scores.append(ndcg(ranking))
        ap_scores.append(ap(ranking))

    # Hitung mean scores
    n = len(rbp_scores)
    print(f"Hasil evaluasi {method_name} terhadap {n} queries")
    print(f"  RBP  score = {sum(rbp_scores) / n:.4f}")
    print(f"  DCG  score = {sum(dcg_scores) / n:.4f}")
    print(f"  NDCG score = {sum(ndcg_scores) / n:.4f}")
    print(f"  MAP  score = {sum(ap_scores) / n:.4f}")
    print()


if __name__ == '__main__':
  qrels = load_qrels()

  # Sanity check qrels
  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  # Sanity check metrics
  assert dcg([1, 0, 1])  > 0, "dcg salah"
  assert ndcg([1, 1, 1]) == 1.0, "ndcg perfect ranking harus 1.0"
  assert ndcg([0, 0, 0]) == 0.0, "ndcg tanpa relevan harus 0.0"
  assert ap([1, 1, 1])   == 1.0, "ap perfect ranking harus 1.0"
  assert ap([0, 0, 0])   == 0.0, "ap tanpa relevan harus 0.0"
  print("Semua sanity check passed!\n")

  eval(qrels)