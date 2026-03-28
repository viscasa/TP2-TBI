import sys
# Remove TP2 dir from path to avoid local compression.py shadowing system packages
sys.path = [p for p in sys.path if not p.endswith('TP2')]
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
print('NLTK data downloaded successfully!')
