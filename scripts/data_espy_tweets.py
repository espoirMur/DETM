from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
import random
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
from pathlib import Path

# Maximum / minimum document frequency
max_df = 0.7
min_df = 100  # choose desired value for min_df


# Read data
print('reading text file...')
data_path = Path.cwd().parent.joinpath("balobi_nini", "data", "cleanned_tweets_2021.csv")
docs = pd.read_csv(data_path,
                   usecols=["created_at", "cleanned_text"],
                   parse_dates=["created_at"])
docs = docs.dropna(axis="rows", subset=["cleanned_text"])

# Create count vectorizer
print('counting document frequency of words...')
vectorizer = TfidfVectorizer(lowercase=True,
                             strip_accents="unicode",
                             ngram_range=(1, 2),
                             max_features=500000)
doc_terms_matrix = vectorizer.fit_transform(docs["cleanned_text"])
terms = vectorizer.get_feature_names()
timestamps = docs.created_at.dt.normalize().values.astype(np.int64) // 10 ** 9

# Get vocabulary
print('building the vocabulary...')

v_size = len(terms)
vocabulary = terms
print('Initial vocabulary size: {}'.format(v_size))

# Create bow representation
print('creating bow representation...')



# Save vocabulary to file
path_save = Path.cwd().joinpath('data', 'preprocess')
Path(path_save).mkdir(exist_ok=True, parents=True)

with open(path_save.joinpath('vocab.pkl'), 'wb') as f:
    pickle.dump(vocabulary, f)
del vocabulary

assert timestamps.shape[0] == doc_terms_matrix.shape[0]
# Split bow intro token/value pairs
print('splitting bow intro token/value pairs and saving to disk...')

savemat(path_save.joinpath('tf_idf_doc_terms_matrix'), {"doc_terms_matrix": doc_terms_matrix}, do_compression=True)
savemat(path_save.joinpath('tf_idf_terms'), {"terms": terms}, do_compression=True)
savemat(path_save.joinpath('tf_idf_timestamps'), {"timestamps": timestamps}, do_compression=True)

print('**' * 10 , doc_terms_matrix.shape)

print('Data ready !!')
print('*************')

