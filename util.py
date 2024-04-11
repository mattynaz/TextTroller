import numpy as np
import requests
import zipfile
import io
import nltk
from nltk.corpus import stopwords
import os

counter_fitted_vectors_file = 'counter-fitted-vectors.txt'
similarity_matrix_file = 'similarity_matrix.npy'
nltk_data_path = os.path.join(os.path.expanduser("~"), 'nltk_data')

if not os.path.exists(nltk_data_path):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')

stop_words_set = set(stopwords.words('english'))

def get_pos(words: list[str], tagset='universal'):
    word_n_pos_list = nltk.pos_tag(words, tagset=tagset)
    _, pos_list = zip(*word_n_pos_list)
    return pos_list

def pos_filter(ori_pos, new_pos_list):
    return [ori_pos == new_pos or {ori_pos, new_pos} <= {'NOUN', 'VERB'} for new_pos in new_pos_list]

if not os.path.exists(counter_fitted_vectors_file):
    url = 'https://github.com/nmrksic/counter-fitting/raw/master/word_vectors/counter-fitted-vectors.txt.zip'
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError('Failed to download counter-fitted vector file.')
    zip_file = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

index_to_word = {}
word_to_index = {}
with open(counter_fitted_vectors_file, 'r') as file:
    for line in file:
        word = line.split()[0]
        if word not in index_to_word:
            index_to_word[len(index_to_word)] = word
            word_to_index[word] = len(index_to_word) - 1

if os.path.exists(similarity_matrix_file):
    similarity_matrix = np.load(similarity_matrix_file)
else:
    embeddings = []
    with open(counter_fitted_vectors_file, 'r') as file:
        for line in file:
            embedding = [float(num) for num in line.strip().split()[1:]]
            embeddings.append(embedding)
    embeddings = np.array(embeddings)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.asarray(embeddings / norm, 'float32')
    similarity_matrix = np.dot(embeddings, embeddings.T)
    np.save('similarity_matrix.npy', similarity_matrix)
