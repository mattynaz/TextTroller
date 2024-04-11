import numpy as np
import argparse
import requests
import zipfile
import requests
import zipfile
import io
import numpy as np
import json

parser = argparse.ArgumentParser(description='Prepare troll.')
parser.add_argument('--idx2word_path', type=str, default='idx2word.json')
parser.add_argument('--word2idx_path', type=str, default='word2idx.json')
parser.add_argument('--cos_similarities_path', type=str, default='cos_similarities.npy')
args = parser.parse_args()


if __name__ == "__main__":
    url = "https://github.com/nmrksic/counter-fitting/raw/master/word_vectors/counter-fitted-vectors.txt.zip"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download the file.")

    zip_file = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall()

    counter_fitted_vectors_file = "counter-fitted-vectors.txt"
    idx2word = {}
    word2idx = {}
    with open(counter_fitted_vectors_file, 'r') as file:
        for line in file:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    embeddings = []
    with open(counter_fitted_vectors_file, 'r') as file:
        for line in file:
            embedding = [float(num) for num in line.strip().split()[1:]]
            embeddings.append(embedding)

    embeddings = np.array(embeddings)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.asarray(embeddings / norm, "float32")
    cos_similarities = np.dot(embeddings, embeddings.T)

    with open("idx2word.json", 'w') as file:
        json.dump(idx2word, file)
        
    with open("word2idx.json", 'w') as file:
        json.dump(word2idx, file)

    np.save(('cos_similarities.npy'), cos_similarities)
