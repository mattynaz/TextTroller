import numpy as np
import nltk
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--counter_fitting_embeddings_path', type=str, required=True,
                    help='Path to the counter-fitted-vectors.txt file')

args = parser.parse_args()

if __name__ == "__main__":
    print("Building vocab...")
    idx2word = {}
    word2idx = {}

    with open(args.counter_fitting_embeddings_path, 'r') as file:
        for line in file:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1


    print('Start computing the cosine similarity matrix!')
    embeddings = []
    with open(args.counter_fitting_embeddings_path, 'r') as file:
        for line in file:
            embedding = [float(num) for num in line.strip().split()[1:]]
            embeddings.append(embedding)

    embeddings = np.array(embeddings)
    product = np.dot(embeddings, embeddings.T)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    cos_sim = product / np.dot(norm, norm.T)
