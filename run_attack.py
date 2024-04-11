import numpy as np
import json
import argparse
import numpy as np
from attack import attack
from similarity_predictor import SimilarityPredictor
from predictor import make_predictor

parser = argparse.ArgumentParser(description='Run TextTroller attack.')
parser.add_argument('--target_model', type=str, required=True, help='The target model')
parser.add_argument('--judge_model', type=str, required=True, help='The judge model')
parser.add_argument('--prompt', type=str, required=True, help='The prompt')
parser.add_argument('--idx2word_path', type=str, default='idx2word.json')
parser.add_argument('--word2idx_path', type=str, default='word2idx.json')
parser.add_argument('--cos_similarities_path', type=str, default='cos_similarities.npy')
parser.add_argument('--synonym_num', type=int, default=48)
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.idx2word_path, 'r') as file:
        idx2word = json.load(file)

    with open(args.word2idx_path, 'r') as file:
        word2idx = json.load(file)

    cos_similarities = np.load(args.cos_similarities_path)

    similarity_predictor = SimilarityPredictor()

    predictor, _, _ = make_predictor(args.target_model, args.judge_model)

    new_prompt, original_label, new_label = attack(
        prompt=args.prompt,
        true_label=1,
        predictor=predictor,
        stop_words_set=None,
        idx2word=idx2word,
        word2idx=word2idx,
        cos_similarities=cos_similarities,
        similarity_predictor=similarity_predictor,
        importance_score_threshold=-1.0,
        sim_score_threshold=0.5,
        sim_score_window=15,
        synonym_num=args.synonym_num,
    )

    print(f"Original prompt: {args.prompt}")
    print(f"Original label: {original_label}")
    print(f"Perturbed prompt: {new_prompt}")
    print(f"New label: {new_label}")
