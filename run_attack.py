import argparse
import numpy as np
from attack import attack

parser = argparse.ArgumentParser(description='Run TextTroller attack.')
parser.add_argument('--prompt', type=str, required=True, help='The prompt')
parser.add_argument('--synonym_num', type=int, default=48)
args = parser.parse_args()

if __name__ == "__main__":
    new_prompt, original_label, new_label = attack(
        prompt=args.prompt,
        true_label=1,
        importance_score_threshold=-1.0,
        sim_score_threshold=0.5,
        sim_score_window=15,
        synonym_num=args.synonym_num,
    )

    print(f"Original prompt: {args.prompt}")
    print(f"Original label: {original_label}")
    print(f"Perturbed prompt: {new_prompt}")
    print(f"New label: {new_label}")
