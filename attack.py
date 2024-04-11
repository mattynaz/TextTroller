from typing import List, Tuple
import torch
import numpy as np
import criteria
import numpy as np
import nltk
from similarity_predictor import SimilarityPredictor

def attack(
    prompt: str,  # The list of words in the original text
    true_label,  # The correct label for the original text
    predictor,  # A function that predicts the label based on the text input
    stop_words_set,  # A set of words to ignore when looking for words to replace
    idx2word,  # A dictionary mapping indices back to words
    word2idx,  # A dictionary mapping words to their indices
    cos_sim,  # A function to compute the cosine similarity between word vectors
    similarity_predictor: SimilarityPredictor,  # A function to compute semantic similarity between texts
    importance_score_threshold=-1.0,  # Threshold for considering a word's importance score
    sim_score_threshold=0.5,  # Threshold for considering semantic similarity as sufficient
    sim_score_window=15,  # The window size to consider for semantic similarity calculation
    synonym_num=50,  # The number of synonyms to consider for each word
):
    # Get the original prediction and its confidence
    original_label_probs = predictor([prompt]).squeeze()
    original_label = original_label_probs.argmax()
    original_label_confidence = original_label_probs.max()

    # Return early if the original prediction is incorrect
    if true_label != original_label:
        return prompt, original_label, original_label
    
    # Get the part-of-speech tagger and compatibility checker
    prompt_tokens = nltk.tokenize.word_tokenize(prompt)
    prompt_tokens_pos = criteria.get_pos(prompt_tokens)

    # Adjust the similarity threshold for shorter texts
    adjusted_sim_score_threshold = 0.1 if len(prompt_tokens) < sim_score_window else sim_score_threshold
    half_sim_score_window = (sim_score_window - 1) // 2

    # Generate perturbed versions of the original text by replacing each token with '<oov>'
    perturbed_prompts_tokens = [prompt_tokens[:i] + ['<oov>'] + prompt_tokens[i+1:] for i in range(len(prompt_tokens))]
    perturbed_prompts = [' '.join(t for t in tokens if t != '<oov>') for tokens in perturbed_prompts_tokens]
    perturbed_labels_probs = predictor(perturbed_prompts)
    perturbed_labels = perturbed_labels_probs.argmax(dim=-1)

    # Calculate importance scores for each token
    confidence_losses = original_label_confidence - perturbed_labels_probs[:, original_label]
    prediction_changes = (perturbed_labels != original_label).float()
    new_pred_confidences = perturbed_labels_probs.max(dim=-1)[0]
    prediction_change_losses = new_pred_confidences - torch.index_select(original_label_probs, 0, perturbed_labels)
    importance_scores = confidence_losses + prediction_changes * prediction_change_losses

    # Filter words based on importance scores and exclusion from stop words set
    importance_scores_sorted = sorted(enumerate(importance_scores), key=lambda x: x[1], reverse=True)
    perturbable_words = [
        (i, prompt_tokens[i]) for (i, score) in importance_scores_sorted
        if score > importance_score_threshold and prompt_tokens[i] not in stop_words_set and prompt_tokens[i] in word2idx
    ]

    # Lookup synonyms for the filtered words
    perturbable_word_indices = [word2idx[word] for _, word in perturbable_words]
    synonyms_list, _ = pick_most_similar_words_batch(perturbable_word_indices, cos_sim, idx2word, synonym_num, adjusted_sim_score_threshold)
    synonyms_for_perturbation = [
        (idx, synonyms) for (idx, _), synonyms in zip(perturbable_words, synonyms_list) if synonyms
    ]
    
    # Attempt to replace words with their synonyms to alter the model's prediction
    modified_prompt_tokens = prompt_tokens.copy()  # Initialize modified text with original text
    modified_prompt_tokens_cache = prompt_tokens.copy()  # Initialize modified text with original text
    for i, synonyms in synonyms_for_perturbation:
        # For each synonym, create new text versions and predict their labels
        perturbed_prompts_tokens = [modified_prompt_tokens[:i] + [synonym] + modified_prompt_tokens[i+1:] for synonym in synonyms]
        perturbed_prompts = [' '.join(tokens) for tokens in perturbed_prompts_tokens]
        perturbed_labels_probs = predictor(perturbed_prompts)

        # Calculate semantic similarity for potential replacements
        text_range_min, text_range_max = calculate_text_range(i, len(prompt_tokens), half_sim_score_window, sim_score_window)
        original_text_segment = ' '.join(modified_prompt_tokens[text_range_min:text_range_max])
        new_variant_segments = [' '.join(prompt[text_range_min:text_range_max]) for prompt in perturbed_prompts]
        semantic_similarities = similarity_predictor.run([original_text_segment] * len(perturbed_prompts), new_variant_segments)

        # Apply filters based on semantic similarity and POS compatibility
        valid_replacements = find_valid_replacements(perturbed_labels_probs, original_label, semantic_similarities, sim_score_threshold, synonyms, prompt_tokens, i)

        if valid_replacements:
            best_replacement = valid_replacements[0]  # Assuming find_valid_replacements returns a sorted list
            modified_prompt_tokens[i] = best_replacement
            break  # Optionally, stop after one successful change

    # Finalize and return the results
    new_label = predictor([modified_prompt_tokens]).argmax().item()
    return ' '.join(modified_prompt_tokens), original_label, new_label


def calculate_text_range(current_index, text_length, half_window_size, full_window_size):
    # If the text is shorter than the full window size, consider the whole text
    if text_length <= full_window_size:
        return 0, text_length

    # Calculate the minimum and maximum index to consider for the window
    start_index = max(0, current_index - half_window_size)
    end_index = min(text_length, current_index + half_window_size + 1)

    # Adjust the start and end index if near the beginning or end of the text
    if current_index < half_window_size:
        end_index = min(full_window_size, text_length)
    elif text_length - current_index <= half_window_size:
        start_index = max(0, text_length - full_window_size)

    return start_index, end_index


def find_valid_replacements(new_probs, original_pred_label, semantic_sims, sim_threshold, synonyms, original_text, idx, pos_tagger, pos_compatibility_checker):
    valid_replacements = []

    for i, (synonym, sim_score) in enumerate(zip(synonyms, semantic_sims)):
        if sim_score >= sim_threshold:
            # Ensure the new prediction differs from the original and meets the semantic similarity threshold
            if new_probs[i].argmax() != original_pred_label:
                # Check part-of-speech compatibility
                original_word_pos = pos_tagger(original_text[idx])
                synonym_pos = pos_tagger(synonym)
                if pos_compatibility_checker(original_word_pos, synonym_pos):
                    valid_replacements.append(synonym)

    return valid_replacements

def pick_most_similar_words_batch(
    source_word_indices: List[int], 
    similarity_matrix: np.ndarray, 
    index_to_word: dict, 
    return_count: int = 10, 
    similarity_threshold: float = 0.0
) -> Tuple[List[List[str]], List[np.ndarray]]:
    # Determine the top most similar word indices for each source word, including the source word itself.
    top_indices = np.argsort(-similarity_matrix[source_word_indices, :], axis=1)[:, :return_count + 1]

    similar_words_batch: List[List[str]] = []
    similarity_scores_batch: List[np.ndarray] = []
    for idx, word_index in enumerate(source_word_indices):
        # Extract indices and scores for the top similar words, excluding the source word itself.
        top_similar_indices = top_indices[idx]
        scores = similarity_matrix[word_index, top_similar_indices]

        # Filter based on the similarity threshold.
        valid_scores_mask = scores >= similarity_threshold
        filtered_indices = top_similar_indices[valid_scores_mask]
        filtered_scores = scores[valid_scores_mask]

        # Convert word indices to their corresponding strings, excluding the source word.
        similar_words = [index_to_word[i] for i in filtered_indices if i != word_index]

        similar_words_batch.append(similar_words)
        similarity_scores_batch.append(filtered_scores)

    return similar_words_batch, similarity_scores_batch
