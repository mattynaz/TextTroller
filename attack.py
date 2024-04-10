import torch
import criteria
import numpy as np
from nltk.tokenize import word_tokenize

def attack(
    prompt: str,  # The list of words in the original text
    true_label,  # The correct label for the original text
    predictor,  # A function that predicts the label based on the text input
    stop_words_set,  # A set of words to ignore when looking for words to replace
    word2idx,  # A dictionary mapping words to their indices
    idx2word,  # A dictionary mapping indices back to words
    cos_sim,  # A function to compute the cosine similarity between word vectors
    sim_predictor,  # A function to compute semantic similarity between texts
    importance_score_threshold=-1.0,  # Threshold for considering a word's importance score
    sim_score_threshold=0.5,  # Threshold for considering semantic similarity as sufficient
    sim_score_window=15,  # The window size to consider for semantic similarity calculation
    synonym_num=50,  # The number of synonyms to consider for each word
) -> tuple:
    
    prompt_tokens = word_tokenize(prompt)
    original_pred_probs = predictor([prompt]).squeeze()
    original_pred_label = original_pred_probs.argmax()
    original_pred_confidence = original_pred_probs.max()

    # Return early if the original prediction is incorrect
    if true_label != original_pred_label:
        return "", 0, original_pred_label, original_pred_label, 0

    # Adjust the similarity threshold for shorter texts
    adjusted_sim_score_threshold = 0.1 if len(prompt_tokens) < sim_score_window else sim_score_threshold
    half_sim_score_window = (sim_score_window - 1) // 2

    # Generate perturbed versions of the original text by replacing each token with '<oov>'
    perturbed_prompt_tokens = [prompt_tokens[:i] + ['<oov>'] + prompt_tokens[i+1:] for i in range(len(prompt_tokens))]
    perturbed_prompts = [' '.join(tokens) for tokens in perturbed_prompt_tokens]
    perturbed_pred_probs = predictor(perturbed_prompts)
    perturbed_pred_labels = perturbed_pred_probs.argmax(dim=-1)

    # Calculate importance scores for each token
    confidence_losses = original_pred_confidence - perturbed_pred_probs[:, original_pred_label]
    prediction_changes = (perturbed_pred_labels != original_pred_label).float()
    new_pred_confidences = perturbed_pred_probs.max(dim=-1)[0]
    prediction_change_losses = new_pred_confidences - torch.index_select(original_pred_probs, 0, perturbed_pred_labels)
    importance_scores = confidence_losses + prediction_changes * prediction_change_losses

    # Filter words based on importance scores and exclusion from stop words set
    importance_scores_sorted = sorted(enumerate(importance_scores), key=lambda x: x[1], reverse=True)
    perturbable_words = [
        (i, prompt_tokens[i]) for (i, score) in importance_scores_sorted
        if score > importance_score_threshold and prompt_tokens[i] not in stop_words_set and prompt_tokens[i] in word2idx
    ]

    # =================================================================================================
    # =================================================================================================
    # =================================================================================================

    # Lookup synonyms for the filtered words
    perturbable_word_indices = [word2idx[word] for _, word in perturbable_words]
    synonyms_list, _ = pick_most_similar_words_batch(perturbable_word_indices, cos_sim, idx2word, synonym_num, adjusted_sim_score_threshold)
    synonyms_for_perturbation = [
        (idx, synonyms) for (idx, _), synonyms in zip(perturbable_words, synonyms_list) if synonyms
    ]
    

    # Attempt to replace words with their synonyms to alter the model's prediction
    modified_text = prompt_tokens[:]  # Initialize modified text with original text
    changes_made = 0
    for idx, synonyms in synonyms_for_perturbation:
        # For each synonym, create new text versions and predict their labels
        new_text_variants = [modified_text[:idx] + [synonym] + modified_text[idx+1:] for synonym in synonyms]
        new_variant_probs = predictor(new_text_variants)

        # Calculate semantic similarity for potential replacements
        text_range_min, text_range_max = calculate_text_range(idx, len(prompt_tokens), half_sim_score_window, sim_score_window)
        original_text_segment = ' '.join(modified_text[text_range_min:text_range_max])
        new_variant_segments = [' '.join(variant[text_range_min:text_range_max]) for variant in new_text_variants]
        semantic_similarities = sim_predictor([original_text_segment] * len(new_text_variants), new_variant_segments)

        # Apply filters based on semantic similarity and POS compatibility
        valid_replacements = find_valid_replacements(new_variant_probs, original_pred_label, semantic_similarities, sim_score_threshold, synonyms, x_tokens, idx)

        if valid_replacements:
            best_replacement = valid_replacements[0]  # Assuming find_valid_replacements returns a sorted list
            modified_text[idx] = best_replacement
            changes_made += 1
            break  # Optionally, stop after one successful change

    # Finalize and return the results
    new_label = predictor([modified_text]).argmax().item()
    return ' '.join(modified_text), changes_made, original_pred_label, new_label, len(synonyms_for_perturbation)


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


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.0):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values
