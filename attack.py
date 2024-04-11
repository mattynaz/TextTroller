import torch
from nltk.tokenize import word_tokenize
import judge_model
import target_model
import semantic_similarity
from util import stop_words_set, get_pos


def attack(
    prompt: str,
    importance_score_threshold: float = -1.0,
    sim_score_threshold: float = 0.05,
    sim_score_window: int = 15,
    num_synonyms: int = 12,
):
    num_calls = 0

    def predict(prompts: list[str]):
        responses = target_model.generate(prompts)
        return judge_model.judge(responses, prompt), responses

    # Get the original prediction and its confidence
    original_label_probs, original_response = predict([prompt])
    num_calls += 1
    original_label_probs = original_label_probs.squeeze()
    original_label = original_label_probs.argmax().item()
    original_label_confidence = original_label_probs.max()

    # Get the part-of-speech tagger and compatibility checker
    prompt_tokens = word_tokenize(prompt)
    prompt_tokens_pos = get_pos(prompt_tokens)

    # Adjust the similarity threshold for shorter texts
    adjusted_sim_score_threshold = 0.1 if len(prompt_tokens) < sim_score_window else sim_score_threshold
    half_sim_score_window = (sim_score_window - 1) // 2

    # Generate perturbed versions of the original text by replacing each token with '<oov>'
    perturbed_prompts_tokens = [prompt_tokens[:i] + ['<oov>'] + prompt_tokens[i+1:] for i in range(len(prompt_tokens))]
    perturbed_prompts = [' '.join(t for t in tokens if t != '<oov>') for tokens in perturbed_prompts_tokens]
    perturbed_labels_probs, _ = predict(perturbed_prompts)
    num_calls += len(perturbed_prompts)
    perturbed_labels = perturbed_labels_probs.argmax(dim=-1)

    # Calculate importance scores for each token
    confidence_losses = original_label_confidence - perturbed_labels_probs[:, original_label]
    prediction_changes = (perturbed_labels != original_label).float()
    new_pred_confidences = perturbed_labels_probs.max(dim=-1)[0]
    prediction_change_losses = new_pred_confidences - torch.index_select(original_label_probs, 0, perturbed_labels)
    importance_scores = confidence_losses + prediction_changes * prediction_change_losses

    # Filter words based on importance scores and exclusion from stop words set
    importance_scores_sorted = sorted(enumerate(importance_scores), key=lambda x: x[1], reverse=True)
    perturbable_words_idxs = [
        i for (i, score) in importance_scores_sorted
        if score > importance_score_threshold and prompt_tokens[i] not in stop_words_set and prompt_tokens[i]
    ]

    # Lookup synonyms for the filtered words
    perturbable_words = [prompt_tokens[i] for i in perturbable_words_idxs]
    perturbable_words_pos = [prompt_tokens_pos[i] for i in perturbable_words_idxs]
    synonyms_list = judge_model.generate_synonyms(perturbable_words, perturbable_words_pos, num_synonyms)
    num_calls += len(synonyms_list)

    # Attempt to replace words with their synonyms to alter the model's prediction
    modified_prompt_tokens = prompt_tokens.copy()
    for i, synonyms in zip(perturbable_words_idxs, synonyms_list):
        # For each synonym, create new text versions and predict their labels
        perturbed_prompts_tokens = [modified_prompt_tokens[:i] + [synonym] + modified_prompt_tokens[i+1:] for synonym in synonyms]
        perturbed_prompts = [' '.join(tokens) for tokens in perturbed_prompts_tokens]
        perturbed_labels_probs, _ = predict(perturbed_prompts)
        num_calls += len(perturbed_prompts)

        # Calculate semantic similarity for potential replacements
        text_range_min, text_range_max = calculate_text_range(i, len(prompt_tokens), half_sim_score_window, sim_score_window)
        original_text_segment = ' '.join(modified_prompt_tokens[text_range_min:text_range_max])
        new_variant_segments = [' '.join(tokens[text_range_min:text_range_max]) for tokens in perturbed_prompts_tokens]
        semantic_similarities = semantic_similarity.similarity([original_text_segment] * len(perturbed_prompts), new_variant_segments)
        
        perturbed_labels_probs_mask = (original_label != perturbed_labels_probs.argmax(dim=-1)).cpu()
        perturbed_labels_probs_mask *= (semantic_similarities >= sim_score_threshold).cpu()

        if perturbed_labels_probs_mask.any().item():
            modified_prompt_tokens[i] = synonyms[(perturbed_labels_probs_mask * semantic_similarities).argmax()]
        else:
            new_label_probs = perturbed_labels_probs[:, original_label].cpu() + (semantic_similarities < sim_score_threshold).float()
            new_label_prob_min, new_label_prob_argmin = new_label_probs.min(dim=-1)
            if new_label_prob_min < original_label_confidence:
                modified_prompt_tokens[i] = synonyms[new_label_prob_argmin]

    # Finalize and return the results
    attacked_prompt = ' '.join(modified_prompt_tokens)
    _, attacked_response = predict([attacked_prompt])
    num_calls += 1
    return original_response, attacked_prompt, attacked_response, num_calls


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
