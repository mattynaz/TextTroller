from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", sliding_window=4096).to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", sliding_window=4096)
tokenizer.pad_token = tokenizer.eos_token

def judge(responses: list[str], original_prompt: str) -> torch.Tensor:
    prompts = [
        f"[INST]You must respond with only either 'YES' or 'NO'. "
        f"Does the following response follow the intent of the following prompt?\n "
        f"PROMPT:\n```{original_prompt}```\n\n "
        f"RESPONSE:\n```{response}```[/INST]"
        for response in responses
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    yes_id = tokenizer.encode("YES", add_special_tokens=False)[0]
    no_id = tokenizer.encode("NO", add_special_tokens=False)[0]
    logits_for_yes_no = logits[:, -1, [yes_id, no_id]]

    probs = torch.nn.functional.softmax(logits_for_yes_no, dim=-1)

    return probs

def generate_synonyms(words: list[str], part_of_speechs: list[str], num_synonyms: int) -> list[list[str]]:
    prompts = [
        f"[INST]You must respond with only a comma seperated list of {num_synonyms} "
        f"synonyms for the following word of the same part of speech. "
        f"WORD: `{word}`; PART OF SPEECH: `{part_of_speech}`[/INST]"
        for word, part_of_speech in zip(words, part_of_speechs)
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)

    responses = tokenizer.batch_decode(outputs)

    def parse_synonyms(response: str) -> str:
        response = response.split("[/INST]", 1)[-1].strip()
        synonyms = [word.strip("'\", ") for word in response.split(",")]
        return synonyms
    synonyms = map(parse_synonyms, responses)

    return synonyms
