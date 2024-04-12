from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5").to(device)
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
tokenizer.pad_token = tokenizer.eos_token

def generate(prompts: list[str]) -> list[str]:
    prompts = [f"[INST]{p}[/INST]" for p in prompts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)

    responses = tokenizer.batch_decode(outputs)
    responses = [r.split("[/INST]", 1)[-1].strip() for r in responses]

    return responses
