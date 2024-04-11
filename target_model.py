from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(device)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

def generate(prompts: list[str]) -> list[str]:
    prompts = [f"[INST]{p}[/INST]" for p in prompts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)

    responses = tokenizer.batch_decode(outputs)
    responses = [r.split("[/INST]", 1)[-1].strip() for r in responses]

    return responses
