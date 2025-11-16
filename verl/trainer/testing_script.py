from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# List of prompts
prompts = [
    "Why is the sky blue?",
    "What causes rainbows?",
    "Explain how airplanes fly.",
    "Why do leaves change color in autumn?"
]

# Tokenize as a batch
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Decoding strategies
decoding_configs = {
    "top_k": dict(top_k=50, temperature=0.7, do_sample=True),
    "top_p": dict(top_p=0.95, temperature=0.8, do_sample=True),
    "min_p": dict(min_p=0.9, temperature=0.7, do_sample=True),
    "greedy": dict(do_sample=False),
    # Beam search is omitted due to batching complexity and streaming limitation
}

# Run batch generation for each decoding strategy
for strategy, gen_args in decoding_configs.items():
    print(f"\n\n==============================")
    print(f"Decoding Strategy: {strategy}")
    print("==============================")

    # Generate
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        **gen_args
    )

    # Decode and print results
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, response in enumerate(responses):
        print(f"\nPrompt {i + 1}: {prompts[i]}")
        print(f"Response: {response}")
