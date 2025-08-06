from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./model/checkpoint-15000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

def chat(prompt, max_length=128):
    input_text = prompt.strip()
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() in {"exit", "quit"}:
            break
        response = chat(prompt)
        print("Response:", response)
