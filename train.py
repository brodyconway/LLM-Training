from datasets import Dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)
import torch
import pandas as pd

def load_and_format_data(parquet_path):
    print("Loading data")

    df = pd.read_parquet(parquet_path)

    prompts = []
    responses = []
    for q, r in zip(df["question"], df["response"]):
        if isinstance(q, str) and isinstance(r, str):
            q_clean = q.strip()
            r_clean = r.strip()
            if q_clean and r_clean:
                prompts.append(q_clean)
                responses.append(r_clean)

    return Dataset.from_dict({"prompt": prompts, "response": responses})


def tokenize_data(example, tokenizer):
    full_text = example["prompt"].strip() + " " + tokenizer.eos_token + " " + example["response"].strip()
    
    tokens = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    model_name = "gpt2"
    parquet_path = "1M-GPT4-Augmented.parquet"

    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config()
    model = GPT2LMHeadModel(config)

    dataset = load_and_format_data(parquet_path)

    print("Tokenizing dataset")
    tokenized_dataset = dataset.map(lambda x: tokenize_data(x, tokenizer), batched=False)

    print("Setting up training")

    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cuda" if torch.cuda.is_available() else "cpu"

    training_args = TrainingArguments(
        output_dir="./model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=5000,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        use_mps_device=use_mps,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting training")
    trainer.train()

    print("Training complete. Model saved in model")


if __name__ == "__main__":
    main()

