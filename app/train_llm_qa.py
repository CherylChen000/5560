# app/train_llm_qa.py
#
# Module 9-style activity: fine-tune a small GPT-2 on a QA dataset.
# This will be the BASE model that will be later post-trained with RL.

import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "openai-community/gpt2"      # small GPT-2
OUTPUT_DIR = "artifacts/llm_qa"           # where to save the fine-tuned model

MAX_LENGTH = 128
BATCH_SIZE = 4
NUM_EPOCHS = 1            # keep small so it runs quickly
LR = 5e-5
WARMUP_RATIO = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 1. Load and format data
# -----------------------------

def format_example(example):
    q = example["prompt"].strip()

    best_answer_entry = min(example["answers"], key=lambda d: d["rank"])
    a = best_answer_entry["answer"].strip()

    text = f"Question: {q}\nAnswer: {a}"
    return {"text": text}


def load_and_prepare_dataset():
    # Use only a small subset for speed; Nectar has ~183k rows.
    ds = load_dataset("berkeley-nest/Nectar", split="train[:2000]")
    ds = ds.map(format_example)
    return ds


# -----------------------------
# 2. Tokenize
# -----------------------------

def tokenize_dataset(ds, tokenizer):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = ds.map(
        tokenize_batch,
        batched=True,
        remove_columns=ds.column_names,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


# -----------------------------
# 3. Training Loop
# -----------------------------

def train():
    print("Loading dataset...")
    ds = load_and_prepare_dataset()

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    tokenized_ds = tokenize_dataset(ds, tokenizer)
    train_loader = DataLoader(tokenized_ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    model.train()
    step_idx = 0
    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            step_idx += 1
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
                      f"Step {step_idx}/{total_steps} "
                      f"Loss {loss.item():.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving fine-tuned model to {OUTPUT_DIR} ...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    train()

