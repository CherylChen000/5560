# app/rl_posttrain_llm.py
#
# Post-train the Module 9 LLM with RL so answers follow a fixed format.

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import optim

from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_PATH = "artifacts/llm_qa"        # base LLM from Module 9
OUTPUT_MODEL_PATH = "artifacts/llm_qa_rl"

# Fixed format 
PREFIX = "That is a great question. "
SUFFIX = " Let me know if you have any other questions."

MAX_NEW_TOKENS = 64
LR = 1e-5
BATCH_SIZE = 4      # episodes per epoch
EPOCHS = 10

QUESTIONS = [
    "What is reinforcement learning?",
    "How does a diffusion model work?",
    "What is the difference between a GAN and an energy-based model?",
    "Why do we fine-tune a language model?",
]


# -----------------------------
# Reward function
# -----------------------------

def format_reward(answer: str) -> float:
    ans = answer.strip()
    reward = 0.0
    if ans.startswith(PREFIX.strip()):
        reward += 0.5
    if ans.endswith(SUFFIX.strip()):
        reward += 0.5
    return reward   # 0.0, 0.5, or 1.0


# -----------------------------
# Environment 
# -----------------------------

@dataclass
class Environment:
    questions: List[str]

    def reset(self) -> str:
        self.current_question = random.choice(self.questions)
        return self.current_question

    def step(self, answer: str) -> Tuple[None, float, bool]:
        r = format_reward(answer)
        done = True
        return None, r, done


# -----------------------------
# Policy wrapper around LLM
# -----------------------------

class Policy:
    def __init__(self, model, tokenizer, optimizer, batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.batch_size = batch_size

    def build_prompt(self, question: str) -> str:
        return f"Question: {question}\nAnswer:"

    def get_action(self, question: str):
        """
        Generate an answer AND compute log-prob of the generated tokens.
        Mirrors the Policy.get_action pattern from Module 11.
        """
        prompt = self.build_prompt(question)
        enc = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # 1) sample answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )

        full_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        answer = full_text[len(prompt):]

        # 2) compute log-prob with grad
        input_ids = generated_ids.to(DEVICE)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits

        log_probs_all = F.log_softmax(logits, dim=-1)
        gen_token_ids = input_ids[:, 1:]
        gen_log_probs = log_probs_all[:, :-1, :].gather(
            dim=-1,
            index=gen_token_ids.unsqueeze(-1),
        ).squeeze(-1)

        # prompt token length
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
        prompt_len = prompt_ids.shape[1]

        # Only use tokens after the prompt
        gen_log_probs_answer = gen_log_probs[:, prompt_len-1:]
        logp = gen_log_probs_answer.sum()

        return answer, logp

    def compute_loss(self, logp_tensor, weights_tensor):
        # REINFORCE: -E[R * log pi(a|s)]
        return -(logp_tensor * weights_tensor).mean()

    def train_one_epoch(self, env: Environment):
        batch_logps = []
        batch_rewards = []

        for _ in range(self.batch_size):
            question = env.reset()
            answer, logp = self.get_action(question)
            _, reward, done = env.step(answer)

            batch_logps.append(logp)
            batch_rewards.append(reward)

        logp_tensor = torch.stack(batch_logps)
        reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=DEVICE)

        if reward_tensor.std() > 1e-6:
            reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)

        loss = self.compute_loss(logp_tensor, reward_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        avg_reward = float(sum(batch_rewards) / len(batch_rewards))
        return loss.item(), avg_reward


# -----------------------------
# Main training loop
# -----------------------------

def main():
    print(f"Loading base model from {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(DEVICE)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    env = Environment(questions=QUESTIONS)
    policy = Policy(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
    )

    for epoch in range(1, EPOCHS + 1):
        loss, avg_reward = policy.train_one_epoch(env)
        print(f"[Epoch {epoch:02d}] loss={loss:.4f}  avg_reward={avg_reward:.3f}")

    print(f"Saving RL-posttrained model to {OUTPUT_MODEL_PATH}")
    tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
    model.save_pretrained(OUTPUT_MODEL_PATH)


if __name__ == "__main__":
    main()
