import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import run_agent

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def compute_normalized_advantages(scores: list[int]) -> list[float]:
    n = len(scores)
    if n == 0:
        return []
    mean = sum(scores) / n
    if n == 1:
        return [0.0]
    var = sum((s - mean) ** 2 for s in scores) / n
    std = var ** 0.5
    if std == 0:
        return [0.0] * n
    return [(s - mean) / std for s in scores]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    results = []
    for i in range(args.num_rollouts):
        print(f"\n=== Rollout {i + 1}/{args.num_rollouts} ===")
        result = run_agent(model, tokenizer)
        print(f"Rollout {i + 1} score: {result.score} ({len(result.tokens)} tokens, {sum(result.mask)} agent-generated)")
        results.append(result)

    scores = [r.score for r in results]
    advantages = compute_normalized_advantages(scores)

    print("\n=== Summary ===")
    for i, (score, adv) in enumerate(zip(scores, advantages)):
        print(f"Rollout {i + 1}: score={score}, advantage={adv:+.4f}")


if __name__ == "__main__":
    main()
