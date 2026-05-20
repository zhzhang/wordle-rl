import argparse

import torch
from transformers import AutoTokenizer
from vllm import LLM

from agent import run_rollouts

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


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
    llm = LLM(
        model=MODEL_ID,
        dtype="float16" if torch.cuda.is_available() else "float32",
    )

    results = run_rollouts(llm, tokenizer, num_rollouts=args.num_rollouts)

    scores = [r.score for r in results]
    advantages = compute_normalized_advantages(scores)

    print("\n=== Summary ===")
    for i, (score, adv) in enumerate(zip(scores, advantages)):
        print(
            f"Rollout {i + 1}: score={score}, advantage={adv:+.4f}, "
            f"tokens={len(results[i].tokens)}, generated={sum(results[i].mask)}"
        )


if __name__ == "__main__":
    main()
