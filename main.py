import argparse
import gc

import torch
from transformers import AutoTokenizer

from agent import run_rollouts
from grpo import GRPOConfig, grpo_step, load_policy, _normalize_advantages

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


def _free_vllm(llm) -> None:
    """Best-effort teardown of a vLLM engine to free GPU memory."""
    try:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception as exc:
        print(f"(vLLM shutdown raised: {exc!r}; continuing)")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _report_mem(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[mem/{tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument(
        "--vllm-mem", type=float, default=0.45,
        help="gpu_memory_utilization for vLLM during rollout phase",
    )
    parser.add_argument(
        "--max-rollout-tokens", type=int, default=512,
        help="max new tokens generated per turn (smaller = less memory + faster)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="vLLM max_model_len; lower this if KV cache fails to fit",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("=== Phase 1: vLLM rollouts ===")
    from vllm import LLM
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
        gpu_memory_utilization=args.vllm_mem,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )
    _report_mem("after_vllm_load")

    rollouts = run_rollouts(
        llm,
        tokenizer,
        num_rollouts=args.num_rollouts,
        max_tokens=args.max_rollout_tokens,
    )
    scores = [r.score for r in rollouts]
    advantages = _normalize_advantages(scores)

    print("\n--- Rollout summary ---")
    for i, (r, s, a) in enumerate(zip(rollouts, scores, advantages)):
        gen = sum(r.mask)
        print(
            f"  Rollout {i + 1}: score={s}, advantage={a:+.4f}, "
            f"tokens={len(r.tokens)}, generated={gen}"
        )

    _free_vllm(llm)
    _report_mem("after_vllm_free")

    print("\n=== Phase 2: GRPO update ===")
    cfg = GRPOConfig(lr=args.lr, clip_eps=args.clip_eps)
    policy = load_policy(MODEL_ID, cfg)
    _report_mem("after_policy_load")

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=cfg.lr,
    )

    metrics = grpo_step(policy, optimizer, rollouts, advantages, cfg)

    print("\n--- GRPO step metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    _report_mem("after_step")


if __name__ == "__main__":
    main()
