import argparse
import gc
import json
import os
import statistics
import time
import traceback

# The system driver lives at /usr/lib/x86_64-linux-gnu/libcuda.so.1 but
# isn't indexed in the local ldconfig cache, so triton (used internally by
# torch) fails its `libcuda.so` probe. Pointing it explicitly at the
# multiarch dir works around that without requiring sudo to refresh the
# linker cache. This must be set before torch/triton import.
os.environ.setdefault("TRITON_LIBCUDA_PATH", "/usr/lib/x86_64-linux-gnu")

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


def _write_rollout_token_viz(
    tokenizer,
    rollout,
    json_path: str,
    html_path: str,
) -> None:
    pairs = [
        [tokenizer.decode([token_id], skip_special_tokens=False), mask]
        for token_id, mask in zip(rollout.tokens, rollout.mask)
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    pairs_json = json.dumps(pairs, ensure_ascii=False).replace("<", "\\u003c")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{ margin: 0; height: 100%; overflow: hidden; }}
.scroll {{ display: flex; overflow-x: auto; height: 100%; align-items: stretch; }}
.cell {{ flex: 0 0 auto; display: flex; flex-direction: column; border-right: 1px solid #ccc; }}
.token, .mask {{ padding: 4px 6px; font-family: monospace; font-size: 12px; }}
.token {{ white-space: pre-wrap; word-break: break-all; border-bottom: 1px solid #eee; }}
.mask-0 {{ background: #fee; }}
.mask-1 {{ background: #efe; }}
</style>
</head>
<body>
<div class="scroll" id="root"></div>
<script>
const pairs = {pairs_json};
const root = document.getElementById("root");
for (const [token, mask] of pairs) {{
  const cell = document.createElement("div");
  cell.className = "cell";
  const tok = document.createElement("div");
  tok.className = "token";
  tok.textContent = token;
  const mk = document.createElement("div");
  mk.className = "mask mask-" + mask;
  mk.textContent = mask;
  cell.appendChild(tok);
  cell.appendChild(mk);
  root.appendChild(cell);
}}
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def _report_mem(tag: str) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[mem/{tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument(
        "--num-rollout-groups", type=int, default=4,
        help=(
            "Number of rollout groups to collect per iteration. Each group "
            "is a fresh batch of `--num-rollouts` rollouts; advantages are "
            "z-scored within each group (preserving GRPO's per-group "
            "baseline), then gradients are accumulated across all groups "
            "before a single optimizer step. Use this to scale effective "
            "batch size without breaking the GRPO group-relative baseline."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument(
        "--max-rollout-tokens", type=int, default=512,
        help="max new tokens generated per turn (smaller = less memory + faster)",
    )
    parser.add_argument(
        "--max-iters", type=int, default=0,
        help="Maximum number of iterations (<= 0 means loop forever).",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="Physical GPU id used for policy training and rollouts.",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="wordle-rl",
        help="W&B project name to log to.",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="Optional run name; defaults to a wandb-generated one.",
    )
    parser.add_argument(
        "--wandb-mode", type=str, default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (default: online; assumes WANDB_API_KEY is set).",
    )
    args = parser.parse_args()

    import torch
    import wandb
    from transformers import AutoTokenizer

    from agent import run_rollouts
    from grpo import GRPOConfig, _normalize_advantages, grpo_step, load_policy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")

    torch.cuda.set_device(args.gpu)

    wandb_mode = args.wandb_mode
    if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print(
            "WARNING: WANDB_API_KEY not set in environment; "
            "falling back to wandb mode='offline'.",
            flush=True,
        )
        wandb_mode = "offline"

    if args.num_rollout_groups < 1:
        raise ValueError(
            "--num-rollout-groups must be >= 1 "
            f"(got {args.num_rollout_groups})."
        )

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=wandb_mode,
        config={
            "model_id": MODEL_ID,
            "num_rollouts": args.num_rollouts,
            "num_rollout_groups": args.num_rollout_groups,
            "effective_batch_size": args.num_rollouts * args.num_rollout_groups,
            "lr": args.lr,
            "clip_eps": args.clip_eps,
            "max_rollout_tokens": args.max_rollout_tokens,
            "gpu": args.gpu,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(
        f"\n=== Loading HF training policy (LoRA) on GPU {args.gpu} ===",
        flush=True,
    )
    cfg = GRPOConfig(lr=args.lr, clip_eps=args.clip_eps)
    policy = load_policy(MODEL_ID, cfg, device=args.gpu)
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=cfg.lr,
    )
    _report_mem("after_policy_load")

    print("\n=== Training loop ===", flush=True)
    iteration = 0
    try:
        while True:
            iteration += 1
            if args.max_iters > 0 and iteration > args.max_iters:
                break

            try:
                t0 = time.time()
                all_rollouts = []
                all_advantages: list[float] = []
                group_scores: list[list[int]] = []

                policy.eval()
                with torch.no_grad():
                    rollouts = run_rollouts(
                        policy,
                        tokenizer,
                        num_rollouts=args.num_rollouts * args.num_rollout_groups,
                        max_tokens=args.max_rollout_tokens,
                    )

                all_rollouts = rollouts
                all_advantages = []
                group_scores = []
                for g in range(args.num_rollout_groups):
                    start = g * args.num_rollouts
                    end = start + args.num_rollouts
                    group = rollouts[start:end]
                    g_scores = [r.score for r in group]
                    g_advs = _normalize_advantages(g_scores)
                    all_advantages.extend(g_advs)
                    group_scores.append(g_scores)
                    if iteration == 1 and g == 0 and group:
                        r0 = group[0]
                        json_path = os.path.join(
                            os.path.dirname(__file__) or ".",
                            "rollout_token_debug.json",
                        )
                        html_path = os.path.join(
                            os.path.dirname(__file__) or ".",
                            "rollout_token_debug.html",
                        )
                        _write_rollout_token_viz(
                            tokenizer, r0, json_path, html_path
                        )
                        with open(json_path, encoding="utf-8") as f:
                            print(f.read(), flush=True)
                        print(f"[iter 1 first rollout] wrote {html_path}", flush=True)
                policy.train()

                rollout_dt = time.time() - t0

                t1 = time.time()
                metrics = grpo_step(
                    policy, optimizer, all_rollouts, all_advantages, cfg
                )
                train_dt = time.time() - t1
            except Exception as exc:
                print(
                    f"[iter {iteration:>4}] FAILED: {exc!r}",
                    flush=True,
                )
                traceback.print_exc()
                wandb.log({"iter": iteration, "failed": 1}, step=iteration)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            scores = [s for gs in group_scores for s in gs]
            score_mean = statistics.fmean(scores) if scores else 0.0
            score_std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            score_min = min(scores) if scores else 0
            score_max = max(scores) if scores else 0
            tokens_per_rollout = [int(sum(r.mask)) for r in all_rollouts]
            tokens_mean = statistics.fmean(tokens_per_rollout) if tokens_per_rollout else 0.0
            adv_mean = statistics.fmean(all_advantages) if all_advantages else 0.0
            adv_std = statistics.pstdev(all_advantages) if len(all_advantages) > 1 else 0.0
            group_means = [
                statistics.fmean(gs) if gs else 0.0 for gs in group_scores
            ]
            group_mean_std = (
                statistics.pstdev(group_means) if len(group_means) > 1 else 0.0
            )

            log_dict = {
                "iter": iteration,
                "loss": metrics["loss"],
                "mean_ratio": metrics["mean_ratio"],
                "clip_frac": metrics["clip_frac"],
                "entropy": metrics["entropy"],
                "grad_norm": metrics["grad_norm"],
                "trained_tokens": metrics["trained_tokens"],
                "score/mean": score_mean,
                "score/std": score_std,
                "score/min": score_min,
                "score/max": score_max,
                "score/group_mean_std": group_mean_std,
                "advantage/mean": adv_mean,
                "advantage/std": adv_std,
                "tokens/per_rollout_mean": tokens_mean,
                "time/rollout_s": rollout_dt,
                "time/train_s": train_dt,
                "groups/num": args.num_rollout_groups,
                "groups/effective_batch_size": len(all_rollouts),
            }
            if torch.cuda.is_available():
                log_dict["mem/allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                log_dict["mem/reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            wandb.log(log_dict, step=iteration)

            scores_repr = (
                f"scores={scores}"
                if args.num_rollout_groups == 1
                else f"groups={group_scores}"
            )
            print(
                f"[iter {iteration:>4}] "
                f"{scores_repr} "
                f"loss={metrics['loss']:+.4f} "
                f"ratio={metrics['mean_ratio']:.3f} "
                f"clip_frac={metrics['clip_frac']:.3f} "
                f"tokens={int(metrics['trained_tokens'])} "
                f"(rollout={rollout_dt:.1f}s train={train_dt:.1f}s)",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\n(interrupted)", flush=True)
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
