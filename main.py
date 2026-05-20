import argparse
import gc
import os
import shutil
import statistics
import time
import traceback

import torch
import wandb
from transformers import AutoTokenizer

from agent import run_rollouts
from grpo import GRPOConfig, _normalize_advantages, grpo_step, load_policy

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


def _spawn_vllm(model_path: str, args):
    """Construct a fresh vLLM engine pointing at `model_path`."""
    from vllm import LLM

    return LLM(
        model=model_path,
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
        gpu_memory_utilization=args.vllm_mem,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )


def _save_snapshot(model, tokenizer, snapshot_dir: str) -> None:
    """Persist the current policy (base + merged LoRA delta) to disk so a
    fresh vLLM engine can load it as a plain Qwen3 checkpoint.

    Steps:
        1. Cache the *un-merged* base parameters to CPU. We use this to
           restore the model after saving so repeated merge/unmerge cycles
           do not accumulate bf16 round-off error.
        2. Clobber any prior snapshot at `snapshot_dir`.
        3. `model.merge_adapter()` fuses the LoRA delta into
           `base_layer.weight` in place.
        4. Build a clean state_dict whose keys match the unwrapped Qwen3
           model (no `.base_layer.`, no `lora_A`/`lora_B` entries).
        5. Hand the state_dict to `save_pretrained` so HF takes care of
           sharding, tied-weight handling, config, and generation_config.
        6. Restore the cached base weights from CPU.
    """
    base = model.get_base_model()
    base_device = next(base.parameters()).device

    base_snapshot = {
        k: v.detach().to("cpu", copy=True)
        for k, v in base.state_dict().items()
        if ".lora_A." not in k and ".lora_B." not in k
    }

    if os.path.isdir(snapshot_dir):
        shutil.rmtree(snapshot_dir)
    os.makedirs(snapshot_dir, exist_ok=True)

    model.merge_adapter()
    try:
        clean_state = {}
        for k, v in base.state_dict().items():
            if ".lora_A." in k or ".lora_B." in k:
                continue
            clean_state[k.replace(".base_layer.", ".")] = v.detach()

        base.save_pretrained(
            snapshot_dir,
            state_dict=clean_state,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(snapshot_dir)
    finally:
        with torch.no_grad():
            base.load_state_dict(
                {k: v.to(base_device) for k, v in base_snapshot.items()},
                strict=False,
            )
        del base_snapshot
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument(
        "--vllm-mem", type=float, default=0.30,
        help=(
            "gpu_memory_utilization for vLLM. Kept low so the HF policy and "
            "optimizer can share the GPU with the rollout engine."
        ),
    )
    parser.add_argument(
        "--max-rollout-tokens", type=int, default=512,
        help="max new tokens generated per turn (smaller = less memory + faster)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=6144,
        help="vLLM max_model_len; lower this if KV cache fails to fit",
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=20,
        help="Refresh the vLLM model from a new HF snapshot every N iterations.",
    )
    parser.add_argument(
        "--snapshot-dir", type=str, default="./vllm_snapshot",
        help="Path to write/clobber the rolling model snapshot.",
    )
    parser.add_argument(
        "--max-iters", type=int, default=0,
        help="Maximum number of iterations (<= 0 means loop forever).",
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

    wandb_mode = args.wandb_mode
    if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        # Don't crash an overnight run because the key wasn't exported. Fall
        # back to offline logging; the directory can be `wandb sync`ed later.
        print(
            "WARNING: WANDB_API_KEY not set in environment; "
            "falling back to wandb mode='offline'.",
            flush=True,
        )
        wandb_mode = "offline"

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=wandb_mode,
        config={
            "model_id": MODEL_ID,
            "num_rollouts": args.num_rollouts,
            "lr": args.lr,
            "clip_eps": args.clip_eps,
            "vllm_mem": args.vllm_mem,
            "max_rollout_tokens": args.max_rollout_tokens,
            "max_model_len": args.max_model_len,
            "snapshot_every": args.snapshot_every,
            "snapshot_dir": args.snapshot_dir,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("=== Loading vLLM (initial: base model) ===")
    llm = _spawn_vllm(MODEL_ID, args)
    _report_mem("after_vllm_load")

    print("\n=== Loading HF training policy (LoRA) ===")
    cfg = GRPOConfig(lr=args.lr, clip_eps=args.clip_eps)
    policy = load_policy(MODEL_ID, cfg)
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=cfg.lr,
    )
    _report_mem("after_policy_load")

    print("\n=== Training loop ===", flush=True)
    iteration = 0
    last_snapshot_iter = 0
    try:
        while True:
            iteration += 1
            if args.max_iters > 0 and iteration > args.max_iters:
                break

            try:
                t0 = time.time()
                rollouts = run_rollouts(
                    llm,
                    tokenizer,
                    num_rollouts=args.num_rollouts,
                    max_tokens=args.max_rollout_tokens,
                )
                scores = [r.score for r in rollouts]
                advantages = _normalize_advantages(scores)
                rollout_dt = time.time() - t0

                t1 = time.time()
                metrics = grpo_step(policy, optimizer, rollouts, advantages, cfg)
                train_dt = time.time() - t1
            except Exception as exc:
                # Resilience: never let a single bad iteration kill the run.
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

            score_mean = statistics.fmean(scores) if scores else 0.0
            score_std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            score_min = min(scores) if scores else 0
            score_max = max(scores) if scores else 0
            tokens_per_rollout = [int(sum(r.mask)) for r in rollouts]
            tokens_mean = statistics.fmean(tokens_per_rollout) if tokens_per_rollout else 0.0
            adv_mean = statistics.fmean(advantages) if advantages else 0.0
            adv_std = statistics.pstdev(advantages) if len(advantages) > 1 else 0.0

            log_dict = {
                "iter": iteration,
                "loss": metrics["loss"],
                "mean_ratio": metrics["mean_ratio"],
                "clip_frac": metrics["clip_frac"],
                "approx_kl": metrics["approx_kl"],
                "entropy": metrics["entropy"],
                "grad_norm": metrics["grad_norm"],
                "trained_tokens": metrics["trained_tokens"],
                "score/mean": score_mean,
                "score/std": score_std,
                "score/min": score_min,
                "score/max": score_max,
                "advantage/mean": adv_mean,
                "advantage/std": adv_std,
                "tokens/per_rollout_mean": tokens_mean,
                "time/rollout_s": rollout_dt,
                "time/train_s": train_dt,
                "iters_since_snapshot": iteration - last_snapshot_iter,
            }
            if torch.cuda.is_available():
                log_dict["mem/allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                log_dict["mem/reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            wandb.log(log_dict, step=iteration)

            print(
                f"[iter {iteration:>4}] "
                f"scores={scores} "
                f"loss={metrics['loss']:+.4f} "
                f"ratio={metrics['mean_ratio']:.3f} "
                f"clip_frac={metrics['clip_frac']:.3f} "
                f"kl={metrics['approx_kl']:.3f} "
                f"tokens={int(metrics['trained_tokens'])} "
                f"(rollout={rollout_dt:.1f}s train={train_dt:.1f}s)",
                flush=True,
            )

            if iteration % args.snapshot_every == 0:
                print(
                    f"\n--- Snapshot @ iter {iteration}: "
                    f"refreshing vLLM from latest policy ---",
                    flush=True,
                )
                snap_t = time.time()
                try:
                    # Free the engine first so the on-disk snapshot can be
                    # reloaded cleanly and the CPU snapshot doesn't crowd the
                    # GPU with two model copies.
                    _free_vllm(llm)
                    _save_snapshot(policy, tokenizer, args.snapshot_dir)
                    _report_mem("after_save_snapshot")
                    llm = _spawn_vllm(args.snapshot_dir, args)
                    _report_mem("after_vllm_reload")
                    snap_dt = time.time() - snap_t
                    last_snapshot_iter = iteration
                    wandb.log(
                        {
                            "iter": iteration,
                            "snapshot": 1,
                            "time/snapshot_s": snap_dt,
                        },
                        step=iteration,
                    )
                    print(f"Snapshot + reload: {snap_dt:.1f}s", flush=True)
                except Exception as exc:
                    print(
                        f"[snapshot @ iter {iteration}] FAILED: {exc!r}",
                        flush=True,
                    )
                    traceback.print_exc()
                    wandb.log(
                        {"iter": iteration, "snapshot_failed": 1},
                        step=iteration,
                    )
    except KeyboardInterrupt:
        print("\n(interrupted)", flush=True)
    finally:
        try:
            _free_vllm(llm)
        except Exception:
            pass
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
