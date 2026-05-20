import argparse
import gc
import multiprocessing as mp
import os
import shutil
import statistics
import time
import traceback

# The system driver lives at /usr/lib/x86_64-linux-gnu/libcuda.so.1 but
# isn't indexed in the local ldconfig cache, so triton (used internally by
# torch and by vLLM) fails its `libcuda.so` probe. Pointing it explicitly
# at the multiarch dir works around that without requiring sudo to refresh
# the linker cache. This must be set before torch/triton import.
os.environ.setdefault("TRITON_LIBCUDA_PATH", "/usr/lib/x86_64-linux-gnu")

import torch
import wandb
from transformers import AutoTokenizer

from grpo import GRPOConfig, _normalize_advantages, grpo_step, load_policy
from rollout_worker import worker_main

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


def _report_mem(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[mem/{tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")


class _SnapshotSaver:
    """Persist the policy (base + merged LoRA delta) as a vanilla Qwen3
    checkpoint on disk so the rollout worker can hot-reload it.

    Caches a CPU copy of the base weights on first use, since:
      * the base weights never change due to LoRA training, so a single
        snapshot is a sufficient restore source;
      * restoring from the CPU snapshot after `merge_adapter` avoids the
        bf16 round-off that `merge` -> `unmerge` would otherwise accumulate
        across many sync iterations.

    Avoiding the per-iter CPU clone of an ~8GB base is essential for
    making `sync_every == 1` practical.
    """

    def __init__(self, policy):
        self._base = policy.get_base_model()
        self._device = next(self._base.parameters()).device
        self._base_cpu = {
            k: v.detach().to("cpu", copy=True)
            for k, v in self._base.state_dict().items()
            if ".lora_A." not in k and ".lora_B." not in k
        }

    def save(self, policy, tokenizer, out_dir: str) -> None:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        policy.merge_adapter()
        try:
            clean_state = {}
            for k, v in self._base.state_dict().items():
                if ".lora_A." in k or ".lora_B." in k:
                    continue
                clean_state[k.replace(".base_layer.", ".")] = v.detach()

            self._base.save_pretrained(
                out_dir,
                state_dict=clean_state,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(out_dir)
        finally:
            with torch.no_grad():
                self._base.load_state_dict(
                    {k: v.to(self._device) for k, v in self._base_cpu.items()},
                    strict=False,
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class RolloutWorker:
    """Parent-side handle to the vLLM rollout subprocess.

    The worker process owns a vLLM engine pinned to a dedicated GPU and
    services rollout + weight-reload requests over a Pipe.
    """

    def __init__(
        self,
        *,
        gpu_id: int,
        model_id: str,
        gpu_memory_utilization: float,
        max_model_len: int,
    ):
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(
            target=worker_main,
            args=(child_conn,),
            kwargs={
                "gpu_id": gpu_id,
                "model_id": model_id,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
            },
            daemon=False,
        )
        self._conn = parent_conn
        self._proc.start()
        # vLLM cold start can take a minute or so; block here until the
        # child reports `ready` (or an explicit error).
        status, payload = self._conn.recv()
        if status != "ready":
            raise RuntimeError(f"Rollout worker failed to start: {payload!r}")

    def rollouts(self, num_rollouts: int, max_tokens: int, targets=None):
        self._conn.send((
            "rollouts",
            {
                "num_rollouts": num_rollouts,
                "targets": targets,
                "max_tokens": max_tokens,
            },
        ))
        status, payload = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"rollouts failed: {payload!r}")
        return payload

    def reload_weights(self, snapshot_dir: str) -> None:
        self._conn.send(("reload_weights", {"snapshot_dir": snapshot_dir}))
        status, payload = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"reload_weights failed: {payload!r}")

    def shutdown(self) -> None:
        try:
            self._conn.send(("shutdown", {}))
            try:
                self._conn.recv()
            except Exception:
                pass
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
        self._proc.join(timeout=20)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.kill()


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
        "--vllm-mem", type=float, default=0.85,
        help=(
            "gpu_memory_utilization for vLLM on its dedicated GPU. With the "
            "rollout engine on its own card we can give it most of the VRAM."
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
        "--sync-every", type=int, default=20,
        help=(
            "Hot-reload the rollout engine with freshly-trained weights "
            "every N iterations. Default 1 (sync after every grpo step)."
        ),
    )
    parser.add_argument(
        "--snapshot-dir", type=str, default="./vllm_snapshot",
        help=(
            "Path to write the rolling merged-policy snapshot before the "
            "rollout worker hot-reloads it. Putting this on /dev/shm or "
            "another tmpfs makes snapshotting an order of magnitude faster."
        ),
    )
    parser.add_argument(
        "--max-iters", type=int, default=0,
        help="Maximum number of iterations (<= 0 means loop forever).",
    )
    parser.add_argument(
        "--train-gpu", type=int, default=0,
        help="Physical GPU id used by the HF training policy.",
    )
    parser.add_argument(
        "--rollout-gpu", type=int, default=1,
        help="Physical GPU id used by the vLLM rollout engine.",
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

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(
            "This training script expects at least two CUDA GPUs: one for "
            "the HF training policy and one for the vLLM rollout engine. "
            f"Detected {torch.cuda.device_count()} GPU(s)."
        )
    if args.train_gpu == args.rollout_gpu:
        raise ValueError(
            "--train-gpu and --rollout-gpu must point at different physical "
            f"GPUs (got both == {args.train_gpu})."
        )

    # The training process keeps the full CUDA topology visible so the HF
    # policy can land on `train_gpu`. The rollout worker masks
    # CUDA_VISIBLE_DEVICES inside its own process so vLLM only sees
    # `rollout_gpu`.
    torch.cuda.set_device(args.train_gpu)

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
            "vllm_mem": args.vllm_mem,
            "max_rollout_tokens": args.max_rollout_tokens,
            "max_model_len": args.max_model_len,
            "sync_every": args.sync_every,
            "snapshot_dir": args.snapshot_dir,
            "train_gpu": args.train_gpu,
            "rollout_gpu": args.rollout_gpu,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(
        f"=== Spawning vLLM rollout worker on GPU {args.rollout_gpu} "
        f"(this may take a minute) ===",
        flush=True,
    )
    rollout_worker = RolloutWorker(
        gpu_id=args.rollout_gpu,
        model_id=MODEL_ID,
        gpu_memory_utilization=args.vllm_mem,
        max_model_len=args.max_model_len,
    )

    print(
        f"\n=== Loading HF training policy (LoRA) on GPU {args.train_gpu} ===",
        flush=True,
    )
    cfg = GRPOConfig(lr=args.lr, clip_eps=args.clip_eps)
    policy = load_policy(MODEL_ID, cfg)
    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=cfg.lr,
    )
    _report_mem("after_policy_load")

    print("\n=== Caching base weights on CPU for snapshot restore ===", flush=True)
    saver = _SnapshotSaver(policy)
    _report_mem("after_snapshot_saver")

    print("\n=== Training loop ===", flush=True)
    iteration = 0
    last_sync_iter = 0
    try:
        while True:
            iteration += 1
            if args.max_iters > 0 and iteration > args.max_iters:
                break

            try:
                # Collect `num_rollout_groups` independent groups. Each group
                # gets its own z-scored advantages (the per-group GRPO
                # baseline), then we hand the concatenated batch to
                # `grpo_step`, which `.backward()`s each rollout in sequence
                # and only calls `optimizer.step()` after the full batch --
                # i.e. gradients accumulate across groups.
                t0 = time.time()
                all_rollouts = []
                all_advantages: list[float] = []
                group_scores: list[list[int]] = []
                for _ in range(args.num_rollout_groups):
                    group = rollout_worker.rollouts(
                        num_rollouts=args.num_rollouts,
                        max_tokens=args.max_rollout_tokens,
                    )
                    g_scores = [r.score for r in group]
                    g_advs = _normalize_advantages(g_scores)
                    all_rollouts.extend(group)
                    all_advantages.extend(g_advs)
                    group_scores.append(g_scores)
                rollout_dt = time.time() - t0

                t1 = time.time()
                metrics = grpo_step(
                    policy, optimizer, all_rollouts, all_advantages, cfg
                )
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

            scores = [s for gs in group_scores for s in gs]
            score_mean = statistics.fmean(scores) if scores else 0.0
            score_std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            score_min = min(scores) if scores else 0
            score_max = max(scores) if scores else 0
            tokens_per_rollout = [int(sum(r.mask)) for r in all_rollouts]
            tokens_mean = statistics.fmean(tokens_per_rollout) if tokens_per_rollout else 0.0
            adv_mean = statistics.fmean(all_advantages) if all_advantages else 0.0
            adv_std = statistics.pstdev(all_advantages) if len(all_advantages) > 1 else 0.0
            # Spread of per-group mean scores -- if this is small, the
            # multi-group batch is just adding samples around the same
            # baseline; if it's large, different groups are exploring very
            # different regions of the score distribution.
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
                "approx_kl": metrics["approx_kl"],
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
                "iters_since_sync": iteration - last_sync_iter,
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
                f"kl={metrics['approx_kl']:.3f} "
                f"tokens={int(metrics['trained_tokens'])} "
                f"(rollout={rollout_dt:.1f}s train={train_dt:.1f}s)",
                flush=True,
            )

            if args.sync_every > 0 and iteration % args.sync_every == 0:
                sync_t = time.time()
                try:
                    save_t = time.time()
                    saver.save(policy, tokenizer, args.snapshot_dir)
                    save_dt = time.time() - save_t
                    _report_mem("after_save_snapshot")

                    reload_t = time.time()
                    rollout_worker.reload_weights(args.snapshot_dir)
                    reload_dt = time.time() - reload_t

                    sync_dt = time.time() - sync_t
                    last_sync_iter = iteration
                    wandb.log(
                        {
                            "iter": iteration,
                            "sync": 1,
                            "time/sync_s": sync_dt,
                            "time/sync_save_s": save_dt,
                            "time/sync_reload_s": reload_dt,
                        },
                        step=iteration,
                    )
                    print(
                        f"Sync @ iter {iteration}: "
                        f"save={save_dt:.1f}s reload={reload_dt:.1f}s "
                        f"(total {sync_dt:.1f}s)",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[sync @ iter {iteration}] FAILED: {exc!r}",
                        flush=True,
                    )
                    traceback.print_exc()
                    wandb.log(
                        {"iter": iteration, "sync_failed": 1},
                        step=iteration,
                    )
    except KeyboardInterrupt:
        print("\n(interrupted)", flush=True)
    finally:
        try:
            rollout_worker.shutdown()
        except Exception:
            pass
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
