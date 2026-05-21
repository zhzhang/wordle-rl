"""Subprocess worker that owns the vLLM rollout engine on a dedicated GPU.

The parent training process drives the worker over a ``multiprocessing.Pipe``
with a small command protocol. Keeping vLLM in its own process lets us pin
it to a specific physical GPU without affecting the training process, and
the worker stays resident across iterations so we can hot-reload weights in
place rather than tearing the engine down every time we want fresh samples.

GPU placement uses a custom vLLM worker that binds to ``cuda:{gpu_id}``
directly. ``CUDA_VISIBLE_DEVICES`` is intentionally *not* used: in many
container / multi-tenant setups it does not isolate GPUs (``CVD=1`` can
still land on physical device 0), while ``torch.device("cuda:1")`` works.

Protocol (parent -> worker):
    ("rollouts", {"num_rollouts": int, "targets": list[str] | None,
                  "max_tokens": int})
    ("reload_weights", {"snapshot_dir": str})
    ("shutdown", {})

Responses (worker -> parent):
    ("ready", None)                              -- once vLLM has loaded
    ("ok", payload | None)                       -- normal completion
    ("error", "<repr of exception>")             -- handled failure
"""

from __future__ import annotations

import glob
import os
import traceback

# Env var read by vllm_pinned_worker.PinnedGpuWorker inside EngineCore.
ROLLOUT_PHYSICAL_GPU_ENV = "ROLLOUT_PHYSICAL_GPU"
PINNED_WORKER_CLS = "vllm_pinned_worker.PinnedGpuWorker"


def _build_llm(
    model_id: str,
    gpu_id: int,
    gpu_memory_utilization: float,
    max_model_len: int,
):
    import torch
    from vllm import LLM

    os.environ[ROLLOUT_PHYSICAL_GPU_ENV] = str(gpu_id)

    # NOTE on `enforce_eager`: leaving it at the vLLM default (False)
    # lets the engine capture CUDA graphs for the decode path, which is
    # a large win for small models like Qwen3-4B (~1.5-2x throughput).
    # This is safe with our hot-reload flow because vLLM's
    # `model.load_weights(...)` uses `param.data.copy_(...)` -- the
    # parameter tensor addresses don't change across reloads, so any
    # captured graphs continue to point at the same memory and just
    # see new values.
    return LLM(
        model=model_id,
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        worker_cls=PINNED_WORKER_CLS,
    )


def _safetensors_weight_iter(snapshot_dir: str):
    """Yield (name, tensor) pairs from every shard in `snapshot_dir`."""
    from safetensors.torch import safe_open

    files = sorted(glob.glob(os.path.join(snapshot_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(
            f"No *.safetensors files found in {snapshot_dir!r}"
        )
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as fp:
            for k in fp.keys():
                yield k, fp.get_tensor(k)


def _resolve_vllm_model(llm):
    """Locate the underlying HF/vLLM model object that exposes
    `load_weights`. Different vLLM versions nest this slightly differently,
    so we probe a few of the known shapes."""
    engine = llm.llm_engine
    candidate_paths = (
        lambda: engine.model_executor.driver_worker.model_runner.model,
        lambda: engine.model_executor.driver_worker.worker.model_runner.model,
        lambda: engine.engine_core.model_executor.driver_worker.model_runner.model,
    )
    last_err: Exception | None = None
    for candidate in candidate_paths:
        try:
            return candidate()
        except AttributeError as err:
            last_err = err
            continue
    raise RuntimeError(
        "Unable to locate vLLM model object for hot-reload; "
        f"last AttributeError: {last_err!r}"
    )


def _hot_reload_weights(llm, gpu_id: int, snapshot_dir: str) -> None:
    """Stream weights from a Qwen3 checkpoint directory into the running
    vLLM model without recreating the engine."""
    import torch

    model = _resolve_vllm_model(llm)
    model.load_weights(_safetensors_weight_iter(snapshot_dir))

    # Prefix-cache entries reference the previous weights; flush so the next
    # rollout actually uses the freshly-loaded policy.
    reset = getattr(llm, "reset_prefix_cache", None)
    if callable(reset):
        try:
            reset()
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.synchronize(gpu_id)


def worker_main(
    conn,
    *,
    gpu_id: int,
    model_id: str,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> None:
    """Entry point invoked by `multiprocessing.Process(target=...)`.

    This function MUST run before any torch / vLLM import in this process,
    which is why the module is intentionally free of top-level GPU imports.
    """
    os.environ[ROLLOUT_PHYSICAL_GPU_ENV] = str(gpu_id)
    # Do not set CUDA_VISIBLE_DEVICES: in this environment it does not
    # isolate GPUs and causes vLLM to land on physical device 0 regardless
    # of the requested index. Pin via vllm_pinned_worker instead.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRITON_LIBCUDA_PATH", "/usr/lib/x86_64-linux-gnu")
    # Isolate torch.compile artifacts per physical GPU so a cache built
    # while running on cuda:0 is not replayed on cuda:1.
    os.environ["VLLM_CACHE_ROOT"] = os.path.join(
        os.path.expanduser("~/.cache/vllm"), f"physical_gpu_{gpu_id}"
    )

    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    try:
        from transformers import AutoTokenizer

        from agent import run_rollouts

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm = _build_llm(model_id, gpu_id, gpu_memory_utilization, max_model_len)
        conn.send(("ready", None))
    except Exception as exc:
        traceback.print_exc()
        try:
            conn.send(("error", repr(exc)))
        except Exception:
            pass
        return

    try:
        while True:
            try:
                cmd, payload = conn.recv()
            except EOFError:
                break

            try:
                if cmd == "rollouts":
                    results = run_rollouts(
                        llm,
                        tokenizer,
                        num_rollouts=payload["num_rollouts"],
                        targets=payload.get("targets"),
                        max_tokens=payload["max_tokens"],
                    )
                    conn.send(("ok", results))
                elif cmd == "reload_weights":
                    _hot_reload_weights(llm, gpu_id, payload["snapshot_dir"])
                    conn.send(("ok", None))
                elif cmd == "shutdown":
                    conn.send(("ok", None))
                    break
                else:
                    conn.send(("error", f"unknown command: {cmd!r}"))
            except Exception as exc:
                traceback.print_exc()
                try:
                    conn.send(("error", repr(exc)))
                except Exception:
                    break
    finally:
        try:
            conn.close()
        except Exception:
            pass
