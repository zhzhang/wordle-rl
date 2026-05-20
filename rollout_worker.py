"""Subprocess worker that owns the vLLM rollout engine on a dedicated GPU.

The parent training process drives the worker over a ``multiprocessing.Pipe``
with a small command protocol. Keeping vLLM in its own process lets us pin
it to a specific GPU via ``CUDA_VISIBLE_DEVICES`` without affecting the
training process, and the worker stays resident across iterations so we can
hot-reload weights in place rather than tearing the engine down every time
we want fresh samples.

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


def _build_llm(model_id: str, gpu_memory_utilization: float, max_model_len: int):
    import torch
    from vllm import LLM

    return LLM(
        model=model_id,
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=max_model_len,
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


def _hot_reload_weights(llm, snapshot_dir: str) -> None:
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
        torch.cuda.synchronize()


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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # See main.py for the rationale. The child is a fresh spawn so we have
    # to set this here too, before vLLM imports triton.
    os.environ.setdefault("TRITON_LIBCUDA_PATH", "/usr/lib/x86_64-linux-gnu")

    try:
        from transformers import AutoTokenizer

        from agent import run_rollouts

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm = _build_llm(model_id, gpu_memory_utilization, max_model_len)
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
                    _hot_reload_weights(llm, payload["snapshot_dir"])
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
