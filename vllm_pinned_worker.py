"""vLLM worker subclass pinned to a physical CUDA device index.

Imported by vLLM's EngineCore subprocess via ``worker_cls=
"vllm_pinned_worker.PinnedGpuWorker"``. The target GPU index comes from
``ROLLOUT_PHYSICAL_GPU`` in the environment (set by ``rollout_worker`` before
``LLM(...)`` is constructed).
"""

from __future__ import annotations

import os

from vllm.v1.worker.gpu_worker import Worker as GpuWorker

ROLLOUT_PHYSICAL_GPU_ENV = "ROLLOUT_PHYSICAL_GPU"


class PinnedGpuWorker(GpuWorker):
    def init_device(self):
        import torch

        physical_gpu = int(os.environ[ROLLOUT_PHYSICAL_GPU_ENV])
        if torch.cuda.is_available():
            torch.cuda.set_device(physical_gpu)
        self.local_rank = physical_gpu
        super().init_device()
