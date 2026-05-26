"""Background CPU-prefetch worker for text-encoder embeddings.

Phase F — adds a 4th text_encoding_mode ("cpu_prefetch") that keeps the
frozen Text Encoder on CPU and runs caption encoding for the *next* batch
(and up to `prefetch_depth` batches ahead) on a daemon thread while the
main thread is busy with GPU forward / backward. PyTorch CPU matmul
kernels release the GIL, so the worker really overlaps with GPU compute.

Backpressure: the worker pushes (batch_index, embeddings_dict) onto a
queue.Queue(maxsize=prefetch_depth). Once `depth` batches are ready the
worker blocks on .put() until the main thread consumes one — the natural
flow-control. Conversely the main thread blocks on .get() if the worker
hasn't caught up; this is the "stall" we want to measure.

The embeddings_dict is keyed by image_path, matching the swap_buffer
contract the rest of base_trainer.py already uses, so the only change at
the consumer site is "instead of synchronously refilling, pull the next
dict from the worker and merge it into swap_buffer".
"""

from __future__ import annotations

import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch


def _detach_to_cpu(obj: Any) -> Any:
    """Move arbitrary {tensor / dict / tuple / None} payloads to CPU + detach.

    Used on the auxiliary value returned by encode_caption — it can be a
    tensor (Z-Image attention_mask, SDXL pooled_embeds), a dict (Anima
    aux bundle), or None (SD1.5).
    """
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=False)
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_detach_to_cpu(v) for v in obj]
        return type(obj)(out)
    return obj


@dataclass
class CpuPrefetchStats:
    """Telemetry for one training run worth of prefetch activity."""
    total_pulls: int = 0
    stalled_pulls: int = 0
    total_stall_seconds: float = 0.0
    cpu_encode_seconds: float = 0.0
    fastest_encode: float = math.inf
    slowest_encode: float = -math.inf
    batches_encoded: int = 0
    samples_encoded: int = 0
    worker_errors: List[str] = field(default_factory=list)

    def record_encode(self, seconds: float, n_samples: int) -> None:
        self.cpu_encode_seconds += seconds
        self.batches_encoded += 1
        self.samples_encoded += n_samples
        if seconds < self.fastest_encode:
            self.fastest_encode = seconds
        if seconds > self.slowest_encode:
            self.slowest_encode = seconds

    def record_pull(self, stall_seconds: float) -> None:
        self.total_pulls += 1
        if stall_seconds > 1e-3:  # >1ms counts as a real stall
            self.stalled_pulls += 1
            self.total_stall_seconds += stall_seconds

    def format(self) -> str:
        if self.batches_encoded == 0:
            return "[CpuPrefetch] no batches were encoded."
        avg_encode = self.cpu_encode_seconds / self.batches_encoded
        stall_ratio = (self.stalled_pulls / self.total_pulls) if self.total_pulls else 0.0
        avg_stall = (self.total_stall_seconds / self.stalled_pulls) if self.stalled_pulls else 0.0
        lines = [
            "[CpuPrefetch] Stats:",
            f"  batches encoded     : {self.batches_encoded}",
            f"  samples encoded     : {self.samples_encoded}",
            f"  cpu encode total    : {self.cpu_encode_seconds:.2f}s "
            f"(avg {avg_encode * 1000:.0f} ms/batch, "
            f"min {self.fastest_encode * 1000:.0f} ms, max {self.slowest_encode * 1000:.0f} ms)",
            f"  main-thread pulls   : {self.total_pulls}",
            f"  stalls (>1ms wait)  : {self.stalled_pulls} "
            f"({stall_ratio * 100:.1f}% of pulls)",
            f"  total stall seconds : {self.total_stall_seconds:.2f}s "
            f"(avg {avg_stall * 1000:.0f} ms per stall)",
        ]
        if self.worker_errors:
            lines.append(f"  worker errors       : {len(self.worker_errors)}")
            for e in self.worker_errors[:3]:
                lines.append(f"    - {e}")
        return "\n".join(lines)


class CpuTextEncoderPrefetcher:
    """Owns a daemon thread that pre-encodes captions on CPU for the next
    `prefetch_depth` batches and ships them to the main thread via Queue.

    Lifetime: created once per epoch (matches the pre-built `batches` list),
    then start() / repeated next() / stop() at the end.
    """

    def __init__(
        self,
        encode_batch_fn,
        batches: List[List[Tuple[Any, Any]]],
        prefetch_depth: int = 4,
        log_prefix: str = "[CpuPrefetch]",
    ):
        """
        Args:
            encode_batch_fn: Callable[[List[str]], List[Tuple[emb, aux]]].
                Receives all captions of one batch and returns one
                (embedding, auxiliary) tuple per caption in order.
                Architectures with a true batched TE forward (e.g. Anima)
                amortise per-call overhead through this API; others fall
                back to a loop internally.
            batches: epoch's pre-built list of [(item, dataset), ...] groups
            prefetch_depth: queue size — number of batches the worker may
                run ahead of the main thread before back-pressure kicks in
        """
        if prefetch_depth < 1:
            raise ValueError("prefetch_depth must be >= 1")
        self._encode_batch_fn = encode_batch_fn
        self._batches = batches
        self._depth = int(prefetch_depth)
        self._log_prefix = log_prefix
        self._queue: "queue.Queue[Tuple[int, dict]]" = queue.Queue(maxsize=self._depth)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats = CpuPrefetchStats()

    # ----- worker -----

    def _worker(self) -> None:
        try:
            for batch_idx, batch in enumerate(self._batches):
                if self._stop_event.is_set():
                    break

                # Collect this batch's (image_path, caption) pairs first, then
                # encode them in ONE call so trainers with a true batched
                # forward (e.g. Anima Qwen3) get the amortisation win.
                paths: List[Any] = []
                captions: List[str] = []
                for entry in batch:
                    if self._stop_event.is_set():
                        break
                    item, _dataset = entry if isinstance(entry, tuple) else (entry, None)
                    if not isinstance(item, dict):
                        continue
                    ip = item.get("image_path")
                    if ip is None:
                        continue
                    paths.append(ip)
                    captions.append(item.get("caption", "") or "")

                t0 = time.time()
                payload: dict = {}
                if captions:
                    try:
                        results = self._encode_batch_fn(captions)
                    except Exception as e:
                        self.stats.worker_errors.append(
                            f"batch {batch_idx} ({len(captions)} captions): {e}"
                        )
                        results = []
                    for ip, cap, res in zip(paths, captions, results):
                        emb, aux = res
                        payload[ip] = (
                            emb.detach().to("cpu", copy=False) if isinstance(emb, torch.Tensor) else emb,
                            _detach_to_cpu(aux),
                            cap,
                        )
                self.stats.record_encode(time.time() - t0, len(payload))

                # put() blocks if the main thread is depth batches behind —
                # this is the backpressure that keeps memory bounded.
                while not self._stop_event.is_set():
                    try:
                        self._queue.put((batch_idx, payload), timeout=0.25)
                        break
                    except queue.Full:
                        continue
        except Exception as e:  # pragma: no cover — defensive
            self.stats.worker_errors.append(f"FATAL: {e}\n{traceback.format_exc()}")
        finally:
            # Sentinel so main thread can drain cleanly even on early exit.
            try:
                self._queue.put((-1, {}), timeout=1.0)
            except queue.Full:
                pass

    # ----- lifecycle -----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._worker, name="CpuTextEncoderPrefetcher", daemon=True,
        )
        self._thread.start()
        print(f"{self._log_prefix} Worker started "
              f"(prefetch_depth={self._depth}, total_batches={len(self._batches)})")

    def next(self, timeout: Optional[float] = None) -> Tuple[int, dict]:
        """Pull the next ready batch. Blocks (and records the stall) when
        the worker hasn't produced anything yet.

        Returns (batch_idx, embeddings_dict). batch_idx == -1 signals the
        worker has finished or errored out — callers should stop pulling.
        """
        t0 = time.time()
        empty_at_call = self._queue.empty()
        item = self._queue.get(timeout=timeout)
        wait = time.time() - t0 if empty_at_call else 0.0
        self.stats.record_pull(wait)
        return item

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        # Drain any pending items so the worker isn't blocked on a full queue.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None
        print(self.stats.format())

    def format_stats(self) -> str:
        return self.stats.format()
