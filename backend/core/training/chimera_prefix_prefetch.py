"""Bounded next-batch prefetch of frozen SenseNova prefix state."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from core.models.sensenova_sdxl_chimera.prefix import (
    capture_chimera_prompt_prefix,
    move_understanding_prefix,
)
from core.models.sensenova_sdxl_chimera.understanding import UnderstandingPrefix


def _pin_prefix(prefix: UnderstandingPrefix) -> UnderstandingPrefix:
    if not torch.cuda.is_available():
        return prefix
    def pin(tensor: torch.Tensor) -> torch.Tensor:
        return tensor if tensor.is_pinned() else tensor.pin_memory()

    return UnderstandingPrefix(
        hidden_states=pin(prefix.hidden_states),
        layer_kv={layer: (pin(key), pin(value))
                  for layer, (key, value) in prefix.layer_kv.items()},
        attention_mask=pin(prefix.attention_mask),
        positions=pin(prefix.positions),
    )


def prefix_nbytes(prefix: UnderstandingPrefix) -> int:
    tensors = [prefix.hidden_states, prefix.attention_mask, prefix.positions]
    for key, value in prefix.layer_kv.values():
        tensors.extend((key, value))
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


@dataclass
class ChimeraPrefixPrefetchStats:
    batches: int = 0
    prefixes: int = 0
    bytes: int = 0
    encode_seconds: float = 0.0
    pull_stall_seconds: float = 0.0
    hits: int = 0
    misses: int = 0


class ChimeraPrefixPrefetcher:
    """Prefetch raw frozen-prefix tensors; the live bridge remains on the main thread."""

    def __init__(
        self,
        *,
        transformer,
        tokenizer,
        selected_layers: tuple[int, ...],
        batches: list,
        device: str,
        depth: int = 1,
        log_prefix: str = "[ChimeraPrefixPrefetch]",
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("Chimera prefix prefetch device must be 'cpu' or 'cuda'")
        if int(depth) < 1:
            raise ValueError("Chimera prefix prefetch depth must be >= 1")
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.selected_layers = tuple(selected_layers)
        self.batches = list(batches)
        self.device = torch.device(device)
        self.log_prefix = log_prefix
        self.queue: queue.Queue[tuple[int, dict[str, UnderstandingPrefix], Any]] = (
            queue.Queue(maxsize=int(depth))
        )
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.active: dict[str, UnderstandingPrefix] = {}
        self.persistent: dict[str, UnderstandingPrefix] = {}
        self.stats = ChimeraPrefixPrefetchStats()
        self.stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    @staticmethod
    def resolve_device(mode: str, *, min_free_gb: float = 10.0) -> str:
        mode = str(mode).strip().lower()
        if mode in {"cpu", "cuda"}:
            if mode == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA Chimera prefix prefetch requested without CUDA")
            return mode
        if mode != "auto":
            raise ValueError("chimera_prefix_prefetch_device must be auto, cpu, or cuda")
        if not torch.cuda.is_available():
            return "cpu"
        free_bytes, _total = torch.cuda.mem_get_info()
        return "cuda" if free_bytes >= float(min_free_gb) * 1024**3 else "cpu"

    def _captions(self, batch: list, batch_idx: int) -> list[str]:
        captions = []
        for entry in batch:
            item = entry[0] if isinstance(entry, tuple) else entry
            if isinstance(item, dict):
                captions.append(str(item.get("caption", "") or ""))
        if batch_idx == 0:
            captions.append("")
        return list(dict.fromkeys(captions))

    def _capture(self, prompt: str) -> UnderstandingPrefix:
        # Prefix tensors feed a trainable bridge in bridge_align/joint. They
        # need ordinary no-grad semantics so autograd may save them for the
        # bridge's weight gradients.
        with self.lock, torch.no_grad():
            if self.stream is None:
                prefix = capture_chimera_prompt_prefix(
                    self.transformer, self.tokenizer, prompt, self.selected_layers
                )
            else:
                with torch.cuda.stream(self.stream):
                    prefix = capture_chimera_prompt_prefix(
                        self.transformer, self.tokenizer, prompt, self.selected_layers
                    )
            if self.device.type == "cpu":
                prefix = _pin_prefix(prefix)
            return prefix

    def _worker(self) -> None:
        try:
            for batch_idx, batch in enumerate(self.batches):
                if self.stop_event.is_set():
                    break
                started = time.perf_counter()
                payload = {caption: self._capture(caption)
                           for caption in self._captions(batch, batch_idx)}
                if batch_idx == 0 and "" in payload:
                    self.persistent[""] = payload[""]
                event = None
                if self.stream is not None:
                    event = torch.cuda.Event()
                    event.record(self.stream)
                self.stats.batches += 1
                self.stats.prefixes += len(payload)
                self.stats.bytes += sum(prefix_nbytes(value) for value in payload.values())
                self.stats.encode_seconds += time.perf_counter() - started
                while not self.stop_event.is_set():
                    try:
                        self.queue.put((batch_idx, payload, event), timeout=0.25)
                        break
                    except queue.Full:
                        pass
        except Exception as exc:
            while not self.stop_event.is_set():
                try:
                    self.queue.put((-1, {"__error__": exc}, None), timeout=0.25)
                    break
                except queue.Full:
                    pass

    def start(self) -> None:
        self.transformer.to(self.device).requires_grad_(False).eval()
        self.thread = threading.Thread(
            target=self._worker, name="ChimeraPrefixPrefetcher", daemon=True
        )
        self.thread.start()
        print(
            f"{self.log_prefix} started on {self.device.type} "
            f"(depth={self.queue.maxsize}, batches={len(self.batches)})"
        )

    def activate_batch(self, batch_idx: int, *, timeout: float = 300.0) -> None:
        started = time.perf_counter()
        pulled_idx, payload, event = self.queue.get(timeout=timeout)
        self.stats.pull_stall_seconds += time.perf_counter() - started
        if pulled_idx == -1:
            error = payload.get("__error__")
            raise RuntimeError(f"Chimera prefix prefetch failed: {error}")
        if pulled_idx != batch_idx:
            raise RuntimeError(
                f"Chimera prefix prefetch order mismatch: expected {batch_idx}, got {pulled_idx}"
            )
        if event is not None:
            event.synchronize()
        self.active = payload

    def take(self, prompt: str, bridge_device: torch.device | str) -> UnderstandingPrefix | None:
        prefix = self.active.get(prompt) or self.persistent.get(prompt)
        if prefix is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        if prefix.hidden_states.device != torch.device(bridge_device):
            prefix = move_understanding_prefix(prefix, bridge_device, non_blocking=True)
        return prefix

    def capture_sync(self, prompt: str, bridge_device: torch.device | str) -> UnderstandingPrefix:
        prefix = self._capture(prompt)
        if self.stream is not None:
            self.stream.synchronize()
        if prefix.hidden_states.device != torch.device(bridge_device):
            prefix = move_understanding_prefix(prefix, bridge_device, non_blocking=True)
        return prefix

    def stop(self) -> None:
        self.stop_event.set()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.thread is not None:
            self.thread.join(timeout=10.0)
            self.thread = None
        mib = self.stats.bytes / 1024**2
        print(
            f"{self.log_prefix} stopped: {self.stats.batches} batches, "
            f"{self.stats.prefixes} prefixes/{mib:.1f} MiB, "
            f"encode={self.stats.encode_seconds:.2f}s, "
            f"stall={self.stats.pull_stall_seconds:.2f}s, "
            f"hits={self.stats.hits}, misses={self.stats.misses}"
        )
