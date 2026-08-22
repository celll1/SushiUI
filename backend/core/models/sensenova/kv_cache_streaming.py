"""Two-slot GPU ring for SenseNova U1.5's per-layer flash KV prefix buffers.

``prepare_flash_kv_cache`` (``vendor/modeling_neo_chat.py:48-97``) normally
allocates ONE buffer per layer of shape ``(B, prefix_len + current_len, H, D)``
with TWO regions of OPPOSITE lifecycle: the head ``[:prefix_len]`` (the
prompt/reference prefix -- written once, read every step, immutable) and the
tail ``[prefix_len:]`` (this step's image-generation tokens -- REWRITTEN IN
PLACE by every layer on every denoise step, see ``vendor/modeling_qwen3.py``'s
``update_cache=False`` path). All 42 layers' buffers are resident
simultaneously today, but the denoise loop is branch-outer/layer-inner (one
full 42-layer pass per CFG branch per step, ``sensenova_pipeline_ops.py``'s
``_predict_v_branch``/``_euler_run``), so only ONE (branch, layer) buffer is
ever live at a time. This module replaces the persistent per-layer buffers
with a 2-slot ring shared across every layer and branch: each slot's prefix
head is streamed in from a pinned CPU master (write-once per generation,
cheap); each slot's tail is written by the layer itself exactly as before, so
the result is numerically identical. Saves ~(num_layers-2)/num_layers of the
whole flash-cache allocation.

TRAP this module's design had to route around: streaming the KV buffer
read-only from CPU (feeding each layer the PREVIOUS step's tail) would be
WRONG -- the tail is per-step scratch, not a cache, and reading a stale tail
changes outputs. Only the prefix HEAD is cacheable; see ``acquire()``'s
docstring for why reassigning a slot's physical memory between denoise steps
is still safe.

Training note (SenseNova LoRA/full-FT, not yet built): this streamer does NOT
apply to training -- a training step is a single-timestep forward/backward
with no multi-step denoise loop, so no persistent read-many KV cache exists to
stream; training-side offload belongs to LayerOffloadConductor. What DOES
transfer is the MoT half-eviction CONCEPT from mot_phase_eviction.py: if
fine-tuning freezes the understanding branch (likely for image-gen tunes),
its weight-half can be CPU-evicted during training for a similar VRAM saving.
Evaluate that when training is built; reuse the layer-selection logic, not
this module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

LABEL = "SenseNova"


class SenseNovaKVCacheStreamer:
    """One instance per generation, installed on ``transformer._kv_cache_streamer``
    (consulted by ``_finalize_prefix_caches``) and, after ``adopt()``, on every
    adopted branch's ``past_key_values`` object as ``._kv_cache_streamer``
    (consulted per-layer by ``vendor/modeling_qwen3.py``)."""

    def __init__(self, transformer: Any, device: Any, num_layers: int):
        self.device = device
        self._torch_device = torch.device(device) if not isinstance(device, torch.device) else device
        self.num_layers = num_layers
        self._copy_stream = torch.cuda.Stream(device=self._torch_device)
        self._current_branch: Optional[str] = None
        self._batch_size = 0
        self._current_len = 0
        self._max_prefix_len = 0
        # branch -> [(pinned_cpu_k, pinned_cpu_v), ...] one pair per layer, [B, S, H, D].
        self._masters: Dict[str, List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = {}
        self._prefix_len: Dict[str, int] = {}
        # Per-layer prefix length (may differ from `_prefix_len`'s branch-wide
        # value for a keyless layer, which has prefix 0 -- see acquire()).
        self._layer_prefix_len: Dict[str, List[int]] = {}
        self._attached_caches: List[Any] = []
        self._slot_k: List[Optional[torch.Tensor]] = [None, None]
        self._slot_v: List[Optional[torch.Tensor]] = [None, None]
        self._slot_meta: List[Optional[Tuple[str, int]]] = [None, None]
        self._copy_done_event = [torch.cuda.Event() for _ in range(2)]
        self._free_event = [torch.cuda.Event() for _ in range(2)]
        self._adopted = False
        self.staged_bytes = 0
        self.saved_bytes = 0
        self.legacy_bytes = 0

    def adopt(self, caches: Dict[str, Any], batch_size: int, current_len: int, phase_notified: bool) -> None:
        """Build pinned CPU masters directly from each branch's existing
        ``layer.keys``/``layer.values`` ([B,H,S,D], already batch-expanded by
        the caller) and free those GPU tensors -- never calls
        ``prepare_flash_kv_cache`` (building all 42 x branches full buffers
        first would hit exactly the peak this feature exists to avoid). Only
        one layer's transpose/CPU-copy transient is alive on GPU at a time.

        ``phase_notified`` is a call-site assertion, not a real signal: the
        only caller (``_finalize_prefix_caches``) must invoke
        ``transformer._notify_layer_offload_phase("denoise")`` immediately
        before this, so MoT eviction's phase flip and this adoption never
        interleave."""
        assert phase_notified, (
            f"{LABEL} KV cache streamer: adopt() must run immediately after "
            f"transformer._notify_layer_offload_phase('denoise')."
        )
        assert not self._adopted, f"{LABEL} KV cache streamer: adopt() called twice in one generation."
        self._adopted = True
        self._batch_size = batch_size
        self._current_len = current_len

        h = d = dtype = None
        for branch, cache in caches.items():
            layer_masters: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = []
            layer_prefix_lens: List[int] = []
            prefix_len = 0
            for layer in cache.layers:
                past_k, past_v = layer.keys, layer.values
                if past_k is None or past_v is None:
                    layer_masters.append((None, None))
                    layer_prefix_lens.append(0)
                    continue
                # Same [B,H,S,D] -> [B,S,H,D] conversion prepare_flash_kv_cache
                # does (modeling_neo_chat.py:74-75), one layer at a time.
                k_flash = past_k.transpose(1, 2).contiguous()
                v_flash = past_v.transpose(1, 2).contiguous()
                prefix_len = k_flash.shape[1]
                layer_prefix_lens.append(prefix_len)
                if h is None:
                    h, d, dtype = k_flash.shape[2], k_flash.shape[3], k_flash.dtype
                k_cpu = k_flash.to("cpu")
                v_cpu = v_flash.to("cpu")
                try:
                    k_cpu, v_cpu = k_cpu.pin_memory(), v_cpu.pin_memory()
                except Exception as exc:
                    print(f"[{LABEL}] KV cache streaming: pin_memory() failed ({exc}); "
                          f"continuing with unpinned CPU staging (slower transfer, same result).")
                layer_masters.append((k_cpu, v_cpu))
                self.staged_bytes += k_cpu.numel() * k_cpu.element_size() * 2
                del k_flash, v_flash
                layer.keys = None
                layer.values = None
                # `layer.is_initialized` is a SEPARATE bool from keys/values
                # being None; transformers 5.1.0's DynamicLayer.get_seq_length()
                # is `if not is_initialized or keys.numel() == 0: return 0`, so
                # leaving it True makes it dereference `None.numel()`. Every
                # denoise step hits this via modeling_qwen3(_moe).py's
                # `past_key_values.get_seq_length()` when cache_position is
                # None. Divergence from baseline (prefix_len -> 0) is inert on
                # this path: forward_gen's RoPE keys off `indexes`, never
                # cache_position/position_ids, and the flash branch's
                # attention_mask is always the dict `{"full_attention": None}`,
                # which skips create_causal_mask/get_mask_sizes entirely.
                if hasattr(layer, "is_initialized"):
                    layer.is_initialized = False
            self._masters[branch] = layer_masters
            self._prefix_len[branch] = prefix_len
            self._layer_prefix_len[branch] = layer_prefix_lens
            cache._kv_cache_streamer = self
            cache._kv_cache_streamer_branch = branch
            self._attached_caches.append(cache)

        assert h is not None, f"{LABEL} KV cache streamer: adopted branches carried no prefix cache to stream."
        self._max_prefix_len = max(self._prefix_len.values())
        total_len = self._max_prefix_len + current_len
        for slot in range(2):
            self._slot_k[slot] = torch.empty((batch_size, total_len, h, d), device=self._torch_device, dtype=dtype)
            self._slot_v[slot] = torch.empty((batch_size, total_len, h, d), device=self._torch_device, dtype=dtype)
        ring_bytes = sum(
            t.numel() * t.element_size() for t in (self._slot_k[0], self._slot_k[1], self._slot_v[0], self._slot_v[1])
        )

        elem_size = dtype.itemsize if hasattr(dtype, "itemsize") else torch.empty((), dtype=dtype).element_size()
        self.legacy_bytes = sum(
            self.num_layers * (self._prefix_len[branch] + current_len) * batch_size * h * d * elem_size * 2
            for branch in caches
        )
        self.saved_bytes = self.legacy_bytes - ring_bytes

        msg = (f"[{LABEL}] KV cache streaming active: {self.staged_bytes / 1024 ** 2:.1f} MiB staged to pinned CPU "
               f"across {len(caches)} branch(es) x {self.num_layers} layer(s); ring allocation is "
               f"{ring_bytes / 1024 ** 2:.1f} MiB vs the {self.legacy_bytes / 1024 ** 3:.2f} GiB full per-layer "
               f"buffer set it replaces (~{self.saved_bytes / 1024 ** 3:.2f} GiB allocation delta) -- excludes "
               f"this call's own transient GPU peak (one layer's expand/transpose/contiguous) and the host-RAM "
               f"cost of the pinned masters.")
        print(msg)
        try:
            from api.generation_status import add_warning
            add_warning(msg, code="sensenova_kv_cache_streaming_active")
        except Exception:
            pass
        self._sanity_check_saving()

    def _sanity_check_saving(self) -> None:
        """Self-check against the eviction-selector bug class documented in
        ``mot_phase_eviction.py`` (a classifier that silently selected almost
        nothing, twice, with no code-level signal). Expected saving ratio is
        ~(num_layers-2)/num_layers (~0.95 at 42 layers); flag well below that."""
        if self.legacy_bytes <= 0:
            return
        ratio = self.saved_bytes / self.legacy_bytes
        expected = (self.num_layers - 2) / self.num_layers
        if ratio < expected * 0.5:
            msg = (f"[{LABEL}] KV cache streaming: measured saving ratio {ratio:.2f} is well below the "
                   f"~{expected:.2f} expected for {self.num_layers} layers -- this feature is probably inert "
                   f"or only partially engaged.")
            print(msg)
            try:
                from api.generation_status import add_warning
                add_warning(msg, code="sensenova_kv_cache_streaming_suspect")
            except Exception:
                pass

    def begin_branch(self, branch: str) -> None:
        """Called at the top of ``_predict_v_branch``, before the transformer
        forward. Selects ``branch``'s masters, invalidates both slots (the
        ring is shared across branches with DIFFERENT prefixes -- the
        prefix head is always re-streamed here, never skipped on a layer-index
        match), and issues layer 0's prefetch so its own ``acquire(0)`` blocks
        on an already-in-flight copy rather than a cold stall. Layer 1's
        prefetch is left to ``acquire(0)`` itself -- prefetching it here too
        would just be redundantly overwritten by that call."""
        assert branch in self._masters, (
            f"{LABEL} KV cache streamer: unknown branch {branch!r} (adopted: {sorted(self._masters)})."
        )
        self._current_branch = branch
        self._slot_meta = [None, None]
        if self.num_layers > 0:
            self._prefetch(0, slot=0)

    def _prefetch(self, layer_idx: int, slot: int) -> None:
        """Copy ``(self._current_branch, layer_idx)``'s prefix head into
        ``slot`` on ``self._copy_stream``, after making the copy stream wait
        on a freshly recorded compute-stream event -- without that wait, the
        copy could overwrite a slot whose reads (2 layers ago, same ring
        parity) are still in flight, corrupting them silently rather than
        crashing."""
        branch = self._current_branch
        k_cpu, v_cpu = self._masters[branch][layer_idx]
        prefix_len = self._layer_prefix_len[branch][layer_idx]

        compute_stream = torch.cuda.current_stream(self._torch_device)
        self._free_event[slot].record(compute_stream)
        self._copy_stream.wait_event(self._free_event[slot])
        with torch.cuda.stream(self._copy_stream):
            if k_cpu is not None:
                self._slot_k[slot][:, :prefix_len].copy_(k_cpu, non_blocking=True)
                self._slot_v[slot][:, :prefix_len].copy_(v_cpu, non_blocking=True)
            self._copy_done_event[slot].record(self._copy_stream)
        self._slot_meta[slot] = (branch, layer_idx)

    def acquire(self, layer_idx: int, branch: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Called from ``vendor/modeling_qwen3.py``'s optimized flash path in
        place of ``layer.flash_k_cache``/``flash_v_cache``. Waits on this
        slot's H2D copy-completion event (so the layer's forward never reads a
        partially-written prefix head), returns narrowed
        ``(k_cache, v_cache, prefix_len)`` views sized to THIS layer's prefix
        (0 for a keyless layer, matching baseline -- see safety condition on
        per-layer prefix tracking), and submits the next layer's prefetch into
        the other slot.

        ``branch``, when given, is the caller's own
        ``past_key_values._kv_cache_streamer_branch`` -- the actual branch the
        forward pass is running against, independent of this streamer's own
        ``_current_branch`` bookkeeping. Cross-checking against it is the only
        way to catch a real ``begin_branch("cond")`` / forward-against-uncond-
        cache mismatch; comparing ``_current_branch`` to itself (the previous
        implementation) cannot detect that class of bug.

        Safe to reassign the slot's physical memory between denoise steps
        because the tail region ``[prefix_len:]`` is unconditionally
        overwritten with this step's k_cur/v_cur by the caller immediately
        after this returns, BEFORE the attention read -- pre-call slot
        contents are never observed (see ``vendor/modeling_qwen3.py:722-726``,
        unchanged by this module)."""
        if branch is not None and branch != self._current_branch:
            raise RuntimeError(
                f"{LABEL} KV cache streamer: acquire() called for branch {branch!r} but the streamer is "
                f"currently staged for {self._current_branch!r} -- a begin_branch()/forward branch mismatch."
            )
        slot = layer_idx % 2
        expected = (self._current_branch, layer_idx)
        if self._slot_meta[slot] != expected:
            raise RuntimeError(
                f"{LABEL} KV cache streamer: slot {slot} holds {self._slot_meta[slot]!r}, expected {expected!r} -- "
                f"prefetch/acquire order was violated."
            )
        compute_stream = torch.cuda.current_stream(self._torch_device)
        compute_stream.wait_event(self._copy_done_event[slot])

        prefix_len = self._layer_prefix_len[self._current_branch][layer_idx]
        total = prefix_len + self._current_len
        k_view = self._slot_k[slot][:, :total]
        v_view = self._slot_v[slot][:, :total]

        next_layer = layer_idx + 1
        if next_layer < self.num_layers:
            self._prefetch(next_layer, slot=next_layer % 2)
        return k_view, v_view, prefix_len

    def teardown(self) -> None:
        """Idempotent (all state resets to empty/None). Free the ring slots
        and drop the pinned master references; detach the streamer attribute
        from every adopted cache so no stale alias survives this generation
        (safety condition 3). Waits for the copy stream first -- at most one
        prefetch (~tens of MiB) can be in flight. NOTE: torch's caching host
        allocator pools freed pinned blocks rather than returning them to the
        OS (measured in ``mot_phase_eviction.py``), so dropping references
        makes the pool reusable but host RSS may not fall -- not claimed
        here."""
        if torch.cuda.is_available():
            self._copy_stream.synchronize()
        for cache in self._attached_caches:
            if getattr(cache, "_kv_cache_streamer", None) is self:
                cache._kv_cache_streamer = None
                cache._kv_cache_streamer_branch = None
        self._attached_caches = []
        self._slot_k = [None, None]
        self._slot_v = [None, None]
        self._slot_meta = [None, None]
        self._masters = {}
        self._prefix_len = {}
        self._layer_prefix_len = {}
        self._current_branch = None


def install(transformer: Any, device: Any) -> Optional[SenseNovaKVCacheStreamer]:
    """Build a ``SenseNovaKVCacheStreamer`` for this generation and register it
    as ``transformer._kv_cache_streamer`` -- consulted by
    ``_finalize_prefix_caches``, which calls ``adopt()`` once real prefix
    caches exist. Returns None (feature silently inert, warned once) without
    CUDA -- there is nothing to stream when the model never leaves the CPU."""
    if not torch.cuda.is_available():
        try:
            from api.generation_status import add_warning
            add_warning(
                f"[{LABEL}] sensenova_kv_cache_streaming requested but CUDA is unavailable; the mechanism is a "
                f"GPU<->pinned-CPU stream, so it has nothing to do here.",
                code="sensenova_kv_cache_streaming_no_cuda",
            )
        except Exception:
            pass
        return None
    try:
        num_layers = len(transformer.language_model.model.layers)
        streamer = SenseNovaKVCacheStreamer(transformer, device, num_layers)
    except Exception as exc:
        # Never take a generation down over an optional VRAM-saving feature.
        print(f"[{LABEL}] KV cache streaming: failed to install ({exc}); continuing without it.")
        return None
    transformer._kv_cache_streamer = streamer
    return streamer


def uninstall(transformer: Any, streamer: Optional[SenseNovaKVCacheStreamer]) -> None:
    """Tear down and unregister. Safe to call even if ``install`` returned
    None or was never called. Always call from the generation's ``finally``,
    after the whole-model CPU restore -- ``teardown()`` is idempotent, so a
    prior call from ``clear_prefix_caches`` (the defence-in-depth path) makes
    this a no-op."""
    if streamer is not None:
        try:
            streamer.teardown()
        except Exception as exc:
            print(f"[{LABEL}] KV cache streaming teardown raised (non-fatal): {exc}")
    if getattr(transformer, "_kv_cache_streamer", None) is not None:
        transformer._kv_cache_streamer = None
