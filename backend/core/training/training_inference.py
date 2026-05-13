"""On-demand generation using the in-training model.

The API writes a request file in ``<output_dir>/.preview_request_<id>.json``
and the trainer picks it up at the next batch boundary (next to the
existing ``.stop_training`` flag).  This module is responsible for the
trainer-side handling: read the request, run inference using the
in-training UNet + VAE + text_encoders + (LoRA) and write the result
back to disk.

Inference reuses ``BaseTrainer.generate_sample()`` for txt2img which
already builds a temporary pipeline around the training UNet and calls
``custom_sampling_loop`` (the exact same function as the production
``/generate/txt2img`` path).

Future enhancements (Phase 2-5):
  - img2img / inpaint variants
  - LoRA stack on top of the training LoRA
  - ControlNet conditioning
  - Negative prompt, sampler choice, advanced CFG / NAG params
"""
from __future__ import annotations

import io
import traceback
from typing import Any, Dict, TYPE_CHECKING

from .training_preview_rpc import write_result

if TYPE_CHECKING:
    from .base_trainer import BaseTrainer


class TrainingPreviewGenerator:
    """Bound to a BaseTrainer; processes one preview request at a time.

    Caller (the trainer loop) invokes :meth:`process_request` at a batch
    boundary.  No state is retained between requests except a reference
    to the trainer itself.
    """

    def __init__(self, trainer: "BaseTrainer"):
        self.trainer = trainer

    # ------------------------------------------------------------------
    # Public API — invoked by the trainer at batch boundaries
    # ------------------------------------------------------------------

    def process_request(self, request_id: str, params: Dict[str, Any]) -> None:
        """Run one preview generation and write result files to disk.

        Wraps every failure mode so a bad request never crashes training.
        """
        output_dir = str(self.trainer.output_dir)
        meta: Dict[str, Any] = {"request_id": request_id, "ok": False}
        png_bytes: bytes | None = None
        try:
            mode = (params.get("mode") or "txt2img").lower()
            if mode != "txt2img":
                # Phase 2 will add img2img / inpaint.
                raise NotImplementedError(
                    f"Preview mode '{mode}' is not yet supported in this trainer. "
                    f"Phase 2 adds img2img and inpaint preview."
                )
            image, seed = self._generate_txt2img(params)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            meta.update({
                "ok": True,
                "seed": seed,
                "width": image.width,
                "height": image.height,
                "mode": mode,
            })
        except Exception as e:   # noqa: BLE001  (we report any failure)
            print(f"[TrainingPreview:{request_id}] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            meta.update({
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            })
            png_bytes = None
        finally:
            try:
                write_result(output_dir, request_id, png_bytes, meta)
            except Exception as we:   # noqa: BLE001
                print(f"[TrainingPreview:{request_id}] failed to write result: {we}")

    # ------------------------------------------------------------------
    # txt2img path (Phase 1) — delegates to BaseTrainer.generate_sample
    # ------------------------------------------------------------------

    def _generate_txt2img(self, params: Dict[str, Any]):
        """Run txt2img by reusing the trainer's generate_sample() machinery.

        ``generate_sample`` already:
          - flips models into eval mode
          - builds a TempPipeline wrapping unet/vae/text_encoders
          - calls custom_sampling_loop (the same function /generate/txt2img uses)
          - restores train mode in its finally block

        Phase 1 scope:
          - prompt, width, height, steps, cfg_scale, seed, schedule_type
          - negative_prompt and non-default samplers are NOT yet exposed
            via generate_sample (it hard-codes "" and "euler"); will be
            wired in Phase 2 along with a generate_sample refactor.
        """
        prompt = params.get("prompt") or ""
        if not prompt:
            raise ValueError("Preview request missing 'prompt'")

        # Pull a seed deterministically from params (-1 => random)
        seed = int(params.get("seed", -1))

        image = self.trainer.generate_sample(
            prompt=prompt,
            height=int(params.get("height", 1024)),
            width=int(params.get("width", 1024)),
            num_inference_steps=int(params.get("steps", 28)),
            guidance_scale=float(params.get("cfg_scale", 3.5)),
            seed=seed,
            current_step=int(params.get("current_step", 0)),
            schedule_type=str(params.get("schedule_type", "uniform")),
        )

        # generate_sample resolves -1 to a random seed internally but
        # does not currently return it.  Best-effort: echo the request
        # seed back; refactor of generate_sample (Phase 2) will return
        # the actual seed.
        return image, seed
