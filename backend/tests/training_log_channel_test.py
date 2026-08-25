"""The training log's channel to the user: emit -> stdout -> parent -> client.

The trainer runs in a subprocess whose stdout reached only the backend
console, so every notice about a setting the run overrode or ignored was
invisible to the product. These tests exercise the whole path on the
mechanism -- a real stdout line pumped through ``TrainingProcess._monitor_logs``
-- rather than on any flag, and pin the bounds that keep a chatty run from
flooding either the socket or the run row.
"""

import asyncio
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from api.websocket import ConnectionManager  # noqa: E402
from core.training.training_events import (  # noqa: E402
    MAX_EVENT_MESSAGE_CHARS,
    MAX_PERSISTED_WARNINGS_PER_RUN,
    TRAINING_EVENT_SENTINEL,
    emit_training_event,
    emit_training_warning,
    merge_run_warnings,
    parse_training_event,
)
from core.training.training_process import TrainingProcess  # noqa: E402


# --------------------------------------------------------------------------
# Harness: a real _monitor_logs run over a scripted stdout.
# --------------------------------------------------------------------------

class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    async def readline(self):
        if not self._lines:
            return b""
        return (self._lines.pop(0) + "\n").encode("utf-8")

    async def read(self, _n):
        return b""


class _FakeProc:
    def __init__(self, lines):
        self.stdout = _FakeStdout(lines)
        self.returncode = 0

    async def wait(self):
        return 0


def pump(lines, max_events=None):
    """Run the real log monitor over *lines*. Returns (logged, events)."""
    proc = TrainingProcess(run_id=1, config_path="c.yaml", output_dir="o",
                           venv_python="python")
    if max_events is not None:
        proc.MAX_EVENTS_PER_RUN = max_events
    proc.process = _FakeProc(lines)
    logged, events = [], []
    # _monitor_logs prints its own end-of-run line; keep the test output clean.
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(proc._monitor_logs(None, logged.append, events.append))
    return logged, events


def capture(fn, *args, **kwargs):
    """Run *fn* with stdout captured; return the lines it printed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue().splitlines()


class FullParameterTrainer:  # name-matched on purpose: the gate keys on it
    pass


def _run_fp8_full_finetune_load(arch):
    """Run the arch's real ``load_components`` as a full FT with fp8 requested.

    Returns (trainer, quantiser spy, printed lines). Mirrors the harness in
    training_method_gate_test.py; kept local so this file runs standalone.
    """
    import torch
    from torch import nn
    from unittest.mock import patch
    from core.training.ops import anima_ops, lens_ops, ltx2_ops

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

    trainer = FullParameterTrainer()
    trainer.log_prefix = "[T]"
    trainer.model_path = "model.safetensors"
    trainer.weight_dtype = torch.float32
    trainer.training_dtype = torch.float32
    trainer.vae_dtype = torch.float32
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer.config = {"fp8_base_dtype": "e4m3"}
    trainer.gradient_checkpointing = False
    trainer.blocks_to_swap = 0
    trainer.use_flash_attention = False

    module, loader_target, components = {
        "anima": (anima_ops, "core.model_loader.ModelLoader.load_anima_from_files",
                  {"transformer": _Stub(), "vae": _Stub(), "text_encoder": _Stub(),
                   "tokenizer": object(), "t5_tokenizer": object(),
                   "scheduler": object()}),
        "lens": (lens_ops, "core.models.lens.lens_loader.load_lens_components",
                 {"transformer": _Stub(), "vae": _Stub(), "text_encoder": _Stub(),
                  "tokenizer": object(), "scheduler": object()}),
        "ltx2": (ltx2_ops, "core.model_loader.ModelLoader.load_ltx2_from_path",
                 {"pipeline": SimpleNamespace(), "transformer": _Stub(),
                  "vae": _Stub(), "audio_vae": _Stub(), "text_encoder": _Stub(),
                  "tokenizer": object(), "connectors": _Stub(), "vocoder": None,
                  "scheduler": object()}),
    }[arch]

    buf = io.StringIO()
    with patch(loader_target, return_value=components), \
            patch("core.vram_optimization._anima_quantize_fp8",
                  side_effect=lambda m, *a, **k: m) as quantize, \
            contextlib.redirect_stdout(buf):
        module.load_components(trainer)
    return trainer, quantize, buf.getvalue().splitlines()


# --------------------------------------------------------------------------
# Negative control: the shipped behaviour.
# --------------------------------------------------------------------------

class NegativeControlTest(unittest.TestCase):
    """What the SenseNova stochastic-rounding override did before this change.

    ``enforce_full_finetune_stochastic_rounding`` reverses an explicit user
    setting and announced it with a bare ``print()``. The parent's only
    consumer of that line was ``log_callback``, which prints it again to the
    backend's own console. Reproduced here verbatim so the gap is a test, not
    a claim.
    """

    SHIPPED_TEXT = (
        "[SenseNova] SenseNova full fine-tuning: optimizer_stochastic_rounding was "
        "off and has been turned on for this run. It is not optional here."
    )

    def test_a_plain_print_reaches_no_client(self):
        logged, events = pump([self.SHIPPED_TEXT])
        # The console sees it...
        self.assertEqual(logged, [self.SHIPPED_TEXT])
        # ...and nothing else does. No event, so nothing to broadcast and
        # nothing to persist: a connected client learns nothing at all.
        self.assertEqual(events, [])

    def test_and_no_amount_of_reading_the_line_recovers_it(self):
        """There is no structure in the shipped line to recover.

        A parent that wanted to forward it would have to match on prose, which
        is why the emitter carries the code instead.
        """
        self.assertIsNone(parse_training_event(self.SHIPPED_TEXT))


# --------------------------------------------------------------------------
# The channel carries it, asserted on the emit path.
# --------------------------------------------------------------------------

class OverrideReachesTheChannelTest(unittest.TestCase):
    def _run_enforce(self):
        from core.training.ops.sensenova_ops import (
            enforce_full_finetune_stochastic_rounding,
        )
        trainer = SimpleNamespace(
            optimizer_stochastic_rounding=False, log_prefix="[SenseNova]"
        )
        lines = capture(enforce_full_finetune_stochastic_rounding, trainer)
        return trainer, lines

    def test_the_real_override_emits_a_line_the_parent_lifts_off_stdout(self):
        trainer, lines = self._run_enforce()
        # Not asserting on trainer.optimizer_stochastic_rounding here: that is
        # the flag. This is the transport -- the bytes the child actually wrote,
        # fed to the real parent-side monitor.
        logged, events = pump(lines)

        self.assertEqual(len(events), 1, events)
        self.assertEqual(events[0]["level"], "warning")
        self.assertEqual(events[0]["code"], "sensenova_stochastic_rounding_forced")
        self.assertIn("optimizer_stochastic_rounding", events[0]["message"])
        self.assertIn("84.5%", events[0]["message"])

        # The human line still goes to the console exactly as before; only the
        # machine line is consumed by the channel.
        self.assertTrue(any("84.5%" in line for line in logged))
        self.assertFalse(any(TRAINING_EVENT_SENTINEL in line for line in logged))

    def test_the_event_survives_the_broadcast_layer_intact(self):
        _, lines = self._run_enforce()
        _, events = pump(lines)

        manager = ConnectionManager()
        manager._notify_sender = lambda: None
        manager.send_training_log(
            run_id=42,
            level=events[0]["level"],
            message=events[0]["message"],
            code=events[0]["code"],
        )
        sent = manager.message_queue.get_nowait()
        self.assertEqual(sent["type"], "training_log")
        self.assertEqual(sent["run_id"], 42)
        self.assertEqual(sent["level"], "warning")
        self.assertEqual(sent["code"], "sensenova_stochastic_rounding_forced")
        # Whatever the sender queues has to be JSON: start_sender json.dumps it.
        json.loads(json.dumps(sent))

    def test_it_lands_on_the_run_row_so_a_disconnected_user_still_sees_it(self):
        _, lines = self._run_enforce()
        _, events = pump(lines)
        merged = merge_run_warnings(None, events[0])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["code"], "sensenova_stochastic_rounding_forced")


class AllFourKnownWarningsAreWiredTest(unittest.TestCase):
    """Each of the four named notices produces a parseable event."""

    def test_fused_gradient_accumulation(self):
        from core.training.base_trainer import BaseTrainer
        trainer = SimpleNamespace(
            log_prefix="[Trainer]", use_fused_backward=True,
            fused_optimizer_groups=None,
        )
        lines = capture(
            BaseTrainer._warn_gradient_accumulation_ignored_under_fused,
            trainer, 4, 1, 1,
        )
        _, events = pump(lines)
        self.assertEqual([e["code"] for e in events],
                         ["fused_gradient_accumulation_ignored"])
        self.assertIn("IGNORED", events[0]["message"])

    def test_fused_grad_clipping(self):
        from core.training.base_trainer import BaseTrainer
        trainer = SimpleNamespace(
            log_prefix="[Trainer]", use_fused_backward=True,
            fused_optimizer_groups=None,
        )
        lines = capture(
            BaseTrainer._warn_grad_clipping_ignored_under_fused, trainer, 1.0
        )
        _, events = pump(lines)
        self.assertEqual([e["code"] for e in events],
                         ["fused_grad_clipping_ignored"])

    def test_fp8_base_dtype_ignored_under_a_full_finetune(self):
        """Drives the arch's real ``load_components``, not a copy of its text.

        A copied message would keep passing after the ops file's wording drifted
        away from it, so the notice is produced by the shipped call site with a
        stubbed loader and a spied quantiser.
        """
        for arch in ("anima", "lens", "ltx2"):
            with self.subTest(arch=arch):
                trainer, quantize, lines = _run_fp8_full_finetune_load(arch)
                _, events = pump(lines)
                self.assertEqual([e["code"] for e in events],
                                 ["fp8_base_dtype_ignored"])
                self.assertIn("fp8_base_dtype=e4m3", events[0]["message"])
                # The gate's decision is untouched: the base a full FT trains is
                # still never handed to the quantiser.
                quantize.assert_not_called()
                # And the console text the existing gate test asserts on stands.
                self.assertTrue(any(
                    "WARNING: fp8_base_dtype=e4m3 requires a frozen" in line
                    for line in lines))

    def test_the_shipped_fp8_call_sites_all_route_through_the_channel(self):
        """Source check: no arch still uses a bare print for this notice."""
        import ast
        for name in ("anima_ops", "lens_ops", "ltx2_ops"):
            path = Path(_BACKEND) / "core" / "training" / "ops" / f"{name}.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            printed = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "print"
                and "fp8_base_dtype" in ast.dump(node)
                and "ignored" in ast.dump(node)
            ]
            self.assertEqual(printed, [], f"{name} still prints the notice only")


# --------------------------------------------------------------------------
# Volume: the bound holds under a flood.
# --------------------------------------------------------------------------

class FloodBoundTest(unittest.TestCase):
    def _sentinel(self, i):
        return (f"{TRAINING_EVENT_SENTINEL} "
                + json.dumps({"level": "warning", "code": f"c{i}", "message": f"m{i}"}))

    def test_distinct_events_stop_at_the_cap_and_say_so(self):
        cap = 20
        logged, events = pump([self._sentinel(i) for i in range(5000)],
                              max_events=cap)
        self.assertEqual(len(events), cap + 1)
        self.assertEqual(events[-1]["code"], "training_event_cap_reached")
        # Sentinel lines never fall through to the console pump either.
        self.assertEqual(logged, [])

    def test_a_repeating_emitter_is_forwarded_once(self):
        line = self._sentinel(0)
        _, events = pump([line] * 5000)
        self.assertEqual(len(events), 1)

    def test_ordinary_output_never_reaches_the_channel_at_all(self):
        noise = [f"step: {i} loss: 0.01 lr: 0.0001" for i in range(1000)]
        logged, events = pump(noise)
        self.assertEqual(events, [])
        self.assertEqual(len(logged), 1000)

    def test_one_notice_cannot_be_a_payload(self):
        lines = capture(emit_training_warning, "x" * 100_000, code="big")
        _, events = pump(lines)
        self.assertEqual(len(events), 1)
        self.assertLessEqual(len(events[0]["message"]), MAX_EVENT_MESSAGE_CHARS)

    def test_the_run_row_is_capped_and_deduped(self):
        stored = None
        for i in range(500):
            merged = merge_run_warnings(
                stored, {"level": "warning", "code": f"c{i}", "message": f"m{i}"})
            if merged is not None:
                stored = merged
        self.assertEqual(len(stored), MAX_PERSISTED_WARNINGS_PER_RUN)
        # Earliest kept: the notice explaining what the run has done since step 0.
        self.assertEqual(stored[0]["code"], "c0")
        # And a truncated list says so rather than looking complete.
        self.assertEqual(stored[-1]["code"], "warnings_truncated")
        self.assertEqual([e["code"] for e in stored[:-1]],
                         [f"c{i}" for i in range(MAX_PERSISTED_WARNINGS_PER_RUN - 1)])

        dup = merge_run_warnings(stored, {"level": "warning", "code": "c0",
                                          "message": "m0"})
        self.assertIsNone(dup)

    def test_a_run_that_fits_carries_no_truncation_marker(self):
        stored = None
        for i in range(MAX_PERSISTED_WARNINGS_PER_RUN - 1):
            stored = merge_run_warnings(
                stored, {"level": "warning", "code": f"c{i}", "message": f"m{i}"})
        self.assertEqual(len(stored), MAX_PERSISTED_WARNINGS_PER_RUN - 1)
        self.assertNotIn("warnings_truncated", [e["code"] for e in stored])

    def test_info_is_live_only(self):
        self.assertIsNone(
            merge_run_warnings([], {"level": "info", "code": None, "message": "m"}))


# --------------------------------------------------------------------------
# Parsing robustness and the other message types.
# --------------------------------------------------------------------------

class ParsingTest(unittest.TestCase):
    def test_round_trip(self):
        lines = capture(emit_training_event, "error", "boom", code="e1",
                        prefix="[X]")
        events = [e for e in (parse_training_event(l) for l in lines) if e]
        self.assertEqual(events,
                         [{"level": "error", "code": "e1", "message": "boom"}])

    def test_a_malformed_sentinel_line_is_ordinary_output(self):
        for bad in (f"{TRAINING_EVENT_SENTINEL} not json",
                    f"{TRAINING_EVENT_SENTINEL} []",
                    f"{TRAINING_EVENT_SENTINEL} {{}}",
                    f'{TRAINING_EVENT_SENTINEL} {{"message": ""}}'):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_training_event(bad))
                logged, events = pump([bad])
                self.assertEqual(events, [])
                self.assertEqual(logged, [bad])

    def test_an_unknown_level_degrades_to_info_rather_than_being_dropped(self):
        line = (f"{TRAINING_EVENT_SENTINEL} "
                + json.dumps({"level": "catastrophe", "message": "m"}))
        self.assertEqual(parse_training_event(line)["level"], "info")


class CarriageReturnGluingTest(unittest.TestCase):
    """A tqdm bar sharing the pipe must not be able to eat a notice.

    ``StreamReader.readline`` splits on ``\\n`` only, and ``_spawn`` merges the
    child's stderr into stdout, where tqdm writes carriage-return-only frames.
    An unterminated frame is delivered PREPENDED to the next newline-terminated
    line -- ours. Anchoring the sentinel at position 0 dropped the notice AND
    printed its raw JSON to the console, which is the exact failure this feature
    exists to remove.
    """

    def _notice(self):
        return (f"{TRAINING_EVENT_SENTINEL} "
                + json.dumps({"level": "warning", "code": "fused_grad_clipping_ignored",
                              "message": "max_grad_norm=1.0 is IGNORED"}))

    def test_a_tqdm_frame_glued_in_front_still_yields_the_event(self):
        bar = "Epoch 1/10:  10%|#         | 1/10\r"
        logged, events = pump([bar + self._notice()])
        self.assertEqual(len(events), 1, events)
        self.assertEqual(events[0]["code"], "fused_grad_clipping_ignored")
        # The bar text is still forwarded to the console, and no raw JSON is.
        self.assertEqual(logged, [bar])
        self.assertFalse(any(TRAINING_EVENT_SENTINEL in line for line in logged))

    def test_the_console_less_emitter_has_no_human_line_to_shield_it(self):
        """``console=False`` had no first line to absorb the CR; verify it now."""
        lines = capture(emit_training_warning, "shieldless", code="c",
                        console=False)
        self.assertEqual(len(lines), 1)
        _, events = pump(["Epoch 1/10:  10%|# | 1/10\r" + lines[0]])
        self.assertEqual([e["code"] for e in events], ["c"])

    def test_a_real_subprocess_interleaving_stderr_and_stdout(self):
        """End to end over a real merged pipe, not a scripted list of lines."""
        import subprocess
        child = (
            "import sys, os\n"
            f"sys.path.insert(0, {_BACKEND!r})\n"
            "from core.training.training_events import emit_training_warning\n"
            "sys.stderr.write('Epoch 1/10:  10%|# | 1/10\\r')\n"
            "sys.stderr.flush()\n"
            "emit_training_warning('interleaved', code='real_pipe', console=False)\n"
        )
        out = subprocess.run(
            [sys.executable, "-u", "-c", child],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout.decode("utf-8", errors="replace")
        # One physical line: the CR frame and the notice arrive glued.
        physical = [l for l in out.split("\n") if l.strip()]
        self.assertTrue(any("\r" in l and TRAINING_EVENT_SENTINEL in l
                            for l in physical), physical)
        _, events = pump([l.rstrip("\r\n") for l in physical])
        self.assertEqual([e["code"] for e in events], ["real_pipe"], events)

    def test_a_line_merely_mentioning_the_sentinel_is_still_ordinary_output(self):
        for bad in (f"see {TRAINING_EVENT_SENTINEL} for details",
                    f"x\r{TRAINING_EVENT_SENTINEL} not-json",
                    f'{TRAINING_EVENT_SENTINEL} {{"message": "m"}} trailing junk'):
            with self.subTest(bad=bad):
                logged, events = pump([bad])
                self.assertEqual(events, [])
                self.assertEqual(logged, [bad])


class ExistingMessageTypesUnaffectedTest(unittest.TestCase):
    """The new type is additive: every other sender still produces its own."""

    def _manager(self):
        m = ConnectionManager()
        m._notify_sender = lambda: None
        return m

    def _drain(self, m):
        out = []
        while not m.message_queue.empty():
            out.append(m.message_queue.get_nowait())
        return out

    def test_each_sender_still_produces_exactly_its_documented_type(self):
        m = self._manager()
        m.send_progress_sync(3, 28, "Step 3/28")
        m.send_training_metrics(run_id=1, step=5, loss=0.1)
        m.send_training_log(run_id=1, level="warning", message="m", code="c")
        m.send_tagger_metrics(run_id="t1", event_type="step", step=5)
        m.send_dataset_scan_progress(scope="training", run_id=1, dataset_id=2,
                                     phase="drift_walk")
        self.assertEqual(
            [d["type"] for d in self._drain(m)],
            ["progress", "training_metrics", "training_log", "tagger_metrics",
             "dataset_scan_progress"],
        )

    def test_the_progress_and_metrics_payloads_are_byte_identical_to_before(self):
        m = self._manager()
        m.send_progress_sync(3, 28, "Step 3/28")
        m.send_training_metrics(run_id=1, step=5, loss=0.1, learning_rate=1e-4)
        progress, metrics = self._drain(m)
        self.assertEqual(progress, {
            "type": "progress", "step": 3, "total_steps": 28,
            "progress": (3 / 28) * 100, "message": "Step 3/28",
        })
        self.assertEqual(metrics, {
            "type": "training_metrics", "run_id": 1, "step": 5, "loss": 0.1,
            "resume_seq": 0, "learning_rate": 1e-4,
        })

    def test_a_training_log_with_no_code_omits_the_key(self):
        m = self._manager()
        m.send_training_log(run_id=1, level="info", message="m")
        self.assertNotIn("code", self._drain(m)[0])


# --------------------------------------------------------------------------
# Nothing any warning DECIDES has changed.
# --------------------------------------------------------------------------

class NoBehaviourChangeTest(unittest.TestCase):
    def test_the_sensenova_override_still_overrides(self):
        from core.training.ops.sensenova_ops import (
            enforce_full_finetune_stochastic_rounding,
        )
        trainer = SimpleNamespace(optimizer_stochastic_rounding=False,
                                  log_prefix="[SenseNova]")
        with contextlib.redirect_stdout(io.StringIO()):
            changed = enforce_full_finetune_stochastic_rounding(trainer)
        self.assertTrue(changed)
        self.assertTrue(trainer.optimizer_stochastic_rounding)

    def test_an_explicit_true_is_still_silent_and_untouched(self):
        from core.training.ops.sensenova_ops import (
            enforce_full_finetune_stochastic_rounding,
        )
        trainer = SimpleNamespace(optimizer_stochastic_rounding=True,
                                  log_prefix="[SenseNova]")
        lines = capture(enforce_full_finetune_stochastic_rounding, trainer)
        self.assertEqual(lines, [])
        _, events = pump(lines)
        self.assertEqual(events, [])

    def test_the_fused_warnings_still_fire_only_under_a_fused_path(self):
        from core.training.base_trainer import BaseTrainer
        unfused = SimpleNamespace(log_prefix="[T]", use_fused_backward=False,
                                  fused_optimizer_groups=None)
        self.assertEqual(
            capture(BaseTrainer._warn_grad_clipping_ignored_under_fused,
                    unfused, 1.0), [])
        self.assertEqual(
            capture(BaseTrainer._warn_gradient_accumulation_ignored_under_fused,
                    unfused, 4, 1, 1), [])

    def test_the_fused_warnings_still_fire_at_most_once(self):
        from core.training.base_trainer import BaseTrainer
        trainer = SimpleNamespace(log_prefix="[T]", use_fused_backward=True,
                                  fused_optimizer_groups=None)
        first = capture(BaseTrainer._warn_grad_clipping_ignored_under_fused,
                        trainer, 1.0)
        second = capture(BaseTrainer._warn_grad_clipping_ignored_under_fused,
                         trainer, 1.0)
        self.assertTrue(first)
        self.assertEqual(second, [])
        self.assertTrue(trainer._fused_clipping_warned)

    def test_the_clipping_warning_is_still_skipped_when_clipping_is_off(self):
        from core.training.base_trainer import BaseTrainer
        trainer = SimpleNamespace(log_prefix="[T]", use_fused_backward=True,
                                  fused_optimizer_groups=None)
        self.assertEqual(
            capture(BaseTrainer._warn_grad_clipping_ignored_under_fused,
                    trainer, 0.0), [])


if __name__ == "__main__":
    unittest.main()
