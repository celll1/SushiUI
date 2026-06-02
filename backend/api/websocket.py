from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import json
import queue
import threading

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.message_queue = queue.Queue()
        self.sender_task = None
        self._send_event: asyncio.Event = None
        self._loop: asyncio.AbstractEventLoop = None

    def _notify_sender(self):
        """Wake the drain loop from any thread."""
        if self._loop and self._send_event:
            self._loop.call_soon_threadsafe(self._send_event.set)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_progress(self, step: int, total_steps: int, message: str = ""):
        """Send generation progress to all connected clients"""
        data = {
            "type": "progress",
            "step": step,
            "total_steps": total_steps,
            "progress": (step / total_steps) * 100,
            "message": message
        }
        await self.broadcast(json.dumps(data))

    def send_progress_sync(self, step: int, total_steps: int, message: str = "", preview_image: str = None, cfg_metrics: dict = None):
        """Send progress synchronously from callback thread - uses queue"""
        data = {
            "type": "progress",
            "step": step,
            "total_steps": total_steps,
            "progress": (step / total_steps) * 100 if total_steps > 0 else 0,
            "message": message
        }
        if preview_image:
            data["preview_image"] = preview_image
        if cfg_metrics:
            data["cfg_metrics"] = cfg_metrics
        self.message_queue.put(data)
        self._notify_sender()

    def send_training_metrics(
        self,
        run_id: int,
        step: int,
        loss: float,
        recon_loss: float = None,
        learning_rate: float = None,
        grad_norm: float = None,
        grad_norm_text_encoder: float = None,
        grad_norm_text_encoder_1: float = None,
        grad_norm_text_encoder_2: float = None,
        grad_norm_unet: float = None,
        grad_norm_vision_encoder: float = None,
    ):
        """Send training metrics (loss, recon_loss, lr, grad_norm) to all connected clients.

        Called from training loop (base_trainer.py) after each step.
        Thread-safe: uses message queue.
        """
        data = {
            "type": "training_metrics",
            "run_id": run_id,
            "step": step,
            "loss": loss,
        }
        if recon_loss is not None:
            data["recon_loss"] = recon_loss
        if learning_rate is not None:
            data["learning_rate"] = learning_rate
        if grad_norm is not None:
            data["grad_norm"] = grad_norm
        if grad_norm_text_encoder is not None:
            data["grad_norm_text_encoder"] = grad_norm_text_encoder
        if grad_norm_text_encoder_1 is not None:
            data["grad_norm_text_encoder_1"] = grad_norm_text_encoder_1
        if grad_norm_text_encoder_2 is not None:
            data["grad_norm_text_encoder_2"] = grad_norm_text_encoder_2
        if grad_norm_unet is not None:
            data["grad_norm_unet"] = grad_norm_unet
        if grad_norm_vision_encoder is not None:
            data["grad_norm_vision_encoder"] = grad_norm_vision_encoder

        self.message_queue.put(data)
        self._notify_sender()

    def send_tagger_metrics(
        self,
        run_id: str,
        event_type: str,
        step: int,
        epoch: int = None,
        loss: float = None,
        lr: float = None,
        f1: float = None,
        train_f1: float = None,
        threshold: float = None,
        progress: float = None,
        resume_seq: int = 0,
        precision: float = None,
        recall: float = None,
        fp_fn_scatter: dict = None,
    ):
        """Send tagger training metrics to all connected clients.

        Called from tagger progress callback after each step/epoch.
        Thread-safe: uses message queue.

        ``resume_seq`` identifies which resume of the run this metric
        belongs to (0 = initial, 1+ = subsequent resumes), so the chart
        can render each resume as its own colored curve.
        """
        data = {
            "type": "tagger_metrics",
            "run_id": run_id,
            "event": event_type,
            "step": step,
            "resume_seq": resume_seq,
        }
        if epoch is not None:
            data["epoch"] = epoch
        if loss is not None:
            data["loss"] = loss
        if lr is not None:
            data["lr"] = lr
        if f1 is not None:
            data["f1"] = f1
        if train_f1 is not None:
            data["train_f1"] = train_f1
        if threshold is not None:
            data["threshold"] = threshold
        if progress is not None:
            data["progress"] = progress
        if precision is not None:
            data["precision"] = precision
        if recall is not None:
            data["recall"] = recall
        if fp_fn_scatter is not None:
            data["fp_fn_scatter"] = fp_fn_scatter
        self.message_queue.put(data)
        self._notify_sender()

    def send_dataset_scan_progress(
        self,
        *,
        scope: str,        # "tagger" | "training"
        run_id,            # str for tagger, int for LoRA/Full-FT
        dataset_id: int,
        phase: str,        # "drift_walk" | "drift_done" | "rescan" | "cleanup"
        files_walked: int = 0,
        items_in_db: int = 0,
        items_missing: int = 0,
        items_new: int = 0,
        message: str = "",
    ):
        """Broadcast a dataset-scan / drift-check progress event.

        Emitted at intervals during the pre-flight directory walk so the
        UI can show progress without waiting for the whole walk to
        finish (which can take minutes for multi-million-item datasets).
        Thread-safe (uses the message queue).
        """
        data = {
            "type": "dataset_scan_progress",
            "scope": scope,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "phase": phase,
            "files_walked": files_walked,
            "items_in_db": items_in_db,
            "items_missing": items_missing,
            "items_new": items_new,
        }
        if message:
            data["message"] = message
        self.message_queue.put(data)
        self._notify_sender()

    async def start_sender(self):
        """Background task to drain the message queue — event-driven, no polling.
        Sends a heartbeat every 30 seconds when idle to keep NAT/VPN tunnels alive."""
        self._send_event = asyncio.Event()
        self._loop = asyncio.get_event_loop()
        HEARTBEAT_INTERVAL = 30
        while True:
            try:
                try:
                    await asyncio.wait_for(self._send_event.wait(), timeout=HEARTBEAT_INTERVAL)
                    self._send_event.clear()
                except asyncio.TimeoutError:
                    # No messages for 30s — send heartbeat to keep NAT/VPN alive
                    if self.active_connections:
                        await self.broadcast(json.dumps({"type": "ping"}))
                    continue
                while not self.message_queue.empty():
                    try:
                        data = self.message_queue.get_nowait()
                        message_str = json.dumps(data)
                        await self.broadcast(message_str)
                    except queue.Empty:
                        break
            except Exception as e:
                print(f"WebSocket sender error: {e}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket):
    # Accept connection regardless of origin (bypass CORS)
    await websocket.accept()
    manager.active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)
