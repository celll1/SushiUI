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
        # Put message in queue (thread-safe)
        self.message_queue.put(data)

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

        # Put message in queue (thread-safe)
        self.message_queue.put(data)

    async def start_sender(self):
        """Background task to send queued messages"""
        while True:
            try:
                # Check queue every 10ms
                await asyncio.sleep(0.01)
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
