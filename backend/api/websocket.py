from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

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

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)
