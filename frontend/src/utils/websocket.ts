type ProgressCallback = (step: number, totalSteps: number, message: string) => void;

class WebSocketClient {
  private ws: WebSocket | null = null;
  private callbacks: Set<ProgressCallback> = new Set();
  private reconnectTimer: NodeJS.Timeout | null = null;

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    const wsUrl = `ws://localhost:8000/ws/progress`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("WebSocket connected");
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "progress") {
          this.callbacks.forEach((callback) => {
            callback(data.step, data.total_steps, data.message);
          });
        }
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    this.ws.onclose = () => {
      console.log("WebSocket disconnected");
      // Auto-reconnect after 3 seconds
      this.reconnectTimer = setTimeout(() => {
        this.connect();
      }, 3000);
    };
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  subscribe(callback: ProgressCallback) {
    this.callbacks.add(callback);
  }

  unsubscribe(callback: ProgressCallback) {
    this.callbacks.delete(callback);
  }
}

export const wsClient = new WebSocketClient();
