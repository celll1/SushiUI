type ProgressCallback = (step: number, totalSteps: number, message: string, previewImage?: string) => void;

class WebSocketClient {
  private ws: WebSocket | null = null;
  private callbacks: Set<ProgressCallback> = new Set();
  private reconnectTimer: NodeJS.Timeout | null = null;
  private backendPort: number = 8000; // Default port

  async fetchBackendPort(): Promise<number> {
    try {
      // Try to get port info from backend's port-info endpoint
      const response = await fetch('/api/port-info');
      if (response.ok) {
        const data = await response.json();
        return data.port;
      }
    } catch (error) {
      console.log('[WebSocket] Could not fetch backend port, using default:', error);
    }
    return 8000; // Default fallback
  }

  async connect() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    // Fetch backend port if not already fetched
    this.backendPort = await this.fetchBackendPort();

    const wsUrl = `ws://localhost:${this.backendPort}/ws/progress`;
    console.log(`[WebSocket] Connecting to ${wsUrl}`);
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("WebSocket connected");
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "progress") {
          this.callbacks.forEach((callback) => {
            callback(data.step, data.total_steps, data.message, data.preview_image);
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
