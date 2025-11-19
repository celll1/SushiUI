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

    // Use current hostname instead of hardcoded localhost for mobile compatibility
    const hostname = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
    const protocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${hostname}:${this.backendPort}/ws/progress`;
    console.log(`[WebSocket] Connecting to ${wsUrl}`);
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("[WebSocket] Connected successfully");
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("[WebSocket] Received message:", data);
        if (data.type === "progress") {
          console.log(`[WebSocket] Progress: ${data.step}/${data.total_steps}, has preview: ${!!data.preview_image}`);
          this.callbacks.forEach((callback) => {
            callback(data.step, data.total_steps, data.message, data.preview_image);
          });
        }
      } catch (error) {
        console.error("[WebSocket] Failed to parse message:", error, event.data);
      }
    };

    this.ws.onerror = (error) => {
      console.error("[WebSocket] Error:", error);
    };

    this.ws.onclose = (event) => {
      console.log(`[WebSocket] Disconnected (code: ${event.code}, reason: ${event.reason})`);
      // Auto-reconnect after 3 seconds
      this.reconnectTimer = setTimeout(() => {
        console.log("[WebSocket] Attempting to reconnect...");
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
