export interface CFGMetrics {
  // Primary metrics
  cosine_similarity: number;
  relative_diff: number;
  snr: number;

  // L2 norms (for reference)
  uncond_norm: number;
  text_norm: number;
  diff_norm: number;

  // Statistics
  uncond_mean: number;
  text_mean: number;
  diff_mean: number;
  uncond_std: number;
  text_std: number;
  diff_std: number;

  guidance_scale: number;
  timestep: number;
  step: number;
  sigma?: number;
}

type ProgressCallback = (step: number, totalSteps: number, message: string, previewImage?: string, cfgMetrics?: CFGMetrics) => void;

class ProgressClient {
  private eventSource: EventSource | null = null;
  private callbacks: Set<ProgressCallback> = new Set();
  private reconnectTimer: NodeJS.Timeout | null = null;

  connect() {
    if (this.eventSource && this.eventSource.readyState === EventSource.OPEN) {
      console.log('[SSE] Already connected, skipping');
      return;
    }

    // Close any existing connection first
    if (this.eventSource) {
      console.log('[SSE] Closing existing connection');
      this.eventSource.close();
      this.eventSource = null;
    }

    // Connect to Next.js API route which will proxy to backend WebSocket
    // This works for both localhost and external network access
    const sseUrl = '/api/progress';
    console.log(`[SSE] Connecting to ${sseUrl}`);

    this.eventSource = new EventSource(sseUrl);

    this.eventSource.onopen = () => {
      console.log("[SSE] Connected successfully");
    };

    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "progress") {
          // Log without base64 data to avoid console spam
          const preview_length = data.preview_image ? data.preview_image.length : 0;
          console.log(`[SSE] Progress: ${data.step}/${data.total_steps}, preview: ${preview_length > 0 ? `${preview_length} chars` : 'none'}, CFG metrics: ${!!data.cfg_metrics}`);

          this.callbacks.forEach((callback) => {
            callback(data.step, data.total_steps, data.message, data.preview_image, data.cfg_metrics);
          });
        } else if (data.type === "error") {
          console.error("[SSE] Error from server:", data.message);
        } else if (data.type === "closed") {
          console.log("[SSE] Backend WebSocket closed:", data.reason);
          this.handleDisconnect();
        }
      } catch (error) {
        console.error("[SSE] Failed to parse message:", error);
      }
    };

    this.eventSource.onerror = (error) => {
      console.error("[SSE] Error:", error);
      this.handleDisconnect();
    };
  }

  private handleDisconnect() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }

    // Auto-reconnect after 3 seconds
    if (!this.reconnectTimer) {
      this.reconnectTimer = setTimeout(() => {
        console.log("[SSE] Attempting to reconnect...");
        this.reconnectTimer = null;
        this.connect();
      }, 3000);
    }
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  subscribe(callback: ProgressCallback) {
    this.callbacks.add(callback);
  }

  unsubscribe(callback: ProgressCallback) {
    this.callbacks.delete(callback);
  }
}

// Export with same name for backwards compatibility
export const wsClient = new ProgressClient();
