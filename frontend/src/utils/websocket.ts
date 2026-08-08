export interface CFGMetrics {
  // Primary metrics
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

interface TrainingMetrics {
  run_id: number;
  step: number;
  loss: number;
  recon_loss?: number;
  // Bespoke arch/method-specific per-step scalars keyed by name (REPA, outpaint
  // gen_loss, …). Live equivalent of the DB extra_metrics channel.
  extra_metrics?: Record<string, number>;
  learning_rate?: number;
  grad_norm?: number;
  grad_norm_text_encoder?: number;
  grad_norm_text_encoder_1?: number;
  grad_norm_text_encoder_2?: number;
  grad_norm_unet?: number;
  grad_norm_vision_encoder?: number;
  epoch?: number;
  resume_seq?: number;
}

export interface FpFnScatterData {
  fp: number[];
  fn: number[];
  n_pos: number[];
  n_tags: number;
  total_images: number;
}

export interface TaggerMetrics {
  run_id: string;
  event: "step" | "epoch" | "train_f1";
  step: number;
  resume_seq?: number;
  epoch?: number;
  loss?: number;
  lr?: number;
  f1?: number;
  train_f1?: number;
  threshold?: number;
  progress?: number;
  precision?: number;
  recall?: number;
  fp_fn_scatter?: FpFnScatterData;
}

export interface DatasetScanProgress {
  scope: "tagger" | "training";
  run_id: string | number;
  dataset_id: number;
  dataset_name?: string;
  /** "scan_start" – this dataset's skippable window opened (Skip button on);
   *  "drift_walk" – walking the directory tree;
   *  "drift_done" – walk finished, report counts;
   *  "rescan" – running full /datasets/scan;
   *  "cleanup" – cleaning orphan latent cache (LoRA only; not cancellable);
   *  "skipped" – user skipped this dataset's rescan;
   *  "scan_end" – skippable window closed (Skip button off). */
  phase: "scan_start" | "drift_walk" | "drift_done" | "rescan" | "cleanup" | "skipped" | "scan_end";
  files_walked?: number;
  items_in_db?: number;
  items_missing?: number;
  items_new?: number;
  message?: string;
}

type TrainingMetricsCallback = (metrics: TrainingMetrics) => void;
type TaggerMetricsCallback = (metrics: TaggerMetrics) => void;
type DatasetScanProgressCallback = (progress: DatasetScanProgress) => void;

class ProgressClient {
  private eventSource: EventSource | null = null;
  private callbacks: Set<ProgressCallback> = new Set();
  private trainingMetricsCallbacks: Set<TrainingMetricsCallback> = new Set();
  private taggerMetricsCallbacks: Set<TaggerMetricsCallback> = new Set();
  private datasetScanProgressCallbacks: Set<DatasetScanProgressCallback> = new Set();
  private reconnectTimer: NodeJS.Timeout | null = null;
  // Timestamp of the last message of ANY type (including "ping"), used as a
  // liveness signal by long-running generation requests -- see api.ts's
  // postGenerationRequest(). The server sends a "ping" every 30s whenever no
  // real message went out, so this is a heartbeat even during phases (text
  // encode, VAE decode) that emit no "progress" message of their own.
  private lastMessageAt = 0;

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
      // Count "just connected" as activity so a caller checking
      // msSinceLastMessage() right after connect() doesn't see the
      // no-message-yet Infinity and mistake it for staleness.
      this.lastMessageAt = Date.now();
    };

    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.lastMessageAt = Date.now();

        if (data.type === "progress") {
          // Log without base64 data to avoid console spam
          const preview_length = data.preview_image ? data.preview_image.length : 0;
          console.log(`[SSE] Progress: ${data.step}/${data.total_steps}, preview: ${preview_length > 0 ? `${preview_length} chars` : 'none'}, CFG metrics: ${!!data.cfg_metrics}`);

          this.callbacks.forEach((callback) => {
            callback(data.step, data.total_steps, data.message, data.preview_image, data.cfg_metrics);
          });
        } else if (data.type === "training_metrics") {
          // Training metrics: loss, recon_loss, learning_rate
          console.log(`[SSE] Training metrics: run_id=${data.run_id}, step=${data.step}, loss=${data.loss?.toFixed(6) || 'N/A'}`);

          const metrics: TrainingMetrics = {
            run_id: data.run_id,
            step: data.step,
            loss: data.loss,
            recon_loss: data.recon_loss,
            extra_metrics: data.extra_metrics,
            learning_rate: data.learning_rate,
            grad_norm: data.grad_norm,
            grad_norm_text_encoder: data.grad_norm_text_encoder,
            grad_norm_text_encoder_1: data.grad_norm_text_encoder_1,
            grad_norm_text_encoder_2: data.grad_norm_text_encoder_2,
            grad_norm_unet: data.grad_norm_unet,
            grad_norm_vision_encoder: data.grad_norm_vision_encoder,
            epoch: data.epoch,
            resume_seq: data.resume_seq,
          };

          this.trainingMetricsCallbacks.forEach((callback) => {
            callback(metrics);
          });
        } else if (data.type === "tagger_metrics") {
          const metrics: TaggerMetrics = {
            run_id: data.run_id,
            event: data.event,
            step: data.step,
            resume_seq: data.resume_seq,
            epoch: data.epoch,
            loss: data.loss,
            lr: data.lr,
            f1: data.f1,
            train_f1: data.train_f1,
            threshold: data.threshold,
            progress: data.progress,
            precision: data.precision,
            recall: data.recall,
            fp_fn_scatter: data.fp_fn_scatter,
          };
          this.taggerMetricsCallbacks.forEach((cb) => cb(metrics));
        } else if (data.type === "dataset_scan_progress") {
          const ev: DatasetScanProgress = {
            scope: data.scope,
            run_id: data.run_id,
            dataset_id: data.dataset_id,
            dataset_name: data.dataset_name,
            phase: data.phase,
            files_walked: data.files_walked,
            items_in_db: data.items_in_db,
            items_missing: data.items_missing,
            items_new: data.items_new,
            message: data.message,
          };
          this.datasetScanProgressCallbacks.forEach((cb) => cb(ev));
        } else if (data.type === "ping") {
          // Heartbeat from server — keep NAT/VPN tunnel alive, no action needed
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

  /** True while the SSE channel to the backend is open (i.e. its heartbeat
   *  is a meaningful liveness signal right now). */
  isConnected(): boolean {
    return !!this.eventSource && this.eventSource.readyState === EventSource.OPEN;
  }

  /** ms since any message (including a "ping") was last received. Infinity
   *  if none has ever arrived on this connection. */
  msSinceLastMessage(): number {
    return this.lastMessageAt ? Date.now() - this.lastMessageAt : Infinity;
  }

  subscribe(callback: ProgressCallback) {
    this.callbacks.add(callback);
  }

  unsubscribe(callback: ProgressCallback) {
    this.callbacks.delete(callback);
  }

  subscribeToTrainingMetrics(callback: TrainingMetricsCallback) {
    this.trainingMetricsCallbacks.add(callback);
  }

  unsubscribeFromTrainingMetrics(callback: TrainingMetricsCallback) {
    this.trainingMetricsCallbacks.delete(callback);
  }

  subscribeToTaggerMetrics(callback: TaggerMetricsCallback) {
    this.taggerMetricsCallbacks.add(callback);
  }

  unsubscribeFromTaggerMetrics(callback: TaggerMetricsCallback) {
    this.taggerMetricsCallbacks.delete(callback);
  }

  subscribeToDatasetScanProgress(callback: DatasetScanProgressCallback) {
    this.datasetScanProgressCallbacks.add(callback);
  }

  unsubscribeFromDatasetScanProgress(callback: DatasetScanProgressCallback) {
    this.datasetScanProgressCallbacks.delete(callback);
  }
}

// Export with same name for backwards compatibility
export const wsClient = new ProgressClient();

// Export types
export type { TrainingMetrics, TrainingMetricsCallback, TaggerMetricsCallback };
