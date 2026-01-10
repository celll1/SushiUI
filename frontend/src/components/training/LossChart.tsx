"use client";

import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { RefreshCw } from "lucide-react";
import { getTrainingMetrics } from "@/utils/api";
import { wsClient, type TrainingMetrics } from "@/utils/websocket";

interface MetricPoint {
  step: number;
  value: number;
  wall_time: number;
}

// Calculate smoothed data using exponential moving average
const calculateSmoothing = (data: MetricPoint[], smoothingFactor: number): MetricPoint[] => {
  if (data.length === 0 || smoothingFactor === 0) return data;

  const smoothed: MetricPoint[] = [];
  let lastSmoothed = data[0].value;

  for (const point of data) {
    lastSmoothed = lastSmoothed * smoothingFactor + point.value * (1 - smoothingFactor);
    smoothed.push({
      step: point.step,
      value: lastSmoothed,
      wall_time: point.wall_time
    });
  }

  return smoothed;
};

interface LossChartProps {
  runId: number;
  isRunning: boolean;
}

export default function LossChart({ runId, isRunning }: LossChartProps) {
  const [lossData, setLossData] = useState<MetricPoint[]>([]);
  const [reconLossData, setReconLossData] = useState<MetricPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastStep, setLastStep] = useState<number>(-1);

  // UI controls
  const [smoothingFactor, setSmoothingFactor] = useState(0.9);
  const [pollingInterval, setPollingInterval] = useState<number>(0); // 0 = off
  const [showLoss, setShowLoss] = useState(true);
  const [showReconLoss, setShowReconLoss] = useState(true);
  const [yAxisMode, setYAxisMode] = useState<"auto" | "custom">("auto");
  const [customYMin, setCustomYMin] = useState<number>(0);
  const [customYMax, setCustomYMax] = useState<number>(1);
  const [xAxisMode, setXAxisMode] = useState<"auto" | "custom">("auto");
  const [customXMin, setCustomXMin] = useState<number>(0);
  const [customXMax, setCustomXMax] = useState<number>(1000);

  // Tooltip state
  const [tooltip, setTooltip] = useState<{ x: number; y: number; step: number; loss: number; smoothLoss: number; reconLoss?: number; smoothReconLoss?: number } | null>(null);

  // SVG ref for responsive width
  const svgRef = useRef<SVGSVGElement>(null);
  const [svgWidth, setSvgWidth] = useState<number>(550);

  // Calculate smooth loss on client side
  const smoothLossData = useMemo(() => {
    return calculateSmoothing(lossData, smoothingFactor);
  }, [lossData, smoothingFactor]);

  const smoothReconLossData = useMemo(() => {
    return calculateSmoothing(reconLossData, smoothingFactor);
  }, [reconLossData, smoothingFactor]);

  const fetchMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch all metrics with uniform sampling (backend handles sampling)
      const data = await getTrainingMetrics(runId);

      console.log(`[LossChart] Fetched metrics: ${data.loss.length} loss points (uniform sampling)`);

      // Replace data entirely (no merging needed - backend does uniform sampling)
      setLossData(data.loss);
      setReconLossData(data.recon_loss || []);

      // Update lastStep for display
      if (data.loss.length > 0) {
        const maxStep = Math.max(...data.loss.map((d) => d.step));
        setLastStep(maxStep);
      }
    } catch (err: any) {
      console.error("[LossChart] Error fetching metrics:", err);
      setError(err.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [runId]);

  // Initial fetch
  useEffect(() => {
    console.log(`[LossChart] Initial fetch for runId=${runId}`);
    fetchMetrics();
  }, [runId, fetchMetrics]);

  // Auto-refresh based on polling interval
  useEffect(() => {
    if (pollingInterval > 0) {
      console.log(`[LossChart] Starting auto-refresh (every ${pollingInterval}s)`);
      const interval = setInterval(() => {
        console.log(`[LossChart] Auto-refresh triggered (interval=${pollingInterval}s)`);
        fetchMetrics();
      }, pollingInterval * 1000);
      return () => {
        console.log(`[LossChart] Stopping auto-refresh`);
        clearInterval(interval);
      };
    }
  }, [pollingInterval, fetchMetrics]);

  // Real-time SSE update for training metrics (when training is running)
  useEffect(() => {
    if (!isRunning) {
      console.log(`[LossChart] Training not running, skipping SSE subscription`);
      return;
    }

    console.log(`[LossChart] Training is running, subscribing to SSE for runId=${runId}`);

    // Connect to SSE if not already connected
    wsClient.connect();

    // Subscribe to training metrics SSE messages
    const handleTrainingMetrics = (metrics: TrainingMetrics) => {
      console.log(`[LossChart] SSE metric received: run_id=${metrics.run_id}, step=${metrics.step}, loss=${metrics.loss?.toFixed(6)}`);

      // Only update if this metric is for the current run
      if (metrics.run_id !== runId) {
        console.log(`[LossChart] Ignoring metric for different run (expected ${runId}, got ${metrics.run_id})`);
        return;
      }

      console.log(`[LossChart] Applying real-time metric: step=${metrics.step}, loss=${metrics.loss?.toFixed(6)}`);

      // Add new metric point to lossData
      const newLossPoint: MetricPoint = {
        step: metrics.step,
        value: metrics.loss,
        wall_time: Date.now() / 1000 // Current timestamp in seconds
      };

      setLossData((prevData) => {
        // Check if this step already exists (UPSERT behavior)
        const existingIndex = prevData.findIndex((p) => p.step === metrics.step);
        if (existingIndex >= 0) {
          // Update existing point
          const newData = [...prevData];
          newData[existingIndex] = newLossPoint;
          return newData;
        } else {
          // Add new point
          return [...prevData, newLossPoint];
        }
      });

      // Add recon_loss if available
      if (metrics.recon_loss !== undefined && metrics.recon_loss !== null) {
        const newReconLossPoint: MetricPoint = {
          step: metrics.step,
          value: metrics.recon_loss,
          wall_time: Date.now() / 1000
        };

        setReconLossData((prevData) => {
          const existingIndex = prevData.findIndex((p) => p.step === metrics.step);
          if (existingIndex >= 0) {
            const newData = [...prevData];
            newData[existingIndex] = newReconLossPoint;
            return newData;
          } else {
            return [...prevData, newReconLossPoint];
          }
        });
      }

      // Update lastStep (both state and ref)
      setLastStep((prevLastStep) => {
        const newLastStep = Math.max(prevLastStep, metrics.step);
        lastStepRef.current = newLastStep;
        return newLastStep;
      });
    };

    wsClient.subscribeToTrainingMetrics(handleTrainingMetrics);
    console.log(`[LossChart] Subscribed to SSE training metrics`);

    return () => {
      console.log(`[LossChart] Unsubscribing from SSE training metrics`);
      wsClient.unsubscribeFromTrainingMetrics(handleTrainingMetrics);
    };
  }, [isRunning, runId]);

  // Monitor SVG width for responsive layout
  useEffect(() => {
    if (!svgRef.current) return;

    const updateWidth = () => {
      if (svgRef.current) {
        const rect = svgRef.current.getBoundingClientRect();
        if (rect.width > 0) {
          setSvgWidth(rect.width);
        }
      }
    };

    // Initial width immediately and after render
    updateWidth();
    const timeoutId = setTimeout(updateWidth, 10);

    // Watch for resize
    const resizeObserver = new ResizeObserver(updateWidth);
    resizeObserver.observe(svgRef.current);

    return () => {
      clearTimeout(timeoutId);
      resizeObserver.disconnect();
    };
  }, [lossData.length]); // Update when data changes

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500 text-red-400 rounded p-3 text-sm">
        {error}
      </div>
    );
  }

  if (loading && lossData.length === 0) {
    return (
      <div className="text-gray-400 text-sm">Loading metrics...</div>
    );
  }

  if (lossData.length === 0) {
    return (
      <div className="text-gray-400 text-sm">No training data available yet</div>
    );
  }

  // Calculate chart dimensions and scaling
  const height = 300;
  // Responsive padding: reduce right padding on narrow screens for better chart visibility
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 640;
  const padding = {
    top: 20,
    right: isMobile ? 10 : 180, // Minimal padding on mobile, tooltip space on desktop
    bottom: 40,
    left: isMobile ? 45 : 60 // Slightly less left padding on mobile
  };
  const chartWidth = svgWidth - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const autoMaxStep = Math.max(...lossData.map((d) => d.step));
  const autoMinStep = Math.min(...lossData.map((d) => d.step));

  // Calculate min/max considering both loss and recon_loss
  const allValues = [
    ...(showLoss ? lossData.map((d) => d.value) : []),
    ...(showReconLoss ? reconLossData.map((d) => d.value) : [])
  ];
  const autoMaxLoss = allValues.length > 0 ? Math.max(...allValues) : 1;
  const autoMinLoss = allValues.length > 0 ? Math.min(...allValues) : 0;

  // Use custom scale if enabled, otherwise use auto scale
  const maxStep = xAxisMode === "custom" ? customXMax : autoMaxStep;
  const minStep = xAxisMode === "custom" ? customXMin : autoMinStep;
  const maxLoss = yAxisMode === "custom" ? customYMax : autoMaxLoss;
  const minLoss = yAxisMode === "custom" ? customYMin : autoMinLoss;

  const scaleX = (step: number) =>
    padding.left + ((step - minStep) / (maxStep - minStep || 1)) * chartWidth;

  const scaleY = (loss: number) =>
    padding.top + chartHeight - ((loss - minLoss) / (maxLoss - minLoss || 1)) * chartHeight;

  // Generate paths for line charts
  const rawLinePath = lossData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  const smoothLinePath = smoothLossData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  // Generate recon_loss paths
  const rawReconLinePath = reconLossData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  const smoothReconLinePath = smoothReconLossData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  // Generate Y-axis ticks
  const yTicks = 5;
  const yTickValues = Array.from({ length: yTicks }, (_, i) =>
    minLoss + ((maxLoss - minLoss) / (yTicks - 1)) * i
  );

  // Generate X-axis ticks
  const xTicks = 5;
  const xTickValues = Array.from({ length: xTicks }, (_, i) =>
    Math.round(minStep + ((maxStep - minStep) / (xTicks - 1)) * i)
  );

  // Handle mouse move to show tooltip
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const svgRect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - svgRect.left;

    // Check if mouse is within chart area
    if (mouseX < padding.left || mouseX > svgWidth - padding.right) {
      setTooltip(null);
      return;
    }

    // Find nearest data point by step
    const hoveredStep = minStep + ((mouseX - padding.left) / chartWidth) * (maxStep - minStep);

    // Find closest data point
    let closestIndex = 0;
    let minDistance = Math.abs(lossData[0].step - hoveredStep);

    for (let i = 1; i < lossData.length; i++) {
      const distance = Math.abs(lossData[i].step - hoveredStep);
      if (distance < minDistance) {
        minDistance = distance;
        closestIndex = i;
      }
    }

    const closestPoint = lossData[closestIndex];
    const closestSmooth = smoothLossData[closestIndex];

    // Find recon loss for the same step (not by index, as arrays may have different lengths)
    const step = closestSmooth.step;
    const closestReconPoint = reconLossData.find(d => d.step === step);
    const closestSmoothRecon = smoothReconLossData.find(d => d.step === step);

    const pointX = scaleX(closestSmooth.step);
    const pointY = scaleY(closestSmooth.value);  // Use smooth loss Y position

    setTooltip({
      x: pointX,
      y: pointY,
      step: closestSmooth.step,
      loss: closestPoint.value,
      smoothLoss: closestSmooth.value,
      reconLoss: closestReconPoint?.value,
      smoothReconLoss: closestSmoothRecon?.value
    });
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  return (
    <div className="bg-gray-800 border border-gray-700 rounded p-4">
      {/* Header with controls */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Training Loss</h3>

        <div className="flex items-center gap-3">
          {/* Polling interval selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400">Auto-refresh:</label>
            <select
              value={pollingInterval}
              onChange={(e) => {
                const newInterval = Number(e.target.value);
                console.log(`[LossChart] Auto-refresh interval changed: ${pollingInterval}s → ${newInterval}s`);
                setPollingInterval(newInterval);
              }}
              className="text-xs px-2 py-1 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500"
            >
              <option value="0">Off</option>
              <option value="5">5s</option>
              <option value="10">10s</option>
              <option value="30">30s</option>
              <option value="60">60s</option>
            </select>
          </div>

          {/* Manual refresh button */}
          <button
            onClick={() => {
              console.log("[LossChart] Manual refresh button clicked");
              fetchMetrics();
            }}
            disabled={loading}
            className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Refresh data"
          >
            <RefreshCw className={`h-4 w-4 text-gray-300 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Smoothing slider */}
      <div className="flex items-center gap-3 mb-3">
        <label className="text-xs text-gray-400 whitespace-nowrap w-20">
          Smoothing:
        </label>
        <input
          type="range"
          min="0"
          max="0.99"
          step="0.01"
          value={smoothingFactor}
          onChange={(e) => setSmoothingFactor(parseFloat(e.target.value))}
          className="flex-1 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer"
        />
        <span className="text-xs text-gray-400 w-12 text-right">
          {(smoothingFactor * 100).toFixed(0)}%
        </span>
      </div>

      {/* Visibility toggles */}
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
          <input
            type="checkbox"
            checked={showLoss}
            onChange={(e) => setShowLoss(e.target.checked)}
            className="w-4 h-4"
          />
          <span>Prediction Loss</span>
        </label>
        {reconLossData.length > 0 && (
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showReconLoss}
              onChange={(e) => setShowReconLoss(e.target.checked)}
              className="w-4 h-4"
            />
            <span>Reconstruction Loss</span>
          </label>
        )}
      </div>

      {/* Axis Scale Controls (2-column layout) */}
      <div className="grid grid-cols-2 gap-4 mb-3">
        {/* X-axis Scale Controls */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input
                type="checkbox"
                checked={xAxisMode === "custom"}
                onChange={(e) => {
                  const newMode = e.target.checked ? "custom" : "auto";
                  setXAxisMode(newMode);
                  if (newMode === "custom" && xAxisMode === "auto") {
                    setCustomXMin(autoMinStep);
                    setCustomXMax(autoMaxStep);
                  }
                }}
                className="w-4 h-4"
              />
              <span>X-axis</span>
            </label>
            {xAxisMode === "custom" && (
              <button
                onClick={() => {
                  setCustomXMin(autoMinStep);
                  setCustomXMax(autoMaxStep);
                }}
                className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
                title="Reset to auto values"
              >
                Reset
              </button>
            )}
          </div>

          {xAxisMode === "custom" && (
            <div className="flex items-center gap-2 pl-6">
              <div className="flex-1 relative h-8">
                {/* Visual track background */}
                <div className="absolute top-1/2 -translate-y-1/2 w-full h-1 bg-gray-700 rounded-full" />
                {/* Active track */}
                <div
                  className="absolute top-1/2 -translate-y-1/2 h-1 bg-blue-500 rounded-full"
                  style={{
                    left: `${(customXMin / autoMaxStep) * 100}%`,
                    right: `${100 - (customXMax / autoMaxStep) * 100}%`,
                  }}
                />
                {/* Min slider */}
                <input
                  type="range"
                  min={0}
                  max={autoMaxStep}
                  step={Math.max(1, Math.floor(autoMaxStep / 1000))}
                  value={customXMin}
                  onChange={(e) => {
                    const newMin = parseFloat(e.target.value);
                    if (newMin < customXMax) {
                      setCustomXMin(newMin);
                    }
                  }}
                  className="absolute top-1/2 -translate-y-1/2 w-full h-1 appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-blue-500 [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-blue-500"
                  style={{ zIndex: customXMin > autoMaxStep - (autoMaxStep / 2) ? 5 : 3 }}
                />
                {/* Max slider */}
                <input
                  type="range"
                  min={0}
                  max={autoMaxStep}
                  step={Math.max(1, Math.floor(autoMaxStep / 1000))}
                  value={customXMax}
                  onChange={(e) => {
                    const newMax = parseFloat(e.target.value);
                    if (newMax > customXMin) {
                      setCustomXMax(newMax);
                    }
                  }}
                  className="absolute top-1/2 -translate-y-1/2 w-full h-1 appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-blue-500 [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-blue-500"
                  style={{ zIndex: 4 }}
                />
              </div>
              <span className="text-xs text-gray-400 w-28 text-right">
                {customXMin.toFixed(0)} - {customXMax.toFixed(0)}
              </span>
            </div>
          )}
        </div>

        {/* Y-axis Scale Controls */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input
                type="checkbox"
                checked={yAxisMode === "custom"}
                onChange={(e) => {
                  const newMode = e.target.checked ? "custom" : "auto";
                  setYAxisMode(newMode);
                  if (newMode === "custom" && yAxisMode === "auto") {
                    setCustomYMin(autoMinLoss);
                    setCustomYMax(autoMaxLoss);
                  }
                }}
                className="w-4 h-4"
              />
              <span>Y-axis</span>
            </label>
            {yAxisMode === "custom" && (
              <button
                onClick={() => {
                  setCustomYMin(autoMinLoss);
                  setCustomYMax(autoMaxLoss);
                }}
                className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
                title="Reset to auto values"
              >
                Reset
              </button>
            )}
          </div>

          {yAxisMode === "custom" && (
            <div className="flex items-center gap-2 pl-6">
              <div className="flex-1 relative h-8">
                {/* Visual track background */}
                <div className="absolute top-1/2 -translate-y-1/2 w-full h-1 bg-gray-700 rounded-full" />
                {/* Active track */}
                <div
                  className="absolute top-1/2 -translate-y-1/2 h-1 bg-blue-500 rounded-full"
                  style={{
                    left: `${(customYMin / (autoMaxLoss * 1.1)) * 100}%`,
                    right: `${100 - (customYMax / (autoMaxLoss * 1.1)) * 100}%`,
                  }}
                />
                {/* Min slider */}
                <input
                  type="range"
                  min={0}
                  max={autoMaxLoss * 1.1}
                  step={autoMaxLoss / 1000}
                  value={customYMin}
                  onChange={(e) => {
                    const newMin = parseFloat(e.target.value);
                    if (newMin < customYMax) {
                      setCustomYMin(newMin);
                    }
                  }}
                  className="absolute top-1/2 -translate-y-1/2 w-full h-1 appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-blue-500 [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-blue-500"
                  style={{ zIndex: customYMin > (autoMaxLoss * 1.1) - ((autoMaxLoss * 1.1) / 2) ? 5 : 3 }}
                />
                {/* Max slider */}
                <input
                  type="range"
                  min={0}
                  max={autoMaxLoss * 1.1}
                  step={autoMaxLoss / 1000}
                  value={customYMax}
                  onChange={(e) => {
                    const newMax = parseFloat(e.target.value);
                    if (newMax > customYMin) {
                      setCustomYMax(newMax);
                    }
                  }}
                  className="absolute top-1/2 -translate-y-1/2 w-full h-1 appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-blue-500 [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-blue-500"
                  style={{ zIndex: 4 }}
                />
              </div>
              <span className="text-xs text-gray-400 w-28 text-right">
                {customYMin.toFixed(3)} - {customYMax.toFixed(3)}
              </span>
            </div>
          )}
        </div>
      </div>

      <svg
        ref={svgRef}
        width="100%"
        height={height}
        className="text-gray-400"
        style={{ fontFamily: "monospace", fontSize: "10px" }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        {/* Clip path to restrict drawing to chart area */}
        <defs>
          <clipPath id="chart-clip-loss">
            <rect
              x={padding.left}
              y={padding.top}
              width={chartWidth}
              height={chartHeight}
            />
          </clipPath>
        </defs>

        {/* Y-axis */}
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={height - padding.bottom}
          stroke="currentColor"
          strokeWidth="1"
        />

        {/* X-axis */}
        <line
          x1={padding.left}
          y1={height - padding.bottom}
          x2={svgWidth - padding.right}
          y2={height - padding.bottom}
          stroke="currentColor"
          strokeWidth="1"
        />

        {/* Y-axis ticks and labels */}
        {yTickValues.map((value, i) => {
          const y = scaleY(value);
          return (
            <g key={i}>
              <line
                x1={padding.left - 5}
                y1={y}
                x2={padding.left}
                y2={y}
                stroke="currentColor"
                strokeWidth="1"
              />
              <text
                x={padding.left - 10}
                y={y}
                textAnchor="end"
                dominantBaseline="middle"
                fill="currentColor"
              >
                {value.toFixed(3)}
              </text>
              {/* Grid line */}
              <line
                x1={padding.left}
                y1={y}
                x2={svgWidth - padding.right}
                y2={y}
                stroke="currentColor"
                strokeWidth="0.5"
                opacity="0.2"
              />
            </g>
          );
        })}

        {/* X-axis ticks and labels */}
        {xTickValues.map((value, i) => {
          const x = scaleX(value);
          return (
            <g key={i}>
              <line
                x1={x}
                y1={height - padding.bottom}
                x2={x}
                y2={height - padding.bottom + 5}
                stroke="currentColor"
                strokeWidth="1"
              />
              <text
                x={x}
                y={height - padding.bottom + 20}
                textAnchor="middle"
                fill="currentColor"
              >
                {value}
              </text>
              {/* Grid line */}
              <line
                x1={x}
                y1={padding.top}
                x2={x}
                y2={height - padding.bottom}
                stroke="currentColor"
                strokeWidth="0.5"
                opacity="0.2"
              />
            </g>
          );
        })}

        {/* Axis labels */}
        <text
          x={svgWidth / 2}
          y={height - 5}
          textAnchor="middle"
          fill="currentColor"
          fontSize="12"
        >
          Step
        </text>
        <text
          x={padding.left - 45}
          y={height / 2}
          textAnchor="middle"
          fill="currentColor"
          fontSize="12"
          transform={`rotate(-90, ${padding.left - 45}, ${height / 2})`}
        >
          Loss
        </text>

        {/* Prediction Loss lines */}
        {showLoss && (
          <>
            {/* Raw loss line (behind) */}
            <path
              d={rawLinePath}
              fill="none"
              stroke="#3b82f6"
              strokeWidth="1.5"
              strokeLinejoin="round"
              opacity="0.3"
              clipPath="url(#chart-clip-loss)"
            />

            {/* Smooth loss line */}
            {smoothingFactor > 0 && (
              <path
                d={smoothLinePath}
                fill="none"
                stroke="#60a5fa"
                strokeWidth="2.5"
                strokeLinejoin="round"
                opacity="0.9"
                clipPath="url(#chart-clip-loss)"
              />
            )}
          </>
        )}

        {/* Reconstruction Loss lines */}
        {showReconLoss && reconLossData.length > 0 && (
          <>
            {/* Raw recon loss line (behind) */}
            <path
              d={rawReconLinePath}
              fill="none"
              stroke="#10b981"
              strokeWidth="1.5"
              strokeLinejoin="round"
              opacity="0.3"
              clipPath="url(#chart-clip-loss)"
            />

            {/* Smooth recon loss line */}
            {smoothingFactor > 0 && (
              <path
                d={smoothReconLinePath}
                fill="none"
                stroke="#34d399"
                strokeWidth="2.5"
                strokeLinejoin="round"
                opacity="0.9"
                clipPath="url(#chart-clip-loss)"
              />
            )}
          </>
        )}

        {/* Tooltip */}
        {tooltip && (
          <g>
            {/* Crosshair vertical line */}
            <line
              x1={tooltip.x}
              y1={padding.top}
              x2={tooltip.x}
              y2={height - padding.bottom}
              stroke="#94a3b8"
              strokeWidth="1"
              strokeDasharray="4 2"
              opacity="0.5"
            />

            {/* Tooltip point indicator */}
            <circle
              cx={tooltip.x}
              cy={tooltip.y}
              r="4"
              fill="#3b82f6"
              stroke="#fff"
              strokeWidth="2"
            />

            {/* Tooltip box - responsive positioning */}
            {(() => {
              const tooltipWidth = 160;
              const tooltipHeight = tooltip.reconLoss !== undefined ? 85 : 50;
              // Show tooltip on left if it would overflow on right
              const showLeft = tooltip.x + tooltipWidth + 20 > svgWidth;
              const tooltipX = showLeft ? tooltip.x - tooltipWidth - 10 : tooltip.x + 10;
              const textX = showLeft ? tooltip.x - tooltipWidth - 5 : tooltip.x + 15;

              return (
                <g>
                  {/* Background */}
                  <rect
                    x={tooltipX}
                    y={tooltip.y - 55}
                    width={tooltipWidth}
                    height={tooltipHeight}
                    fill="#1f2937"
                    stroke="#4b5563"
                    strokeWidth="1"
                    rx="4"
                  />

                  {/* Text content */}
                  <text
                    x={textX}
                    y={tooltip.y - 40}
                    fill="#e5e7eb"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    Step: {tooltip.step}
                  </text>
                  {showLoss && (
                    <>
                      <text
                        x={textX}
                        y={tooltip.y - 25}
                        fill="#3b82f6"
                        fontSize="11"
                        fontFamily="monospace"
                      >
                        Pred Loss: {tooltip.loss.toFixed(4)}
                      </text>
                      <text
                        x={textX}
                        y={tooltip.y - 10}
                        fill="#60a5fa"
                        fontSize="11"
                        fontFamily="monospace"
                      >
                        Smooth: {tooltip.smoothLoss.toFixed(4)}
                      </text>
                    </>
                  )}
                  {showReconLoss && tooltip.reconLoss !== undefined && (
                    <>
                      <text
                        x={textX}
                        y={tooltip.y + 5}
                        fill="#10b981"
                        fontSize="11"
                        fontFamily="monospace"
                      >
                        Recon Loss: {tooltip.reconLoss.toFixed(4)}
                      </text>
                      <text
                        x={textX}
                        y={tooltip.y + 20}
                        fill="#34d399"
                        fontSize="11"
                        fontFamily="monospace"
                      >
                        Smooth: {tooltip.smoothReconLoss?.toFixed(4)}
                      </text>
                    </>
                  )}
                </g>
              );
            })()}
          </g>
        )}
      </svg>

      {/* Legend and stats */}
      <div className="mt-3 text-xs text-gray-500">
        <div className="flex items-center gap-6 mb-2">
          {showLoss && (
            <div className="flex items-center gap-3">
              <span className="text-gray-400">Prediction Loss:</span>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-blue-500 opacity-30"></div>
                <span>Raw</span>
              </div>
              {smoothingFactor > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-blue-400"></div>
                  <span>Smooth</span>
                </div>
              )}
            </div>
          )}
          {showReconLoss && reconLossData.length > 0 && (
            <div className="flex items-center gap-3">
              <span className="text-gray-400">Reconstruction Loss:</span>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-emerald-500 opacity-30"></div>
                <span>Raw</span>
              </div>
              {smoothingFactor > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-emerald-400"></div>
                  <span>Smooth</span>
                </div>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-4 text-gray-400">
          {showLoss && smoothLossData.length > 0 && (
            <span>
              Latest Pred: {smoothLossData[smoothLossData.length - 1]?.value.toFixed(4)}
            </span>
          )}
          {showReconLoss && smoothReconLossData.length > 0 && (
            <span>
              Latest Recon: {smoothReconLossData[smoothReconLossData.length - 1]?.value.toFixed(4)}
            </span>
          )}
          <span className="ml-auto">
            Step {smoothLossData[smoothLossData.length - 1]?.step || 0} / {lossData.length} points
          </span>
        </div>
      </div>
    </div>
  );
}
