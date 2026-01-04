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

interface GradNormChartProps {
  runId: number;
  isRunning: boolean;
}

export default function GradNormChart({ runId, isRunning }: GradNormChartProps) {
  const [gradNormData, setGradNormData] = useState<MetricPoint[]>([]);
  const [gradNormTEData, setGradNormTEData] = useState<MetricPoint[]>([]);
  const [gradNormUNetData, setGradNormUNetData] = useState<MetricPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastStep, setLastStep] = useState<number>(-1);

  // Use ref to track lastStep without causing useCallback dependency issues
  const lastStepRef = useRef<number>(-1);

  // UI controls
  const [smoothingFactor, setSmoothingFactor] = useState(0.9);
  const [pollingInterval, setPollingInterval] = useState<number>(0); // 0 = off
  const [showTotal, setShowTotal] = useState(true);
  const [showTextEncoder, setShowTextEncoder] = useState(true);
  const [showUNet, setShowUNet] = useState(true);

  // Tooltip state
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    step: number;
    total: number;
    smoothTotal: number;
    textEncoder?: number;
    smoothTextEncoder?: number;
    unet?: number;
    smoothUNet?: number;
  } | null>(null);

  // SVG ref for responsive width
  const svgRef = useRef<SVGSVGElement>(null);
  const [svgWidth, setSvgWidth] = useState<number>(550);

  // Calculate smooth grad norms on client side
  const smoothGradNormData = useMemo(() => {
    return calculateSmoothing(gradNormData, smoothingFactor);
  }, [gradNormData, smoothingFactor]);

  const smoothGradNormTEData = useMemo(() => {
    return calculateSmoothing(gradNormTEData, smoothingFactor);
  }, [gradNormTEData, smoothingFactor]);

  const smoothGradNormUNetData = useMemo(() => {
    return calculateSmoothing(gradNormUNetData, smoothingFactor);
  }, [gradNormUNetData, smoothingFactor]);

  const fetchMetrics = useCallback(async (isIncremental: boolean = false) => {
    try {
      setLoading(true);
      setError(null);

      // Use lastStep ref to avoid dependency issues
      const sinceStep = isIncremental && lastStepRef.current >= 0 ? lastStepRef.current : undefined;
      const data = await getTrainingMetrics(runId, sinceStep);

      console.log(`[GradNormChart] Fetched metrics: ${data.grad_norm.length} grad_norm points, incremental=${isIncremental}, sinceStep=${sinceStep}`);

      // Merge new data with existing data and limit total points to prevent memory accumulation
      const MAX_POINTS = 1000; // Limit frontend memory usage for long-duration training

      setGradNormData((prevData) => {
        let newData = sinceStep !== undefined ? [...prevData, ...data.grad_norm] : data.grad_norm;

        // Update lastStep
        if (newData.length > 0) {
          const maxStep = Math.max(...newData.map((d) => d.step));
          lastStepRef.current = maxStep;
          setLastStep(maxStep);
        }

        // Decimate if too many points (keep recent data dense, old data sparse)
        if (newData.length > MAX_POINTS) {
          const keepRecent = Math.floor(MAX_POINTS * 0.3); // Keep last 30% at full resolution
          const decimateOld = newData.length - keepRecent;
          const decimationFactor = Math.ceil(decimateOld / (MAX_POINTS - keepRecent));

          const decimatedOld = newData.slice(0, decimateOld).filter((_, i) => i % decimationFactor === 0);
          const recentData = newData.slice(decimateOld);
          newData = [...decimatedOld, ...recentData];
        }

        return newData;
      });

      // Update text encoder grad norm data
      setGradNormTEData((prevData) => {
        let newData = sinceStep !== undefined ? [...prevData, ...(data.grad_norm_text_encoder || [])] : (data.grad_norm_text_encoder || []);

        // Decimate if too many points
        if (newData.length > MAX_POINTS) {
          const keepRecent = Math.floor(MAX_POINTS * 0.3);
          const decimateOld = newData.length - keepRecent;
          const decimationFactor = Math.ceil(decimateOld / (MAX_POINTS - keepRecent));

          const decimatedOld = newData.slice(0, decimateOld).filter((_, i) => i % decimationFactor === 0);
          const recentData = newData.slice(decimateOld);
          newData = [...decimatedOld, ...recentData];
        }

        return newData;
      });

      // Update UNet grad norm data
      setGradNormUNetData((prevData) => {
        let newData = sinceStep !== undefined ? [...prevData, ...(data.grad_norm_unet || [])] : (data.grad_norm_unet || []);

        // Decimate if too many points
        if (newData.length > MAX_POINTS) {
          const keepRecent = Math.floor(MAX_POINTS * 0.3);
          const decimateOld = newData.length - keepRecent;
          const decimationFactor = Math.ceil(decimateOld / (MAX_POINTS - keepRecent));

          const decimatedOld = newData.slice(0, decimateOld).filter((_, i) => i % decimationFactor === 0);
          const recentData = newData.slice(decimateOld);
          newData = [...decimatedOld, ...recentData];
        }

        return newData;
      });
    } catch (err: any) {
      console.error("[GradNormChart] Error fetching metrics:", err);
      setError(err.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [runId]);

  // Initial fetch
  useEffect(() => {
    console.log(`[GradNormChart] Initial fetch for runId=${runId}`);
    fetchMetrics(false);
  }, [runId, fetchMetrics]);

  // Auto-refresh based on polling interval
  useEffect(() => {
    if (pollingInterval > 0) {
      console.log(`[GradNormChart] Starting auto-refresh (every ${pollingInterval}s)`);
      const interval = setInterval(() => {
        console.log(`[GradNormChart] Auto-refresh triggered (interval=${pollingInterval}s)`);
        fetchMetrics(true);
      }, pollingInterval * 1000);
      return () => {
        console.log(`[GradNormChart] Stopping auto-refresh`);
        clearInterval(interval);
      };
    }
  }, [pollingInterval, fetchMetrics]);

  // Real-time SSE update for training metrics (when training is running)
  useEffect(() => {
    if (!isRunning) {
      console.log(`[GradNormChart] Training not running, skipping SSE subscription`);
      return;
    }

    console.log(`[GradNormChart] Training is running, subscribing to SSE for runId=${runId}`);

    // Connect to SSE if not already connected
    wsClient.connect();

    // Subscribe to training metrics SSE messages
    const handleTrainingMetrics = (metrics: TrainingMetrics) => {
      console.log(`[GradNormChart] SSE metric received: run_id=${metrics.run_id}, step=${metrics.step}, grad_norm=${metrics.grad_norm?.toFixed(6)}`);

      // Only update if this metric is for the current run
      if (metrics.run_id !== runId) {
        console.log(`[GradNormChart] Ignoring metric for different run (expected ${runId}, got ${metrics.run_id})`);
        return;
      }

      console.log(`[GradNormChart] Applying real-time metric: step=${metrics.step}, grad_norm=${metrics.grad_norm?.toFixed(6)}`);

      // Add new metric point to gradNormData
      if (metrics.grad_norm !== undefined && metrics.grad_norm !== null) {
        const newGradNormPoint: MetricPoint = {
          step: metrics.step,
          value: metrics.grad_norm,
          wall_time: Date.now() / 1000 // Current timestamp in seconds
        };

        setGradNormData((prevData) => {
          // Check if this step already exists (UPSERT behavior)
          const existingIndex = prevData.findIndex((p) => p.step === metrics.step);
          if (existingIndex >= 0) {
            // Update existing point
            const newData = [...prevData];
            newData[existingIndex] = newGradNormPoint;
            return newData;
          } else {
            // Add new point
            return [...prevData, newGradNormPoint];
          }
        });
      }

      // Add grad_norm_text_encoder if available
      if (metrics.grad_norm_text_encoder !== undefined && metrics.grad_norm_text_encoder !== null) {
        const newGradNormTEPoint: MetricPoint = {
          step: metrics.step,
          value: metrics.grad_norm_text_encoder,
          wall_time: Date.now() / 1000
        };

        setGradNormTEData((prevData) => {
          const existingIndex = prevData.findIndex((p) => p.step === metrics.step);
          if (existingIndex >= 0) {
            const newData = [...prevData];
            newData[existingIndex] = newGradNormTEPoint;
            return newData;
          } else {
            return [...prevData, newGradNormTEPoint];
          }
        });
      }

      // Add grad_norm_unet if available
      if (metrics.grad_norm_unet !== undefined && metrics.grad_norm_unet !== null) {
        const newGradNormUNetPoint: MetricPoint = {
          step: metrics.step,
          value: metrics.grad_norm_unet,
          wall_time: Date.now() / 1000
        };

        setGradNormUNetData((prevData) => {
          const existingIndex = prevData.findIndex((p) => p.step === metrics.step);
          if (existingIndex >= 0) {
            const newData = [...prevData];
            newData[existingIndex] = newGradNormUNetPoint;
            return newData;
          } else {
            return [...prevData, newGradNormUNetPoint];
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
    console.log(`[GradNormChart] Subscribed to SSE training metrics`);

    return () => {
      console.log(`[GradNormChart] Unsubscribing from SSE training metrics`);
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
  }, [gradNormData.length]); // Update when data changes

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500 text-red-400 rounded p-3 text-sm">
        {error}
      </div>
    );
  }

  if (loading && gradNormData.length === 0) {
    return (
      <div className="text-gray-400 text-sm">Loading metrics...</div>
    );
  }

  if (gradNormData.length === 0) {
    return (
      <div className="text-gray-400 text-sm">No gradient norm data available yet</div>
    );
  }

  // Calculate chart dimensions and scaling
  const height = 300;
  const padding = { top: 20, right: 200, bottom: 40, left: 70 }; // right: 200 for tooltip space, left: 70 for scientific notation
  const chartWidth = svgWidth - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const maxStep = Math.max(...gradNormData.map((d) => d.step));
  const minStep = Math.min(...gradNormData.map((d) => d.step));

  // Calculate min/max considering all grad norms
  const allValues = [
    ...(showTotal ? gradNormData.map((d) => d.value) : []),
    ...(showTextEncoder ? gradNormTEData.map((d) => d.value) : []),
    ...(showUNet ? gradNormUNetData.map((d) => d.value) : [])
  ];
  const maxGradNorm = allValues.length > 0 ? Math.max(...allValues) : 1;
  const minGradNorm = allValues.length > 0 ? Math.min(...allValues) : 0;

  const scaleX = (step: number) =>
    padding.left + ((step - minStep) / (maxStep - minStep || 1)) * chartWidth;

  const scaleY = (gradNorm: number) =>
    padding.top + chartHeight - ((gradNorm - minGradNorm) / (maxGradNorm - minGradNorm || 1)) * chartHeight;

  // Generate paths for line charts
  const rawTotalPath = gradNormData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  const smoothTotalPath = smoothGradNormData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  // Generate Text Encoder paths
  const rawTEPath = gradNormTEData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  const smoothTEPath = smoothGradNormTEData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  // Generate UNet paths
  const rawUNetPath = gradNormUNetData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  const smoothUNetPath = smoothGradNormUNetData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  // Generate Y-axis ticks
  const yTicks = 5;
  const yTickValues = Array.from({ length: yTicks }, (_, i) =>
    minGradNorm + ((maxGradNorm - minGradNorm) / (yTicks - 1)) * i
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
    let minDistance = Math.abs(gradNormData[0].step - hoveredStep);

    for (let i = 1; i < gradNormData.length; i++) {
      const distance = Math.abs(gradNormData[i].step - hoveredStep);
      if (distance < minDistance) {
        minDistance = distance;
        closestIndex = i;
      }
    }

    const closestPoint = gradNormData[closestIndex];
    const closestSmooth = smoothGradNormData[closestIndex];

    // Find other grad norms for the same step
    const step = closestSmooth.step;
    const closestTEPoint = gradNormTEData.find(d => d.step === step);
    const closestSmoothTE = smoothGradNormTEData.find(d => d.step === step);
    const closestUNetPoint = gradNormUNetData.find(d => d.step === step);
    const closestSmoothUNet = smoothGradNormUNetData.find(d => d.step === step);

    const pointX = scaleX(closestSmooth.step);
    const pointY = scaleY(closestSmooth.value);  // Use smooth total grad norm Y position

    setTooltip({
      x: pointX,
      y: pointY,
      step: closestSmooth.step,
      total: closestPoint.value,
      smoothTotal: closestSmooth.value,
      textEncoder: closestTEPoint?.value,
      smoothTextEncoder: closestSmoothTE?.value,
      unet: closestUNetPoint?.value,
      smoothUNet: closestSmoothUNet?.value
    });
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  return (
    <div className="bg-gray-800 border border-gray-700 rounded p-4">
      {/* Header with controls */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Gradient Norm</h3>

        <div className="flex items-center gap-3">
          {/* Polling interval selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400">Auto-refresh:</label>
            <select
              value={pollingInterval}
              onChange={(e) => {
                const newInterval = Number(e.target.value);
                console.log(`[GradNormChart] Auto-refresh interval changed: ${pollingInterval}s → ${newInterval}s`);
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
              console.log("[GradNormChart] Manual refresh button clicked");
              fetchMetrics(true);
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
            checked={showTotal}
            onChange={(e) => setShowTotal(e.target.checked)}
            className="w-4 h-4"
          />
          <span>Total</span>
        </label>
        {gradNormTEData.length > 0 && (
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showTextEncoder}
              onChange={(e) => setShowTextEncoder(e.target.checked)}
              className="w-4 h-4"
            />
            <span>Text Encoder</span>
          </label>
        )}
        {gradNormUNetData.length > 0 && (
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showUNet}
              onChange={(e) => setShowUNet(e.target.checked)}
              className="w-4 h-4"
            />
            <span>U-Net/Transformer</span>
          </label>
        )}
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
                {value.toExponential(1)}
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
          x={padding.left - 55}
          y={height / 2}
          textAnchor="middle"
          fill="currentColor"
          fontSize="12"
          transform={`rotate(-90, ${padding.left - 55}, ${height / 2})`}
        >
          Gradient Norm
        </text>

        {/* Total Grad Norm lines */}
        {showTotal && (
          <>
            {/* Raw total line (behind) */}
            <path
              d={rawTotalPath}
              fill="none"
              stroke="#8b5cf6"
              strokeWidth="1.5"
              strokeLinejoin="round"
              opacity="0.3"
            />

            {/* Smooth total line */}
            {smoothingFactor > 0 && (
              <path
                d={smoothTotalPath}
                fill="none"
                stroke="#a78bfa"
                strokeWidth="2.5"
                strokeLinejoin="round"
                opacity="0.9"
              />
            )}
          </>
        )}

        {/* Text Encoder Grad Norm lines */}
        {showTextEncoder && gradNormTEData.length > 0 && (
          <>
            {/* Raw TE line (behind) */}
            <path
              d={rawTEPath}
              fill="none"
              stroke="#10b981"
              strokeWidth="1.5"
              strokeLinejoin="round"
              opacity="0.3"
            />

            {/* Smooth TE line */}
            {smoothingFactor > 0 && (
              <path
                d={smoothTEPath}
                fill="none"
                stroke="#34d399"
                strokeWidth="2.5"
                strokeLinejoin="round"
                opacity="0.9"
              />
            )}
          </>
        )}

        {/* UNet Grad Norm lines */}
        {showUNet && gradNormUNetData.length > 0 && (
          <>
            {/* Raw UNet line (behind) */}
            <path
              d={rawUNetPath}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="1.5"
              strokeLinejoin="round"
              opacity="0.3"
            />

            {/* Smooth UNet line */}
            {smoothingFactor > 0 && (
              <path
                d={smoothUNetPath}
                fill="none"
                stroke="#fbbf24"
                strokeWidth="2.5"
                strokeLinejoin="round"
                opacity="0.9"
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
              fill="#8b5cf6"
              stroke="#fff"
              strokeWidth="2"
            />

            {/* Tooltip box - always on right side */}
            <g>
              {/* Background */}
              <rect
                x={tooltip.x + 10}
                y={tooltip.y - 65}
                width="180"
                height={120}
                fill="#1f2937"
                stroke="#4b5563"
                strokeWidth="1"
                rx="4"
              />

              {/* Text content */}
              <text
                x={tooltip.x + 15}
                y={tooltip.y - 50}
                fill="#e5e7eb"
                fontSize="11"
                fontFamily="monospace"
              >
                Step: {tooltip.step}
              </text>
              {showTotal && (
                <>
                  <text
                    x={tooltip.x + 15}
                    y={tooltip.y - 35}
                    fill="#8b5cf6"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    Total: {tooltip.total.toExponential(3)}
                  </text>
                  <text
                    x={tooltip.x + 15}
                    y={tooltip.y - 20}
                    fill="#a78bfa"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    Smooth: {tooltip.smoothTotal.toExponential(3)}
                  </text>
                </>
              )}
              {showTextEncoder && tooltip.textEncoder !== undefined && (
                <text
                  x={tooltip.x + 15}
                  y={tooltip.y - 5}
                  fill="#34d399"
                  fontSize="11"
                  fontFamily="monospace"
                >
                  Text Enc: {tooltip.textEncoder.toExponential(3)}
                </text>
              )}
              {showUNet && tooltip.unet !== undefined && (
                <text
                  x={tooltip.x + 15}
                  y={tooltip.y + 10}
                  fill="#fbbf24"
                  fontSize="11"
                  fontFamily="monospace"
                >
                  UNet: {tooltip.unet.toExponential(3)}
                </text>
              )}
            </g>
          </g>
        )}
      </svg>

      {/* Legend and stats */}
      <div className="mt-3 text-xs text-gray-500">
        <div className="flex items-center gap-6 mb-2 flex-wrap">
          {showTotal && (
            <div className="flex items-center gap-3">
              <span className="text-gray-400">Total:</span>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-violet-500 opacity-30"></div>
                <span>Raw</span>
              </div>
              {smoothingFactor > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-violet-400"></div>
                  <span>Smooth</span>
                </div>
              )}
            </div>
          )}
          {showTextEncoder && gradNormTEData.length > 0 && (
            <div className="flex items-center gap-3">
              <span className="text-gray-400">Text Encoder:</span>
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
          {showUNet && gradNormUNetData.length > 0 && (
            <div className="flex items-center gap-3">
              <span className="text-gray-400">U-Net/Transformer:</span>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-amber-500 opacity-30"></div>
                <span>Raw</span>
              </div>
              {smoothingFactor > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-amber-400"></div>
                  <span>Smooth</span>
                </div>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-4 text-gray-400 flex-wrap">
          {showTotal && smoothGradNormData.length > 0 && (
            <span>
              Latest Total: {smoothGradNormData[smoothGradNormData.length - 1]?.value.toExponential(3)}
            </span>
          )}
          {showTextEncoder && smoothGradNormTEData.length > 0 && (
            <span>
              Latest TE: {smoothGradNormTEData[smoothGradNormTEData.length - 1]?.value.toExponential(3)}
            </span>
          )}
          {showUNet && smoothGradNormUNetData.length > 0 && (
            <span>
              Latest UNet: {smoothGradNormUNetData[smoothGradNormUNetData.length - 1]?.value.toExponential(3)}
            </span>
          )}
          <span className="ml-auto">
            Step {smoothGradNormData[smoothGradNormData.length - 1]?.step || 0} / {gradNormData.length} points
          </span>
        </div>
      </div>
    </div>
  );
}
