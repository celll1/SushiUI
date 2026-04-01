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

// Calculate robust Y-axis range using percentiles (excludes outliers)
const calculateRobustYRange = (
  values: number[],
  lowerPercentile: number = 1,
  upperPercentile: number = 99
): { min: number; max: number } => {
  if (values.length === 0) return { min: 0, max: 1 };

  // Filter out invalid values (NaN, Infinity)
  const validValues = values.filter(v => isFinite(v) && !isNaN(v));
  if (validValues.length === 0) return { min: 0, max: 1 };

  // Sort values
  const sorted = [...validValues].sort((a, b) => a - b);

  // Calculate percentile indices
  const lowerIndex = Math.floor(sorted.length * (lowerPercentile / 100));
  const upperIndex = Math.ceil(sorted.length * (upperPercentile / 100)) - 1;

  const pMin = sorted[Math.max(0, lowerIndex)];
  const pMax = sorted[Math.min(sorted.length - 1, upperIndex)];

  // Add 5% padding
  const range = pMax - pMin;
  const padding = range * 0.05;

  return {
    min: Math.max(0, pMin - padding), // Don't go below 0 for grad norm values
    max: pMax + padding
  };
};

interface GradNormChartProps {
  runId: number;
  isRunning: boolean;
}

export default function GradNormChart({ runId, isRunning }: GradNormChartProps) {
  const [gradNormData, setGradNormData] = useState<MetricPoint[]>([]);
  const [gradNormTEData, setGradNormTEData] = useState<MetricPoint[]>([]);
  const [gradNormTE1Data, setGradNormTE1Data] = useState<MetricPoint[]>([]);
  const [gradNormTE2Data, setGradNormTE2Data] = useState<MetricPoint[]>([]);
  const [gradNormUNetData, setGradNormUNetData] = useState<MetricPoint[]>([]);
  const [gradNormVEData, setGradNormVEData] = useState<MetricPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastStep, setLastStep] = useState<number>(-1);

  // UI controls
  const [smoothingFactor, setSmoothingFactor] = useState(0.9);
  const [pollingInterval, setPollingInterval] = useState<number>(0); // 0 = off
  const [showTotal, setShowTotal] = useState(true);
  const [showTextEncoder, setShowTextEncoder] = useState(true);
  const [showTE1, setShowTE1] = useState(true);
  const [showTE2, setShowTE2] = useState(true);
  const [showUNet, setShowUNet] = useState(true);
  const [showVisionEncoder, setShowVisionEncoder] = useState(true);
  const [yAxisMode, setYAxisMode] = useState<"auto" | "custom">("auto");
  const [customYMin, setCustomYMin] = useState<number>(0);
  const [customYMax, setCustomYMax] = useState<number>(1);
  const [xAxisMode, setXAxisMode] = useState<"auto" | "custom">("auto");
  const [customXMin, setCustomXMin] = useState<number>(0);
  const [customXMax, setCustomXMax] = useState<number>(1000);

  // Tooltip state
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    step: number;
    total: number;
    smoothTotal: number;
    textEncoder?: number;
    smoothTextEncoder?: number;
    te1?: number;
    smoothTE1?: number;
    te2?: number;
    smoothTE2?: number;
    unet?: number;
    smoothUNet?: number;
    visionEncoder?: number;
    smoothVisionEncoder?: number;
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

  const smoothGradNormTE1Data = useMemo(() => {
    return calculateSmoothing(gradNormTE1Data, smoothingFactor);
  }, [gradNormTE1Data, smoothingFactor]);

  const smoothGradNormTE2Data = useMemo(() => {
    return calculateSmoothing(gradNormTE2Data, smoothingFactor);
  }, [gradNormTE2Data, smoothingFactor]);

  const smoothGradNormUNetData = useMemo(() => {
    return calculateSmoothing(gradNormUNetData, smoothingFactor);
  }, [gradNormUNetData, smoothingFactor]);

  const smoothGradNormVEData = useMemo(() => {
    return calculateSmoothing(gradNormVEData, smoothingFactor);
  }, [gradNormVEData, smoothingFactor]);

  const fetchMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch all metrics with uniform sampling (backend handles sampling)
      const data = await getTrainingMetrics(runId);

      console.log(`[GradNormChart] Fetched metrics: ${data.grad_norm.length} grad_norm points (uniform sampling)`);

      // Replace data entirely (no merging needed - backend does uniform sampling)
      setGradNormData(data.grad_norm);
      setGradNormTEData(data.grad_norm_text_encoder || []);
      setGradNormTE1Data(data.grad_norm_text_encoder_1 ?? []);
      setGradNormTE2Data(data.grad_norm_text_encoder_2 ?? []);
      setGradNormUNetData(data.grad_norm_unet || []);
      setGradNormVEData(data.grad_norm_vision_encoder || []);

      // Update lastStep for display
      if (data.grad_norm.length > 0) {
        const maxStep = Math.max(...data.grad_norm.map((d) => d.step));
        setLastStep(maxStep);
      }
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
    fetchMetrics();
  }, [runId, fetchMetrics]);

  // Auto-refresh based on polling interval
  useEffect(() => {
    if (pollingInterval > 0) {
      console.log(`[GradNormChart] Starting auto-refresh (every ${pollingInterval}s)`);
      const interval = setInterval(() => {
        console.log(`[GradNormChart] Auto-refresh triggered (interval=${pollingInterval}s)`);
        fetchMetrics();
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

      // Add grad_norm_text_encoder_1 if available (SDXL TE1)
      if (metrics.grad_norm_text_encoder_1 !== undefined && metrics.grad_norm_text_encoder_1 !== null) {
        const newPoint: MetricPoint = {
          step: metrics.step,
          value: metrics.grad_norm_text_encoder_1,
          wall_time: Date.now() / 1000
        };
        setGradNormTE1Data((prevData) => {
          const existingIndex = prevData.findIndex((p) => p.step === metrics.step);
          if (existingIndex >= 0) {
            const newData = [...prevData];
            newData[existingIndex] = newPoint;
            return newData;
          } else {
            return [...prevData, newPoint];
          }
        });
      }

      // Add grad_norm_text_encoder_2 if available (SDXL TE2)
      if (metrics.grad_norm_text_encoder_2 !== undefined && metrics.grad_norm_text_encoder_2 !== null) {
        const newPoint: MetricPoint = {
          step: metrics.step,
          value: metrics.grad_norm_text_encoder_2,
          wall_time: Date.now() / 1000
        };
        setGradNormTE2Data((prevData) => {
          const existingIndex = prevData.findIndex((p) => p.step === metrics.step);
          if (existingIndex >= 0) {
            const newData = [...prevData];
            newData[existingIndex] = newPoint;
            return newData;
          } else {
            return [...prevData, newPoint];
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

      // Add grad_norm_vision_encoder if available
      if (metrics.grad_norm_vision_encoder !== undefined && metrics.grad_norm_vision_encoder !== null) {
        const newGradNormVEPoint: MetricPoint = {
          step: metrics.step,
          value: metrics.grad_norm_vision_encoder,
          wall_time: Date.now() / 1000
        };

        setGradNormVEData((prevData) => {
          const existingIndex = prevData.findIndex((p) => p.step === metrics.step);
          if (existingIndex >= 0) {
            const newData = [...prevData];
            newData[existingIndex] = newGradNormVEPoint;
            return newData;
          } else {
            return [...prevData, newGradNormVEPoint];
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
  // Responsive padding: reduce right padding on narrow screens for better chart visibility
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 640;
  const padding = {
    top: 20,
    right: isMobile ? 10 : 200, // Minimal padding on mobile, tooltip space on desktop
    bottom: 40,
    left: isMobile ? 50 : 70 // Slightly less left padding on mobile
  };
  const chartWidth = svgWidth - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const autoMaxStep = Math.max(...gradNormData.map((d) => d.step));
  const autoMinStep = Math.min(...gradNormData.map((d) => d.step));

  // Calculate min/max considering all grad norms using percentile-based range
  // This excludes outliers for more readable charts
  const allValues = [
    ...(showTotal ? gradNormData.map((d) => d.value) : []),
    ...(showTextEncoder ? gradNormTEData.map((d) => d.value) : []),
    ...(showTE1 ? gradNormTE1Data.map((d) => d.value) : []),
    ...(showTE2 ? gradNormTE2Data.map((d) => d.value) : []),
    ...(showUNet ? gradNormUNetData.map((d) => d.value) : []),
    ...(showVisionEncoder ? gradNormVEData.map((d) => d.value) : []),
  ];
  const { min: autoMinGradNorm, max: autoMaxGradNorm } = calculateRobustYRange(allValues);

  // Use custom scale if enabled, otherwise use auto scale
  const maxStep = xAxisMode === "custom" ? customXMax : autoMaxStep;
  const minStep = xAxisMode === "custom" ? customXMin : autoMinStep;
  const maxGradNorm = yAxisMode === "custom" ? customYMax : autoMaxGradNorm;
  const minGradNorm = yAxisMode === "custom" ? customYMin : autoMinGradNorm;

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

  // Generate Vision Encoder paths
  const rawVEPath = gradNormVEData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  const smoothVEPath = smoothGradNormVEData
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  // Generate TE1 paths (SDXL TE1: blue-300, dashed)
  const rawTE1Path = gradNormTE1Data
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  const smoothTE1Path = smoothGradNormTE1Data
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  // Generate TE2 paths (SDXL TE2: violet-300, dashed)
  const rawTE2Path = gradNormTE2Data
    .map((d, i) => {
      const x = scaleX(d.step);
      const y = scaleY(d.value);
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(" ");

  const smoothTE2Path = smoothGradNormTE2Data
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
    const closestTE1Point = gradNormTE1Data.find(d => d.step === step);
    const closestSmoothTE1 = smoothGradNormTE1Data.find(d => d.step === step);
    const closestTE2Point = gradNormTE2Data.find(d => d.step === step);
    const closestSmoothTE2 = smoothGradNormTE2Data.find(d => d.step === step);
    const closestUNetPoint = gradNormUNetData.find(d => d.step === step);
    const closestSmoothUNet = smoothGradNormUNetData.find(d => d.step === step);
    const closestVEPoint = gradNormVEData.find(d => d.step === step);
    const closestSmoothVE = smoothGradNormVEData.find(d => d.step === step);

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
      te1: closestTE1Point?.value,
      smoothTE1: closestSmoothTE1?.value,
      te2: closestTE2Point?.value,
      smoothTE2: closestSmoothTE2?.value,
      unet: closestUNetPoint?.value,
      smoothUNet: closestSmoothUNet?.value,
      visionEncoder: closestVEPoint?.value,
      smoothVisionEncoder: closestSmoothVE?.value,
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
        {gradNormTE1Data.length > 0 && (
          <label className="flex items-center gap-2 text-xs cursor-pointer" style={{color: "#93c5fd"}}>
            <input
              type="checkbox"
              checked={showTE1}
              onChange={(e) => setShowTE1(e.target.checked)}
              className="w-4 h-4"
            />
            <span>TE1</span>
          </label>
        )}
        {gradNormTE2Data.length > 0 && (
          <label className="flex items-center gap-2 text-xs cursor-pointer" style={{color: "#c4b5fd"}}>
            <input
              type="checkbox"
              checked={showTE2}
              onChange={(e) => setShowTE2(e.target.checked)}
              className="w-4 h-4"
            />
            <span>TE2</span>
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
        {gradNormVEData.length > 0 && (
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showVisionEncoder}
              onChange={(e) => setShowVisionEncoder(e.target.checked)}
              className="w-4 h-4"
            />
            <span>Vision Encoder</span>
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
                    setCustomYMin(autoMinGradNorm);
                    setCustomYMax(autoMaxGradNorm);
                  }
                }}
                className="w-4 h-4"
              />
              <span>Y-axis</span>
            </label>
            {yAxisMode === "custom" && (
              <button
                onClick={() => {
                  setCustomYMin(autoMinGradNorm);
                  setCustomYMax(autoMaxGradNorm);
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
                    left: `${(customYMin / (autoMaxGradNorm * 1.1)) * 100}%`,
                    right: `${100 - (customYMax / (autoMaxGradNorm * 1.1)) * 100}%`,
                  }}
                />
                {/* Min slider */}
                <input
                  type="range"
                  min={0}
                  max={autoMaxGradNorm * 1.1}
                  step={autoMaxGradNorm / 1000}
                  value={customYMin}
                  onChange={(e) => {
                    const newMin = parseFloat(e.target.value);
                    if (newMin < customYMax) {
                      setCustomYMin(newMin);
                    }
                  }}
                  className="absolute top-1/2 -translate-y-1/2 w-full h-1 appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-blue-500 [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-blue-500"
                  style={{ zIndex: customYMin > (autoMaxGradNorm * 1.1) - ((autoMaxGradNorm * 1.1) / 2) ? 5 : 3 }}
                />
                {/* Max slider */}
                <input
                  type="range"
                  min={0}
                  max={autoMaxGradNorm * 1.1}
                  step={autoMaxGradNorm / 1000}
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
          <clipPath id="chart-clip-gradnorm">
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
              clipPath="url(#chart-clip-gradnorm)"
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
                clipPath="url(#chart-clip-gradnorm)"
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
              clipPath="url(#chart-clip-gradnorm)"
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
                clipPath="url(#chart-clip-gradnorm)"
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
              clipPath="url(#chart-clip-gradnorm)"
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
                clipPath="url(#chart-clip-gradnorm)"
              />
            )}
          </>
        )}

        {/* Vision Encoder Grad Norm lines */}
        {showVisionEncoder && gradNormVEData.length > 0 && (
          <>
            {/* Raw VE line (behind) */}
            <path
              d={rawVEPath}
              fill="none"
              stroke="#a78bfa"
              strokeWidth="1.5"
              strokeLinejoin="round"
              opacity="0.3"
              clipPath="url(#chart-clip-gradnorm)"
            />

            {/* Smooth VE line */}
            {smoothingFactor > 0 && (
              <path
                d={smoothVEPath}
                fill="none"
                stroke="#c4b5fd"
                strokeWidth="2.5"
                strokeLinejoin="round"
                opacity="0.9"
                clipPath="url(#chart-clip-gradnorm)"
              />
            )}
          </>
        )}

        {/* TE1 Grad Norm lines (SDXL TE1: blue-300, dashed) */}
        {showTE1 && gradNormTE1Data.length > 0 && (
          <>
            <path
              d={rawTE1Path}
              fill="none"
              stroke="#93c5fd"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeDasharray="4 2"
              opacity="0.3"
              clipPath="url(#chart-clip-gradnorm)"
            />
            {smoothingFactor > 0 && (
              <path
                d={smoothTE1Path}
                fill="none"
                stroke="#93c5fd"
                strokeWidth="2"
                strokeLinejoin="round"
                strokeDasharray="6 2"
                opacity="0.9"
                clipPath="url(#chart-clip-gradnorm)"
              />
            )}
          </>
        )}

        {/* TE2 Grad Norm lines (SDXL TE2: violet-300, dashed) */}
        {showTE2 && gradNormTE2Data.length > 0 && (
          <>
            <path
              d={rawTE2Path}
              fill="none"
              stroke="#c4b5fd"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeDasharray="4 2"
              opacity="0.3"
              clipPath="url(#chart-clip-gradnorm)"
            />
            {smoothingFactor > 0 && (
              <path
                d={smoothTE2Path}
                fill="none"
                stroke="#c4b5fd"
                strokeWidth="2"
                strokeLinejoin="round"
                strokeDasharray="6 2"
                opacity="0.9"
                clipPath="url(#chart-clip-gradnorm)"
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

            {/* Tooltip box - responsive positioning */}
            {(() => {
              const tooltipWidth = 160;
              const lineSpacing = 15;

              // Build lines dynamically so Y positions are always sequential
              type TooltipLine = { color: string; text: string };
              const lines: TooltipLine[] = [];
              lines.push({ color: "#e5e7eb", text: `Step: ${tooltip.step}` });
              if (showTotal) {
                lines.push({ color: "#8b5cf6", text: `Total: ${tooltip.total.toExponential(3)}` });
                lines.push({ color: "#a78bfa", text: `Smooth: ${tooltip.smoothTotal.toExponential(3)}` });
              }
              if (showTextEncoder && tooltip.textEncoder !== undefined)
                lines.push({ color: "#34d399", text: `Text Enc: ${tooltip.textEncoder.toExponential(3)}` });
              if (showTE1 && tooltip.te1 !== undefined)
                lines.push({ color: "#93c5fd", text: `TE1: ${tooltip.te1.toExponential(3)}` });
              if (showTE2 && tooltip.te2 !== undefined)
                lines.push({ color: "#c4b5fd", text: `TE2: ${tooltip.te2.toExponential(3)}` });
              if (showUNet && tooltip.unet !== undefined)
                lines.push({ color: "#fbbf24", text: `UNet: ${tooltip.unet.toExponential(3)}` });
              if (showVisionEncoder && tooltip.visionEncoder !== undefined)
                lines.push({ color: "#f9a8d4", text: `VE: ${tooltip.visionEncoder.toExponential(3)}` });

              const tooltipHeight = lines.length * lineSpacing + 10;
              const startY = tooltip.y - 40 + 15; // 15px top padding inside box

              // Show tooltip on left if it would overflow on right
              const showLeft = tooltip.x + tooltipWidth + 20 > svgWidth;
              const tooltipX = showLeft ? tooltip.x - tooltipWidth - 10 : tooltip.x + 10;
              const textX = showLeft ? tooltip.x - tooltipWidth - 5 : tooltip.x + 15;

              return (
                <g>
                  <rect
                    x={tooltipX}
                    y={tooltip.y - 40}
                    width={tooltipWidth}
                    height={tooltipHeight}
                    fill="rgba(31, 41, 55, 0.95)"
                    stroke="#4b5563"
                    strokeWidth="1"
                    rx="4"
                  />
                  {lines.map((line, i) => (
                    <text
                      key={i}
                      x={textX}
                      y={startY + i * lineSpacing}
                      fill={line.color}
                      fontSize="11"
                      fontFamily="monospace"
                    >
                      {line.text}
                    </text>
                  ))}
                </g>
              );
            })()}
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
          {showTE1 && gradNormTE1Data.length > 0 && (
            <div className="flex items-center gap-3">
              <span style={{color: "#93c5fd"}}>TE1:</span>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 opacity-30" style={{background: "#93c5fd", borderTop: "2px dashed #93c5fd"}}></div>
                <span className="text-gray-400">Raw</span>
              </div>
              {smoothingFactor > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5" style={{borderTop: "2px dashed #93c5fd"}}></div>
                  <span className="text-gray-400">Smooth</span>
                </div>
              )}
            </div>
          )}
          {showTE2 && gradNormTE2Data.length > 0 && (
            <div className="flex items-center gap-3">
              <span style={{color: "#c4b5fd"}}>TE2:</span>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 opacity-30" style={{borderTop: "2px dashed #c4b5fd"}}></div>
                <span className="text-gray-400">Raw</span>
              </div>
              {smoothingFactor > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5" style={{borderTop: "2px dashed #c4b5fd"}}></div>
                  <span className="text-gray-400">Smooth</span>
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
          {showVisionEncoder && gradNormVEData.length > 0 && (
            <div className="flex items-center gap-3">
              <span className="text-gray-400">Vision Encoder:</span>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-violet-500 opacity-30"></div>
                <span>Raw</span>
              </div>
              {smoothingFactor > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-violet-300"></div>
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
          {showTE1 && smoothGradNormTE1Data.length > 0 && (
            <span style={{color: "#93c5fd"}}>
              Latest TE1: {smoothGradNormTE1Data[smoothGradNormTE1Data.length - 1]?.value.toExponential(3)}
            </span>
          )}
          {showTE2 && smoothGradNormTE2Data.length > 0 && (
            <span style={{color: "#c4b5fd"}}>
              Latest TE2: {smoothGradNormTE2Data[smoothGradNormTE2Data.length - 1]?.value.toExponential(3)}
            </span>
          )}
          {showUNet && smoothGradNormUNetData.length > 0 && (
            <span>
              Latest UNet: {smoothGradNormUNetData[smoothGradNormUNetData.length - 1]?.value.toExponential(3)}
            </span>
          )}
          {showVisionEncoder && smoothGradNormVEData.length > 0 && (
            <span>
              Latest VE: {smoothGradNormVEData[smoothGradNormVEData.length - 1]?.value.toExponential(3)}
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
