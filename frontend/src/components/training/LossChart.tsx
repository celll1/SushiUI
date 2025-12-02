"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { RefreshCw } from "lucide-react";
import { getTrainingMetrics } from "@/utils/api";

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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastStep, setLastStep] = useState<number>(-1);

  // UI controls
  const [smoothingFactor, setSmoothingFactor] = useState(0.9);
  const [pollingInterval, setPollingInterval] = useState<number>(0); // 0 = off

  // Tooltip state
  const [tooltip, setTooltip] = useState<{ x: number; y: number; step: number; loss: number; smoothLoss: number } | null>(null);

  // Calculate smooth loss on client side
  const smoothLossData = useMemo(() => {
    return calculateSmoothing(lossData, smoothingFactor);
  }, [lossData, smoothingFactor]);

  const fetchMetrics = useCallback(async (isIncremental: boolean = false) => {
    try {
      setLoading(true);
      setError(null);

      const sinceStep = isIncremental && lastStep >= 0 ? lastStep : undefined;
      const data = await getTrainingMetrics(runId, sinceStep);

      // Merge new data with existing data
      setLossData((prevData) => {
        const newData = sinceStep !== undefined ? [...prevData, ...data.loss] : data.loss;

        // Update lastStep
        if (newData.length > 0) {
          const maxStep = Math.max(...newData.map((d) => d.step));
          setLastStep(maxStep);
        }

        return newData;
      });
    } catch (err: any) {
      console.error("Error fetching metrics:", err);
      setError(err.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [runId, lastStep]);

  // Initial fetch
  useEffect(() => {
    fetchMetrics(false);
  }, [runId]);

  // Auto-refresh based on polling interval
  useEffect(() => {
    if (pollingInterval > 0) {
      const interval = setInterval(() => fetchMetrics(true), pollingInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [pollingInterval, fetchMetrics]);

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
  const width = 100; // Use percentage width for responsiveness
  const height = 300;
  const padding = { top: 20, right: 20, bottom: 40, left: 60 };
  const actualWidth = 550; // Actual pixel width for calculations
  const chartWidth = actualWidth - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const maxStep = Math.max(...lossData.map((d) => d.step));
  const minStep = Math.min(...lossData.map((d) => d.step));
  const maxLoss = Math.max(...lossData.map((d) => d.value));
  const minLoss = Math.min(...lossData.map((d) => d.value));

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

    // Convert to viewBox coordinates
    const scaleFactorX = actualWidth / svgRect.width;
    const viewBoxX = mouseX * scaleFactorX;

    // Check if mouse is within chart area
    if (viewBoxX < padding.left || viewBoxX > actualWidth - padding.right) {
      setTooltip(null);
      return;
    }

    // Find nearest data point by step
    const hoveredStep = minStep + ((viewBoxX - padding.left) / chartWidth) * (maxStep - minStep);

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
    const pointX = scaleX(closestSmooth.step);
    const pointY = scaleY(closestSmooth.value);  // Use smooth loss Y position

    setTooltip({
      x: pointX,
      y: pointY,
      step: closestSmooth.step,
      loss: closestPoint.value,
      smoothLoss: closestSmooth.value
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
              onChange={(e) => setPollingInterval(Number(e.target.value))}
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
            onClick={() => fetchMetrics(true)}
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

      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${actualWidth} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="text-gray-400"
        style={{ fontFamily: "monospace", fontSize: "10px", maxWidth: "100%" }}
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
          x2={actualWidth - padding.right}
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
                x2={actualWidth - padding.right}
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
          x={actualWidth / 2}
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

        {/* Raw loss line (behind) */}
        <path
          d={rawLinePath}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="1.5"
          strokeLinejoin="round"
          opacity="0.3"
        />

        {/* Smooth loss line (always shown) */}
        {smoothingFactor > 0 && (
          <path
            d={smoothLinePath}
            fill="none"
            stroke="#60a5fa"
            strokeWidth="2.5"
            strokeLinejoin="round"
            opacity="0.9"
          />
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

            {/* Tooltip box */}
            <g>
              {/* Background */}
              <rect
                x={tooltip.x + 10}
                y={tooltip.y - 40}
                width="140"
                height="50"
                fill="#1f2937"
                stroke="#4b5563"
                strokeWidth="1"
                rx="4"
              />

              {/* Text content */}
              <text
                x={tooltip.x + 15}
                y={tooltip.y - 25}
                fill="#e5e7eb"
                fontSize="11"
                fontFamily="monospace"
              >
                Step: {tooltip.step}
              </text>
              <text
                x={tooltip.x + 15}
                y={tooltip.y - 10}
                fill="#3b82f6"
                fontSize="11"
                fontFamily="monospace"
              >
                Loss: {tooltip.loss.toFixed(4)}
              </text>
              <text
                x={tooltip.x + 15}
                y={tooltip.y + 5}
                fill="#60a5fa"
                fontSize="11"
                fontFamily="monospace"
              >
                Smooth: {tooltip.smoothLoss.toFixed(4)}
              </text>
            </g>
          </g>
        )}
      </svg>

      {/* Legend and stats */}
      <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-blue-500 opacity-30"></div>
            <span>Raw ({lossData.length} points)</span>
          </div>
          {smoothingFactor > 0 && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-blue-400"></div>
              <span>Smooth (EMA {(smoothingFactor * 100).toFixed(0)}%)</span>
            </div>
          )}
        </div>
        <span>
          Latest: {smoothLossData[smoothLossData.length - 1]?.value.toFixed(4)} (Step{" "}
          {smoothLossData[smoothLossData.length - 1]?.step})
        </span>
      </div>
    </div>
  );
}
