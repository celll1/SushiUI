"use client";

import { useEffect, useState, useCallback } from "react";
import { getTrainingMetrics } from "@/utils/api";

interface MetricPoint {
  step: number;
  value: number;
  wall_time: number;
}

interface LossChartProps {
  runId: number;
  isRunning: boolean;
}

export default function LossChart({ runId, isRunning }: LossChartProps) {
  const [lossData, setLossData] = useState<MetricPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastStep, setLastStep] = useState<number>(-1);

  const fetchMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const sinceStep = lastStep >= 0 ? lastStep : undefined;
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

  useEffect(() => {
    fetchMetrics();

    // Auto-refresh if training is running
    if (isRunning) {
      const interval = setInterval(fetchMetrics, 10000); // 10 seconds
      return () => clearInterval(interval);
    }
  }, [fetchMetrics, isRunning]);

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
  const width = 600;
  const height = 300;
  const padding = { top: 20, right: 20, bottom: 40, left: 60 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const maxStep = Math.max(...lossData.map((d) => d.step));
  const minStep = Math.min(...lossData.map((d) => d.step));
  const maxLoss = Math.max(...lossData.map((d) => d.value));
  const minLoss = Math.min(...lossData.map((d) => d.value));

  const scaleX = (step: number) =>
    padding.left + ((step - minStep) / (maxStep - minStep || 1)) * chartWidth;

  const scaleY = (loss: number) =>
    padding.top + chartHeight - ((loss - minLoss) / (maxLoss - minLoss || 1)) * chartHeight;

  // Generate path for line chart
  const linePath = lossData
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

  return (
    <div className="bg-gray-800 border border-gray-700 rounded p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Training Loss</h3>
        {isRunning && (
          <span className="text-xs text-gray-500">Auto-updating every 10s</span>
        )}
      </div>

      <svg
        width={width}
        height={height}
        className="text-gray-400"
        style={{ fontFamily: "monospace", fontSize: "10px" }}
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
          x2={width - padding.right}
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
                x2={width - padding.right}
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
          x={width / 2}
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

        {/* Loss line */}
        <path
          d={linePath}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
          strokeLinejoin="round"
        />

        {/* Data points */}
        {lossData.map((d, i) => (
          <circle
            key={i}
            cx={scaleX(d.step)}
            cy={scaleY(d.value)}
            r="2"
            fill="#3b82f6"
          />
        ))}
      </svg>

      <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
        <span>{lossData.length} data points</span>
        <span>
          Latest: {lossData[lossData.length - 1]?.value.toFixed(4)} (Step{" "}
          {lossData[lossData.length - 1]?.step})
        </span>
      </div>
    </div>
  );
}
