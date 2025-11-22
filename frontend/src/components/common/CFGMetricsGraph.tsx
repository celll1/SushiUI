"use client";

import { useEffect, useRef, useState } from "react";
import { CFGMetrics } from "@/utils/websocket";

interface CFGMetricsGraphProps {
  metrics: CFGMetrics[];
  className?: string;
}

type MetricType = "relative_diff" | "cosine_similarity" | "snr" | "norms";

export default function CFGMetricsGraph({ metrics, className = "" }: CFGMetricsGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<CFGMetrics | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<MetricType>("relative_diff");

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || metrics.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size with devicePixelRatio for crisp rendering
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 30, right: 60, bottom: 60, left: 70 };
    const graphWidth = width - padding.left - padding.right;
    const graphHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, width, height);

    // Determine x-axis (prefer sigma, fallback to timestep)
    const useTimestep = metrics[0].sigma === undefined;
    const xValues = metrics.map(m => useTimestep ? m.timestep : m.sigma!);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);

    // Determine y-axis range based on selected metric
    let yMin = 0;
    let yMax = 1;
    let yLabel = "";
    let lineColor = "#10b981";
    let dataValues: number[] = [];

    switch (selectedMetric) {
      case "relative_diff":
        dataValues = metrics.map(m => m.relative_diff);
        yMin = 0;
        yMax = Math.max(...dataValues) * 1.1;
        yLabel = "Relative Diff";
        lineColor = "#10b981"; // green
        break;
      case "cosine_similarity":
        dataValues = metrics.map(m => m.cosine_similarity);
        yMin = -1;
        yMax = 1;
        yLabel = "Cosine Similarity";
        lineColor = "#f59e0b"; // amber
        break;
      case "snr":
        dataValues = metrics.map(m => m.snr);
        yMin = 0;
        yMax = Math.max(...dataValues) * 1.1;
        yLabel = "SNR";
        lineColor = "#a855f7"; // purple
        break;
      case "norms":
        // For norms, we'll show diff_norm
        dataValues = metrics.map(m => m.diff_norm);
        yMin = 0;
        yMax = Math.max(...dataValues) * 1.1;
        yLabel = "Diff Norm";
        lineColor = "#ef4444"; // red
        break;
    }

    // Helper functions
    const xScale = (val: number) => padding.left + ((val - xMin) / (xMax - xMin)) * graphWidth;
    const yScale = (val: number) => height - padding.bottom - ((val - yMin) / (yMax - yMin)) * graphHeight;

    // Draw grid lines
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;

    // Horizontal grid lines (y-axis)
    const ySteps = 5;
    for (let i = 0; i <= ySteps; i++) {
      const y = yMin + (yMax - yMin) * (i / ySteps);
      const yPos = yScale(y);

      ctx.beginPath();
      ctx.moveTo(padding.left, yPos);
      ctx.lineTo(width - padding.right, yPos);
      ctx.stroke();

      // Y-axis labels
      ctx.fillStyle = lineColor;
      ctx.font = "11px sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(y.toFixed(4), padding.left - 5, yPos);
    }

    // Vertical grid lines (x-axis)
    const xSteps = Math.min(metrics.length - 1, 10);
    for (let i = 0; i <= xSteps; i++) {
      const x = xMin + (xMax - xMin) * (i / xSteps);
      const xPos = xScale(x);

      ctx.beginPath();
      ctx.moveTo(xPos, padding.top);
      ctx.lineTo(xPos, height - padding.bottom);
      ctx.stroke();

      // X-axis labels
      ctx.fillStyle = "#888";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      // Show more decimal places for sigma
      const xLabel = useTimestep ? x.toFixed(0) : x.toFixed(2);
      ctx.fillText(xLabel, xPos, height - padding.bottom + 8);
    }

    // Draw axis labels
    ctx.fillStyle = "#aaa";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(useTimestep ? "Timestep" : "Sigma", width / 2, height - 15);

    // Y-axis label
    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = lineColor;
    ctx.font = "13px sans-serif";
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // Draw line
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 3;
    ctx.beginPath();

    dataValues.forEach((val, i) => {
      const xVal = xValues[i];
      const x = xScale(xVal);
      const y = yScale(val);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

  }, [metrics, selectedMetric]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || metrics.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
    const graphWidth = rect.width - padding.left - padding.right;

    // Find closest point
    const useTimestep = metrics[0].sigma === undefined;
    const xValues = metrics.map(m => useTimestep ? m.timestep : m.sigma!);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);

    const xScale = (val: number) => padding.left + ((val - xMin) / (xMax - xMin)) * graphWidth;

    let closestIndex = 0;
    let closestDist = Infinity;

    metrics.forEach((m, i) => {
      const xPos = xScale(xValues[i]);
      const dist = Math.abs(xPos - x);
      if (dist < closestDist) {
        closestDist = dist;
        closestIndex = i;
      }
    });

    if (closestDist < 30) {
      setHoveredPoint(metrics[closestIndex]);
    } else {
      setHoveredPoint(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredPoint(null);
  };

  return (
    <div className={`relative ${className}`}>
      {/* Metric selector dropdown */}
      <div className="mb-2">
        <select
          value={selectedMetric}
          onChange={(e) => setSelectedMetric(e.target.value as MetricType)}
          className="bg-gray-800 border border-gray-600 rounded px-3 py-1 text-sm text-white focus:outline-none focus:border-blue-500"
        >
          <option value="relative_diff">Relative Diff (CFG strength)</option>
          <option value="cosine_similarity">Cosine Similarity (direction alignment)</option>
          <option value="snr">SNR (signal-to-noise ratio)</option>
          <option value="norms">Diff Norm (||yp - yn||)</option>
        </select>
      </div>

      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ width: "100%", height: "300px" }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      {hoveredPoint && (
        <div
          className="absolute bg-gray-800 border border-gray-600 rounded p-2 text-xs pointer-events-none"
          style={{
            bottom: "50px",
            right: "20px",
          }}
        >
          <div className="grid grid-cols-2 gap-x-3 gap-y-1">
            <span className="text-gray-400">Step:</span>
            <span className="text-white">{hoveredPoint.step}</span>

            {hoveredPoint.sigma !== undefined && (
              <>
                <span className="text-gray-400">Sigma:</span>
                <span className="text-white">{hoveredPoint.sigma.toFixed(3)}</span>
              </>
            )}

            <span className="text-gray-400">Timestep:</span>
            <span className="text-white">{hoveredPoint.timestep}</span>

            <span className="text-green-400">Rel. Diff:</span>
            <span className="text-white">{hoveredPoint.relative_diff.toFixed(4)}</span>

            <span className="text-amber-400">Cos Sim:</span>
            <span className="text-white">{hoveredPoint.cosine_similarity.toFixed(4)}</span>

            <span className="text-purple-400">SNR:</span>
            <span className="text-white">{hoveredPoint.snr.toFixed(4)}</span>

            <span className="text-gray-400">CFG:</span>
            <span className="text-white">{hoveredPoint.guidance_scale}</span>
          </div>
        </div>
      )}
    </div>
  );
}
