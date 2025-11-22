"use client";

import { useEffect, useRef, useState } from "react";
import { CFGMetrics } from "@/utils/websocket";

interface CFGMetricsGraphProps {
  metrics: CFGMetrics[];
  className?: string;
}

export default function CFGMetricsGraph({ metrics, className = "" }: CFGMetricsGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<CFGMetrics | null>(null);

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
    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
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

    // Determine y-axis range (use relative_diff and cosine_similarity)
    const allRelativeDiff = metrics.map(m => m.relative_diff);
    const allCosineSim = metrics.map(m => m.cosine_similarity);

    // For dual Y-axis: left = relative_diff (0 to max), right = cosine_similarity (-1 to 1)
    const yMinLeft = 0;
    const yMaxLeft = Math.max(...allRelativeDiff) * 1.1;
    const yMinRight = -1;
    const yMaxRight = 1;

    // Helper functions
    const xScale = (val: number) => padding.left + ((val - xMin) / (xMax - xMin)) * graphWidth;
    const yScaleLeft = (val: number) => height - padding.bottom - ((val - yMinLeft) / (yMaxLeft - yMinLeft)) * graphHeight;
    const yScaleRight = (val: number) => height - padding.bottom - ((val - yMinRight) / (yMaxRight - yMinRight)) * graphHeight;

    // Draw grid lines
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;

    // Horizontal grid lines (left y-axis - relative_diff)
    const ySteps = 5;
    for (let i = 0; i <= ySteps; i++) {
      const y = yMinLeft + (yMaxLeft - yMinLeft) * (i / ySteps);
      const yPos = yScaleLeft(y);

      ctx.beginPath();
      ctx.moveTo(padding.left, yPos);
      ctx.lineTo(width - padding.right, yPos);
      ctx.stroke();

      // Left Y-axis labels (relative_diff)
      ctx.fillStyle = "#10b981"; // green
      ctx.font = "11px sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(y.toFixed(3), padding.left - 5, yPos);
    }

    // Right Y-axis labels (cosine_similarity)
    for (let i = 0; i <= ySteps; i++) {
      const y = yMinRight + (yMaxRight - yMinRight) * (i / ySteps);
      const yPos = yScaleRight(y);

      ctx.fillStyle = "#f59e0b"; // amber
      ctx.font = "11px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(y.toFixed(2), width - padding.right + 5, yPos);
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
      ctx.fillText(x.toFixed(0), xPos, height - padding.bottom + 5);
    }

    // Draw axis labels
    ctx.fillStyle = "#aaa";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(useTimestep ? "Timestep" : "Sigma", width / 2, height - 5);

    // Left Y-axis label
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = "#10b981";
    ctx.fillText("Relative Diff", 0, 0);
    ctx.restore();

    // Right Y-axis label
    ctx.save();
    ctx.translate(width - 10, height / 2);
    ctx.rotate(Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = "#f59e0b";
    ctx.fillText("Cosine Sim", 0, 0);
    ctx.restore();

    // Draw lines
    const drawLineLeft = (data: number[], color: string, lineWidth: number = 2) => {
      if (data.length === 0) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();

      data.forEach((val, i) => {
        const xVal = xValues[i];
        const x = xScale(xVal);
        const y = yScaleLeft(val);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    };

    const drawLineRight = (data: number[], color: string, lineWidth: number = 2) => {
      if (data.length === 0) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();

      data.forEach((val, i) => {
        const xVal = xValues[i];
        const x = xScale(xVal);
        const y = yScaleRight(val);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    };

    // Draw relative_diff (left axis) - green
    drawLineLeft(metrics.map(m => m.relative_diff), "#10b981", 3);

    // Draw cosine_similarity (right axis) - amber
    drawLineRight(metrics.map(m => m.cosine_similarity), "#f59e0b", 3);

    // Draw legend
    const legendX = width - padding.right - 150;
    const legendY = padding.top + 10;

    // relative_diff - green
    ctx.fillStyle = "#10b981";
    ctx.fillRect(legendX, legendY, 20, 3);
    ctx.fillStyle = "#aaa";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Relative Diff", legendX + 25, legendY + 2);

    // cosine_similarity - amber
    ctx.fillStyle = "#f59e0b";
    ctx.fillRect(legendX, legendY + 20, 20, 3);
    ctx.fillStyle = "#aaa";
    ctx.fillText("Cosine Sim", legendX + 25, legendY + 22);

  }, [metrics]);

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
