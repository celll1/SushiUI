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

    // Determine y-axis range (norms)
    const allNorms = metrics.flatMap(m => [m.uncond_norm, m.text_norm]);
    const yMin = 0;
    const yMax = Math.max(...allNorms) * 1.1; // 10% padding

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
      ctx.fillStyle = "#888";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(y.toFixed(2), padding.left - 5, yPos);
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

    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("L2 Norm", 0, 0);
    ctx.restore();

    // Draw lines
    const drawLine = (data: number[], color: string, lineWidth: number = 2) => {
      if (data.length === 0) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();

      data.forEach((norm, i) => {
        const xVal = xValues[i];
        const x = xScale(xVal);
        const y = yScale(norm);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();
    };

    // Draw uncond_norm (yn) - blue
    drawLine(metrics.map(m => m.uncond_norm), "#3b82f6", 2);

    // Draw text_norm (yp) - red
    drawLine(metrics.map(m => m.text_norm), "#ef4444", 2);

    // Draw legend
    const legendX = width - padding.right - 100;
    const legendY = padding.top + 10;

    // yn (uncond) - blue
    ctx.fillStyle = "#3b82f6";
    ctx.fillRect(legendX, legendY, 15, 3);
    ctx.fillStyle = "#aaa";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("yn (uncond)", legendX + 20, legendY + 2);

    // yp (text) - red
    ctx.fillStyle = "#ef4444";
    ctx.fillRect(legendX, legendY + 20, 15, 3);
    ctx.fillStyle = "#aaa";
    ctx.fillText("yp (text)", legendX + 20, legendY + 22);

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

            <span className="text-blue-400">yn:</span>
            <span className="text-white">{hoveredPoint.uncond_norm.toFixed(4)}</span>

            <span className="text-red-400">yp:</span>
            <span className="text-white">{hoveredPoint.text_norm.toFixed(4)}</span>

            <span className="text-gray-400">CFG:</span>
            <span className="text-white">{hoveredPoint.guidance_scale}</span>
          </div>
        </div>
      )}
    </div>
  );
}
