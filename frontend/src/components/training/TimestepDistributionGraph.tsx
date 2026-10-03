"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { densityOf } from "@/utils/timestepDistribution";

/** One pre-computed density, for the morph overlay (see `curves` below). */
export interface TimestepGraphCurve {
  points: { x: number; y: number }[];
  color: string;
  dashed?: boolean;
  label?: string;
}

interface TimestepDistributionGraphProps {
  distribution: string;
  minTimestep: number;
  maxTimestep: number;
  mean?: number;
  std?: number;
  alpha?: number;
  beta?: number;
  width?: number;
  height?: number;
  /**
   * Draw these densities instead of the one implied by the props above, all on
   * ONE shared y-scale so a morph's source, target and current law are
   * comparable. Used by the training monitor, where the law being sampled is an
   * interpolation no single parameter set describes.
   */
  curves?: TimestepGraphCurve[];
}

export default function TimestepDistributionGraph({
  distribution,
  minTimestep,
  maxTimestep,
  mean = 0.0,
  std = 1.0,
  alpha = 2.0,
  beta = 2.0,
  width = 300,
  height = 80,
  curves,
}: TimestepDistributionGraphProps) {
  const graphContainerRef = useRef<HTMLDivElement>(null);
  const [measuredWidth, setMeasuredWidth] = useState<number | null>(null);

  useEffect(() => {
    const element = graphContainerRef.current;
    if (!element) return;

    const updateWidth = () => {
      const nextWidth = Math.max(16, Math.round(element.getBoundingClientRect().width));
      setMeasuredWidth((current) => current === nextWidth ? current : nextWidth);
    };

    updateWidth();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(updateWidth);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const renderWidth = measuredWidth ?? width;

  // Validate and sanitize numeric inputs
  const safeNum = (val: any, fallback: number): number => {
    const num = typeof val === 'string' ? parseFloat(val) : val;
    return typeof num === 'number' && isFinite(num) ? num : fallback;
  };

  const safeMean = safeNum(mean, 0.0);
  const safeStd = safeNum(std, 1.0);
  const safeAlpha = safeNum(alpha, 2.0);
  const safeBeta = safeNum(beta, 2.0);
  const safeMinTimestep = safeNum(minTimestep, 0.0);
  const safeMaxTimestep = safeNum(maxTimestep, 1.0);

  const isValidForRender =
    safeMinTimestep < safeMaxTimestep &&
    safeStd > 0 &&
    safeAlpha > 0 &&
    safeBeta > 0;

  const path = useMemo(() => {
    if (!isValidForRender) return "";
    const pts = densityOf({
      version: 1,
      kind: "config",
      config: {
        distribution,
        min_timestep: safeMinTimestep,
        max_timestep: safeMaxTimestep,
        mean: safeMean,
        std: safeStd,
        alpha: safeAlpha,
        beta: safeBeta,
      },
    });
    if (!pts) return "";

    const maxYVal = Math.max(...pts.map(p => p.y), 0.001);

    const padding = 4;
    const graphWidth = renderWidth - padding * 2;
    const graphHeight = height - padding * 2 - 15; // Leave room for axis labels

    let pathStr = "";
    pts.forEach((pt, i) => {
      const px = padding + pt.x * graphWidth;
      const py = padding + graphHeight - (pt.y / maxYVal) * graphHeight;
      if (i === 0) {
        pathStr += `M ${px} ${py}`;
      } else {
        pathStr += ` L ${px} ${py}`;
      }
    });

    return pathStr;
  }, [distribution, isValidForRender, safeMinTimestep, safeMaxTimestep, safeMean, safeStd, safeAlpha, safeBeta, renderWidth, height]);

  const overlayPaths = useMemo(() => {
    if (!curves || curves.length === 0) return null;
    const pad = 4;
    const w = renderWidth - pad * 2;
    const h = height - pad * 2 - 15;
    const peak = Math.max(
      0.001,
      ...curves.flatMap((curve) => curve.points.map((p) => p.y)),
    );
    return curves.map((curve) => ({
      ...curve,
      d: curve.points
        .map((point, i) => {
          const px = pad + point.x * w;
          const py = pad + h - (point.y / peak) * h;
          return `${i === 0 ? "M" : "L"} ${px} ${py}`;
        })
        .join(" "),
    }));
  }, [curves, renderWidth, height]);

  const expectedMean = useMemo(() => {
    if (!isValidForRender) return 0.5;

    switch (distribution) {
      case "uniform":
        return (safeMinTimestep + safeMaxTimestep) / 2;
      case "logit_normal":
      case "lognormal":
        // Approximate: sigmoid(mean) gives the mode, not mean
        // For visualization, use sigmoid(mean) as indicator
        return 1 / (1 + Math.exp(-safeMean));
      case "normal":
        return Math.max(safeMinTimestep, Math.min(safeMaxTimestep, safeMean));
      case "beta":
        return safeAlpha / (safeAlpha + safeBeta);
      default:
        return 0.5;
    }
  }, [distribution, isValidForRender, safeMinTimestep, safeMaxTimestep, safeMean, safeAlpha, safeBeta]);

  const padding = 4;
  const graphWidth = renderWidth - padding * 2;
  const graphHeight = height - padding * 2 - 15;

  // Min/max range indicator positions
  const minX = padding + safeMinTimestep * graphWidth;
  const maxX = padding + safeMaxTimestep * graphWidth;
  const meanX = padding + expectedMean * graphWidth;

  if (overlayPaths) {
    return (
      <div className="bg-gray-900 rounded p-2">
        <div ref={graphContainerRef}>
          <svg width={renderWidth} height={height} className="block">
            <defs>
              <pattern id="grid-overlay" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#374151" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect x={padding} y={padding} width={graphWidth} height={graphHeight} fill="url(#grid-overlay)" />
            {overlayPaths.map((curve, i) => (
              <path
                key={i}
                d={curve.d}
                fill="none"
                stroke={curve.color}
                strokeWidth={curve.dashed ? 1 : 2}
                strokeDasharray={curve.dashed ? "3,2" : undefined}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ))}
            <line
              x1={padding} y1={padding + graphHeight}
              x2={padding + graphWidth} y2={padding + graphHeight}
              stroke="#6b7280" strokeWidth="1"
            />
            <text x={padding} y={height - 2} fontSize="9" fill="#9ca3af" textAnchor="start">0</text>
            <text x={padding + graphWidth / 2} y={height - 2} fontSize="9" fill="#9ca3af" textAnchor="middle">0.5</text>
            <text x={padding + graphWidth} y={height - 2} fontSize="9" fill="#9ca3af" textAnchor="end">1</text>
          </svg>
        </div>
        <div className="flex justify-center gap-3 text-[10px] text-gray-400 mt-1">
          {overlayPaths.filter((c) => c.label).map((curve, i) => (
            <span key={i} className="flex items-center gap-1">
              <span
                className="w-3 h-0.5 inline-block"
                style={{ backgroundColor: curve.color, opacity: curve.dashed ? 0.6 : 1 }}
              />
              {curve.label}
            </span>
          ))}
        </div>
      </div>
    );
  }

  // Show placeholder if invalid parameters
  if (!isValidForRender) {
    return (
      <div className="bg-gray-900 rounded p-2">
        <div ref={graphContainerRef}>
          <svg width={renderWidth} height={height} className="block">
            <rect x={padding} y={padding} width={graphWidth} height={graphHeight} fill="#1f2937" />
            <text x={renderWidth / 2} y={height / 2} fontSize="10" fill="#6b7280" textAnchor="middle">
              Enter valid parameters
            </text>
          </svg>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded p-2">
      <div ref={graphContainerRef}>
        <svg width={renderWidth} height={height} className="block">
        {/* Background grid */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#374151" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect x={padding} y={padding} width={graphWidth} height={graphHeight} fill="url(#grid)" />

        {/* Active range highlight */}
        <rect
          x={minX}
          y={padding}
          width={maxX - minX}
          height={graphHeight}
          fill="rgba(59, 130, 246, 0.1)"
        />

        {/* Distribution curve */}
        <path
          d={path}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Fill under curve */}
        <path
          d={`${path} L ${padding + graphWidth} ${padding + graphHeight} L ${padding} ${padding + graphHeight} Z`}
          fill="rgba(59, 130, 246, 0.2)"
        />

        {/* Min/Max range lines */}
        <line
          x1={minX} y1={padding}
          x2={minX} y2={padding + graphHeight}
          stroke="#22c55e"
          strokeWidth="1.5"
          strokeDasharray="3,2"
        />
        <line
          x1={maxX} y1={padding}
          x2={maxX} y2={padding + graphHeight}
          stroke="#ef4444"
          strokeWidth="1.5"
          strokeDasharray="3,2"
        />

        {/* Mean/Mode indicator */}
        {distribution !== "uniform" && (
          <line
            x1={meanX} y1={padding}
            x2={meanX} y2={padding + graphHeight}
            stroke="#f59e0b"
            strokeWidth="1.5"
          />
        )}

        {/* X-axis */}
        <line
          x1={padding} y1={padding + graphHeight}
          x2={padding + graphWidth} y2={padding + graphHeight}
          stroke="#6b7280"
          strokeWidth="1"
        />

        {/* X-axis labels */}
        <text x={padding} y={height - 2} fontSize="9" fill="#9ca3af" textAnchor="start">0</text>
        <text x={padding + graphWidth / 2} y={height - 2} fontSize="9" fill="#9ca3af" textAnchor="middle">0.5</text>
        <text x={padding + graphWidth} y={height - 2} fontSize="9" fill="#9ca3af" textAnchor="end">1</text>

        {/* Legend */}
        <text x={padding + graphWidth - 2} y={padding + 10} fontSize="8" fill="#9ca3af" textAnchor="end">
          {distribution === "logit_normal" || distribution === "lognormal"
            ? `logit-normal(${safeMean.toFixed(1)}, ${safeStd.toFixed(1)})`
            : distribution === "normal"
            ? `normal(${safeMean.toFixed(2)}, ${safeStd.toFixed(2)})`
            : distribution === "beta"
            ? `beta(${safeAlpha.toFixed(1)}, ${safeBeta.toFixed(1)})`
            : "uniform"}
        </text>
        </svg>
      </div>

      {/* Legend below graph */}
      <div className="flex justify-center gap-4 text-[10px] text-gray-400 mt-1">
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-green-500 inline-block" style={{ borderTop: "1.5px dashed #22c55e" }}></span>
          min
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-red-500 inline-block" style={{ borderTop: "1.5px dashed #ef4444" }}></span>
          max
        </span>
        {distribution !== "uniform" && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-amber-500 inline-block"></span>
            {distribution === "logit_normal" || distribution === "lognormal" ? "mode" : "mean"}
          </span>
        )}
      </div>
    </div>
  );
}
