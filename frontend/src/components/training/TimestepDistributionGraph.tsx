"use client";

import { useMemo } from "react";

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
}

/**
 * Calculate probability density for different distributions.
 * These are normalized for visualization purposes (not exact PDFs).
 */

// Standard normal PDF
function normalPdf(x: number, mean: number, std: number): number {
  const z = (x - mean) / std;
  return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
}

// Logit-normal PDF: derivative of sigmoid(normal)
// PDF(t) = (1 / (std * sqrt(2*pi))) * (1 / (t * (1-t))) * exp(-0.5 * ((logit(t) - mean) / std)^2)
function logitNormalPdf(t: number, mean: number, std: number): number {
  // Avoid division by zero at boundaries
  if (t <= 0.001 || t >= 0.999) return 0;

  const logit = Math.log(t / (1 - t));
  const z = (logit - mean) / std;
  const jacobian = 1 / (t * (1 - t));
  return normalPdf(logit, mean, std) * jacobian;
}

// Beta PDF using the gamma function approximation
function betaPdf(x: number, alpha: number, beta: number): number {
  if (x <= 0 || x >= 1) return 0;

  // Use log-gamma for numerical stability
  const logBeta = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
  const logPdf = (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - logBeta;
  return Math.exp(logPdf);
}

// Log-gamma approximation (Stirling's approximation for large values)
function logGamma(x: number): number {
  if (x <= 0) return 0;
  if (x < 0.5) {
    // Reflection formula
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  // Lanczos approximation
  const g = 7;
  const c = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];

  x -= 1;
  let sum = c[0];
  for (let i = 1; i < g + 2; i++) {
    sum += c[i] / (x + i);
  }
  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(sum);
}

// Uniform PDF (constant)
function uniformPdf(x: number, min: number, max: number): number {
  if (x < min || x > max) return 0;
  return 1 / (max - min);
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
}: TimestepDistributionGraphProps) {
  // Generate points for the distribution curve
  const { path, maxY, points } = useMemo(() => {
    const numPoints = 100;
    const pts: { x: number; y: number }[] = [];

    for (let i = 0; i <= numPoints; i++) {
      const t = i / numPoints; // 0 to 1
      let y = 0;

      switch (distribution) {
        case "uniform":
          // Uniform within [minTimestep, maxTimestep]
          if (t >= minTimestep && t <= maxTimestep) {
            y = 1;
          }
          break;

        case "logit_normal":
        case "lognormal":
          y = logitNormalPdf(t, mean, std);
          break;

        case "normal":
          // Normal distribution clamped to [0, 1]
          if (t >= minTimestep && t <= maxTimestep) {
            y = normalPdf(t, mean, std);
          }
          break;

        case "beta":
          y = betaPdf(t, alpha, beta);
          break;

        default:
          y = 1; // Default to uniform
      }

      pts.push({ x: t, y: isFinite(y) ? y : 0 });
    }

    // Find max Y for normalization
    const maxYVal = Math.max(...pts.map(p => p.y), 0.001);

    // Build SVG path
    const padding = 4;
    const graphWidth = width - padding * 2;
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

    return { path: pathStr, maxY: maxYVal, points: pts };
  }, [distribution, minTimestep, maxTimestep, mean, std, alpha, beta, width, height]);

  // Calculate mean timestep for the indicator
  const expectedMean = useMemo(() => {
    switch (distribution) {
      case "uniform":
        return (minTimestep + maxTimestep) / 2;
      case "logit_normal":
      case "lognormal":
        // Approximate: sigmoid(mean) gives the mode, not mean
        // For visualization, use sigmoid(mean) as indicator
        return 1 / (1 + Math.exp(-mean));
      case "normal":
        return Math.max(minTimestep, Math.min(maxTimestep, mean));
      case "beta":
        return alpha / (alpha + beta);
      default:
        return 0.5;
    }
  }, [distribution, minTimestep, maxTimestep, mean, alpha, beta]);

  const padding = 4;
  const graphWidth = width - padding * 2;
  const graphHeight = height - padding * 2 - 15;

  // Min/max range indicator positions
  const minX = padding + minTimestep * graphWidth;
  const maxX = padding + maxTimestep * graphWidth;
  const meanX = padding + expectedMean * graphWidth;

  return (
    <div className="bg-gray-900 rounded p-2">
      <svg width={width} height={height} className="w-full">
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
            ? `logit-normal(${mean.toFixed(1)}, ${std.toFixed(1)})`
            : distribution === "normal"
            ? `normal(${mean.toFixed(2)}, ${std.toFixed(2)})`
            : distribution === "beta"
            ? `beta(${alpha.toFixed(1)}, ${beta.toFixed(1)})`
            : "uniform"}
        </text>
      </svg>

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
