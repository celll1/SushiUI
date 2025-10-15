import React, { useEffect, useRef, useState } from "react";

interface LayerWeightGraphProps {
  layers: string[];
  weights: { [layerName: string]: number };
  onChange: (weights: { [layerName: string]: number }) => void;
  disabled?: boolean;
}

const LayerWeightGraph: React.FC<LayerWeightGraphProps> = ({
  layers,
  weights,
  onChange,
  disabled = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hoveredLayer, setHoveredLayer] = useState<string | null>(null);

  const CANVAS_WIDTH = 360;
  const CANVAS_HEIGHT = 220;
  const PADDING = { top: 15, right: 10, bottom: 35, left: 35 };
  const GRAPH_WIDTH = CANVAS_WIDTH - PADDING.left - PADDING.right;
  const GRAPH_HEIGHT = CANVAS_HEIGHT - PADDING.top - PADDING.bottom;
  const MIN_WEIGHT = 0;
  const MAX_WEIGHT = 2;

  // Initialize weights for all layers if not present
  useEffect(() => {
    const newWeights = { ...weights };
    let hasChanges = false;

    layers.forEach((layer) => {
      if (!(layer in newWeights)) {
        newWeights[layer] = 1.0; // Default weight
        hasChanges = true;
      }
    });

    if (hasChanges) {
      onChange(newWeights);
    }
  }, [layers]);

  // Draw the graph
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || layers.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Draw background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Draw graph area
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(PADDING.left, PADDING.top, GRAPH_WIDTH, GRAPH_HEIGHT);

    // Draw grid lines
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;

    // Horizontal grid lines (weight levels)
    const weightSteps = 5;
    for (let i = 0; i <= weightSteps; i++) {
      const y = PADDING.top + (GRAPH_HEIGHT * i) / weightSteps;
      ctx.beginPath();
      ctx.moveTo(PADDING.left, y);
      ctx.lineTo(PADDING.left + GRAPH_WIDTH, y);
      ctx.stroke();

      // Weight labels
      const weight = MAX_WEIGHT - (i * (MAX_WEIGHT - MIN_WEIGHT)) / weightSteps;
      ctx.fillStyle = "#888";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText(weight.toFixed(1), PADDING.left - 5, y + 3);
    }

    // Draw layer lines and labels
    const layerStep = GRAPH_WIDTH / Math.max(layers.length - 1, 1);
    ctx.fillStyle = "#888";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";

    layers.forEach((layer, index) => {
      const x = PADDING.left + index * layerStep;

      // Vertical grid line
      ctx.strokeStyle = "#333";
      ctx.beginPath();
      ctx.moveTo(x, PADDING.top);
      ctx.lineTo(x, PADDING.top + GRAPH_HEIGHT);
      ctx.stroke();

      // Block label (straight text for better readability)
      ctx.fillStyle = hoveredLayer === layer ? "#fff" : "#888";
      ctx.font = "9px monospace";
      ctx.fillText(layer, x, PADDING.top + GRAPH_HEIGHT + 12);
    });

    // Draw weight curve
    if (layers.length > 0) {
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 2;
      ctx.beginPath();

      layers.forEach((layer, index) => {
        const x = PADDING.left + index * layerStep;
        const weight = weights[layer] ?? 1.0;
        const normalizedWeight = (weight - MIN_WEIGHT) / (MAX_WEIGHT - MIN_WEIGHT);
        const y = PADDING.top + GRAPH_HEIGHT * (1 - normalizedWeight);

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // Draw points
      layers.forEach((layer, index) => {
        const x = PADDING.left + index * layerStep;
        const weight = weights[layer] ?? 1.0;
        const normalizedWeight = (weight - MIN_WEIGHT) / (MAX_WEIGHT - MIN_WEIGHT);
        const y = PADDING.top + GRAPH_HEIGHT * (1 - normalizedWeight);

        ctx.fillStyle = hoveredLayer === layer ? "#60a5fa" : "#3b82f6";
        ctx.beginPath();
        ctx.arc(x, y, hoveredLayer === layer ? 6 : 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Draw axes
    ctx.strokeStyle = "#666";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(PADDING.left, PADDING.top);
    ctx.lineTo(PADDING.left, PADDING.top + GRAPH_HEIGHT);
    ctx.lineTo(PADDING.left + GRAPH_WIDTH, PADDING.top + GRAPH_HEIGHT);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = "#aaa";
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Blocks", CANVAS_WIDTH / 2, CANVAS_HEIGHT - 3);

    ctx.save();
    ctx.translate(12, CANVAS_HEIGHT / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Weight", 0, 0);
    ctx.restore();
  }, [layers, weights, hoveredLayer]);

  const getLayerFromPosition = (x: number, y: number): { layer: string; index: number } | null => {
    if (layers.length === 0) return null;

    const graphX = x - PADDING.left;
    const graphY = y - PADDING.top;

    if (graphX < 0 || graphX > GRAPH_WIDTH || graphY < 0 || graphY > GRAPH_HEIGHT) {
      return null;
    }

    const layerStep = GRAPH_WIDTH / Math.max(layers.length - 1, 1);
    const index = Math.round(graphX / layerStep);

    if (index < 0 || index >= layers.length) return null;

    return { layer: layers[index], index };
  };

  const getWeightFromY = (y: number): number => {
    const graphY = y - PADDING.top;
    const normalizedWeight = 1 - graphY / GRAPH_HEIGHT;
    const weight = MIN_WEIGHT + normalizedWeight * (MAX_WEIGHT - MIN_WEIGHT);
    return Math.max(MIN_WEIGHT, Math.min(MAX_WEIGHT, weight));
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const result = getLayerFromPosition(x, y);
    if (result) {
      setIsDrawing(true);
      const weight = getWeightFromY(y);
      const newWeights = { ...weights, [result.layer]: weight };
      onChange(newWeights);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const result = getLayerFromPosition(x, y);
    setHoveredLayer(result ? result.layer : null);

    if (isDrawing && !disabled && result) {
      const weight = getWeightFromY(y);
      const newWeights = { ...weights, [result.layer]: weight };
      onChange(newWeights);
    }
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const handleMouseLeave = () => {
    setIsDrawing(false);
    setHoveredLayer(null);
  };

  const resetWeights = () => {
    if (disabled) return;
    const newWeights: { [key: string]: number } = {};
    layers.forEach((layer) => {
      newWeights[layer] = 1.0;
    });
    onChange(newWeights);
  };

  if (layers.length === 0) {
    return (
      <div className="p-4 bg-gray-800 rounded text-gray-400 text-center text-sm">
        Select a LoRA to configure block weights
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="text-sm font-medium">U-Net Block Weights</label>
        <button
          onClick={resetWeights}
          disabled={disabled}
          className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded"
        >
          Reset All to 1.0
        </button>
      </div>
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          className={`border border-gray-700 rounded ${
            disabled ? "opacity-50 cursor-not-allowed" : "cursor-crosshair"
          }`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        />
        {hoveredLayer && (
          <div className="absolute top-2 right-2 bg-black bg-opacity-75 px-2 py-1 rounded text-xs">
            <div className="text-gray-400">{hoveredLayer}</div>
            <div className="text-white font-mono">
              Weight: {(weights[hoveredLayer] ?? 1.0).toFixed(3)}
            </div>
          </div>
        )}
      </div>
      <div className="text-xs text-gray-500">
        Click and drag on the graph to adjust block weights. Each point represents a U-Net block (IN=input, MID=middle, OUT=output).
      </div>
    </div>
  );
};

export default LayerWeightGraph;
