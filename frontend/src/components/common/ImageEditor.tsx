"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Button from "./Button";

interface ImageEditorProps {
  imageUrl: string;
  onSave: (editedImageUrl: string) => void;
  onClose: () => void;
  onSaveMask?: (maskUrl: string) => void; // Optional callback for mask export
  mode?: "edit" | "inpaint"; // Editor mode (default: "edit")
}

type Tool = "pen" | "eraser" | "blur" | "eyedropper" | "bucket" | "pan";
type BrushType = "normal" | "pencil" | "gpen" | "fude";

interface LayerInfo {
  id: string;
  name: string;
  visible: boolean;
  opacity: number;
  editable: boolean; // Can this layer be drawn on?
  deletable: boolean; // Can this layer be deleted?
}

interface HistoryState {
  layerId: string; // Which layer this history belongs to
  layerData: ImageData;
}

export default function ImageEditor({ imageUrl, onSave, onClose, onSaveMask, mode = "edit" }: ImageEditorProps) {
  // Canvas refs - we'll use Map to store multiple layer canvases
  const baseLayerRef = useRef<HTMLCanvasElement>(null); // Original image layer (not editable)
  const layerCanvasRefs = useRef<Map<string, HTMLCanvasElement>>(new Map()); // Editable layers
  const compositeCanvasRef = useRef<HTMLCanvasElement>(null); // For display
  const containerRef = useRef<HTMLDivElement>(null);
  const tempStrokeCanvasRef = useRef<HTMLCanvasElement | null>(null); // Temporary canvas for current stroke

  // Layer management state
  const [layers, setLayers] = useState<LayerInfo[]>(() => {
    const baseLayers = [
      { id: "base", name: "Base", visible: true, opacity: 1, editable: false, deletable: false },
      { id: "layer1", name: "Layer 1", visible: true, opacity: 1, editable: true, deletable: false }, // First layer not deletable
    ];
    // Add inpaint mask layer in inpaint mode
    if (mode === "inpaint") {
      baseLayers.push({ id: "mask", name: "Inpaint Mask", visible: true, opacity: 0.5, editable: true, deletable: false });
    }
    return baseLayers;
  });
  const [activeLayerId, setActiveLayerId] = useState<string>(mode === "inpaint" ? "mask" : "layer1");

  const [tool, setTool] = useState<Tool>("pen");
  const [brushType, setBrushType] = useState<BrushType>("normal");
  const [brushSize, setBrushSize] = useState(5);
  const [rgb, setRgb] = useState({ r: 0, g: 0, b: 0 });
  const [alpha, setAlpha] = useState(1); // 0-1
  const [hue, setHue] = useState(0);
  const [saturation, setSaturation] = useState(100);
  const [lightness, setLightness] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isTapering, setIsTapering] = useState(false); // Tapering mode after pointer release
  const [history, setHistory] = useState<HistoryState[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const updatingSourceRef = useRef<'rgb' | 'hsl' | null>(null);
  const strokeSnapshotRef = useRef<ImageData | null>(null); // Snapshot before stroke starts

  // Brush stroke tracking
  const strokeStartRef = useRef<{ x: number; y: number; time: number } | null>(null);
  const lastPointRef = useRef<{ x: number; y: number; time: number } | null>(null);
  const prevPointRef = useRef<{ x: number; y: number } | null>(null); // Point before last, for direction
  const strokeDistanceRef = useRef(0);
  const taperProgressRef = useRef(0); // 0 to 1, for gradual tapering
  const strokePathRef = useRef<Array<{ x: number; y: number; pressure: number; velocity: number; distance: number; taperProgress: number }>>([]); // Store stroke path

  // Calculate color from RGB with alpha
  const getColorFromRGB = () => {
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
  };

  // Calculate color from RGB with full opacity (for drawing preview)
  const getColorFromRGBOpaque = () => {
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 1)`;
  };

  // Calculate color from HSL and lightness with alpha
  const getColorFromHSL = () => {
    return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
  };

  // Convert RGB to HSL
  const rgbToHsl = (r: number, g: number, b: number) => {
    r /= 255;
    g /= 255;
    b /= 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0, s = 0, l = (max + min) / 2;

    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

      switch (max) {
        case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
        case g: h = ((b - r) / d + 2) / 6; break;
        case b: h = ((r - g) / d + 4) / 6; break;
      }
    }

    return {
      h: Math.round(h * 360),
      s: Math.round(s * 100),
      l: Math.round(l * 100)
    };
  };

  // Update RGB when HSL changes
  useEffect(() => {
    if (updatingSourceRef.current === 'rgb') {
      updatingSourceRef.current = null;
      return;
    }

    updatingSourceRef.current = 'hsl';
    const color = getColorFromHSL();
    // Convert HSL to RGB for display
    const temp = document.createElement('div');
    temp.style.color = color;
    document.body.appendChild(temp);
    const computed = window.getComputedStyle(temp).color;
    document.body.removeChild(temp);

    const match = computed.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (match) {
      setRgb({ r: parseInt(match[1]), g: parseInt(match[2]), b: parseInt(match[3]) });
    }
  }, [hue, saturation, lightness]);

  // Update HSL when RGB changes
  useEffect(() => {
    if (updatingSourceRef.current === 'hsl') {
      updatingSourceRef.current = null;
      return;
    }

    updatingSourceRef.current = 'rgb';
    const hsl = rgbToHsl(rgb.r, rgb.g, rgb.b);
    setHue(hsl.h);
    setSaturation(hsl.s);
    setLightness(hsl.l);
  }, [rgb]);

  // Helper function to get canvas for a layer
  const getLayerCanvas = useCallback((layerId: string): HTMLCanvasElement | null => {
    if (layerId === "base") {
      return baseLayerRef.current;
    }
    return layerCanvasRefs.current.get(layerId) || null;
  }, []);

  // Composite layers to display canvas
  const composeLayers = useCallback(() => {
    const composite = compositeCanvasRef.current;
    if (!composite) return;

    const ctx = composite.getContext("2d");
    if (!ctx) return;

    // Clear composite
    ctx.clearRect(0, 0, composite.width, composite.height);

    // Draw all visible layers in order
    for (const layer of layers) {
      if (!layer.visible) continue;

      const layerCanvas = getLayerCanvas(layer.id);
      if (!layerCanvas) continue;

      ctx.globalAlpha = layer.opacity;
      ctx.drawImage(layerCanvas, 0, 0);
      ctx.globalAlpha = 1;
    }

    // Draw temporary stroke canvas on top (if drawing)
    if (tempStrokeCanvasRef.current) {
      ctx.drawImage(tempStrokeCanvasRef.current, 0, 0);
    }
  }, [layers, getLayerCanvas]);

  // Re-composite when layer visibility or opacity changes
  useEffect(() => {
    composeLayers();
  }, [layers, composeLayers]);

  // Load image and initialize layers
  useEffect(() => {
    const baseLayer = baseLayerRef.current;
    const composite = compositeCanvasRef.current;
    const container = containerRef.current;
    if (!baseLayer || !composite || !container) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // Set canvas sizes
      const width = img.width;
      const height = img.height;
      baseLayer.width = width;
      baseLayer.height = height;
      composite.width = width;
      composite.height = height;

      // Draw original image to base layer
      const baseCtx = baseLayer.getContext("2d");
      if (baseCtx) {
        baseCtx.drawImage(img, 0, 0);
      }

      // Initialize editable layer canvases
      const editableLayers = layers.filter(l => l.editable);
      for (const layer of editableLayers) {
        let canvas = layerCanvasRefs.current.get(layer.id);
        if (!canvas) {
          // Create new canvas only if it doesn't exist
          canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          layerCanvasRefs.current.set(layer.id, canvas);

          // Clear new canvas to transparent
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, width, height);
          }
        } else {
          // Canvas already exists - check if resize is needed
          if (canvas.width !== width || canvas.height !== height) {
            // Only resize if dimensions actually changed
            // Save existing content first
            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = canvas.width;
            tempCanvas.height = canvas.height;
            const tempCtx = tempCanvas.getContext("2d");
            if (tempCtx) {
              tempCtx.drawImage(canvas, 0, 0);
            }

            // Resize canvas (this clears content)
            canvas.width = width;
            canvas.height = height;

            // Restore content
            const ctx = canvas.getContext("2d");
            if (ctx && tempCtx) {
              ctx.drawImage(tempCanvas, 0, 0);
            }
          }
          // Canvas already has correct size, no action needed
        }
      }

      // Composite layers
      composeLayers();

      // Calculate initial zoom to fit canvas in container
      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;
      const scaleX = containerWidth / width;
      const scaleY = containerHeight / height;
      const initialZoom = Math.min(scaleX, scaleY, 1);
      setZoom(initialZoom);

      // Center the image
      const displayWidth = width * initialZoom;
      const displayHeight = height * initialZoom;
      setPanOffset({
        x: (containerWidth - displayWidth) / 2,
        y: (containerHeight - displayHeight) / 2,
      });

      // Save initial state for active layer
      const activeLayer = layers.find(l => l.id === activeLayerId);
      if (activeLayer && activeLayer.editable) {
        const canvas = layerCanvasRefs.current.get(activeLayer.id);
        const ctx = canvas?.getContext("2d");
        if (canvas && ctx) {
          const initialData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          setHistory([{ layerId: activeLayer.id, layerData: initialData }]);
          setHistoryIndex(0);
        }
      }
    };
    img.src = imageUrl;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl, layers.length, layers.map(l => l.id).join(','), composeLayers, activeLayerId]);

  const saveToHistory = useCallback((layerId: string, ctx: CanvasRenderingContext2D) => {
    const layerCanvas = getLayerCanvas(layerId);
    if (!layerCanvas) return;

    const layerData = ctx.getImageData(0, 0, layerCanvas.width, layerCanvas.height);

    // Remove any history after current index
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({ layerId, layerData });

    // Limit history to 50 states
    if (newHistory.length > 50) {
      newHistory.shift();
    } else {
      setHistoryIndex(historyIndex + 1);
    }

    setHistory(newHistory);
  }, [history, historyIndex, getLayerCanvas]);

  const undo = useCallback(() => {
    if (historyIndex <= 0) return;

    const newIndex = historyIndex - 1;
    const historyState = history[newIndex];
    const layerCanvas = getLayerCanvas(historyState.layerId);
    const ctx = layerCanvas?.getContext("2d");

    if (!layerCanvas || !ctx) return;

    setHistoryIndex(newIndex);
    ctx.putImageData(historyState.layerData, 0, 0);
    composeLayers();
  }, [history, historyIndex, composeLayers, getLayerCanvas]);

  const redo = useCallback(() => {
    if (historyIndex >= history.length - 1) return;

    const newIndex = historyIndex + 1;
    const historyState = history[newIndex];
    const layerCanvas = getLayerCanvas(historyState.layerId);
    const ctx = layerCanvas?.getContext("2d");

    if (!layerCanvas || !ctx) return;

    setHistoryIndex(newIndex);
    ctx.putImageData(historyState.layerData, 0, 0);
    composeLayers();
  }, [history, historyIndex, composeLayers, getLayerCanvas]);

  const getCanvasPoint = (e: React.PointerEvent<HTMLCanvasElement> | React.MouseEvent<HTMLCanvasElement>) => {
    const composite = compositeCanvasRef.current;
    const container = containerRef.current;
    if (!composite || !container) return { x: 0, y: 0 };

    const containerRect = container.getBoundingClientRect();

    // Screen coordinates relative to container
    const screenX = e.clientX - containerRect.left - panOffset.x;
    const screenY = e.clientY - containerRect.top - panOffset.y;

    // Center of the rotated canvas
    const centerX = (composite.width * zoom) / 2;
    const centerY = (composite.height * zoom) / 2;

    // Apply inverse rotation
    const rad = (-rotation * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    const relX = screenX - centerX;
    const relY = screenY - centerY;

    const rotatedX = relX * cos - relY * sin + centerX;
    const rotatedY = relX * sin + relY * cos + centerY;

    // Convert to canvas coordinates
    const x = rotatedX / zoom;
    const y = rotatedY / zoom;

    return { x, y };
  };

  const applyBlur = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, brushSize: number) => {
    const blurKernel = Math.max(1, Math.floor(brushSize / 8)); // Small kernel for subtle blur
    const blurRadius = brushSize * 0.5; // brushSize is diameter, so radius is half

    const startX = Math.max(0, Math.floor(centerX - blurRadius));
    const startY = Math.max(0, Math.floor(centerY - blurRadius));
    const regionWidth = Math.min(ctx.canvas.width - startX, Math.ceil(blurRadius * 2));
    const regionHeight = Math.min(ctx.canvas.height - startY, Math.ceil(blurRadius * 2));

    if (regionWidth <= 0 || regionHeight <= 0) return;

    const imageData = ctx.getImageData(startX, startY, regionWidth, regionHeight);
    const pixels = imageData.data;
    const width = regionWidth;
    const height = regionHeight;
    const original = new Uint8ClampedArray(pixels);

    // Apply blur only within circular area matching brush size
    for (let py = 0; py < height; py++) {
      for (let px = 0; px < width; px++) {
        // Calculate distance from center
        const worldX = startX + px;
        const worldY = startY + py;
        const dist = Math.sqrt((worldX - centerX) ** 2 + (worldY - centerY) ** 2);

        // Only blur within brush radius
        if (dist > blurRadius) continue;

        // Smooth falloff at edges
        const edgeFalloff = 1 - Math.pow(dist / blurRadius, 2);
        const blendFactor = edgeFalloff * 0.5; // Max 50% blur

        const idx = (py * width + px) * 4;

        // Average surrounding pixels
        let r = 0, g = 0, b = 0, a = 0, count = 0;
        for (let dy = -blurKernel; dy <= blurKernel; dy++) {
          for (let dx = -blurKernel; dx <= blurKernel; dx++) {
            const ny = py + dy;
            const nx = px + dx;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const nidx = (ny * width + nx) * 4;
              r += original[nidx];
              g += original[nidx + 1];
              b += original[nidx + 2];
              a += original[nidx + 3];
              count++;
            }
          }
        }

        // Blend blurred with original
        const blurredR = r / count;
        const blurredG = g / count;
        const blurredB = b / count;
        const blurredA = a / count;

        pixels[idx] = original[idx] * (1 - blendFactor) + blurredR * blendFactor;
        pixels[idx + 1] = original[idx + 1] * (1 - blendFactor) + blurredG * blendFactor;
        pixels[idx + 2] = original[idx + 2] * (1 - blendFactor) + blurredB * blendFactor;
        pixels[idx + 3] = original[idx + 3] * (1 - blendFactor) + blurredA * blendFactor;
      }
    }

    ctx.putImageData(imageData, startX, startY);
  };

  // Apply alpha to completed stroke
  const applyAlphaToStroke = (ctx: CanvasRenderingContext2D) => {
    if (!strokeSnapshotRef.current || alpha >= 1) {
      // No need to apply alpha if it's 100%
      strokeSnapshotRef.current = null;
      return;
    }

    const canvas = ctx.canvas;
    const width = canvas.width;
    const height = canvas.height;

    // Get current state (with stroke)
    const currentData = ctx.getImageData(0, 0, width, height);
    const current = currentData.data;
    const snapshot = strokeSnapshotRef.current.data;

    // Apply alpha only to the new pixels (difference from snapshot)
    for (let i = 0; i < current.length; i += 4) {
      // Check if pixel changed from snapshot
      if (current[i] !== snapshot[i] ||
          current[i+1] !== snapshot[i+1] ||
          current[i+2] !== snapshot[i+2] ||
          current[i+3] !== snapshot[i+3]) {

        // This pixel is part of the new stroke
        // Blend with snapshot using alpha
        const newAlpha = current[i+3] / 255;
        const finalAlpha = newAlpha * alpha;

        current[i+3] = finalAlpha * 255;
      }
    }

    ctx.putImageData(currentData, 0, 0);
    strokeSnapshotRef.current = null;
  };

  // Flood fill algorithm for bucket tool with tolerance for anti-aliased edges
  const floodFill = (ctx: CanvasRenderingContext2D, startX: number, startY: number, fillColor: string, tolerance: number = 100) => {
    const canvas = ctx.canvas;
    const width = canvas.width;
    const height = canvas.height;

    // Parse fill color to rgba values
    const temp = document.createElement('canvas');
    temp.width = 1;
    temp.height = 1;
    const tempCtx = temp.getContext('2d');
    if (!tempCtx) return;

    tempCtx.fillStyle = fillColor;
    tempCtx.fillRect(0, 0, 1, 1);
    const fillData = tempCtx.getImageData(0, 0, 1, 1).data;
    const fillR = fillData[0];
    const fillG = fillData[1];
    const fillB = fillData[2];
    const fillA = fillData[3];

    // Get image data
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    // Get target color at start position
    const startIdx = (startY * width + startX) * 4;
    const targetR = data[startIdx];
    const targetG = data[startIdx + 1];
    const targetB = data[startIdx + 2];
    const targetA = data[startIdx + 3];

    // Calculate color distance
    const colorDistance = (r: number, g: number, b: number, a: number): number => {
      const dr = r - targetR;
      const dg = g - targetG;
      const db = b - targetB;
      const da = a - targetA;
      return Math.sqrt(dr*dr + dg*dg + db*db + da*da);
    };

    // If target color is same as fill color, no need to fill
    if (colorDistance(fillR, fillG, fillB, fillA) <= tolerance * 2) {
      return;
    }

    // Stack-based flood fill with 8-directional search
    const stack: Array<[number, number]> = [[startX, startY]];
    const visited = new Set<number>();

    const colorMatch = (x: number, y: number): boolean => {
      const idx = (y * width + x) * 4;
      const distance = colorDistance(data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);
      return distance <= tolerance * 2; // Scale tolerance for distance metric
    };

    while (stack.length > 0) {
      const pos = stack.pop();
      if (!pos) break;

      const [x, y] = pos;

      // Check bounds
      if (x < 0 || x >= width || y < 0 || y >= height) continue;

      // Check if already visited
      const key = y * width + x;
      if (visited.has(key)) continue;
      visited.add(key);

      // Check if color matches
      if (!colorMatch(x, y)) continue;

      // Fill pixel
      const idx = (y * width + x) * 4;
      data[idx] = fillR;
      data[idx + 1] = fillG;
      data[idx + 2] = fillB;
      data[idx + 3] = fillA;

      // Add neighbors to stack (8-directional for better gap filling)
      stack.push([x + 1, y]);     // right
      stack.push([x - 1, y]);     // left
      stack.push([x, y + 1]);     // down
      stack.push([x, y - 1]);     // up
      stack.push([x + 1, y + 1]); // bottom-right
      stack.push([x + 1, y - 1]); // top-right
      stack.push([x - 1, y + 1]); // bottom-left
      stack.push([x - 1, y - 1]); // top-left
    }

    // Put modified data back
    ctx.putImageData(imageData, 0, 0);
  };

  // Draw with different brush types
  const drawWithBrush = (
    ctx: CanvasRenderingContext2D,
    fromX: number,
    fromY: number,
    toX: number,
    toY: number,
    size: number,
    color: string,
    pressure: number,
    velocity: number,
    strokeDistance: number,
    taperProgress: number = 0 // 0 = no taper, 1 = full taper
  ) => {
    ctx.globalCompositeOperation = "source-over";

    switch (brushType) {
      case "normal":
        // Normal pen - solid, uniform stroke, no tapering
        ctx.strokeStyle = color;
        ctx.lineWidth = size * pressure;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.globalAlpha = 1;
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        break;

      case "pencil":
        // Pencil - textured, random opacity variations with gradual exit tapering
        const distance = Math.hypot(toX - fromX, toY - fromY);
        const steps = Math.max(1, Math.floor(distance / 2));

        for (let i = 0; i <= steps; i++) {
          const t = i / steps;
          const x = fromX + (toX - fromX) * t;
          const y = fromY + (toY - fromY) * t;

          // Gradual exit tapering based on taperProgress
          // taperProgress: 0 = normal, 1 = fully tapered
          const exitFactor = Math.max(0.05, 1 - taperProgress * 0.95); // Taper to 5% size

          const randomOpacity = Math.max(0.05, (0.3 + Math.random() * 0.4) * exitFactor);
          const randomSize = Math.max(0.5, size * pressure * (0.8 + Math.random() * 0.4) * exitFactor);

          ctx.globalAlpha = randomOpacity;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, randomSize / 2, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
        break;

      case "gpen":
        // G-pen - varies thickness based on velocity with gradual exit tapering
        const velocityFactor = Math.max(0.3, 1 - velocity * 0.01);
        // taperProgress: 0 = normal, 1 = fully tapered
        const taperFactor = Math.max(0.1, 1 - taperProgress * 0.9); // Taper down to 10% size
        const gpenPressure = pressure * taperFactor;

        const gpenSize = Math.max(0.5, size * gpenPressure * velocityFactor);
        ctx.strokeStyle = color;
        ctx.lineWidth = gpenSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.globalAlpha = Math.max(0.1, 1 - taperProgress * 0.7);
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        ctx.globalAlpha = 1;
        break;

      case "fude":
        // Fude/brush - soft edges with blur, tapers at entry and gradual exit with trailing fade
        const segmentDistance = Math.hypot(toX - fromX, toY - fromY);
        const segmentSteps = Math.max(1, Math.ceil(segmentDistance / 1));

        for (let i = 0; i <= segmentSteps; i++) {
          const t = i / segmentSteps;
          const x = fromX + (toX - fromX) * t;
          const y = fromY + (toY - fromY) * t;

          // Entry taper
          let tapering = 1;
          if (strokeDistance < 20) {
            tapering = Math.pow(strokeDistance / 20, 0.7);
          }

          // Gradual exit tapering based on taperProgress
          // taperProgress: 0 = normal, 1 = fully tapered
          const exitTaper = Math.max(0.05, 1 - taperProgress * 0.95); // Taper down to 5% size
          tapering *= exitTaper;

          const currentSize = Math.max(0.5, size * pressure * tapering);

          // Draw soft brush with multiple layers for blur effect
          const layers = 4;
          for (let layer = 0; layer < layers; layer++) {
            const layerRatio = (layer + 1) / layers;
            const layerSize = Math.max(0.5, currentSize * (0.4 + layerRatio * 0.6));
            const layerAlpha = (0.15 / layers) * (1 - layer / layers) * exitTaper;

            ctx.globalAlpha = Math.max(0, layerAlpha);
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, layerSize, 0, Math.PI * 2);
            ctx.fill();
          }

          // Core solid part
          ctx.globalAlpha = Math.max(0.05, 0.8 * tapering * exitTaper);
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, Math.max(0.5, currentSize * 0.5), 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.globalAlpha = 1;
        break;
    }
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const composite = compositeCanvasRef.current;
    const compositeCtx = composite?.getContext("2d");
    if (!composite || !compositeCtx) return;

    // Get active editable layer
    const activeLayer = layers.find(l => l.id === activeLayerId);
    if (!activeLayer || !activeLayer.editable || !activeLayer.visible) return;

    const layerCanvas = getLayerCanvas(activeLayerId);
    const layerCtx = layerCanvas?.getContext("2d");
    if (!layerCanvas || !layerCtx) return;

    // Capture the pointer
    e.currentTarget.setPointerCapture(e.pointerId);

    const point = getCanvasPoint(e);

    if (tool === "pan") {
      setIsPanning(true);
      setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
      return;
    }

    if (tool === "eyedropper") {
      // Pick color from composite (both layers)
      const x = Math.floor(point.x);
      const y = Math.floor(point.y);

      // Ensure coordinates are within canvas bounds
      if (x >= 0 && x < composite.width && y >= 0 && y < composite.height) {
        const imageData = compositeCtx.getImageData(x, y, 1, 1);
        const pixel = imageData.data;
        setRgb({ r: pixel[0], g: pixel[1], b: pixel[2] });
      }

      // Switch back to pen tool after picking
      setTool("pen");
      return;
    }

    if (tool === "bucket") {
      // Bucket fill
      const x = Math.floor(point.x);
      const y = Math.floor(point.y);

      if (x >= 0 && x < layerCanvas.width && y >= 0 && y < layerCanvas.height) {
        floodFill(layerCtx, x, y, getColorFromRGB());
        composeLayers();
        saveToHistory(activeLayerId, layerCtx);
      }
      return;
    }

    setIsDrawing(true);

    // Initialize stroke tracking
    const now = Date.now();
    strokeStartRef.current = { x: point.x, y: point.y, time: now };
    lastPointRef.current = { x: point.x, y: point.y, time: now };
    prevPointRef.current = { x: point.x, y: point.y }; // Initialize with same point
    strokeDistanceRef.current = 0;
    taperProgressRef.current = 0; // Reset taper progress

    // Get pressure from pointer event (0.5 default for mouse, varies for pen/touch)
    const pressure = e.pressure > 0 ? e.pressure : 0.5;

    if (tool === "pen") {
      // Save layer state before starting stroke
      strokeSnapshotRef.current = layerCtx.getImageData(0, 0, layerCanvas.width, layerCanvas.height);

      // Initialize stroke path
      strokePathRef.current = [{
        x: point.x,
        y: point.y,
        pressure: pressure,
        velocity: 0,
        distance: 0,
        taperProgress: 0
      }];

      // Draw initial point with full opacity (alpha will be applied at end)
      drawWithBrush(
        layerCtx,
        point.x,
        point.y,
        point.x,
        point.y,
        brushSize,
        getColorFromRGBOpaque(),
        pressure,
        0,
        0,
        0 // No taper at start
      );
      composeLayers();
    } else if (tool === "eraser") {
      // Eraser erases active layer only (not the base image)
      layerCtx.globalCompositeOperation = "destination-out";
      layerCtx.strokeStyle = "rgba(0,0,0,1)";
      layerCtx.lineWidth = brushSize * pressure;
      layerCtx.lineCap = "round";
      layerCtx.lineJoin = "round";
      layerCtx.beginPath();
      layerCtx.moveTo(point.x, point.y);
    } else if (tool === "blur") {
      applyBlur(layerCtx, point.x, point.y, brushSize);
      composeLayers();
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const container = containerRef.current;
    if (!container) return;

    if (isPanning) {
      setPanOffset({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
      return;
    }

    const point = getCanvasPoint(e);

    // Store screen coordinates for cursor preview (relative to container)
    const containerRect = container.getBoundingClientRect();
    setCursorPos({
      x: e.clientX - containerRect.left,
      y: e.clientY - containerRect.top
    });

    if (isDrawing || isTapering) {
      const lastPoint = lastPointRef.current;
      if (!lastPoint) return;

      // Get active layer
      const activeLayer = layers.find(l => l.id === activeLayerId);
      if (!activeLayer || !activeLayer.editable) return;

      const layerCanvas = getLayerCanvas(activeLayerId);
      const layerCtx = layerCanvas?.getContext("2d");
      if (!layerCanvas || !layerCtx) return;

      const now = Date.now();
      const pressure = e.pressure > 0 ? e.pressure : 0.5;

      // Check for low pressure to trigger exit tapering (for pen tablets)
      const isLowPressure = e.pressure > 0 && e.pressure < 0.15;

      // Calculate velocity (pixels per millisecond)
      const dx = point.x - lastPoint.x;
      const dy = point.y - lastPoint.y;
      const distance = Math.hypot(dx, dy);
      const timeDelta = now - lastPoint.time;
      const velocity = timeDelta > 0 ? distance / timeDelta : 0;

      // Update stroke distance
      strokeDistanceRef.current += distance;

      if (tool === "pen") {
        // Gradually increase taper when in tapering mode or pressure is low
        if (isTapering || (isLowPressure && brushType !== "normal")) {
          taperProgressRef.current = Math.min(1, taperProgressRef.current + 0.08);
        }

        // Calculate current pressure (reduce during tapering)
        const currentPressure = isTapering ? Math.max(0.1, pressure * (1 - taperProgressRef.current)) : pressure;

        // Store point in stroke path
        strokePathRef.current.push({
          x: point.x,
          y: point.y,
          pressure: currentPressure,
          velocity: velocity,
          distance: strokeDistanceRef.current,
          taperProgress: taperProgressRef.current
        });

        // Draw preview with full opacity (alpha will be applied at end)
        drawWithBrush(
          layerCtx,
          lastPoint.x,
          lastPoint.y,
          point.x,
          point.y,
          brushSize,
          getColorFromRGBOpaque(),
          currentPressure,
          velocity,
          strokeDistanceRef.current,
          taperProgressRef.current
        );
        composeLayers();

        // If taper is complete, apply alpha and end stroke
        if (taperProgressRef.current >= 1 && brushType !== "normal") {
          applyAlphaToStroke(layerCtx);
          setIsDrawing(false);
          setIsTapering(false);
          saveToHistory(activeLayerId, layerCtx);
          composeLayers();
          return;
        }
      } else if (tool === "eraser") {
        if (isDrawing) {
          layerCtx.lineWidth = brushSize * pressure;
          layerCtx.lineTo(point.x, point.y);
          layerCtx.stroke();
          composeLayers();
        }
      } else if (tool === "blur") {
        if (isDrawing) {
          applyBlur(layerCtx, point.x, point.y, brushSize);
          composeLayers();
        }
      }

      // Update points for tracking direction
      prevPointRef.current = lastPoint;
      lastPointRef.current = { x: point.x, y: point.y, time: now };
    }
  };

  const handlePointerUp = (e?: React.PointerEvent<HTMLCanvasElement>) => {
    if (isPanning) {
      setIsPanning(false);
    }

    if (isDrawing) {
      // Get active layer
      const activeLayer = layers.find(l => l.id === activeLayerId);
      if (!activeLayer || !activeLayer.editable) return;

      const layerCanvas = getLayerCanvas(activeLayerId);
      const layerCtx = layerCanvas?.getContext("2d");
      if (!layerCanvas || !layerCtx) return;

      // For non-normal brushes, enter tapering mode instead of ending immediately
      if (tool === "pen" && brushType !== "normal") {
        setIsDrawing(false);
        setIsTapering(true);
        // Don't save to history yet - will save when tapering completes
      } else {
        // Normal pen or other tools: apply alpha and end immediately
        if (tool === "pen") {
          applyAlphaToStroke(layerCtx);
          composeLayers();
        }
        setIsDrawing(false);
        saveToHistory(activeLayerId, layerCtx);
      }
    }
  };

  const handlePointerLeave = () => {
    setCursorPos(null);

    // If in tapering mode, complete the taper immediately
    if (isTapering) {
      const activeLayer = layers.find(l => l.id === activeLayerId);
      if (activeLayer && activeLayer.editable) {
        const layerCanvas = getLayerCanvas(activeLayerId);
        const layerCtx = layerCanvas?.getContext("2d");
        if (layerCanvas && layerCtx) {
          // Apply alpha to completed stroke
          applyAlphaToStroke(layerCtx);
          setIsTapering(false);
          saveToHistory(activeLayerId, layerCtx);
          composeLayers();
        }
      }
    } else {
      handlePointerUp();
    }
  };

  const handleSave = () => {
    const composite = compositeCanvasRef.current;
    if (!composite) return;

    // In inpaint mode, save mask separately if onSaveMask is provided
    if (mode === "inpaint" && onSaveMask) {
      const maskCanvas = layerCanvasRefs.current.get("mask");
      if (maskCanvas) {
        maskCanvas.toBlob((blob) => {
          if (!blob) return;
          const url = URL.createObjectURL(blob);
          onSaveMask(url);
        }, "image/png");
      }
    }

    // Save the composite (merged layers excluding mask)
    // Create temp composite without mask layer
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = composite.width;
    tempCanvas.height = composite.height;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    // Draw all layers except mask
    for (const layer of layers) {
      if (layer.id === "mask" || !layer.visible) continue;
      const layerCanvas = getLayerCanvas(layer.id);
      if (!layerCanvas) continue;
      tempCtx.globalAlpha = layer.opacity;
      tempCtx.drawImage(layerCanvas, 0, 0);
      tempCtx.globalAlpha = 1;
    }

    tempCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      onSave(url);
    }, "image/png");
  };

  // Layer management functions
  const addLayer = () => {
    const editableLayers = layers.filter(l => l.editable);
    if (editableLayers.length >= 3) {
      alert("Maximum 3 editable layers allowed");
      return;
    }

    // Find the smallest available layer number (1, 2, or 3)
    const existingNumbers = layers
      .filter(l => l.id.startsWith('layer'))
      .map(l => parseInt(l.id.replace('layer', '')))
      .filter(n => !isNaN(n));

    let layerNumber = 1;
    while (existingNumbers.includes(layerNumber) && layerNumber <= 3) {
      layerNumber++;
    }

    const newLayerId = `layer${layerNumber}`;
    const newLayer: LayerInfo = {
      id: newLayerId,
      name: `Layer ${layerNumber}`,
      visible: true,
      opacity: 1,
      editable: true,
      deletable: layerNumber > 1, // Layer 1 is not deletable
    };

    setLayers(prev => [...prev, newLayer]);
    setActiveLayerId(newLayerId);
  };

  const deleteLayer = (layerId: string) => {
    const layer = layers.find(l => l.id === layerId);
    if (!layer || !layer.deletable) return;

    if (!confirm(`Delete ${layer.name}?`)) return;

    // Remove layer from state
    setLayers(prev => prev.filter(l => l.id !== layerId));

    // Remove canvas from map
    layerCanvasRefs.current.delete(layerId);

    // If deleting active layer, switch to another editable layer
    if (activeLayerId === layerId) {
      const remainingEditableLayers = layers.filter(l => l.editable && l.id !== layerId);
      if (remainingEditableLayers.length > 0) {
        setActiveLayerId(remainingEditableLayers[0].id);
      }
    }
  };

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();

    const container = containerRef.current;
    if (!container) return;

    if (e.ctrlKey || e.metaKey) {
      // Ctrl + Wheel: Zoom centered on cursor position
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.1, Math.min(10, zoom * delta));

      // Get cursor position relative to container
      const rect = container.getBoundingClientRect();
      const cursorX = e.clientX - rect.left;
      const cursorY = e.clientY - rect.top;

      // Calculate the point under cursor in canvas coordinates (before zoom)
      const canvasX = (cursorX - panOffset.x) / zoom;
      const canvasY = (cursorY - panOffset.y) / zoom;

      // Calculate new pan offset to keep the same point under cursor
      const newPanX = cursorX - canvasX * newZoom;
      const newPanY = cursorY - canvasY * newZoom;

      setZoom(newZoom);
      setPanOffset({ x: newPanX, y: newPanY });
    } else if (e.shiftKey) {
      // Shift + Wheel: Rotate
      const delta = e.deltaY > 0 ? -1 : 1;
      setRotation((prev) => (prev + delta + 360) % 360);
    } else {
      // Wheel: Adjust brush size (larger steps for faster adjustment)
      const delta = e.deltaY > 0 ? -3 : 3;
      setBrushSize((prev) => Math.max(1, Math.min(256, prev + delta)));
    }
  }, [zoom, panOffset]);

  const resetViewTransform = () => {
    const composite = compositeCanvasRef.current;
    const container = containerRef.current;
    if (!composite || !container) return;

    // Reset zoom to fit
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const scaleX = containerWidth / composite.width;
    const scaleY = containerHeight / composite.height;
    const initialZoom = Math.min(scaleX, scaleY, 1);
    setZoom(initialZoom);

    // Reset rotation
    setRotation(0);

    // Center the image
    const displayWidth = composite.width * initialZoom;
    const displayHeight = composite.height * initialZoom;
    setPanOffset({
      x: (containerWidth - displayWidth) / 2,
      y: (containerHeight - displayHeight) / 2,
    });
  };

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Pan tool with spacebar
    if (e.code === "Space" && tool !== "pan") {
      e.preventDefault();
      setTool("pan");
    }

    if (e.ctrlKey || e.metaKey) {
      // Ctrl+0: Reset view transform
      if (e.key === "0") {
        e.preventDefault();
        resetViewTransform();
      }
      // Undo/Redo
      else if (e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if (e.key === "z" && e.shiftKey || e.key === "y") {
        e.preventDefault();
        redo();
      }
    }
  }, [undo, redo, tool]);

  const handleKeyUp = useCallback((e: KeyboardEvent) => {
    // Release pan tool when spacebar is released
    if (e.code === "Space" && tool === "pan") {
      e.preventDefault();
      setTool("pen");
    }
  }, [tool]);

  useEffect(() => {
    const container = containerRef.current;
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    if (container) {
      container.addEventListener("wheel", handleWheel, { passive: false });
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      if (container) {
        container.removeEventListener("wheel", handleWheel);
      }
    };
  }, [handleKeyDown, handleKeyUp, handleWheel]);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 z-50 flex">
      {/* Left Toolbox */}
      <div
        className="bg-gray-900 w-80 h-screen overflow-y-auto p-4 space-y-4"
        onWheel={(e) => {
          const element = e.currentTarget;
          const atTop = element.scrollTop === 0;
          const atBottom = element.scrollTop + element.clientHeight >= element.scrollHeight - 1;

          // If scrolling down at bottom, or scrolling up at top, prevent default
          if ((e.deltaY > 0 && atBottom) || (e.deltaY < 0 && atTop)) {
            e.preventDefault();
          }
        }}
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-white">Image Editor</h2>
          <Button onClick={onClose} variant="secondary" size="sm">
            ×
          </Button>
        </div>

        {/* Tools */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-300">Tools</h3>
          <div className="flex gap-1">
            <button
              onClick={() => setTool("pen")}
              className={`flex-1 py-2 px-1 text-xs rounded transition-colors ${
                tool === "pen" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-300 hover:bg-gray-700"
              }`}
              title="Pen"
            >
              ✏️
            </button>
            <button
              onClick={() => setTool("eraser")}
              className={`flex-1 py-2 px-1 text-xs rounded transition-colors ${
                tool === "eraser" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-300 hover:bg-gray-700"
              }`}
              title="Eraser"
            >
              🧹
            </button>
            <button
              onClick={() => setTool("blur")}
              className={`flex-1 py-2 px-1 text-xs rounded transition-colors ${
                tool === "blur" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-300 hover:bg-gray-700"
              }`}
              title="Blur"
            >
              🌫️
            </button>
            <button
              onClick={() => setTool("bucket")}
              className={`flex-1 py-2 px-1 text-xs rounded transition-colors ${
                tool === "bucket" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-300 hover:bg-gray-700"
              }`}
              title="Bucket Fill"
            >
              🪣
            </button>
            <button
              onClick={() => setTool("eyedropper")}
              className={`flex-1 py-2 px-1 text-xs rounded transition-colors ${
                tool === "eyedropper" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-300 hover:bg-gray-700"
              }`}
              title="Eyedropper"
            >
              💧
            </button>
            <button
              onClick={() => setTool("pan")}
              className={`flex-1 py-2 px-1 text-xs rounded transition-colors ${
                tool === "pan" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-300 hover:bg-gray-700"
              }`}
              title="Pan (Space)"
            >
              ✋
            </button>
          </div>
        </div>

        {/* Brush Type - only show when pen tool is selected */}
        {tool === "pen" && (
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-300">Brush Type</h3>
            <div className="grid grid-cols-2 gap-2">
              <Button
                onClick={() => setBrushType("normal")}
                variant={brushType === "normal" ? "primary" : "secondary"}
                size="sm"
              >
                🖊️ Normal
              </Button>
              <Button
                onClick={() => setBrushType("pencil")}
                variant={brushType === "pencil" ? "primary" : "secondary"}
                size="sm"
              >
                ✏️ Pencil
              </Button>
              <Button
                onClick={() => setBrushType("gpen")}
                variant={brushType === "gpen" ? "primary" : "secondary"}
                size="sm"
              >
                🖋️ G-Pen
              </Button>
              <Button
                onClick={() => setBrushType("fude")}
                variant={brushType === "fude" ? "primary" : "secondary"}
                size="sm"
              >
                🖌️ Fude
              </Button>
            </div>
          </div>
        )}

        {/* Brush Size */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-300">Brush Size</h3>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="1"
              max="256"
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value))}
              onWheel={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const delta = e.deltaY < 0 ? 1 : -1;
                setBrushSize(Math.max(1, Math.min(256, brushSize + delta)));
              }}
              className="flex-1"
            />
            <span className="text-sm text-gray-300 w-12">{brushSize}</span>
          </div>
        </div>

        {/* Color Picker */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-300">Color</h3>
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded border-2 border-gray-600" style={{ backgroundColor: getColorFromRGB() }} />
            <div className="flex-1 space-y-1">
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-400 w-6">R:</label>
                <input
                  type="range"
                  min="0"
                  max="255"
                  value={rgb.r}
                  onChange={(e) => {
                    setRgb({ ...rgb, r: parseInt(e.target.value) });
                  }}
                  onWheel={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const delta = e.deltaY < 0 ? 1 : -1;
                    setRgb({ ...rgb, r: Math.max(0, Math.min(255, rgb.r + delta)) });
                  }}
                  className="flex-1"
                />
                <span className="text-xs text-gray-300 w-8">{rgb.r}</span>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-400 w-6">G:</label>
                <input
                  type="range"
                  min="0"
                  max="255"
                  value={rgb.g}
                  onChange={(e) => {
                    setRgb({ ...rgb, g: parseInt(e.target.value) });
                  }}
                  onWheel={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const delta = e.deltaY < 0 ? 1 : -1;
                    setRgb({ ...rgb, g: Math.max(0, Math.min(255, rgb.g + delta)) });
                  }}
                  className="flex-1"
                />
                <span className="text-xs text-gray-300 w-8">{rgb.g}</span>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-400 w-6">B:</label>
                <input
                  type="range"
                  min="0"
                  max="255"
                  value={rgb.b}
                  onChange={(e) => {
                    setRgb({ ...rgb, b: parseInt(e.target.value) });
                  }}
                  onWheel={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const delta = e.deltaY < 0 ? 1 : -1;
                    setRgb({ ...rgb, b: Math.max(0, Math.min(255, rgb.b + delta)) });
                  }}
                  className="flex-1"
                />
                <span className="text-xs text-gray-300 w-8">{rgb.b}</span>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-400 w-6">A:</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={Math.round(alpha * 100)}
                  onChange={(e) => {
                    setAlpha(parseInt(e.target.value) / 100);
                  }}
                  onWheel={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const delta = e.deltaY < 0 ? 0.01 : -0.01;
                    setAlpha(Math.max(0, Math.min(1, alpha + delta)));
                  }}
                  className="flex-1"
                />
                <span className="text-xs text-gray-300 w-8">{Math.round(alpha * 100)}%</span>
              </div>
            </div>
          </div>

          {/* Gradient Palette */}
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-400 w-8">H:</label>
              <input
                type="range"
                min="0"
                max="360"
                value={hue}
                onChange={(e) => setHue(parseInt(e.target.value))}
                onWheel={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  const delta = e.deltaY < 0 ? 1 : -1;
                  setHue(Math.max(0, Math.min(360, hue + delta)));
                }}
                className="flex-1 h-2 rounded-lg appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none
                  [&::-webkit-slider-thumb]:w-4
                  [&::-webkit-slider-thumb]:h-4
                  [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-white
                  [&::-webkit-slider-thumb]:border-2
                  [&::-webkit-slider-thumb]:border-gray-800
                  [&::-webkit-slider-thumb]:cursor-pointer
                  [&::-moz-range-thumb]:w-4
                  [&::-moz-range-thumb]:h-4
                  [&::-moz-range-thumb]:rounded-full
                  [&::-moz-range-thumb]:bg-white
                  [&::-moz-range-thumb]:border-2
                  [&::-moz-range-thumb]:border-gray-800
                  [&::-moz-range-thumb]:cursor-pointer"
                style={{
                  background: `linear-gradient(to right,
                    hsl(0, 100%, 50%),
                    hsl(60, 100%, 50%),
                    hsl(120, 100%, 50%),
                    hsl(180, 100%, 50%),
                    hsl(240, 100%, 50%),
                    hsl(300, 100%, 50%),
                    hsl(360, 100%, 50%))`
                }}
              />
              <span className="text-xs text-gray-300 w-8">{hue}</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-400 w-8">S:</label>
              <input
                type="range"
                min="0"
                max="100"
                value={saturation}
                onChange={(e) => setSaturation(parseInt(e.target.value))}
                onWheel={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  const delta = e.deltaY < 0 ? 1 : -1;
                  setSaturation(Math.max(0, Math.min(100, saturation + delta)));
                }}
                className="flex-1 h-2 rounded-lg appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none
                  [&::-webkit-slider-thumb]:w-4
                  [&::-webkit-slider-thumb]:h-4
                  [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-white
                  [&::-webkit-slider-thumb]:border-2
                  [&::-webkit-slider-thumb]:border-gray-800
                  [&::-webkit-slider-thumb]:cursor-pointer
                  [&::-moz-range-thumb]:w-4
                  [&::-moz-range-thumb]:h-4
                  [&::-moz-range-thumb]:rounded-full
                  [&::-moz-range-thumb]:bg-white
                  [&::-moz-range-thumb]:border-2
                  [&::-moz-range-thumb]:border-gray-800
                  [&::-moz-range-thumb]:cursor-pointer"
                style={{
                  background: `linear-gradient(to right,
                    hsl(${hue}, 0%, ${lightness}%),
                    hsl(${hue}, 100%, ${lightness}%))`
                }}
              />
              <span className="text-xs text-gray-300 w-8">{saturation}</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-400 w-8">L:</label>
              <input
                type="range"
                min="0"
                max="100"
                value={lightness}
                onChange={(e) => setLightness(parseInt(e.target.value))}
                onWheel={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  const delta = e.deltaY < 0 ? 1 : -1;
                  setLightness(Math.max(0, Math.min(100, lightness + delta)));
                }}
                className="flex-1 h-2 rounded-lg appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none
                  [&::-webkit-slider-thumb]:w-4
                  [&::-webkit-slider-thumb]:h-4
                  [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-white
                  [&::-webkit-slider-thumb]:border-2
                  [&::-webkit-slider-thumb]:border-gray-800
                  [&::-webkit-slider-thumb]:cursor-pointer
                  [&::-moz-range-thumb]:w-4
                  [&::-moz-range-thumb]:h-4
                  [&::-moz-range-thumb]:rounded-full
                  [&::-moz-range-thumb]:bg-white
                  [&::-moz-range-thumb]:border-2
                  [&::-moz-range-thumb]:border-gray-800
                  [&::-moz-range-thumb]:cursor-pointer"
                style={{
                  background: `linear-gradient(to right,
                    hsl(${hue}, ${saturation}%, 0%),
                    hsl(${hue}, ${saturation}%, 50%),
                    hsl(${hue}, ${saturation}%, 100%))`
                }}
              />
              <span className="text-xs text-gray-300 w-8">{lightness}</span>
            </div>
          </div>
        </div>

        {/* View Transform */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-semibold text-gray-300">View</h3>
            <Button onClick={resetViewTransform} variant="secondary" size="sm" className="text-xs">
              Reset (Ctrl+0)
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-300 w-12">Zoom:</label>
            <input
              type="range"
              min="0.1"
              max="10"
              step="0.1"
              value={zoom}
              onChange={(e) => setZoom(parseFloat(e.target.value))}
              onWheel={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const delta = e.deltaY < 0 ? 0.1 : -0.1;
                setZoom(Math.max(0.1, Math.min(10, zoom + delta)));
              }}
              className="flex-1"
            />
            <span className="text-sm text-gray-300 w-12">{zoom.toFixed(1)}x</span>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-300 w-12">Rotate:</label>
            <input
              type="range"
              min="0"
              max="360"
              step="1"
              value={rotation}
              onChange={(e) => setRotation(parseInt(e.target.value))}
              onWheel={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const delta = e.deltaY < 0 ? 1 : -1;
                setRotation((prev) => (prev + delta + 360) % 360);
              }}
              className="flex-1"
            />
            <span className="text-sm text-gray-300 w-12">{rotation}°</span>
          </div>
        </div>

        {/* Undo/Redo */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-300">History</h3>
          <div className="flex gap-2">
            <Button
              onClick={undo}
              disabled={historyIndex <= 0}
              variant="secondary"
              size="sm"
              className="flex-1"
            >
              ↶ Undo
            </Button>
            <Button
              onClick={redo}
              disabled={historyIndex >= history.length - 1}
              variant="secondary"
              size="sm"
              className="flex-1"
            >
              ↷ Redo
            </Button>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-4 border-t border-gray-700">
          <Button onClick={onClose} variant="secondary" className="flex-1">
            Cancel
          </Button>
          <Button onClick={handleSave} variant="primary" className="flex-1">
            Save & Use
          </Button>
        </div>
      </div>

      {/* Canvas Area */}
      <div
        ref={containerRef}
        className="flex-1 bg-gray-800 relative overflow-hidden"
      >
        {/* Hidden base layer canvas */}
        <canvas ref={baseLayerRef} className="hidden" />
        {/* Editable layer canvases are created dynamically in memory */}

        <div
          className="absolute"
          style={{
            transform: `translate(${panOffset.x}px, ${panOffset.y}px)`,
          }}
        >
          <div
            style={{
              transform: `rotate(${rotation}deg)`,
              transformOrigin: compositeCanvasRef.current ? `${compositeCanvasRef.current.width * zoom / 2}px ${compositeCanvasRef.current.height * zoom / 2}px` : 'center',
            }}
          >
            <canvas
              ref={compositeCanvasRef}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerLeave}
              className={tool === "pan" ? "cursor-grab" : "cursor-none"}
              style={{
                imageRendering: "pixelated",
                width: compositeCanvasRef.current ? `${compositeCanvasRef.current.width * zoom}px` : undefined,
                height: compositeCanvasRef.current ? `${compositeCanvasRef.current.height * zoom}px` : undefined,
                touchAction: "none", // Disable default touch behaviors
              }}
            />
          </div>
        </div>
        {/* Brush Preview Cursor - positioned relative to container */}
        {cursorPos && compositeCanvasRef.current && tool !== "pan" && (() => {
          const canvas = compositeCanvasRef.current;
          // brushSize is used as lineWidth, which represents diameter
          // With average pressure of 0.5, actual size is brushSize * 0.5
          const scaledSize = brushSize * 0.5 * zoom;

          return (
            <div
              className="absolute pointer-events-none rounded-full border-2"
              style={{
                left: `${cursorPos.x}px`,
                top: `${cursorPos.y}px`,
                width: `${scaledSize}px`,
                height: `${scaledSize}px`,
                transform: 'translate(-50%, -50%)',
                borderColor: tool === "eraser" ? "#ffffff" : getColorFromRGB(),
                backgroundColor: tool === "eraser" ? "rgba(255,255,255,0.2)" : `${getColorFromRGB()}33`,
              }}
            />
          );
        })()}

        {/* Layer Panel - Bottom Right */}
        <div className="absolute bottom-4 right-4 bg-gray-900 bg-opacity-95 rounded-lg p-3 min-w-[200px] max-w-[300px]">
          <div className="flex justify-between items-center mb-2">
            <div className="text-xs font-semibold text-gray-300">Layers</div>
            <button
              onClick={addLayer}
              className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white transition-colors"
              title="Add new layer (max 3)"
            >
              + Add
            </button>
          </div>
          <div className="space-y-1">
            {[...layers].reverse().map((layer) => (
              <div
                key={layer.id}
                className={`flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
                  activeLayerId === layer.id
                    ? 'bg-blue-600 bg-opacity-50'
                    : 'bg-gray-800 hover:bg-gray-700'
                }`}
                onClick={() => layer.editable && setActiveLayerId(layer.id)}
              >
                {/* Visibility Toggle */}
                <button
                  className="w-4 h-4 flex items-center justify-center text-gray-400 hover:text-white"
                  onClick={(e) => {
                    e.stopPropagation();
                    setLayers(prev => prev.map(l =>
                      l.id === layer.id ? { ...l, visible: !l.visible } : l
                    ));
                  }}
                >
                  {layer.visible ? '👁️' : '👁️‍🗨️'}
                </button>

                {/* Layer Thumbnail */}
                <div className="w-8 h-8 bg-gray-700 rounded flex items-center justify-center text-xs">
                  {layer.id === 'base' ? '🖼️' : '📝'}
                </div>

                {/* Layer Info */}
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-white truncate">{layer.name}</div>
                  {!layer.editable && (
                    <div className="text-[10px] text-gray-500">Locked</div>
                  )}
                </div>

                {/* Active Indicator */}
                {activeLayerId === layer.id && layer.editable && (
                  <div className="w-2 h-2 bg-blue-400 rounded-full" />
                )}

                {/* Layer Action Buttons */}
                {layer.editable && (
                  <div className="flex gap-1">
                    {/* Clear Layer Button */}
                    <button
                      className="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-yellow-400 transition-colors"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Clear all content on ${layer.name}?`)) {
                          const canvas = getLayerCanvas(layer.id);
                          const ctx = canvas?.getContext("2d");
                          if (canvas && ctx) {
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            composeLayers();
                            // Save to history
                            saveToHistory(layer.id, ctx);
                          }
                        }
                      }}
                      title="Clear layer"
                    >
                      🧹
                    </button>

                    {/* Delete Layer Button (only for deletable layers) */}
                    {layer.deletable && (
                      <button
                        className="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-red-400 transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteLayer(layer.id);
                        }}
                        title="Delete layer"
                      >
                        ❌
                      </button>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
