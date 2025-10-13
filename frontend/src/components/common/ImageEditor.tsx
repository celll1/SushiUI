"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Button from "./Button";

interface ImageEditorProps {
  imageUrl: string;
  onSave: (editedImageUrl: string) => void;
  onClose: () => void;
}

type Tool = "pen" | "eraser" | "blur" | "eyedropper" | "pan";

interface HistoryState {
  imageData: ImageData;
}

export default function ImageEditor({ imageUrl, onSave, onClose }: ImageEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tool, setTool] = useState<Tool>("pen");
  const [brushSize, setBrushSize] = useState(5);
  const [rgb, setRgb] = useState({ r: 0, g: 0, b: 0 });
  const [hue, setHue] = useState(0);
  const [saturation, setSaturation] = useState(100);
  const [lightness, setLightness] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [history, setHistory] = useState<HistoryState[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const updatingSourceRef = useRef<'rgb' | 'hsl' | null>(null);

  // Calculate color from RGB
  const getColorFromRGB = () => {
    return `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
  };

  // Calculate color from HSL and lightness
  const getColorFromHSL = () => {
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
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

  // Load image and initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Calculate initial zoom to fit canvas in container
      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;
      const scaleX = containerWidth / img.width;
      const scaleY = containerHeight / img.height;
      const initialZoom = Math.min(scaleX, scaleY, 1);
      setZoom(initialZoom);

      // Center the image
      const displayWidth = img.width * initialZoom;
      const displayHeight = img.height * initialZoom;
      setPanOffset({
        x: (containerWidth - displayWidth) / 2,
        y: (containerHeight - displayHeight) / 2,
      });

      // Save initial state
      saveToHistory(ctx);
    };
    img.src = imageUrl;
  }, [imageUrl]);

  const saveToHistory = useCallback((ctx: CanvasRenderingContext2D) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Remove any history after current index
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({ imageData });

    // Limit history to 50 states
    if (newHistory.length > 50) {
      newHistory.shift();
    } else {
      setHistoryIndex(historyIndex + 1);
    }

    setHistory(newHistory);
  }, [history, historyIndex]);

  const undo = useCallback(() => {
    if (historyIndex <= 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const newIndex = historyIndex - 1;
    setHistoryIndex(newIndex);
    ctx.putImageData(history[newIndex].imageData, 0, 0);
  }, [history, historyIndex]);

  const redo = useCallback(() => {
    if (historyIndex >= history.length - 1) return;

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const newIndex = historyIndex + 1;
    setHistoryIndex(newIndex);
    ctx.putImageData(history[newIndex].imageData, 0, 0);
  }, [history, historyIndex]);

  const getCanvasPoint = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return { x: 0, y: 0 };

    const containerRect = container.getBoundingClientRect();

    // Screen coordinates relative to container
    const screenX = e.clientX - containerRect.left - panOffset.x;
    const screenY = e.clientY - containerRect.top - panOffset.y;

    // Center of the rotated canvas
    const centerX = (canvas.width * zoom) / 2;
    const centerY = (canvas.height * zoom) / 2;

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

  const applyBlur = (ctx: CanvasRenderingContext2D, x: number, y: number, radius: number) => {
    const imageData = ctx.getImageData(
      Math.max(0, x - radius),
      Math.max(0, y - radius),
      radius * 2,
      radius * 2
    );

    // Simple box blur
    const pixels = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const blurred = new Uint8ClampedArray(pixels);

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;

        // Average surrounding pixels
        let r = 0, g = 0, b = 0, count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nidx = ((y + dy) * width + (x + dx)) * 4;
            r += pixels[nidx];
            g += pixels[nidx + 1];
            b += pixels[nidx + 2];
            count++;
          }
        }

        blurred[idx] = r / count;
        blurred[idx + 1] = g / count;
        blurred[idx + 2] = b / count;
      }
    }

    const blurredImageData = new ImageData(blurred, width, height);
    ctx.putImageData(blurredImageData, Math.max(0, x - radius), Math.max(0, y - radius));
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const point = getCanvasPoint(e);

    if (tool === "pan") {
      setIsPanning(true);
      setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
      return;
    }

    if (tool === "eyedropper") {
      // Pick color from canvas
      const x = Math.floor(point.x);
      const y = Math.floor(point.y);

      // Ensure coordinates are within canvas bounds
      if (x >= 0 && x < canvas.width && y >= 0 && y < canvas.height) {
        const imageData = ctx.getImageData(x, y, 1, 1);
        const pixel = imageData.data;
        setRgb({ r: pixel[0], g: pixel[1], b: pixel[2] });
      }

      // Switch back to pen tool after picking
      setTool("pen");
      return;
    }

    setIsDrawing(true);

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brushSize;

    if (tool === "pen") {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = getColorFromRGB();
      ctx.beginPath();
      ctx.moveTo(point.x, point.y);
    } else if (tool === "eraser") {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "rgba(0,0,0,1)"; // Eraser needs a stroke style
      ctx.beginPath();
      ctx.moveTo(point.x, point.y);
    } else if (tool === "blur") {
      applyBlur(ctx, point.x, point.y, brushSize);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !container || !ctx) return;

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

    if (isDrawing) {
      if (tool === "pen" || tool === "eraser") {
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
      } else if (tool === "blur") {
        applyBlur(ctx, point.x, point.y, brushSize);
      }
    }
  };

  const handleMouseUp = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    if (isPanning) {
      setIsPanning(false);
    }

    if (isDrawing) {
      setIsDrawing(false);
      saveToHistory(ctx);
    }
  };

  const handleMouseLeave = () => {
    setCursorPos(null);
    handleMouseUp();
  };

  const handleSave = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      onSave(url);
    }, "image/png");
  };

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();

    if (e.shiftKey) {
      // Shift + Wheel: Rotate
      const delta = e.deltaY > 0 ? -1 : 1;
      setRotation((prev) => (prev + delta + 360) % 360);
    } else {
      // Wheel: Zoom
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.1, Math.min(10, zoom * delta));
      setZoom(newZoom);
    }
  }, [zoom]);

  const resetViewTransform = () => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    // Reset zoom to fit
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const scaleX = containerWidth / canvas.width;
    const scaleY = containerHeight / canvas.height;
    const initialZoom = Math.min(scaleX, scaleY, 1);
    setZoom(initialZoom);

    // Reset rotation
    setRotation(0);

    // Center the image
    const displayWidth = canvas.width * initialZoom;
    const displayHeight = canvas.height * initialZoom;
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
          <div className="grid grid-cols-2 gap-2">
            <Button
              onClick={() => setTool("pen")}
              variant={tool === "pen" ? "primary" : "secondary"}
              size="sm"
            >
              ✏️ Pen
            </Button>
            <Button
              onClick={() => setTool("eraser")}
              variant={tool === "eraser" ? "primary" : "secondary"}
              size="sm"
            >
              🧹 Eraser
            </Button>
            <Button
              onClick={() => setTool("blur")}
              variant={tool === "blur" ? "primary" : "secondary"}
              size="sm"
            >
              🌫️ Blur
            </Button>
            <Button
              onClick={() => setTool("eyedropper")}
              variant={tool === "eyedropper" ? "primary" : "secondary"}
              size="sm"
            >
              💧 Eyedropper
            </Button>
            <Button
              onClick={() => setTool("pan")}
              variant={tool === "pan" ? "primary" : "secondary"}
              size="sm"
              className="col-span-2"
            >
              ✋ Pan (Space)
            </Button>
          </div>
        </div>

        {/* Brush Size */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-300">Brush Size</h3>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="1"
              max="50"
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value))}
              onWheel={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const delta = e.deltaY < 0 ? 1 : -1;
                setBrushSize(Math.max(1, Math.min(50, brushSize + delta)));
              }}
              className="flex-1"
            />
            <span className="text-sm text-gray-300 w-8">{brushSize}</span>
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
        <div
          className="absolute"
          style={{
            transform: `translate(${panOffset.x}px, ${panOffset.y}px)`,
          }}
        >
          <div
            style={{
              transform: `rotate(${rotation}deg)`,
              transformOrigin: canvasRef.current ? `${canvasRef.current.width * zoom / 2}px ${canvasRef.current.height * zoom / 2}px` : 'center',
            }}
          >
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseLeave}
              className={tool === "pan" ? "cursor-grab" : "cursor-none"}
              style={{
                imageRendering: "pixelated",
                width: canvasRef.current ? `${canvasRef.current.width * zoom}px` : undefined,
                height: canvasRef.current ? `${canvasRef.current.height * zoom}px` : undefined,
              }}
            />
          </div>
        </div>
        {/* Brush Preview Cursor - positioned relative to container */}
        {cursorPos && canvasRef.current && tool !== "pan" && (() => {
          const canvas = canvasRef.current;
          const scaledSize = brushSize * zoom;

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
      </div>
    </div>
  );
}
