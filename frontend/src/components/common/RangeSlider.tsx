"use client";

import { useState, useEffect, useRef } from "react";

interface RangeSliderProps {
  min?: number;
  max?: number;
  step?: number;
  value?: [number, number];
  onChange?: (value: [number, number]) => void;
  onCommit?: (value: [number, number]) => void;
  label?: string;
  disabled?: boolean;
  className?: string;
}

export default function RangeSlider({
  min = 0,
  max = 100,
  step = 1,
  value = [0, 100],
  onChange,
  onCommit,
  label,
  disabled = false,
  className = "",
}: RangeSliderProps) {
  const [minValue, setMinValue] = useState(value[0]);
  const [maxValue, setMaxValue] = useState(value[1]);
  const minRef = useRef<HTMLInputElement>(null);
  const maxRef = useRef<HTMLInputElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMinValue(value[0]);
    setMaxValue(value[1]);
  }, [value]);

  const handleMinChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newMin = Math.min(Number(e.target.value), maxValue - step);
    setMinValue(newMin);
    onChange?.([newMin, maxValue]);
  };

  const handleMaxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newMax = Math.max(Number(e.target.value), minValue + step);
    setMaxValue(newMax);
    onChange?.([minValue, newMax]);
  };

  const handleCommit = () => {
    onCommit?.([minValue, maxValue]);
  };

  const getPercentage = (value: number) => {
    return ((value - min) / (max - min)) * 100;
  };

  return (
    <div className={`space-y-2 ${className}`}>
      {label && (
        <div className="flex justify-between items-center text-sm">
          <label className="text-gray-300">{label}</label>
          <span className="text-gray-400">
            {minValue} - {maxValue}
          </span>
        </div>
      )}
      <div className="relative h-8">
        {/* Track background */}
        <div className="absolute top-1/2 -translate-y-1/2 w-full h-1 bg-gray-700 rounded-full" />

        {/* Active track */}
        <div
          ref={trackRef}
          className="absolute top-1/2 -translate-y-1/2 h-1 bg-blue-500 rounded-full"
          style={{
            left: `${getPercentage(minValue)}%`,
            right: `${100 - getPercentage(maxValue)}%`,
          }}
        />

        {/* Min slider */}
        <input
          ref={minRef}
          type="range"
          min={min}
          max={max}
          step={step}
          value={minValue}
          onChange={handleMinChange}
          onMouseUp={handleCommit}
          onTouchEnd={handleCommit}
          disabled={disabled}
          className="absolute top-1/2 -translate-y-1/2 w-full h-1 appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-blue-500 [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-blue-500"
          style={{ zIndex: minValue > max - (max - min) / 2 ? 5 : 3 }}
        />

        {/* Max slider */}
        <input
          ref={maxRef}
          type="range"
          min={min}
          max={max}
          step={step}
          value={maxValue}
          onChange={handleMaxChange}
          onMouseUp={handleCommit}
          onTouchEnd={handleCommit}
          disabled={disabled}
          className="absolute top-1/2 -translate-y-1/2 w-full h-1 appearance-none bg-transparent pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-blue-500 [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-blue-500"
          style={{ zIndex: 4 }}
        />
      </div>
    </div>
  );
}
