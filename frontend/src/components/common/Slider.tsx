import { InputHTMLAttributes } from "react";

interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export default function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  className = "",
  ...props
}: SliderProps) {
  const handleWheel = (e: React.WheelEvent<HTMLInputElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const delta = e.deltaY < 0 ? step : -step;
    const newValue = Math.max(min, Math.min(max, value + delta));

    // Create synthetic event for onChange
    const syntheticEvent = {
      target: { value: newValue.toString() },
      currentTarget: { value: newValue.toString() }
    } as React.ChangeEvent<HTMLInputElement>;

    onChange(syntheticEvent);
  };

  return (
    <div className={className}>
      {label && (
        <div className="flex justify-between items-center mb-2">
          <label className="block text-sm font-medium text-gray-300">
            {label}
          </label>
          <span className="text-sm text-gray-400 font-mono">{value}</span>
        </div>
      )}
      <div className="flex items-center space-x-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={onChange}
          onWheel={handleWheel}
          className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-blue-600
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:hover:bg-blue-700
            [&::-moz-range-thumb]:w-4
            [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:bg-blue-600
            [&::-moz-range-thumb]:cursor-pointer
            [&::-moz-range-thumb]:hover:bg-blue-700
            [&::-moz-range-thumb]:border-0"
          {...props}
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={onChange}
          onWheel={handleWheel}
          className="w-20 px-2 py-1 text-sm bg-gray-800 border border-gray-700 rounded-md text-white
            focus:outline-none focus:ring-2 focus:ring-blue-500"
          {...props}
        />
      </div>
    </div>
  );
}
