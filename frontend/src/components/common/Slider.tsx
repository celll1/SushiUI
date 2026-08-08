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
        <div className="mb-1 flex items-center justify-between">
          <label className="block text-xs font-medium text-gray-400">
            {label}
          </label>
          <span className="text-xs font-mono text-gray-400">{value}</span>
        </div>
      )}
      <div className="flex items-center space-x-2">
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
            [&::-webkit-slider-thumb]:bg-violet-500
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:hover:bg-violet-400
            [&::-moz-range-thumb]:w-4
            [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:bg-violet-500
            [&::-moz-range-thumb]:cursor-pointer
            [&::-moz-range-thumb]:hover:bg-violet-400
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
          className="h-7 w-16 rounded-md border border-gray-700 bg-gray-800 px-1 text-xs text-white focus:outline-none focus:ring-1 focus:ring-violet-500 sm:w-20 sm:px-2"
          {...props}
        />
      </div>
    </div>
  );
}
