import { SelectHTMLAttributes } from "react";

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: Array<{ value: string; label: string }>;
}

export default function Select({
  label,
  options,
  className = "",
  ...props
}: SelectProps) {
  return (
    <div className={className}>
      {label && (
        <label className="mb-1 block text-xs font-medium text-gray-400">
          {label}
        </label>
      )}
      <select
        className="h-8 w-full rounded-md border border-gray-700 bg-gray-800 px-2.5 text-sm text-white
          focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500
          disabled:opacity-50 disabled:cursor-not-allowed"
        {...props}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}
