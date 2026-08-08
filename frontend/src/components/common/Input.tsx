import { InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export default function Input({ label, className, ...props }: InputProps) {
  return (
    <div className="space-y-1">
      {label && (
        <label className="block text-xs font-medium text-gray-400">
          {label}
        </label>
      )}
      <input
        className={cn(
          "h-8 w-full rounded-md border border-gray-700 bg-gray-800 px-2.5 text-sm text-gray-100 placeholder-gray-500 focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500",
          className
        )}
        {...props}
      />
    </div>
  );
}
