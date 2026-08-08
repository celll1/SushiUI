import { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: "primary" | "secondary" | "danger";
  size?: "xs" | "sm" | "md" | "lg";
}

export default function Button({
  children,
  variant = "primary",
  size = "md",
  className,
  ...props
}: ButtonProps) {
  const baseStyles = "rounded-md border border-transparent font-medium transition-colors focus:outline-none focus:ring-1 focus:ring-violet-400 disabled:opacity-50 disabled:cursor-not-allowed";

  const variantStyles = {
    primary: "border-violet-400/30 bg-violet-600 hover:bg-violet-500 text-white shadow-[0_0_12px_rgba(124,92,255,0.14)]",
    secondary: "border-gray-700 bg-gray-800 hover:bg-gray-700 text-gray-200",
    danger: "border-red-500/30 bg-red-600 hover:bg-red-500 text-white",
  };

  const sizeStyles = {
    xs: "px-2 py-1 text-xs",
    sm: "px-3 py-1.5 text-sm",
    md: "px-3 py-1.5 text-sm",
    lg: "px-4 py-2 text-base",
  };

  return (
    <button
      className={cn(baseStyles, variantStyles[variant], sizeStyles[size], className)}
      {...props}
    >
      {children}
    </button>
  );
}
