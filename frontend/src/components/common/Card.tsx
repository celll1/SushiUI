import { ReactNode, useState, useEffect } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

interface CardProps {
  title?: string;
  children?: ReactNode;
  className?: string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  collapsedPreview?: ReactNode;
  storageKey?: string;
}

export default function Card({
  title,
  children,
  className = "",
  collapsible = false,
  defaultCollapsed = false,
  collapsedPreview,
  storageKey
}: CardProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [isMounted, setIsMounted] = useState(false);

  // Load collapsed state from localStorage
  useEffect(() => {
    setIsMounted(true);
    if (storageKey && typeof window !== "undefined") {
      const saved = localStorage.getItem(storageKey);
      if (saved !== null) {
        setCollapsed(saved === "true");
      }
    }
  }, [storageKey]);

  // Save collapsed state to localStorage
  useEffect(() => {
    if (isMounted && storageKey && typeof window !== "undefined") {
      localStorage.setItem(storageKey, collapsed.toString());
    }
  }, [collapsed, storageKey, isMounted]);

  return (
    <section className={`space-y-2 rounded-md border border-gray-800 bg-gray-900 p-3 shadow-sm ${className}`}>
      {title && (
        <div
          className={`flex min-h-5 items-center justify-between gap-2 ${collapsible ? 'cursor-pointer select-none' : ''}`}
          onClick={() => collapsible && setCollapsed(!collapsed)}
        >
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.1em] text-gray-400">{title}</h3>
          {collapsible && (
            collapsed ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronUp className="w-4 h-4 text-gray-400" />
          )}
        </div>
      )}
      {collapsed && collapsedPreview}
      {!collapsed && children}
    </section>
  );
}
