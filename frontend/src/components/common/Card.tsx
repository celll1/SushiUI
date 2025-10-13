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
    <section className={`space-y-2 px-4 pb-4 pt-2 bg-gray-900 rounded-lg ${className}`}>
      {title && (
        <div
          className={`flex items-center justify-between ${collapsible ? 'cursor-pointer' : ''}`}
          onClick={() => collapsible && setCollapsed(!collapsed)}
        >
          <h3 className="text-sm font-semibold uppercase text-gray-400">{title}</h3>
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
