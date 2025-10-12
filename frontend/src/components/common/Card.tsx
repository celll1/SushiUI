import { ReactNode } from "react";

interface CardProps {
  title?: string;
  children?: ReactNode;
  className?: string;
}

export default function Card({ title, children, className = "" }: CardProps) {
  return (
    <section className={`space-y-2 px-4 pb-4 pt-2 bg-gray-900 rounded-lg ${className}`}>
      {title && (
        <h3 className="text-sm font-semibold uppercase text-gray-400">{title}</h3>
      )}
      {children}
    </section>
  );
}
