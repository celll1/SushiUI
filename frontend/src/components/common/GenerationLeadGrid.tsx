import type { CSSProperties, ReactNode } from "react";

interface GenerationLeadGridProps {
  prompt: ReactNode;
  conditioning?: ReactNode;
}

export default function GenerationLeadGrid({
  prompt,
  conditioning,
}: GenerationLeadGridProps) {
  const gridStyle = conditioning
    ? ({
        gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 18rem), 1fr))",
      } satisfies CSSProperties)
    : undefined;

  return (
    <div className="grid items-start gap-2" style={gridStyle}>
        <div className="min-w-0">{prompt}</div>
        {conditioning && (
          <div className="min-w-0 space-y-2">{conditioning}</div>
        )}
    </div>
  );
}
