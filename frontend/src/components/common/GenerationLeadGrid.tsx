import type { ReactNode } from "react";

interface GenerationLeadGridProps {
  prompt: ReactNode;
  conditioning?: ReactNode;
}

export default function GenerationLeadGrid({
  prompt,
  conditioning,
}: GenerationLeadGridProps) {
  return (
    <div className="@container">
      <div className={`grid items-start gap-2.5 ${
        conditioning
          ? "@[700px]:grid-cols-[minmax(300px,0.9fr)_minmax(340px,1.1fr)]"
          : ""
      }`}>
        <div className="min-w-0">{prompt}</div>
        {conditioning && (
          <div className="min-w-0 space-y-2.5">{conditioning}</div>
        )}
      </div>
    </div>
  );
}
