import type { ReactNode } from "react";

interface GenerationLeadGridProps {
  prompt: ReactNode;
  conditioning?: ReactNode;
  primaryDetails?: ReactNode;
}

export default function GenerationLeadGrid({
  prompt,
  conditioning,
  primaryDetails,
}: GenerationLeadGridProps) {
  return (
    <div className="generation-lead">
      <div
        className={`grid items-start gap-2 ${
          conditioning ? "generation-lead-has-conditioning" : ""
        }`}
      >
        <div className="generation-lead-prompt min-w-0">{prompt}</div>
        {conditioning && (
          <div className="generation-lead-conditioning min-w-0 space-y-2">
            {conditioning}
          </div>
        )}
        {primaryDetails && (
          <div className="generation-lead-primary-details min-w-0 space-y-2">
            {primaryDetails}
          </div>
        )}
      </div>
    </div>
  );
}
