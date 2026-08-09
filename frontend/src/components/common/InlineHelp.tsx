import { Info } from "lucide-react";
import type { ReactNode } from "react";

interface InlineHelpProps {
  label: string;
  children: ReactNode;
}

export default function InlineHelp({ label, children }: InlineHelpProps) {
  return (
    <details className="group relative inline-flex">
      <summary
        className="flex h-6 w-6 cursor-help list-none items-center justify-center rounded text-gray-500 transition-colors hover:bg-gray-800 hover:text-gray-200 focus:outline-none focus:ring-1 focus:ring-violet-400 [&::-webkit-details-marker]:hidden"
        aria-label={label}
        title={label}
      >
        <Info className="h-4 w-4" aria-hidden="true" />
      </summary>
      <div className="pointer-events-none invisible absolute left-0 top-6 z-50 w-[min(22rem,calc(100vw-2rem))] space-y-1.5 rounded-md border border-gray-700 bg-gray-900 p-2.5 text-left text-xs font-normal leading-4 text-gray-300 opacity-0 shadow-xl transition-opacity group-hover:pointer-events-auto group-hover:visible group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:visible group-focus-within:opacity-100 group-open:pointer-events-auto group-open:visible group-open:opacity-100">
        {children}
      </div>
    </details>
  );
}
