"use client";

import { useGenerationQueue } from "@/contexts/GenerationQueueContext";
import Button from "./Button";
import { useEffect, useState } from "react";

interface LastGenerationInfo {
  width: number;
  height: number;
  steps: number;
  sampler: string;
  elapsedTime: number;
  currentStep: number;
}

interface GenerationQueueProps {
  currentStep?: number;
}

export default function GenerationQueue({ currentStep = 0 }: GenerationQueueProps) {
  const { queue, currentItem, removeFromQueue } = useGenerationQueue();
  const [elapsedTime, setElapsedTime] = useState(0);
  const [lastGenInfo, setLastGenInfo] = useState<LastGenerationInfo | null>(null);

  // Update elapsed time every 100ms when generating
  useEffect(() => {
    if (!currentItem || currentItem.status !== "generating" || !currentItem.startTime) {
      setElapsedTime(0);
      return;
    }

    const interval = setInterval(() => {
      const elapsed = (Date.now() - currentItem.startTime!) / 1000;
      setElapsedTime(elapsed);
    }, 100);

    return () => clearInterval(interval);
  }, [currentItem]);

  // Save last generation info when generation completes
  useEffect(() => {
    if (currentItem && currentItem.status === "generating") {
      // Store current generation info while it's running
      const info = {
        width: currentItem.params.width || 0,
        height: currentItem.params.height || 0,
        steps: currentItem.params.steps || 0,
        sampler: currentItem.params.sampler || "",
        elapsedTime: elapsedTime,
        currentStep: currentStep,
      };
      setLastGenInfo(info);
    }
  }, [currentItem, elapsedTime, currentStep]);

  const pendingItems = queue.filter((item) => item.status === "pending");
  const completedItems = queue.filter((item) => item.status === "completed");
  const failedItems = queue.filter((item) => item.status === "failed");

  return (
    <div className="flex flex-col h-full bg-gray-800/50 border-l border-gray-700">
      {/* Header */}
      <div className="p-3 border-b border-gray-700 flex items-center justify-between flex-shrink-0">
        <h3 className="text-sm font-semibold">Queue</h3>
        {queue.length > 0 && (
          <Button
            onClick={() => {
              queue.forEach((item) => {
                if (item.status === "completed" || item.status === "failed") {
                  removeFromQueue(item.id);
                }
              });
            }}
            variant="secondary"
            size="sm"
            className="text-xs py-1 px-2"
          >
            Clear
          </Button>
        )}
      </div>

      {/* Generation Info */}
      {lastGenInfo && (
        <div className="p-2 border-b border-gray-700 bg-gray-800/80 flex-shrink-0">
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Size:</span>
              <span className="text-gray-200 font-mono">
                {lastGenInfo.width}×{lastGenInfo.height}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Steps:</span>
              <span className="text-gray-200 font-mono">
                {lastGenInfo.steps}
              </span>
            </div>
            <div className="flex items-center justify-between col-span-2">
              <span className="text-gray-400">Sampler:</span>
              <span className="text-gray-200 font-mono">
                {lastGenInfo.sampler}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Time:</span>
              <span className="text-blue-400 font-mono">
                {lastGenInfo.elapsedTime.toFixed(2)}s
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-400">Speed:</span>
              <span className="text-blue-400 font-mono">
                {lastGenInfo.elapsedTime > 0 && (lastGenInfo.currentStep > 0 || lastGenInfo.steps > 0)
                  ? `${(lastGenInfo.elapsedTime / (lastGenInfo.currentStep > 0 ? lastGenInfo.currentStep : lastGenInfo.steps)).toFixed(3)}s/it`
                  : "—"}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Queue Items */}
      <div className="overflow-y-auto flex-1 p-2 space-y-2">
        {/* Current item */}
        {currentItem && (
          <div className="bg-blue-900/30 border border-blue-700 rounded p-2">
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1 mb-1">
                  <span className="text-xs font-semibold text-blue-400 uppercase">
                    {currentItem.type}
                  </span>
                  <span className="text-xs text-blue-400">●</span>
                </div>
                <p className="text-xs text-gray-300 truncate" title={currentItem.prompt}>
                  {currentItem.prompt || "No prompt"}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Pending items */}
        {pendingItems.map((item, index) => (
          <div
            key={item.id}
            className="bg-gray-700/50 border border-gray-600 rounded p-2"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1 mb-1">
                  <span className="text-xs font-semibold text-gray-400 uppercase">
                    {item.type}
                  </span>
                  <span className="text-xs text-gray-400">
                    #{index + 1}
                  </span>
                </div>
                <p className="text-xs text-gray-300 truncate" title={item.prompt}>
                  {item.prompt || "No prompt"}
                </p>
              </div>
              <button
                onClick={() => removeFromQueue(item.id)}
                className="ml-1 text-gray-400 hover:text-red-500 flex-shrink-0"
                title="Remove from queue"
              >
                <svg
                  className="w-3 h-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          </div>
        ))}

        {/* Completed items */}
        {completedItems.map((item) => (
          <div
            key={item.id}
            className="bg-green-900/20 border border-green-700/50 rounded p-2"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1 mb-1">
                  <span className="text-xs font-semibold text-green-400 uppercase">
                    {item.type}
                  </span>
                  <span className="text-xs text-green-400">✓</span>
                </div>
                <p className="text-xs text-gray-300 truncate" title={item.prompt}>
                  {item.prompt || "No prompt"}
                </p>
              </div>
              <button
                onClick={() => removeFromQueue(item.id)}
                className="ml-1 text-gray-400 hover:text-red-500 flex-shrink-0"
                title="Remove"
              >
                <svg
                  className="w-3 h-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          </div>
        ))}

        {/* Failed items */}
        {failedItems.map((item) => (
          <div
            key={item.id}
            className="bg-red-900/20 border border-red-700/50 rounded p-2"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1 mb-1">
                  <span className="text-xs font-semibold text-red-400 uppercase">
                    {item.type}
                  </span>
                  <span className="text-xs text-red-400">✗</span>
                </div>
                <p className="text-xs text-gray-300 truncate" title={item.prompt}>
                  {item.prompt || "No prompt"}
                </p>
              </div>
              <button
                onClick={() => removeFromQueue(item.id)}
                className="ml-1 text-gray-400 hover:text-red-500 flex-shrink-0"
                title="Remove"
              >
                <svg
                  className="w-3 h-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
          </div>
        ))}

        {queue.length === 0 && !currentItem && (
          <p className="text-xs text-gray-400 text-center py-4">
            No items in queue
          </p>
        )}
      </div>
    </div>
  );
}
