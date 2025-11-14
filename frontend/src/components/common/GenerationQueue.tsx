"use client";

import { useGenerationQueue } from "@/contexts/GenerationQueueContext";
import Button from "./Button";

export default function GenerationQueue() {
  const { queue, currentItem, removeFromQueue } = useGenerationQueue();

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
              if (confirm("Clear all completed and failed items?")) {
                queue.forEach((item) => {
                  if (item.status === "completed" || item.status === "failed") {
                    removeFromQueue(item.id);
                  }
                });
              }
            }}
            variant="secondary"
            size="sm"
            className="text-xs py-1 px-2"
          >
            Clear
          </Button>
        )}
      </div>

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
