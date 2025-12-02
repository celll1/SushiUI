"use client";

import { Download, FileText } from "lucide-react";

interface CheckpointListProps {
  checkpoints: string[];
  runId: number;
}

export default function CheckpointList({ checkpoints, runId }: CheckpointListProps) {
  if (!checkpoints || checkpoints.length === 0) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded p-3">
        <h3 className="text-sm font-medium text-gray-300 mb-2">Saved Checkpoints</h3>
        <p className="text-xs text-gray-500">No checkpoints saved yet</p>
      </div>
    );
  }

  // Extract checkpoint info from path
  const getCheckpointInfo = (path: string) => {
    const filename = path.split(/[/\\]/).pop() || path;
    const stepMatch = filename.match(/step-(\d+)/i);
    const epochMatch = filename.match(/epoch-(\d+)/i);

    return {
      filename,
      step: stepMatch ? parseInt(stepMatch[1]) : null,
      epoch: epochMatch ? parseInt(epochMatch[1]) : null,
    };
  };

  const handleDownload = (checkpointPath: string) => {
    // Download via backend API (assumes checkpoint files are served)
    const filename = checkpointPath.split(/[/\\]/).pop();
    const downloadUrl = `/api/training/runs/${runId}/checkpoints/${encodeURIComponent(filename || "")}`;

    // Open download link
    window.open(downloadUrl, "_blank");
  };

  return (
    <div className="bg-gray-800 border border-gray-700 rounded p-3">
      <h3 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
        <FileText className="h-4 w-4" />
        Saved Checkpoints ({checkpoints.length})
      </h3>

      <div className="space-y-1.5">
        {checkpoints.map((checkpoint, index) => {
          const info = getCheckpointInfo(checkpoint);
          return (
            <div
              key={index}
              className="flex items-center justify-between bg-gray-700/50 rounded px-3 py-2 text-xs hover:bg-gray-700 transition-colors"
            >
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <span className="text-gray-400 font-mono">
                  {info.step !== null && `Step ${info.step}`}
                  {info.epoch !== null && `Epoch ${info.epoch}`}
                  {!info.step && !info.epoch && "Checkpoint"}
                </span>
                <span className="text-gray-500 truncate">{info.filename}</span>
              </div>

              <button
                onClick={() => handleDownload(checkpoint)}
                className="flex items-center gap-1 px-2 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded transition-colors ml-2"
                title={`Download ${info.filename}`}
              >
                <Download className="h-3 w-3" />
                <span>Download</span>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
