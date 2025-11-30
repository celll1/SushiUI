"use client";

import { TrainingRun } from "@/utils/api";
import { Play, Square, Clock, CheckCircle, XCircle, Loader2 } from "lucide-react";

interface TrainingListProps {
  runs: TrainingRun[];
  selectedRunId: number | null;
  onSelectRun: (id: number) => void;
  onRefresh: () => void;
  loading: boolean;
}

export default function TrainingList({ runs, selectedRunId, onSelectRun, onRefresh, loading }: TrainingListProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />;
      case "pending":
        return <Clock className="h-4 w-4 text-gray-400" />;
      case "paused":
        return <Square className="h-4 w-4 text-yellow-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  if (loading) {
    return (
      <div className="p-4 text-center text-gray-400 text-sm">
        Loading training runs...
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="p-4 text-center text-gray-400">
        <p className="text-sm font-medium">No training runs yet</p>
        <p className="text-xs mt-1">Create a new training run to get started</p>
      </div>
    );
  }

  return (
    <div className="p-2">
      <div className="space-y-1.5">
        {runs.map((run) => (
          <button
            key={run.id}
            onClick={() => onSelectRun(run.id)}
            className={`w-full text-left p-2.5 rounded transition-colors ${
              selectedRunId === run.id
                ? "bg-blue-600 text-white"
                : "bg-gray-800 hover:bg-gray-700"
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-medium truncate">{run.run_name}</span>
              {getStatusIcon(run.status)}
            </div>

            <div className="text-xs text-gray-400 space-y-0.5">
              <div className="flex items-center justify-between">
                <span>{run.training_method === "lora" ? "LoRA" : "Full"}</span>
                <span>{run.current_step} / {run.total_steps}</span>
              </div>

              {/* Progress bar */}
              {run.status === "running" && (
                <div className="h-1 bg-gray-700 rounded-full overflow-hidden mt-1">
                  <div
                    className="h-full bg-blue-500 transition-all"
                    style={{ width: `${run.progress}%` }}
                  />
                </div>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
