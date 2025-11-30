"use client";

import { useState } from "react";
import { TrainingRun, deleteTrainingRun } from "@/utils/api";
import { Play, Square, Clock, CheckCircle, XCircle, Loader2, Trash2 } from "lucide-react";

interface TrainingListProps {
  runs: TrainingRun[];
  selectedRunId: number | null;
  onSelectRun: (id: number) => void;
  onRefresh: () => void;
  loading: boolean;
}

export default function TrainingList({ runs, selectedRunId, onSelectRun, onRefresh, loading }: TrainingListProps) {
  const [deletingId, setDeletingId] = useState<number | null>(null);

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

  const handleDelete = async (e: React.MouseEvent, runId: number, runName: string) => {
    e.stopPropagation(); // Prevent selecting the run when clicking delete

    if (!confirm(`Are you sure you want to delete training run "${runName}"?`)) {
      return;
    }

    setDeletingId(runId);
    try {
      await deleteTrainingRun(runId);
      onRefresh(); // Refresh the list
    } catch (err: any) {
      console.error("Failed to delete training run:", err);
      alert(err.response?.data?.detail || "Failed to delete training run");
    } finally {
      setDeletingId(null);
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
          <div
            key={run.id}
            className={`relative group rounded transition-colors ${
              selectedRunId === run.id
                ? "bg-blue-600 text-white"
                : "bg-gray-800 hover:bg-gray-700"
            }`}
          >
            <button
              onClick={() => onSelectRun(run.id)}
              className="w-full text-left p-2.5 pr-10"
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

            {/* Delete button - only show when not running */}
            {run.status !== "running" && run.status !== "starting" && (
              <button
                onClick={(e) => handleDelete(e, run.id, run.run_name)}
                disabled={deletingId === run.id}
                className={`absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded transition-colors opacity-0 group-hover:opacity-100 ${
                  selectedRunId === run.id
                    ? "hover:bg-blue-700"
                    : "hover:bg-gray-600"
                } disabled:opacity-50`}
                title="Delete training run"
              >
                {deletingId === run.id ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5 text-red-400" />
                )}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
