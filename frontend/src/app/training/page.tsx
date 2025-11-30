"use client";

import { useState, useEffect } from "react";
import Sidebar from "@/components/common/Sidebar";
import ProtectedRoute from "@/components/common/ProtectedRoute";
import TrainingList from "@/components/training/TrainingList";
import TrainingConfig from "@/components/training/TrainingConfig";
import TrainingMonitor from "@/components/training/TrainingMonitor";
import { listTrainingRuns, TrainingRun } from "@/utils/api";

export default function TrainingPage() {
  return (
    <ProtectedRoute>
      <TrainingPageContent />
    </ProtectedRoute>
  );
}

function TrainingPageContent() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [showConfig, setShowConfig] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRuns();
  }, []);

  // Poll running trainings to update list
  useEffect(() => {
    const hasRunningTraining = runs.some(r => r.status === "running" || r.status === "starting");
    if (!hasRunningTraining) return;

    const interval = setInterval(() => {
      loadRuns();
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(interval);
  }, [runs]);

  const loadRuns = async () => {
    setLoading(true);
    try {
      const response = await listTrainingRuns();
      setRuns(response.runs);
    } catch (err) {
      console.error("Failed to load training runs:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateRun = () => {
    setSelectedRunId(null);
    setShowConfig(true);
  };

  const handleRunCreated = (newRun: TrainingRun) => {
    setRuns([newRun, ...runs]);
    setShowConfig(false);
    setSelectedRunId(newRun.id);
  };

  const handleSelectRun = (id: number) => {
    setSelectedRunId(id);
    setShowConfig(false);
  };

  const handleStatusChange = (updatedRun: TrainingRun) => {
    setRuns((prevRuns) =>
      prevRuns.map((r) => (r.id === updatedRun.id ? updatedRun : r))
    );
  };

  const selectedRun = runs.find(r => r.id === selectedRunId);

  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden pt-16 lg:pt-0">
        {/* Header */}
        <div className="flex-shrink-0 p-3 sm:p-4 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <h1 className="text-lg sm:text-xl font-bold">Training</h1>
            <button
              onClick={handleCreateRun}
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-sm transition-colors"
            >
              New Training Run
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex">
          {/* Left: Training Runs List */}
          <div className="w-80 flex-shrink-0 border-r border-gray-700 overflow-y-auto">
            <TrainingList
              runs={runs}
              selectedRunId={selectedRunId}
              onSelectRun={handleSelectRun}
              onRefresh={loadRuns}
              loading={loading}
            />
          </div>

          {/* Right: Config or Monitor */}
          <div className="flex-1 overflow-y-auto">
            {showConfig ? (
              <TrainingConfig
                onClose={() => setShowConfig(false)}
                onRunCreated={handleRunCreated}
              />
            ) : selectedRun ? (
              <TrainingMonitor
                key={selectedRun.id}
                run={selectedRun}
                onClose={() => setSelectedRunId(null)}
                onStatusChange={handleStatusChange}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                <div className="text-center">
                  <p className="text-lg font-medium">No training run selected</p>
                  <p className="text-sm mt-2">Select a run from the list or create a new one</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
