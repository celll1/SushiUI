"use client";

import { useState, useEffect } from "react";
import { getLatents, scoreLatent, getLatentStats, type LatentRecord, type LatentStats } from "@/utils/api";

export default function Home() {
  const [records, setRecords] = useState<LatentRecord[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [score, setScore] = useState(0.5);
  const [stats, setStats] = useState<LatentStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [unscoredOnly, setUnscoredOnly] = useState(true);

  const currentRecord = records[currentIndex];

  // Load records
  useEffect(() => {
    loadRecords();
    loadStats();
  }, [unscoredOnly]);

  const loadRecords = async () => {
    setLoading(true);
    try {
      const data = await getLatents({
        skip: 0,
        limit: 1000,
        unscored_only: unscoredOnly,
      });
      setRecords(data.records);
      setCurrentIndex(0);
    } catch (error) {
      console.error("Failed to load records:", error);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const data = await getLatentStats();
      setStats(data);
    } catch (error) {
      console.error("Failed to load stats:", error);
    }
  };

  const handleScoreSubmit = async () => {
    if (!currentRecord) return;

    try {
      await scoreLatent(currentRecord.id, score);

      // Move to next
      if (currentIndex < records.length - 1) {
        setCurrentIndex(currentIndex + 1);
        setScore(0.5); // Reset score
      } else {
        // Reload records when finished
        loadRecords();
      }

      loadStats();
    } catch (error) {
      console.error("Failed to score latent:", error);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
      setScore(records[currentIndex - 1].user_score ?? 0.5);
    }
  };

  const handleNext = () => {
    if (currentIndex < records.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setScore(records[currentIndex + 1].user_score ?? 0.5);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Number keys 0-9 for quick scoring
      if (e.key >= "0" && e.key <= "9") {
        const numScore = e.key === "0" ? 1.0 : parseInt(e.key) / 10;
        setScore(numScore);
      }
      // Space: Submit and next
      else if (e.key === " ") {
        e.preventDefault();
        handleScoreSubmit();
      }
      // Arrow keys: Navigation
      else if (e.key === "ArrowLeft") {
        handlePrevious();
      } else if (e.key === "ArrowRight") {
        handleNext();
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [currentIndex, score, records]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="text-xl">Loading...</p>
      </div>
    );
  }

  if (!currentRecord) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-xl mb-4">No records to score</p>
          <button
            onClick={() => setUnscoredOnly(!unscoredOnly)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
          >
            {unscoredOnly ? "Show All Records" : "Show Unscored Only"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Aesthetic Scorer</h1>
        {stats && (
          <div className="flex gap-6 text-sm text-gray-400">
            <span>Total: {stats.total}</span>
            <span>Scored: {stats.scored}</span>
            <span>Unscored: {stats.unscored}</span>
            <span>Progress: {stats.scored_percentage.toFixed(1)}%</span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="mb-4 flex gap-4 items-center">
        <button
          onClick={handlePrevious}
          disabled={currentIndex === 0}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded"
        >
          ← Previous
        </button>
        <span className="text-gray-400">
          {currentIndex + 1} / {records.length}
        </span>
        <button
          onClick={handleNext}
          disabled={currentIndex === records.length - 1}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded"
        >
          Next →
        </button>
        <label className="flex items-center gap-2 ml-auto">
          <input
            type="checkbox"
            checked={unscoredOnly}
            onChange={(e) => setUnscoredOnly(e.target.checked)}
          />
          <span className="text-sm">Unscored only</span>
        </label>
      </div>

      {/* Image Comparison */}
      <div className="grid grid-cols-2 gap-8 mb-8">
        {/* True Latent */}
        <div>
          <h3 className="text-xl font-semibold mb-4 text-green-400">
            Ground Truth (True Latent)
          </h3>
          {currentRecord.true_latent_image_path ? (
            <img
              src={`http://localhost:8001${currentRecord.true_latent_image_path.replace("subapps/aesthetic_scorer/data", "")}`}
              alt="True Latent"
              className="w-full border border-green-400/30 rounded"
            />
          ) : (
            <div className="w-full aspect-square bg-gray-800 flex items-center justify-center rounded">
              <p className="text-gray-500">Not decoded yet</p>
            </div>
          )}
        </div>

        {/* Predicted Latent */}
        <div>
          <h3 className="text-xl font-semibold mb-4 text-blue-400">
            Predicted Latent (t={currentRecord.timestep.toFixed(4)})
          </h3>
          {currentRecord.predicted_latent_image_path ? (
            <img
              src={`http://localhost:8001${currentRecord.predicted_latent_image_path.replace("subapps/aesthetic_scorer/data", "")}`}
              alt="Predicted Latent"
              className="w-full border border-blue-400/30 rounded"
            />
          ) : (
            <div className="w-full aspect-square bg-gray-800 flex items-center justify-center rounded">
              <p className="text-gray-500">Not decoded yet</p>
            </div>
          )}
        </div>
      </div>

      {/* Scoring Panel */}
      <div className="max-w-2xl mx-auto p-6 bg-gray-900 rounded-lg border border-gray-700">
        <h3 className="text-xl font-semibold mb-4">Quality Score</h3>

        {/* Score Slider */}
        <div className="mb-6">
          <div className="flex justify-between mb-2">
            <span className="text-sm text-green-400">0.0 (Best)</span>
            <span className="text-2xl font-bold">{score.toFixed(2)}</span>
            <span className="text-sm text-red-400">1.0 (Worst)</span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={score}
            onChange={(e) => setScore(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="text-xs text-gray-500 mt-2">
            Keyboard: 1-9 for quick score, 0 for 1.0 (worst), Space to submit
          </div>
        </div>

        {/* Metadata */}
        <div className="mb-6 text-sm text-gray-400 space-y-1">
          <p><span className="text-gray-500">Recon Loss:</span> {currentRecord.recon_loss.toFixed(6)}</p>
          <p><span className="text-gray-500">Timestep:</span> {currentRecord.timestep.toFixed(4)}</p>
          <p><span className="text-gray-500">Latent Shape:</span> [{currentRecord.latent_shape.join(", ")}]</p>
          <p><span className="text-gray-500">Dataset:</span> {currentRecord.dataset_name}</p>
        </div>

        {/* Caption */}
        <div className="mb-6">
          <p className="text-sm text-gray-500 mb-1">Caption:</p>
          <p className="text-sm bg-gray-800 p-3 rounded border border-gray-700">
            {currentRecord.caption || "(No caption)"}
          </p>
        </div>

        {/* Submit Button */}
        <button
          onClick={handleScoreSubmit}
          className="w-full py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold text-lg transition-colors"
        >
          Submit Score & Next (Space)
        </button>
      </div>

      {/* Instructions */}
      <div className="mt-8 max-w-2xl mx-auto text-sm text-gray-500">
        <h4 className="font-semibold mb-2">Scoring Guidelines:</h4>
        <ul className="list-disc list-inside space-y-1">
          <li><strong>0.0-0.3:</strong> Excellent quality, very close to ground truth</li>
          <li><strong>0.3-0.5:</strong> Good quality, minor artifacts</li>
          <li><strong>0.5-0.7:</strong> Acceptable quality, noticeable degradation</li>
          <li><strong>0.7-0.9:</strong> Poor quality, significant overbaking or artifacts</li>
          <li><strong>0.9-1.0:</strong> Very poor quality, unusable</li>
        </ul>
      </div>
    </div>
  );
}
