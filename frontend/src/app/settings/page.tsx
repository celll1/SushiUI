"use client";

import { useState, useEffect } from "react";
import Sidebar from "@/components/common/Sidebar";
import Card from "@/components/common/Card";
import Button from "@/components/common/Button";
import DirectorySettings from "@/components/settings/DirectorySettings";
import { restartBackend, restartFrontend, restartBoth } from "@/utils/api";

export default function SettingsPage() {
  const [isRestarting, setIsRestarting] = useState(false);
  const [storageInfo, setStorageInfo] = useState({ used: 0, quota: 0 });

  const updateStorageInfo = () => {
    if (typeof window !== 'undefined' && 'storage' in navigator && 'estimate' in navigator.storage) {
      navigator.storage.estimate().then(estimate => {
        setStorageInfo({
          used: estimate.usage || 0,
          quota: estimate.quota || 0,
        });
      });
    }
  };

  const handleClearLocalStorage = () => {
    if (!confirm("Are you sure you want to clear all localStorage data? This will reset all saved settings, images, and panel states.")) {
      return;
    }

    try {
      localStorage.clear();
      alert("localStorage cleared successfully! The page will reload.");
      window.location.reload();
    } catch (error) {
      console.error("Failed to clear localStorage:", error);
      alert("Failed to clear localStorage. Please check the console.");
    }
  };

  const handleClearTempImages = async () => {
    if (!confirm("Are you sure you want to clear all temporary images? This will remove all saved input images and ControlNet references.")) {
      return;
    }

    try {
      const { cleanupTempImages } = await import("@/utils/api");
      const deletedCount = await cleanupTempImages(0); // Delete all images (max age 0 hours)
      alert(`Successfully deleted ${deletedCount} temporary images.`);
      updateStorageInfo();
    } catch (error) {
      console.error("Failed to clear temp images:", error);
      alert("Failed to clear temp images. Please check the console.");
    }
  };

  const handleRestartBackend = async () => {
    if (!confirm("Are you sure you want to restart the backend server?")) {
      return;
    }

    setIsRestarting(true);
    try {
      const result = await restartBackend();
      console.log("Backend restart response:", result);
      alert("Backend restart scheduled. The backend will restart in a moment. You may need to refresh the page in a few seconds.");
    } catch (error: any) {
      console.error("Failed to restart backend:", error);
      console.error("Error details:", error.response?.data);
      const errorMsg = error.response?.data?.detail || error.message || "Unknown error";
      alert(`Failed to restart backend: ${errorMsg}\n\nPlease check the backend console for details.`);
    } finally {
      // Keep the button disabled for a few seconds
      setTimeout(() => {
        setIsRestarting(false);
      }, 5000);
    }
  };

  const handleRestartFrontend = () => {
    if (!confirm("Are you sure you want to restart the frontend? The page will reload.")) {
      return;
    }

    restartFrontend();
  };

  const handleRestartBoth = async () => {
    if (!confirm("Are you sure you want to restart both servers? The page will reload after backend restarts.")) {
      return;
    }

    setIsRestarting(true);
    try {
      await restartBoth();
    } catch (error) {
      console.error("Failed to restart servers:", error);
      alert("Failed to restart servers. Please check the console.");
      setIsRestarting(false);
    }
  };

  useEffect(() => {
    updateStorageInfo();
  }, []);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto p-6">
        <h1 className="text-2xl font-bold mb-6">Settings</h1>

        <div className="space-y-6">
          <Card title="Server Control">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Restart the backend or frontend servers without manually stopping them.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Button
                  onClick={handleRestartBackend}
                  disabled={isRestarting}
                  variant="secondary"
                  className="w-full"
                >
                  {isRestarting ? "Restarting..." : "Restart Backend"}
                </Button>

                <Button
                  onClick={handleRestartFrontend}
                  disabled={isRestarting}
                  variant="secondary"
                  className="w-full"
                >
                  Restart Frontend
                </Button>

                <Button
                  onClick={handleRestartBoth}
                  disabled={isRestarting}
                  className="w-full"
                >
                  {isRestarting ? "Restarting..." : "Restart Both"}
                </Button>
              </div>

              <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                <h3 className="text-sm font-semibold mb-2">Notes:</h3>
                <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
                  <li><strong>Backend:</strong> Restarts the Python FastAPI server. Use this after code changes in backend/.</li>
                  <li><strong>Frontend:</strong> Reloads the page. Use this to refresh the UI state.</li>
                  <li><strong>Both:</strong> Restarts backend first, then reloads the page after 2 seconds.</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card title="Storage Management">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Manage browser storage and temporary files to free up space.
              </p>

              {storageInfo.quota > 0 && (
                <div className="p-4 bg-gray-800 rounded-lg mb-4">
                  <h3 className="text-sm font-semibold mb-2">Storage Usage</h3>
                  <div className="space-y-2 text-sm text-gray-400">
                    <div className="flex justify-between">
                      <span>Used:</span>
                      <span className="font-mono">{formatBytes(storageInfo.used)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Quota:</span>
                      <span className="font-mono">{formatBytes(storageInfo.quota)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Available:</span>
                      <span className="font-mono">{formatBytes(storageInfo.quota - storageInfo.used)}</span>
                    </div>
                    <div className="mt-2 bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: `${(storageInfo.used / storageInfo.quota) * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-center text-gray-500">
                      {((storageInfo.used / storageInfo.quota) * 100).toFixed(1)}% used
                    </div>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Button
                  onClick={handleClearLocalStorage}
                  variant="secondary"
                  className="w-full"
                >
                  Clear localStorage
                </Button>

                <Button
                  onClick={handleClearTempImages}
                  variant="secondary"
                  className="w-full"
                >
                  Clear Temp Images
                </Button>
              </div>

              <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                <h3 className="text-sm font-semibold mb-2">What gets cleared:</h3>
                <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
                  <li><strong>localStorage:</strong> All saved settings, prompts, parameters, and image references. The page will reload after clearing.</li>
                  <li><strong>Temp Images:</strong> All temporary images stored on the server (input images, ControlNet references). References in localStorage will become invalid.</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card title="Model Directories">
            <DirectorySettings />
          </Card>

          <Card title="Tag Suggestions">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Configure tag autocompletion behavior in prompt fields.
              </p>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Minimum Tag Count
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="10000"
                    value={
                      typeof window !== 'undefined'
                        ? parseInt(localStorage.getItem('tag_suggestion_min_count') || '50')
                        : 50
                    }
                    onChange={(e) => {
                      localStorage.setItem('tag_suggestion_min_count', e.target.value);
                    }}
                    className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Only show tags that appear at least this many times in the dataset. Lower values show more tags but may include uncommon or misspelled tags. Default: 50
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Other Settings">
            <p className="text-gray-400">Additional settings will be implemented in future updates.</p>
          </Card>
        </div>
      </main>
    </div>
  );
}
