"use client";

import { useState } from "react";
import Sidebar from "@/components/common/Sidebar";
import Card from "@/components/common/Card";
import Button from "@/components/common/Button";
import { restartBackend, restartFrontend, restartBoth } from "@/utils/api";

export default function SettingsPage() {
  const [isRestarting, setIsRestarting] = useState(false);

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

          <Card title="Other Settings">
            <p className="text-gray-400">Additional settings will be implemented in future updates.</p>
          </Card>
        </div>
      </main>
    </div>
  );
}
