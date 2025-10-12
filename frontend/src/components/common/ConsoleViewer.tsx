"use client";

import { useState, useEffect, useRef } from "react";
import Card from "./Card";
import Button from "./Button";
import { Terminal, Trash2, RefreshCw } from "lucide-react";

export default function ConsoleViewer() {
  const [logs, setLogs] = useState<string[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const fetchLogs = async () => {
    try {
      const response = await fetch("/api/logs");
      const data = await response.json();
      setLogs(data.logs || []);
    } catch (error) {
      console.error("Failed to fetch logs:", error);
    }
  };

  const clearLogs = async () => {
    try {
      await fetch("/api/logs", { method: "DELETE" });
      setLogs([]);
    } catch (error) {
      console.error("Failed to clear logs:", error);
    }
  };

  const scrollToBottom = () => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchLogs, 2000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  useEffect(() => {
    if (autoRefresh) {
      scrollToBottom();
    }
  }, [logs, autoRefresh]);

  return (
    <div className="space-y-4">
      <Card title="Backend Console Logs">
        <div className="space-y-4">
          {/* Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Terminal className="w-5 h-5 text-gray-400" />
              <span className="text-sm text-gray-400">
                {logs.length} log entries
              </span>
            </div>
            <div className="flex space-x-2">
              <Button
                onClick={() => setAutoRefresh(!autoRefresh)}
                variant={autoRefresh ? "primary" : "secondary"}
                size="sm"
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${autoRefresh ? "animate-spin" : ""}`} />
                Auto-refresh: {autoRefresh ? "ON" : "OFF"}
              </Button>
              <Button onClick={fetchLogs} variant="secondary" size="sm">
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
              <Button onClick={clearLogs} variant="danger" size="sm">
                <Trash2 className="w-4 h-4 mr-2" />
                Clear
              </Button>
            </div>
          </div>

          {/* Log Display */}
          <div className="bg-gray-950 rounded-lg p-4 h-96 overflow-y-auto font-mono text-xs">
            {logs.length === 0 ? (
              <p className="text-gray-500">No logs available</p>
            ) : (
              <div className="space-y-1">
                {logs.map((log, index) => (
                  <div
                    key={index}
                    className={`${
                      log.includes("ERROR") || log.includes("Error")
                        ? "text-red-400"
                        : log.includes("WARNING") || log.includes("Warning")
                        ? "text-yellow-400"
                        : log.includes("INFO")
                        ? "text-blue-400"
                        : "text-gray-300"
                    }`}
                  >
                    {log}
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            )}
          </div>

          {/* Info */}
          <div className="text-xs text-gray-500">
            <p>💡 Logs are captured from the backend console output</p>
            <p>💡 Auto-refresh updates every 2 seconds when enabled</p>
          </div>
        </div>
      </Card>

      <Card title="Frontend Console (Browser)">
        <div className="bg-gray-950 rounded-lg p-4">
          <p className="text-sm text-gray-400 mb-2">
            Open browser DevTools (F12) to view frontend console logs
          </p>
          <p className="text-xs text-gray-500">
            Press <kbd className="px-2 py-1 bg-gray-800 rounded">F12</kbd> or{" "}
            <kbd className="px-2 py-1 bg-gray-800 rounded">Ctrl+Shift+I</kbd>{" "}
            (Windows/Linux) /{" "}
            <kbd className="px-2 py-1 bg-gray-800 rounded">Cmd+Option+I</kbd>{" "}
            (Mac)
          </p>
        </div>
      </Card>
    </div>
  );
}
