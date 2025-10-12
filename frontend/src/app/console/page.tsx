"use client";

import Sidebar from "@/components/common/Sidebar";
import ConsoleViewer from "@/components/common/ConsoleViewer";

export default function ConsolePage() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto p-6">
        <h1 className="text-2xl font-bold mb-6">Console Logs</h1>
        <ConsoleViewer />
      </main>
    </div>
  );
}
