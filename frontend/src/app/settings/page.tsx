"use client";

import Sidebar from "@/components/common/Sidebar";
import Card from "@/components/common/Card";

export default function SettingsPage() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto p-6">
        <h1 className="text-2xl font-bold mb-6">Settings</h1>
        <Card title="Coming Soon">
          <p className="text-gray-400">Settings page will be implemented in future updates.</p>
        </Card>
      </main>
    </div>
  );
}
