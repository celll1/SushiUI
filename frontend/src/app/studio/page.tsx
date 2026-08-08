"use client";

import Sidebar from "@/components/common/Sidebar";
import ProtectedRoute from "@/components/common/ProtectedRoute";
import StudioWorkspace from "@/components/studio/StudioWorkspace";

export default function StudioPage() {
  return (
    <ProtectedRoute>
      <div className="flex h-screen overflow-hidden bg-gray-950">
        <Sidebar />
        <StudioWorkspace />
      </div>
    </ProtectedRoute>
  );
}
