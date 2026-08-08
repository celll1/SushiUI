"use client";

import Sidebar from "@/components/common/Sidebar";
import ProtectedRoute from "@/components/common/ProtectedRoute";
import TaggerSection from "@/components/tagger/TaggerSection";

export default function TaggerPage() {
  return (
    <ProtectedRoute>
      <div className="app-shell">
        <Sidebar />
        <main className="app-main compact-workspace overflow-hidden">
          <TaggerSection />
        </main>
      </div>
    </ProtectedRoute>
  );
}
