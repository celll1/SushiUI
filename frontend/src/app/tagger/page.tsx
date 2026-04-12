"use client";

import Sidebar from "@/components/common/Sidebar";
import ProtectedRoute from "@/components/common/ProtectedRoute";
import TaggerSection from "@/components/tagger/TaggerSection";

export default function TaggerPage() {
  return (
    <ProtectedRoute>
      <div className="flex h-screen bg-gray-900 text-white">
        <Sidebar />
        <main className="flex-1 overflow-hidden pt-16 lg:pt-0">
          <TaggerSection />
        </main>
      </div>
    </ProtectedRoute>
  );
}
