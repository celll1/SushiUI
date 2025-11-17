"use client";

import Sidebar from "@/components/common/Sidebar";
import ImageGrid from "@/components/viewer/ImageGrid";
import ProtectedRoute from "@/components/common/ProtectedRoute";

export default function GalleryPage() {
  return (
    <ProtectedRoute>
      <div className="flex h-screen">
        <Sidebar />
        <main className="flex-1 overflow-auto p-6">
          <h1 className="text-2xl font-bold mb-6">Gallery</h1>
          <ImageGrid />
        </main>
      </div>
    </ProtectedRoute>
  );
}
