"use client";

import Sidebar from "@/components/common/Sidebar";
import ImageGrid from "@/components/viewer/ImageGrid";
import ProtectedRoute from "@/components/common/ProtectedRoute";

export default function GalleryPage() {
  return (
    <ProtectedRoute>
      <div className="app-shell">
        <Sidebar />
        <main className="app-main compact-workspace flex flex-col overflow-hidden">
          <header className="app-topbar">
            <div>
              <p className="app-kicker">Library</p>
              <h1 className="app-title">Gallery</h1>
            </div>
          </header>
          <div className="app-content flex-1 overflow-auto">
            <ImageGrid />
          </div>
        </main>
      </div>
    </ProtectedRoute>
  );
}
