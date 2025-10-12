"use client";

import Sidebar from "@/components/common/Sidebar";
import Txt2ImgPanel from "@/components/generation/Txt2ImgPanel";

export default function GeneratePage() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto p-6">
        <h1 className="text-2xl font-bold mb-6">Generate</h1>
        <Txt2ImgPanel />
      </main>
    </div>
  );
}
