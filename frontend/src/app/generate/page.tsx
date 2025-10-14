"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Sidebar from "@/components/common/Sidebar";
import Txt2ImgPanel from "@/components/generation/Txt2ImgPanel";
import Img2ImgPanel from "@/components/generation/Img2ImgPanel";
import InpaintPanel from "@/components/generation/InpaintPanel";

export default function GeneratePage() {
  const searchParams = useSearchParams();
  const tabParam = searchParams.get("tab");
  const [activeTab, setActiveTab] = useState<"txt2img" | "img2img" | "inpaint">("txt2img");

  useEffect(() => {
    if (tabParam === "img2img") {
      setActiveTab("img2img");
    } else if (tabParam === "inpaint") {
      setActiveTab("inpaint");
    }
  }, [tabParam]);

  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto p-6">
        <h1 className="text-2xl font-bold mb-6">Generate</h1>

        {/* Tabs */}
        <div className="flex space-x-2 border-b border-gray-700 mb-6">
          <button
            onClick={() => setActiveTab("txt2img")}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === "txt2img"
                ? "border-b-2 border-blue-500 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            txt2img
          </button>
          <button
            onClick={() => setActiveTab("img2img")}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === "img2img"
                ? "border-b-2 border-blue-500 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            img2img
          </button>
          <button
            onClick={() => setActiveTab("inpaint")}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === "inpaint"
                ? "border-b-2 border-blue-500 text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            inpaint
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === "txt2img" && <Txt2ImgPanel onTabChange={setActiveTab} />}
        {activeTab === "img2img" && <Img2ImgPanel onTabChange={setActiveTab} />}
        {activeTab === "inpaint" && <InpaintPanel onTabChange={setActiveTab} />}
      </main>
    </div>
  );
}
