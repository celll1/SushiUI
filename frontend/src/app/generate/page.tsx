"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Sidebar from "@/components/common/Sidebar";
import Txt2ImgPanel from "@/components/generation/Txt2ImgPanel";
import Img2ImgPanel from "@/components/generation/Img2ImgPanel";
import InpaintPanel from "@/components/generation/InpaintPanel";
import OutpaintPanel from "@/components/generation/OutpaintPanel";
import UpscalePanel from "@/components/generation/UpscalePanel";
import FloatingGallery from "@/components/common/FloatingGallery";
import GenerationQueue from "@/components/common/GenerationQueue";
import GPUMonitor from "@/components/common/GPUMonitor";
import ProtectedRoute from "@/components/common/ProtectedRoute";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";

export default function GeneratePage() {
  return (
    <ProtectedRoute>
      <GeneratePageContent />
    </ProtectedRoute>
  );
}

function GeneratePageContent() {
  const searchParams = useSearchParams();
  const tabParam = searchParams.get("tab");
  const [activeTab, setActiveTab] = useState<"txt2img" | "img2img" | "inpaint" | "outpaint" | "upscale">("txt2img");
  const [galleryImages, setGalleryImages] = useState<Array<{ url: string; timestamp: number }>>([]);
  const [maxGalleryImages, setMaxGalleryImages] = useState(30);
  const { setGenerateForever } = useGenerationQueue();

  useEffect(() => {
    if (tabParam === "img2img") {
      setActiveTab("img2img");
    } else if (tabParam === "inpaint") {
      setActiveTab("inpaint");
    } else if (tabParam === "outpaint") {
      setActiveTab("outpaint");
    } else if (tabParam === "upscale") {
      setActiveTab("upscale");
    }
  }, [tabParam]);

  useEffect(() => {
    // Load max gallery images setting
    const savedMaxImages = localStorage.getItem('floating_gallery_max_images');
    if (savedMaxImages) {
      setMaxGalleryImages(parseInt(savedMaxImages));
    }
  }, []);

  // Stop generate forever when switching panels
  useEffect(() => {
    setGenerateForever(false);
  }, [activeTab, setGenerateForever]);

  const handleImageGenerated = (imageUrl: string) => {
    setGalleryImages(prev => [...prev, { url: imageUrl, timestamp: Date.now() }]);
  };

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="app-main compact-workspace relative flex flex-col overflow-hidden">
        <header className="app-topbar flex-wrap gap-x-5 gap-y-1 py-1 lg:flex-nowrap lg:py-0">
          <div className="hidden shrink-0 lg:block">
            <p className="app-kicker">Create</p>
            <h1 className="app-title">Generate</h1>
          </div>

          {/* Tabs */}
          <div className="app-tabs min-w-0 flex-1 border-b-0">
          <button
            onClick={() => setActiveTab("txt2img")}
            className={`app-tab ${
              activeTab === "txt2img"
                ? "app-tab-active"
                : ""
            }`}
          >
            txt2img
          </button>
          <button
            onClick={() => setActiveTab("img2img")}
            className={`app-tab ${
              activeTab === "img2img"
                ? "app-tab-active"
                : ""
            }`}
          >
            img2img
          </button>
          <button
            onClick={() => setActiveTab("inpaint")}
            className={`app-tab ${
              activeTab === "inpaint"
                ? "app-tab-active"
                : ""
            }`}
          >
            inpaint
          </button>
          <button
            onClick={() => setActiveTab("outpaint")}
            className={`app-tab ${
              activeTab === "outpaint"
                ? "app-tab-active"
                : ""
            }`}
          >
            outpaint
          </button>
          <button
            onClick={() => setActiveTab("upscale")}
            className={`app-tab ${
              activeTab === "upscale"
                ? "app-tab-active"
                : ""
            }`}
          >
            Upscale
          </button>
          </div>
        </header>

        {/* Tab Content */}
        <div className="app-content flex-1 overflow-auto">
          {activeTab === "txt2img" && <Txt2ImgPanel onTabChange={setActiveTab} onImageGenerated={handleImageGenerated} />}
          {activeTab === "img2img" && <Img2ImgPanel onTabChange={setActiveTab} onImageGenerated={handleImageGenerated} />}
          {activeTab === "inpaint" && <InpaintPanel onTabChange={setActiveTab} onImageGenerated={handleImageGenerated} />}
          {activeTab === "outpaint" && <OutpaintPanel onTabChange={setActiveTab} onImageGenerated={handleImageGenerated} />}
          {activeTab === "upscale" && <UpscalePanel onTabChange={setActiveTab} onImageGenerated={handleImageGenerated} />}
        </div>
      </main>

      {/* Floating Gallery - shared across all tabs */}
      <FloatingGallery images={galleryImages} maxImages={maxGalleryImages} />

      {/* GPU Monitor */}
      <GPUMonitor />
    </div>
  );
}
