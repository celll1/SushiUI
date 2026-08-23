"use client";

import { useState, useEffect, useRef } from "react";
import { useSearchParams } from "next/navigation";
import Sidebar from "@/components/common/Sidebar";
import Txt2ImgPanel from "@/components/generation/Txt2ImgPanel";
import Img2ImgPanel from "@/components/generation/Img2ImgPanel";
import InpaintPanel from "@/components/generation/InpaintPanel";
import OutpaintPanel from "@/components/generation/OutpaintPanel";
import UpscalePanel from "@/components/generation/UpscalePanel";
import FloatingGallery, { GalleryEntry } from "@/components/common/FloatingGallery";
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

type GenerateTab = "txt2img" | "img2img" | "inpaint" | "outpaint" | "upscale";

function tabFromParam(value: string | null): GenerateTab {
  return value === "img2img" || value === "inpaint" || value === "outpaint" || value === "upscale"
    ? value
    : "txt2img";
}

function GeneratePageContent() {
  const searchParams = useSearchParams();
  const tabParam = searchParams.get("tab");
  // Honour a deep-linked tab on the first render. Initialising to txt2img and
  // correcting it in an effect mounted the wrong generation panel for one
  // commit, starting its model/candidate requests before immediately unmounting it.
  const [activeTab, setActiveTab] = useState<GenerateTab>(() => tabFromParam(tabParam));
  const [galleryImages, setGalleryImages] = useState<GalleryEntry[]>([]);
  const [maxGalleryImages, setMaxGalleryImages] = useState(30);
  const { setGenerateForever, resultFeed } = useGenerationQueue();
  const lastFeedIdRef = useRef(0);

  useEffect(() => {
    if (tabParam !== null) setActiveTab(tabFromParam(tabParam));
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

  // Results produced by the global queue processor. Panels that still dispatch
  // their own types keep using the onImageGenerated prop below until they are
  // migrated; the two sources never carry the same result.
  useEffect(() => {
    const fresh = resultFeed.filter((entry) => entry.id > lastFeedIdRef.current);
    if (fresh.length === 0) return;
    lastFeedIdRef.current = fresh[fresh.length - 1].id;
    setGalleryImages((prev) => [
      ...prev,
      ...fresh.map((entry) => ({
        url: entry.url,
        timestamp: entry.timestamp,
        kind: entry.kind,
        playbackUrl: entry.playbackUrl,
      })),
    ]);
  }, [resultFeed]);

  const handleImageGenerated = (
    imageUrl: string,
    opts?: { kind?: "image" | "video" | "audio"; playbackUrl?: string }
  ) => {
    setGalleryImages(prev => [
      ...prev,
      { url: imageUrl, timestamp: Date.now(), kind: opts?.kind, playbackUrl: opts?.playbackUrl },
    ]);
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
