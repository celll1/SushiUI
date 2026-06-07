"use client";

import { useState } from "react";
import { SigLIP2StatusResponse } from "@/utils/api";
import ModelLoader from "./ModelLoader";
import ModelTools from "./ModelTools";
import InferencePanel from "./InferencePanel";
import DatasetBrowserPanel from "./DatasetBrowserPanel";

export default function TaggerSection() {
  const [modelStatus, setModelStatus] = useState<SigLIP2StatusResponse>({
    loaded: false,
    checkpoint_path: "",
    vocab_path: "",
    model_type: "",
    num_tags: 0,
  });
  const [taggerTab, setTaggerTab] = useState<"inference" | "browser">(
    "inference"
  );
  const [drawerOpen, setDrawerOpen] = useState(false);

  const modelPanel = (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="p-4 border-b border-gray-700 flex items-center gap-2">
        <h1 className="text-xl font-bold text-white flex-1">Tagger</h1>
        {/* Close button (mobile only) */}
        <button
          onClick={() => setDrawerOpen(false)}
          className="lg:hidden text-gray-400 hover:text-white p-1"
          aria-label="閉じる"
        >
          ✕
        </button>
      </div>
      <ModelLoader onStatusChange={setModelStatus} />
      <ModelTools
        modelLoaded={modelStatus.loaded}
        modelType={modelStatus.model_type}
      />
    </div>
  );

  return (
    <div className="flex h-full min-h-screen relative">
      {/* ── Desktop: fixed left sidebar ── */}
      <div className="hidden lg:flex lg:w-80 lg:shrink-0 border-r border-gray-700 flex-col">
        {modelPanel}
      </div>

      {/* ── Mobile: drawer overlay ── */}
      {/* Backdrop */}
      {drawerOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-40 lg:hidden"
          onClick={() => setDrawerOpen(false)}
        />
      )}
      {/* Drawer panel */}
      <div
        className={`fixed top-0 left-0 h-full w-80 max-w-[90vw] bg-gray-900 border-r border-gray-700 z-50 lg:hidden transition-transform duration-200 ${
          drawerOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {modelPanel}
      </div>

      {/* ── Right panel ── */}
      <div className="flex-1 overflow-hidden flex flex-col min-w-0">
        {/* Top bar */}
        <div className="flex items-center border-b border-gray-700 px-2 flex-shrink-0">
          {/* Hamburger (mobile only) */}
          <button
            onClick={() => setDrawerOpen(true)}
            className="lg:hidden mr-2 p-2 text-gray-400 hover:text-white flex-shrink-0"
            aria-label="モデル設定を開く"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>

          {/* Mobile model status badge */}
          <span className="lg:hidden text-xs text-gray-500 mr-2 truncate">
            {modelStatus.loaded
              ? modelStatus.checkpoint_path.split(/[\\/]/).pop()
              : "モデル未ロード"}
          </span>

          {/* Tab bar */}
          <div className="flex gap-1 flex-1">
            {(["inference", "browser"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTaggerTab(t)}
                className={`px-3 py-2 text-sm font-medium transition-colors ${
                  taggerTab === t
                    ? "text-white border-b-2 border-blue-500"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                {t === "inference" ? "推論" : "ブラウザ"}
              </button>
            ))}
          </div>
        </div>

        {/* Tab content */}
        <div className="flex-1 overflow-hidden min-h-0">
          {taggerTab === "inference" ? (
            <div className="h-full overflow-y-auto">
              <InferencePanel modelLoaded={modelStatus.loaded} />
            </div>
          ) : (
            <DatasetBrowserPanel modelLoaded={modelStatus.loaded} />
          )}
        </div>
      </div>
    </div>
  );
}
