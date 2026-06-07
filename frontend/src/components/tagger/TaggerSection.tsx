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

  return (
    <div className="flex h-full min-h-screen">
      {/* Left panel: Model controls */}
      <div className="w-80 shrink-0 border-r border-gray-700 overflow-y-auto flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h1 className="text-xl font-bold text-white">Tagger</h1>
        </div>
        <ModelLoader onStatusChange={setModelStatus} />
        <ModelTools
          modelLoaded={modelStatus.loaded}
          modelType={modelStatus.model_type}
        />
      </div>

      {/* Right panel */}
      <div className="flex-1 overflow-hidden flex flex-col min-w-0">
        {/* Tab bar */}
        <div className="flex gap-1 border-b border-gray-700 px-4 flex-shrink-0">
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
