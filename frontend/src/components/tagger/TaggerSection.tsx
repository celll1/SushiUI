"use client";

import { useState } from "react";
import { SigLIP2StatusResponse } from "@/utils/api";
import ModelLoader from "./ModelLoader";
import ModelTools from "./ModelTools";
import InferencePanel from "./InferencePanel";

export default function TaggerSection() {
  const [modelStatus, setModelStatus] = useState<SigLIP2StatusResponse>({
    loaded: false,
    checkpoint_path: "",
    vocab_path: "",
    model_type: "",
    num_tags: 0,
  });

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

      {/* Right panel: Inference */}
      <div className="flex-1 overflow-y-auto">
        <InferencePanel modelLoaded={modelStatus.loaded} />
      </div>
    </div>
  );
}
