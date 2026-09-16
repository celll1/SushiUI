"use client";

import { useState } from "react";
import Button from "@/components/common/Button";
import { initializeSenseNovaSDXLChimera } from "@/utils/api";

export default function ChimeraInitializer() {
  const [outputName, setOutputName] = useState("snu1.5_sdxl_chimera");
  const [understandingSource, setUnderstandingSource] = useState("");
  const [sdxlSource, setSdxlSource] = useState("");
  const [initialization, setInitialization] = useState<"scratch" | "sdxl_transplant">("scratch");
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  const initialize = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const result = await initializeSenseNovaSDXLChimera({
        output_name: outputName,
        understanding_source: understandingSource,
        sdxl_source: sdxlSource,
        unet_initialization: initialization,
        initialization_seed: seed,
        context_tokens: 77,
      });
      setMessage({
        ok: true,
        text: `Created ${result.path} (${result.unet_parameter_count.toLocaleString()} U-Net parameters, bridge ${result.bridge_state}).`,
      });
    } catch (error: any) {
      setMessage({
        ok: false,
        text: error?.response?.data?.detail || error?.message || "Initialization failed",
      });
    } finally {
      setBusy(false);
    }
  };

  const fieldClass = "w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm";
  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-500">
        Builds a production-loadable Chimera directory from a frozen SenseNova understanding source and an SDXL donor. The donor VAE is bundled.
      </p>
      <label className="block text-xs text-gray-400">Output directory name
        <input className={`${fieldClass} mt-1`} value={outputName} onChange={(e) => setOutputName(e.target.value)} />
      </label>
      <label className="block text-xs text-gray-400">SenseNova source
        <input className={`${fieldClass} mt-1`} value={understandingSource} onChange={(e) => setUnderstandingSource(e.target.value)} placeholder="M:\\model\\sensenova\\...safetensors" />
      </label>
      <label className="block text-xs text-gray-400">SDXL donor
        <input className={`${fieldClass} mt-1`} value={sdxlSource} onChange={(e) => setSdxlSource(e.target.value)} placeholder="M:\\model\\sdxl\\...safetensors" />
      </label>
      <div className="grid grid-cols-2 gap-3">
        <label className="block text-xs text-gray-400">U-Net initialization
          <select className={`${fieldClass} mt-1`} value={initialization}
            onChange={(e) => setInitialization(e.target.value as "scratch" | "sdxl_transplant")}>
            <option value="scratch">Scratch</option>
            <option value="sdxl_transplant">SDXL transplant (experimental)</option>
          </select>
        </label>
        <label className="block text-xs text-gray-400">Initialization seed
          <input className={`${fieldClass} mt-1`} type="number" value={seed}
            onChange={(e) => setSeed(Number(e.target.value))} />
        </label>
      </div>
      <Button onClick={initialize} disabled={busy || !outputName || !understandingSource || !sdxlSource}>
        {busy ? "Building…" : "Initialize Chimera"}
      </Button>
      {message && <p className={`text-xs ${message.ok ? "text-green-400" : "text-red-400"}`}>{message.text}</p>}
    </div>
  );
}
