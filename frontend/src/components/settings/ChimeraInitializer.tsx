"use client";

import { useState } from "react";
import Button from "@/components/common/Button";
import { initializeSenseNovaSDXLChimera } from "@/utils/api";

export default function ChimeraInitializer() {
  const [outputName, setOutputName] = useState("snu1.5_sdxl_chimera");
  const [targetDir, setTargetDir] = useState("");
  const [understandingSource, setUnderstandingSource] = useState("");
  const [sdxlSource, setSdxlSource] = useState("");
  const [flowVersion, setFlowVersion] = useState<"v1" | "v2" | "v3">("v1");
  const [latentMean, setLatentMean] = useState("");
  const [latentMoment, setLatentMoment] = useState("");
  const [angularEndpointSlope, setAngularEndpointSlope] = useState(2);
  const [initialization, setInitialization] = useState<"scratch" | "sdxl_transplant">("scratch");
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  const initialize = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const mean = latentMean.split(",").map((value) => Number(value.trim()));
      const needsCalibration = flowVersion !== "v1";
      const result = await initializeSenseNovaSDXLChimera({
        output_name: outputName,
        target_dir: targetDir || undefined,
        understanding_source: understandingSource,
        sdxl_source: sdxlSource,
        flow_version: flowVersion,
        latent_mean: needsCalibration ? mean : null,
        latent_centered_second_moment: needsCalibration ? Number(latentMoment) : null,
        angular_endpoint_slope: flowVersion === "v3" ? angularEndpointSlope : 2,
        unet_initialization: initialization,
        initialization_seed: seed,
      });
      setMessage({
        ok: true,
        text: `Created ${result.path} (format ${result.format_version}, ${result.prediction_type}, ${result.unet_parameter_count.toLocaleString()} U-Net parameters, bridge ${result.bridge_state}).`,
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
  const parsedMean = latentMean.split(",").map((value) => Number(value.trim()));
  const calibrationValid = flowVersion === "v1" || (
    parsedMean.length === 4
    && parsedMean.every(Number.isFinite)
    && Number.isFinite(Number(latentMoment))
    && Number(latentMoment) > 0
  );
  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-500">
        Builds a production-loadable Chimera directory from a frozen SenseNova understanding source and an SDXL donor. The donor VAE is bundled.
      </p>
      <label className="block text-xs text-gray-400">Output directory name
        <input className={`${fieldClass} mt-1`} value={outputName} onChange={(e) => setOutputName(e.target.value)} />
      </label>
      <label className="block text-xs text-gray-400">Configured model directory (optional)
        <input className={`${fieldClass} mt-1`} value={targetDir} onChange={(e) => setTargetDir(e.target.value)} placeholder="M:\\models" />
      </label>
      <label className="block text-xs text-gray-400">SenseNova source
        <input className={`${fieldClass} mt-1`} value={understandingSource} onChange={(e) => setUnderstandingSource(e.target.value)} placeholder="M:\\model\\sensenova\\...safetensors" />
      </label>
      <label className="block text-xs text-gray-400">SDXL donor
        <input className={`${fieldClass} mt-1`} value={sdxlSource} onChange={(e) => setSdxlSource(e.target.value)} placeholder="M:\\model\\sdxl\\...safetensors" />
      </label>
      <label className="block text-xs text-gray-400">Flow contract
        <select className={`${fieldClass} mt-1`} value={flowVersion}
          onChange={(e) => setFlowVersion(e.target.value as "v1" | "v2" | "v3")}>
          <option value="v1">v1 direct velocity</option>
          <option value="v2">v2 endpoint-observable residual</option>
          <option value="v3">v3 polar tangent flow</option>
        </select>
      </label>
      {flowVersion !== "v1" && (
        <div className="grid grid-cols-2 gap-3">
          <label className="block text-xs text-gray-400">Latent mean (4 comma-separated values)
            <input className={`${fieldClass} mt-1`} value={latentMean}
              onChange={(e) => setLatentMean(e.target.value)} placeholder="0.0, 0.0, 0.0, 0.0" />
          </label>
          <label className="block text-xs text-gray-400">Centered second moment
            <input className={`${fieldClass} mt-1`} type="number" min={Number.MIN_VALUE} step="any"
              value={latentMoment} onChange={(e) => setLatentMoment(e.target.value)} />
          </label>
          {flowVersion === "v3" && (
            <label className="block text-xs text-gray-400">Noise-end angular slope
              <input className={`${fieldClass} mt-1`} type="number" min={0} max={2} step="any"
                value={angularEndpointSlope}
                onChange={(e) => setAngularEndpointSlope(Number(e.target.value))} />
            </label>
          )}
        </div>
      )}
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
      <Button onClick={initialize} disabled={busy || !outputName || !understandingSource || !sdxlSource || !calibrationValid}>
        {busy ? "Building…" : "Initialize Chimera"}
      </Button>
      {message && <p className={`text-xs ${message.ok ? "text-green-400" : "text-red-400"}`}>{message.text}</p>}
    </div>
  );
}
