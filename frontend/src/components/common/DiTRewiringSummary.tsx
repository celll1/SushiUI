"use client";

import { useEffect, useState } from "react";
import { useStartup } from "@/contexts/StartupContext";
import { previewDiTRewiring, temporalRewiringGrid } from "@/utils/api";
import type { DiTRewiringPlan, DiTRewiringSpec, DiTRewiringTemporalGeometry, DiTRewiringVideoGeometry } from "@/utils/api";
import DiTRewiringGraph from "./DiTRewiringGraph";

export default function DiTRewiringSummary({ spec, tokenStride, temporalGeometry, videoGeometry }: { spec: DiTRewiringSpec; tokenStride?: number; temporalGeometry?: DiTRewiringTemporalGeometry; videoGeometry?: DiTRewiringVideoGeometry }) {
  const { generationDefaults } = useStartup();
  const [plan, setPlan] = useState<DiTRewiringPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const width = generationDefaults?.txt2img.width;
  const height = generationDefaults?.txt2img.height;
  useEffect(() => {
    let cancelled = false;
    setPlan(null); setError(null);
    if (temporalGeometry) {
      try {
        const grid = temporalRewiringGrid(spec, temporalGeometry.preview_frames, temporalGeometry);
        previewDiTRewiring(spec, grid.padded, 1).then(value => { if (!cancelled) setPlan(value.plan); })
          .catch(err => { if (!cancelled) setError(typeof err.response?.data?.detail === "string" ? err.response.data.detail : "Topology preview unavailable"); });
      } catch (err) { setError(err instanceof Error ? err.message : "Topology preview unavailable"); }
      return () => { cancelled = true; };
    }
    if (typeof width !== "number" || typeof height !== "number") return;
    const heightStride = videoGeometry?.height_scale_factor ?? tokenStride;
    const widthStride = videoGeometry?.width_scale_factor ?? tokenStride;
    if (typeof heightStride !== "number" || typeof widthStride !== "number" || !Number.isInteger(widthStride) || widthStride < 1 || !Number.isInteger(heightStride) || heightStride < 1) {
      setError("Native token geometry unavailable"); return;
    }
    previewDiTRewiring(spec, height / heightStride, width / widthStride)
      .then(value => { if (!cancelled) setPlan(value.plan); })
      .catch(err => { if (!cancelled) setError(typeof err.response?.data?.detail === "string" ? err.response.data.detail : "Topology preview unavailable"); });
    return () => { cancelled = true; };
  }, [spec, width, height, tokenStride, temporalGeometry, videoGeometry]);
  return <details className="mt-2 rounded border border-emerald-800 p-2 text-xs">
    <summary className="cursor-pointer text-emerald-300">Residual U-Net / {spec.regions.length} regions / checkpoint topology</summary>
    <p className="my-2 text-gray-400">{temporalGeometry ? `Read-only temporal topology. Preview at ${temporalGeometry.preview_frames / temporalGeometry.frame_rate}s; actual clip length follows the generation request. Decoder KV caching is disabled.`
      : videoGeometry ? `Per-frame spatial topology at ${width}×${height}px. Frames and audio tokens retain native resolution; pooling and skip connections never cross frame boundaries.`
      : `Read-only restored topology. Preview at ${width}×${height}px; image dimensions follow the generation request. Image-token KV caching is disabled during rewired execution; immutable prefix KV remains available.`}</p>
    {plan && <DiTRewiringGraph spec={spec} plan={plan} />}
    {spec.regions.map(region => <p key={region.id} className="mt-1 text-gray-300">
      {region.stream} / {region.id}: [{region.start_block}, {region.end_block}) / parent {region.parent ?? "full grid"} / downsample {region.down_factor.join("×")} / residual skip {region.start_block} → {region.end_block}
    </p>)}
    {error && <p role="alert" className="mt-2 text-red-400">{error}</p>}
  </details>;
}
