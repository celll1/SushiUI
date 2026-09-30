"use client";

import { useEffect, useState } from "react";
import { useStartup } from "@/contexts/StartupContext";
import { previewDiTRewiring } from "@/utils/api";
import type { DiTRewiringPlan, DiTRewiringSpec } from "@/utils/api";
import DiTRewiringGraph from "./DiTRewiringGraph";

export default function DiTRewiringSummary({ spec, tokenStride }: { spec: DiTRewiringSpec; tokenStride?: number }) {
  const { generationDefaults } = useStartup();
  const [plan, setPlan] = useState<DiTRewiringPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const width = generationDefaults?.txt2img.width;
  const height = generationDefaults?.txt2img.height;
  useEffect(() => {
    let cancelled = false;
    setPlan(null); setError(null);
    if (typeof width !== "number" || typeof height !== "number") return;
    if (typeof tokenStride !== "number" || !Number.isInteger(tokenStride) || tokenStride < 1) {
      setError("Native token geometry unavailable"); return;
    }
    previewDiTRewiring(spec, height / tokenStride, width / tokenStride)
      .then(value => { if (!cancelled) setPlan(value.plan); })
      .catch(err => { if (!cancelled) setError(typeof err.response?.data?.detail === "string" ? err.response.data.detail : "Topology preview unavailable"); });
    return () => { cancelled = true; };
  }, [spec, width, height, tokenStride]);
  return <details className="mt-2 rounded border border-emerald-800 p-2 text-xs">
    <summary className="cursor-pointer text-emerald-300">Residual U-Net / {spec.regions.length} regions / checkpoint topology</summary>
    <p className="my-2 text-gray-400">Read-only restored topology. Preview at {width}×{height}px; image dimensions follow the generation request. KV caching is disabled during rewired execution.</p>
    {plan && <DiTRewiringGraph spec={spec} plan={plan} />}
    {spec.regions.map(region => <p key={region.id} className="mt-1 text-gray-300">
      {region.stream} / {region.id}: [{region.start_block}, {region.end_block}) / parent {region.parent ?? "full grid"} / downsample {region.down_factor.join("×")} / residual skip {region.start_block} → {region.end_block}
    </p>)}
    {error && <p role="alert" className="mt-2 text-red-400">{error}</p>}
  </details>;
}
