"use client";

import { useEffect } from "react";
import { useStartup } from "@/contexts/StartupContext";
import type { QwenPartitionConfig } from "@/utils/api";
import QwenPartitionControls from "./QwenPartitionControls";

type Props = { value?: QwenPartitionConfig | null; onChange: (value: QwenPartitionConfig | undefined) => void;
  layers?: number; width?: number; height?: number; background?: string | null };

export default function MingPartitionControls({ layers, ...props }: Props) {
  const { modelInfo, generationDefaults } = useStartup();
  useEffect(() => {
    if (modelInfo?.type && modelInfo.type !== "ming_image_01" && props.value) props.onChange(undefined);
  }, [modelInfo?.type, props.value, props.onChange]);
  if (modelInfo?.type !== "ming_image_01") return null;
  const count = layers ?? generationDefaults?.txt2img.ming_image_01_layers;
  if (modelInfo.profile === "layer_decompose" && typeof count !== "number") return null;
  const frames = modelInfo.profile === "layer_decompose" ? count! + 1 : 1;
  return <QwenPartitionControls {...props} architecture="ming_image_01" targetFrames={frames} />;
}
