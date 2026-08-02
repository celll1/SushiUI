"use client";

import { useEffect, useState } from "react";
import Select from "@/components/common/Select";
import { useStartup } from "@/contexts/StartupContext";
import {
  archSupportsFeature,
  getFp8ScaledMm,
  getInt8Mm,
  type QuantizedGemmMode,
} from "@/utils/api";

interface QuantizedGemmSelectProps {
  /** Loaded model architecture (e.g. "krea2"). Undefined = no model loaded. */
  arch?: string | null;
  /** Current per-generation value; null = inherit the process setting. */
  value?: QuantizedGemmMode;
  onChange: (value: QuantizedGemmMode) => void;
}

/**
 * Per-generation selection of the quantized GEMM path (`quantized_gemm_mode`).
 *
 * Renders NOTHING on architectures that do not consume it. Which those are is
 * decided by the backend capability matrix (`GET /schema/arch-capabilities`,
 * feature `quantized_gemm`), not by an arch list duplicated here — only the
 * architectures whose loaders swap in weight-only quantized Linear layers
 * (Ideogram 4, Krea 2, Anima) have a quantized GEMM to select at all.
 *
 * A DIFFERENT axis from the "Transformer/U-Net Quantization" selector next to
 * it: that one quantizes an unquantized model's weights at load time to reduce
 * VRAM, this one selects how weights that are ALREADY quantized in the
 * checkpoint are multiplied. On Ideogram 4 and Krea 2 the weight-quantization
 * selector is declared not applied at all (the checkpoint format decides), so
 * the two are never interchangeable.
 *
 * When "Default" is selected the current process-level value is shown inline,
 * read from `GET /system/fp8-scaled-mm` and `GET /system/int8-mm`, so the
 * inherited setting is visible instead of invisible.
 *
 * Deliberately makes no speed or quality claim: neither path has an end-to-end
 * measurement, and the two are numerically different functions.
 */
export default function QuantizedGemmSelect({
  arch,
  value,
  onChange,
}: QuantizedGemmSelectProps) {
  const { archCapabilities } = useStartup();
  const supported = archSupportsFeature(
    archCapabilities,
    arch,
    "quantized_gemm"
  );

  const [processLabel, setProcessLabel] = useState<string | null>(null);

  useEffect(() => {
    if (!supported) {
      setProcessLabel(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const [fp8, int8] = await Promise.all([getFp8ScaledMm(), getInt8Mm()]);
        if (cancelled) return;
        setProcessLabel(
          `FP8 ${fp8.enabled ? "W8A8" : "dequant"}, INT8 ${
            int8.enabled ? "W8A8" : "dequant"
          }`
        );
      } catch {
        if (!cancelled) setProcessLabel(null);
      }
    })();
    return () => {
      cancelled = true;
    };
    // `value` is a dependency on purpose: a generation that forced the flags
    // changes the process state, so re-read it whenever the selection moves.
  }, [supported, arch, value]);

  if (!supported) return null;

  const defaultLabel = processLabel
    ? `Default (currently: ${processLabel})`
    : "Default (process setting)";

  return (
    <div className="space-y-2 border-t border-gray-700 pt-3">
      <p className="text-sm font-medium text-gray-300">Quantization</p>
      <Select
        label="Quantized GEMM path"
        value={value ?? "default"}
        onChange={(e) =>
          onChange(
            e.target.value === "default"
              ? null
              : (e.target.value as QuantizedGemmMode)
          )
        }
        options={[
          { value: "default", label: defaultLabel },
          { value: "w8a8", label: "W8A8 (quantized activation, quantized GEMM)" },
          { value: "dequant", label: "Dequant (dequantize weight, normal matmul)" },
        ]}
      />
      <p className="text-xs text-gray-500">
        Applies to Linear layers whose weights are already quantized in the
        checkpoint. FP8 vs INT8 is fixed by the checkpoint, so this selects only
        W8A8 or the dequantized matmul; the two are numerically different
        functions. Separate from the weight-quantization selector above, which
        quantizes an unquantized model&apos;s weights at load time to reduce
        VRAM. &quot;Default&quot; sends nothing and leaves the process-level
        setting (Settings &rarr; Quantized GEMM) in force.
      </p>
    </div>
  );
}
