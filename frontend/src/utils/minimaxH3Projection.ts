import { MiniMaxH3TeAgreement } from "@/utils/api";

// A projection is named by absolute path in some places and by the listing's
// extension-stripped name in others, while `agreement.projection` is a plain
// basename. Reduce all three to the same key before comparing.
export function projectionKey(value: string | null | undefined): string | null {
  if (!value) return null;
  const base = value.split(/[\\/]/).pop() || value;
  return base.replace(/\.safetensors$/i, "").toLowerCase();
}

// The measurement is keyed by the (encoder, projection) PAIR, so a number taken
// on one projection says nothing about the same encoder driven through another.
// The single rule both selectors apply before presenting one.
export function agreementCoversProjection(
  agreement: MiniMaxH3TeAgreement | null | undefined,
  selectedProjection: string | null | undefined,
): boolean {
  const measured = projectionKey(agreement?.projection);
  const selected = projectionKey(selectedProjection);
  return measured != null && selected != null && measured === selected;
}
