// Shared HH:MM:SS:FF formatting, used by every timeline that shows a frame
// position next to a wall-clock-style readout, including
// `components/studio/StudioWorkspace.tsx`.
export function formatTimecode(seconds: number, fps: number): string {
  const safeFps = fps > 0 ? fps : 24;
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const wholeSeconds = Math.floor(safe % 60);
  const frames = Math.floor((safe - Math.floor(safe)) * safeFps);
  return [hours, minutes, wholeSeconds, frames]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

/** "frame 42 · 00:00:01:18" -- the combined label used by both timelines' hover/drag previews and readouts. */
export function formatFrameLabel(frame: number, fps: number): string {
  const safeFps = fps > 0 ? fps : 24;
  return `frame ${frame} · ${formatTimecode(frame / safeFps, safeFps)}`;
}
