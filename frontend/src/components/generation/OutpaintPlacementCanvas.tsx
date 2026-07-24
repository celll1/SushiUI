"use client";

import { useRef, useState } from "react";
import Slider from "../common/Slider";
import Select from "../common/Select";
import Button from "../common/Button";

// ---------------------------------------------------------------------------
// Small local helpers (intentionally NOT shared -- this is the only place
// that needs bbox rounding/clamping for the outpaint placement UI).
// ---------------------------------------------------------------------------
const roundToN = (value: number, n: number): number => (n > 0 ? Math.round(value / n) * n : Math.round(value));
const clamp = (value: number, min: number, max: number): number => Math.min(Math.max(value, min), max);

const MIN_PLACE_SIZE = 8;
const PREVIEW_MAX_DIM = 480;
// Minimum on-screen separation (px) two fingers must reach before a pinch
// gesture "starts" (snapshots dist/axis/place) -- see maybeStartPinch.
const MIN_PINCH_DIST = 12;

// Resolves the backend's "input_crop_w/h <= 0 means to the input's edge"
// sentinel (see validate_and_snap_placement in outpaint_utils.py) to a
// concrete pixel rect. Used anywhere the UI needs a real crop-w/h to derive
// a place:crop scale factor (drag math, preview rendering).
const resolveCropRect = (
  cropX: number,
  cropY: number,
  cropW: number,
  cropH: number,
  inputW: number,
  inputH: number
) => ({
  x: cropX,
  y: cropY,
  w: cropW > 0 ? cropW : Math.max(1, inputW - cropX),
  h: cropH > 0 ? cropH : Math.max(1, inputH - cropY),
});

export interface OutpaintPlacementParams {
  canvas_width: number;
  canvas_height: number;
  place_x: number;
  place_y: number;
  place_width: number;
  place_height: number;
  input_crop_x: number;
  input_crop_y: number;
  input_crop_w: number;
  input_crop_h: number;
  outpaint_fill_mode: string;
  mask_blur: number;
}

interface OutpaintPlacementCanvasProps {
  inputImagePreview: string | null;
  inputImageSize: { width: number; height: number } | null;
  params: OutpaintPlacementParams;
  onChange: (patch: Partial<OutpaintPlacementParams>) => void;
  // Lifted to the parent panel (was local state here) so panel-level
  // handlers -- new-image reset, "Reset Placement" -- can also drive it.
  // Governs Ctrl+drag RESIZE mode AND the default no-Ctrl CROP-drag mode
  // (see onPointerMove); Shift inverts it for the current drag either way.
  maintainAspect: boolean;
  onMaintainAspectChange: (value: boolean) => void;
}

type DragMode = "move" | "resize-nw" | "resize-ne" | "resize-sw" | "resize-se" | null;

interface DragStart {
  mouseX: number;
  mouseY: number;
  placeX: number;
  placeY: number;
  placeWidth: number;
  placeHeight: number;
  // Input-crop geometry at drag start, ALREADY resolved from the backend's
  // "0 = auto/full" sentinel to concrete pixel values (see resolveCropRect
  // below) -- crop-mode math always needs concrete numbers to derive a
  // place:crop scale factor.
  cropX: number;
  cropY: number;
  cropW: number;
  cropH: number;
}

const fillModeOptions = [
  { value: "replicate", label: "Replicate (edge-extend)" },
  { value: "reflect", label: "Reflect (mirror)" },
  { value: "mean", label: "Mean color (solid)" },
  { value: "noise", label: "Noise (uniform RGB)" },
];

export default function OutpaintPlacementCanvas({
  inputImagePreview,
  inputImageSize,
  params,
  onChange,
  maintainAspect,
  onMaintainAspectChange,
}: OutpaintPlacementCanvasProps) {
  const {
    canvas_width: canvasWidth,
    canvas_height: canvasHeight,
    place_x: placeX,
    place_y: placeY,
    place_width: placeWidth,
    place_height: placeHeight,
    input_crop_x: inputCropX,
    input_crop_y: inputCropY,
    input_crop_w: inputCropW,
    input_crop_h: inputCropH,
    outpaint_fill_mode: outpaintFillMode,
    mask_blur: maskBlur,
  } = params;

  const [snapEnabled, setSnapEnabled] = useState(true);
  // maintainAspect/onMaintainAspectChange are now props (lifted to
  // OutpaintPanel) -- see OutpaintPlacementCanvasProps above.
  const [dragMode, setDragMode] = useState<DragMode>(null);
  const dragStartRef = useRef<DragStart | null>(null);
  // The ACTUAL kind of change last applied by onPointerMove during the
  // current drag ("move" never touches size/crop; "resize" is Ctrl+drag,
  // place-only; "crop" is the default no-Ctrl drag, place+crop coupled).
  // onPointerUp reads this (not dragMode, which is just which handle was
  // grabbed) to decide whether the release-time snap may also touch
  // input_crop_w/h -- see onPointerUp.
  const lastDragKindRef = useRef<"move" | "resize" | "crop" | null>(null);

  const inputNativeW = inputImageSize?.width || 0;
  const inputNativeH = inputImageSize?.height || 0;

  // Fit the canvas into a PREVIEW_MAX_DIM x PREVIEW_MAX_DIM box, preserving
  // its aspect ratio -- this is purely a display transform; all committed
  // values stay in canvas pixel space.
  const scale = canvasWidth > 0 && canvasHeight > 0
    ? PREVIEW_MAX_DIM / Math.max(canvasWidth, canvasHeight)
    : 1;
  const previewW = Math.max(1, Math.round(canvasWidth * scale));
  const previewH = Math.max(1, Math.round(canvasHeight * scale));

  // Truthful preview geometry (fix A): render ONLY the input_crop_*
  // sub-rectangle of the source image, stretched to place_width x
  // place_height exactly like build_outpaint_canvas's
  // cropped.resize((place_width, place_height)) -- via CSS
  // background-position/background-size on the placed rect itself, instead
  // of an <img object-cover>, which (a) showed the WHOLE input regardless of
  // input_crop_* and (b) center-cropped to the frame's aspect instead of
  // reflecting an actual per-axis stretch. This is exact for every
  // combination of crop/place, in every drag mode.
  const previewCrop = resolveCropRect(inputCropX, inputCropY, inputCropW, inputCropH, inputNativeW, inputNativeH);
  const previewSx = previewCrop.w > 0 ? (placeWidth / previewCrop.w) * scale : scale;
  const previewSy = previewCrop.h > 0 ? (placeHeight / previewCrop.h) * scale : scale;
  const previewBackgroundStyle = !inputImagePreview
    ? undefined
    : inputNativeW > 0 && inputNativeH > 0
      ? {
          backgroundImage: `url(${inputImagePreview})`,
          backgroundRepeat: "no-repeat" as const,
          backgroundPosition: `${-previewCrop.x * previewSx}px ${-previewCrop.y * previewSy}px`,
          backgroundSize: `${inputNativeW * previewSx}px ${inputNativeH * previewSy}px`,
        }
      : {
          // Native size not known yet (the async img.onload that populates
          // inputImageSize hasn't fired) -- fall back to a simple
          // aspect-preserving cover so the thumbnail still appears
          // immediately during that brief decode window, same as the old
          // <img object-cover>. This ignores input_crop_* until the exact
          // crop/stretch background above takes over on the next render.
          backgroundImage: `url(${inputImagePreview})`,
          backgroundRepeat: "no-repeat" as const,
          backgroundPosition: "center" as const,
          backgroundSize: "cover" as const,
        };

  const commitDrag = (patch: Partial<OutpaintPlacementParams>) => {
    onChange(patch);
  };

  // --- Two-finger pinch resize (touch) -------------------------------------
  // Desktop mouse/Ctrl-drag is entirely untouched by this: a mouse only ever
  // produces a single simultaneous pointer, so activePointersRef never
  // reaches size 2 for it and none of the pinch branches below ever engage.
  const pts = (map: Map<number, { x: number; y: number }>) => Array.from(map.entries());
  const activePointersRef = useRef<Map<number, { x: number; y: number }>>(new Map());
  const pinchStartRef = useRef<{
    dist: number;
    axis: "both" | "w" | "h";
    place: { x: number; y: number; w: number; h: number };
    pointerIds: [number, number];
  } | null>(null);

  // Registers every pointer that goes down, from BOTH the container-level
  // background pointerdown AND startDrag (rect/move + the 4 corner handles)
  // -- a pinch may land its 2nd finger on either. A 2nd finger landing
  // always aborts whatever single-pointer move/resize/crop drag was in
  // progress (dragMode/dragStartRef nulled here); the actual pinch-start
  // SNAPSHOT (dist/axis/place) is deferred to maybeStartPinch below until
  // the fingers have spread apart enough to read a stable axis. A 3rd+
  // pointer is registered (so its later pointerup is a correctly-ignored
  // no-op) but otherwise has zero effect -- handlePinchMove/maybeStartPinch
  // only ever look at the first two REGISTERED pointer ids.
  const registerPointer = (e: React.PointerEvent<HTMLDivElement>) => {
    activePointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (activePointersRef.current.size === 2) {
      setDragMode(null);
      dragStartRef.current = null;
      lastDragKindRef.current = null;
      pinchStartRef.current = null;
    }
  };

  const maybeStartPinch = () => {
    const entries = pts(activePointersRef.current);
    if (entries.length < 2) return;
    const [[idA, a], [idB, b]] = entries; // first two REGISTERED pointers only (3rd+ ignored)
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const distNow = Math.hypot(dx, dy);
    if (distNow < MIN_PINCH_DIST) return; // still too close together -- defer

    // Axis decided ONCE here (not re-evaluated per-move, to avoid jitter):
    // "maintainAspect" pinches both axes together; otherwise the angle of
    // the line BETWEEN the two fingers picks which single axis the pinch
    // targets (near-horizontal spread -> width, near-vertical -> height,
    // diagonal -> both).
    const axis: "both" | "w" | "h" = maintainAspect
      ? "both"
      : (() => {
          const thetaDeg = (Math.atan2(Math.abs(dy), Math.abs(dx)) * 180) / Math.PI;
          if (thetaDeg < 30) return "w";
          if (thetaDeg > 60) return "h";
          return "both";
        })();

    pinchStartRef.current = {
      dist: distNow,
      axis,
      place: { x: placeX, y: placeY, w: placeWidth, h: placeHeight },
      pointerIds: [idA, idB],
    };
  };

  const handlePinchMove = () => {
    const start = pinchStartRef.current;
    if (!start || start.dist <= 0) return;
    const a = activePointersRef.current.get(start.pointerIds[0]);
    const b = activePointersRef.current.get(start.pointerIds[1]);
    if (!a || !b) return; // one of the 2 pinch pointers isn't tracked (shouldn't happen mid-pinch)

    const curDist = Math.hypot(b.x - a.x, b.y - a.y);
    const r = curDist / start.dist;
    const { w, h, x, y } = start.place;
    const cx = x + w / 2;
    const cy = y + h / 2;

    let newW = w;
    let newH = h;
    if (start.axis === "both") {
      const rMin = MIN_PLACE_SIZE / Math.min(w, h);
      const rMax = Math.min(canvasWidth / w, canvasHeight / h);
      const rClamped = clamp(r, rMin, Math.max(rMin, rMax));
      newW = w * rClamped;
      newH = h * rClamped;
    } else if (start.axis === "w") {
      newW = clamp(w * r, MIN_PLACE_SIZE, canvasWidth);
    } else {
      newH = clamp(h * r, MIN_PLACE_SIZE, canvasHeight);
    }

    const newX = clamp(Math.round(cx - newW / 2), 0, Math.max(0, canvasWidth - newW));
    const newY = clamp(Math.round(cy - newH / 2), 0, Math.max(0, canvasHeight - newH));

    lastDragKindRef.current = "resize";
    onChange({
      place_x: newX,
      place_y: newY,
      place_width: Math.round(newW),
      place_height: Math.round(newH),
    });
  };

  const handleContainerPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    registerPointer(e);
  };
  // --- end two-finger pinch resize ------------------------------------------

  const startDrag = (mode: DragMode) => (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    registerPointer(e);
    if (activePointersRef.current.size >= 2) {
      // A 2nd (or 3rd+) finger landing directly on the rect/handle -- a
      // pinch takes over (registerPointer already aborted any in-progress
      // single-pointer drag above); never start a new one for this pointer.
      return;
    }
    setDragMode(mode);
    const resolvedCrop = resolveCropRect(inputCropX, inputCropY, inputCropW, inputCropH, inputNativeW, inputNativeH);
    dragStartRef.current = {
      mouseX: e.clientX,
      mouseY: e.clientY,
      placeX,
      placeY,
      placeWidth,
      placeHeight,
      cropX: resolvedCrop.x,
      cropY: resolvedCrop.y,
      cropW: resolvedCrop.w,
      cropH: resolvedCrop.h,
    };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    // Keep this pointer's live position current for pinch distance/axis math
    // (a harmless no-op for any pointer not involved in a pinch).
    if (activePointersRef.current.has(e.pointerId)) {
      activePointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    }

    if (pinchStartRef.current) {
      handlePinchMove();
      return;
    }
    if (activePointersRef.current.size === 2) {
      // 2 fingers down but not yet spread past MIN_PINCH_DIST -- crop/move
      // never runs mid-(pending-)pinch; just check whether it's time to
      // snapshot the pinch start now.
      maybeStartPinch();
      return;
    }

    if (!dragMode || !dragStartRef.current) return;
    const start = dragStartRef.current;
    const dx = (e.clientX - start.mouseX) / scale;
    const dy = (e.clientY - start.mouseY) / scale;

    if (dragMode === "move") {
      lastDragKindRef.current = "move";
      const newX = clamp(Math.round(start.placeX + dx), 0, Math.max(0, canvasWidth - start.placeWidth));
      const newY = clamp(Math.round(start.placeY + dy), 0, Math.max(0, canvasHeight - start.placeHeight));
      onChange({ place_x: newX, place_y: newY });
      return;
    }

    // Ctrl (or Cmd on Mac) held during an edge/corner drag = RESIZE/SCALE
    // mode: adjusts place_width/place_height only (the placed input is
    // scaled -- and, per the "Maintain aspect ratio" checkbox, may be
    // stretched). This is the OLD default drag behavior, now opt-in.
    if (e.ctrlKey || e.metaKey) {
      lastDragKindRef.current = "resize";
      let newX = start.placeX;
      let newY = start.placeY;
      let newW = start.placeWidth;
      let newH = start.placeHeight;

      if (dragMode.includes("e")) newW = start.placeWidth + dx;
      if (dragMode.includes("w")) { newW = start.placeWidth - dx; newX = start.placeX + dx; }
      if (dragMode.includes("s")) newH = start.placeHeight + dy;
      if (dragMode.includes("n")) { newH = start.placeHeight - dy; newY = start.placeY + dy; }

      // "Maintain aspect ratio" checkbox governs proportional resize; Shift
      // inverts it for the current drag (a temporary override), same
      // interaction pattern as most image editors' free-transform tools.
      const effectiveAspectLock = maintainAspect !== e.shiftKey;
      if (effectiveAspectLock && start.placeHeight > 0) {
        // Width is always the driver axis here (matches the pre-existing
        // newH = newW / aspect derivation); what's new is that ALL bounds
        // (both axes, plus the anchored-edge limits) are intersected in
        // W-space BEFORE deriving H, so a corner drag that would have hit an
        // H-axis bound can no longer silently distort the ratio the way an
        // independent per-axis clamp would.
        const aspect = start.placeWidth / start.placeHeight; // W:H, preserved exactly
        let minW = MIN_PLACE_SIZE;
        let maxW = canvasWidth;
        // H-bounds [MIN_PLACE_SIZE, canvasHeight], expressed in W-space (W = H * aspect).
        minW = Math.max(minW, MIN_PLACE_SIZE * aspect);
        maxW = Math.min(maxW, canvasHeight * aspect);

        if (dragMode.includes("e")) {
          // Left edge anchored at start.placeX -- newX stays start.placeX.
          maxW = Math.min(maxW, canvasWidth - start.placeX);
        } else if (dragMode.includes("w")) {
          // Right edge anchored at start.placeX + start.placeWidth.
          maxW = Math.min(maxW, start.placeX + start.placeWidth);
        }
        if (dragMode.includes("s")) {
          // Top edge anchored at start.placeY (H-bound, in W-space).
          maxW = Math.min(maxW, (canvasHeight - start.placeY) * aspect);
        } else if (dragMode.includes("n")) {
          // Bottom edge anchored at start.placeY + start.placeHeight.
          maxW = Math.min(maxW, (start.placeY + start.placeHeight) * aspect);
        }

        newW = clamp(newW, minW, Math.max(minW, maxW));
        newH = newW / aspect;
        newX = dragMode.includes("w") ? start.placeX + start.placeWidth - newW : start.placeX;
        newY = dragMode.includes("n") ? start.placeY + start.placeHeight - newH : start.placeY;
      } else {
        newW = clamp(newW, MIN_PLACE_SIZE, canvasWidth);
        newH = clamp(newH, MIN_PLACE_SIZE, canvasHeight);
        newX = clamp(newX, 0, canvasWidth - newW);
        newY = clamp(newY, 0, canvasHeight - newH);
      }

      onChange({
        place_x: Math.round(newX),
        place_y: Math.round(newY),
        place_width: Math.round(newW),
        place_height: Math.round(newH),
      });
      return;
    }

    // Default (no Ctrl) = CROP mode: adjusts input_crop_* (which part of the
    // input is used), never place_width:place_height's ratio to
    // input_crop_w:input_crop_h -- i.e. the backend's
    // cropped.resize((place_width, place_height)) in build_outpaint_canvas
    // is always a UNIFORM (non-distorting) scale, matching whatever
    // place:crop scale was already in effect (1:1 -- true native pixels --
    // until the user explicitly scales via Ctrl+drag).
    //
    // Geometry: sx/sy (canvas px per input px) is locked from the
    // drag-start place:crop ratio. Each dragged edge maps its canvas-pixel
    // delta to an input-pixel crop delta via dx/sx (dy/sy), with the
    // OPPOSITE edge staying anchored (mirrors the resize-mode w/n branch
    // above). The crop delta is clamped to the input's own bounds AND
    // (converted back through sx/sy) to the canvas bounds in a SINGLE pass,
    // so place is re-derived from the final crop and can never diverge from
    // the sx/sy ratio -- this is what keeps validate_and_snap_placement's
    // independent per-axis place_w/place_h cap from ever binding in a way
    // that would distort the resize.
    lastDragKindRef.current = "crop";
    const inW = inputNativeW || start.cropX + start.cropW;
    const inH = inputNativeH || start.cropY + start.cropH;
    const sxRaw = start.cropW > 0 ? start.placeWidth / start.cropW : 1;
    const syRaw = start.cropH > 0 ? start.placeHeight / start.cropH : 1;
    const sx = Number.isFinite(sxRaw) && sxRaw > 0 ? sxRaw : 1;
    const sy = Number.isFinite(syRaw) && syRaw > 0 ? syRaw : 1;

    const minCropW = Math.max(1, MIN_PLACE_SIZE / sx);
    const minCropH = Math.max(1, MIN_PLACE_SIZE / sy);

    // "Maintain aspect ratio" (same Shift-invertible flag as the Ctrl-resize
    // branch above) also governs this default crop-drag now -- there are no
    // separate side handles (only the 4 corner handles: nw/ne/sw/se, see
    // `handles` below), so every crop-drag here always touches BOTH axes
    // and a lock is meaningful. When locked, the crop rect's aspect ratio
    // (start.cropW / start.cropH) is preserved exactly; place is then
    // derived from crop at the fixed sx/sy scale as before, so place's
    // aspect ends up exactly preserved too. Unlocked keeps the original,
    // fully independent per-axis crop math (byte-identical).
    const effectiveAspectLock = maintainAspect !== e.shiftKey;

    let newCropW: number;
    let newCropX: number;
    let newCropH: number;
    let newCropY: number;

    if (effectiveAspectLock && start.cropW > 0 && start.cropH > 0) {
      const ar = start.cropW / start.cropH; // crop-space W:H, preserved exactly

      // Desired (unclamped) size along each dragged axis independently --
      // used ONLY to pick the DRIVER axis (whichever the user moved further,
      // relative to its own start size), so a diagonal drag doesn't
      // arbitrarily prefer one axis over the other.
      let desiredW = start.cropW;
      if (dragMode.includes("e")) desiredW = start.cropW + dx / sx;
      else if (dragMode.includes("w")) desiredW = start.cropW - dx / sx;
      let desiredH = start.cropH;
      if (dragMode.includes("s")) desiredH = start.cropH + dy / sy;
      else if (dragMode.includes("n")) desiredH = start.cropH - dy / sy;

      const relW = Math.abs(desiredW - start.cropW) / start.cropW;
      const relH = Math.abs(desiredH - start.cropH) / start.cropH;
      const wantW = relH > relW ? desiredH * ar : desiredW;

      // Joint-clamp in W-space: intersect the W-axis's own bounds with the
      // H-axis's bounds converted through `ar` (mirrors the Ctrl-resize
      // branch's W-space joint clamp), using the SAME per-edge input/canvas
      // bounds as the unlocked branch below.
      let minW = Math.max(minCropW, minCropH * ar);
      let maxW = inW;

      if (dragMode.includes("e")) {
        maxW = Math.min(maxW, inW - start.cropX, (canvasWidth - start.placeX) / sx);
      } else if (dragMode.includes("w")) {
        const rightEdgeInput = start.cropX + start.cropW;
        const rightEdgeCanvas = start.placeX + start.placeWidth;
        maxW = Math.min(maxW, rightEdgeInput, rightEdgeCanvas / sx);
      }
      if (dragMode.includes("s")) {
        const maxCropH = Math.min(inH - start.cropY, (canvasHeight - start.placeY) / sy);
        maxW = Math.min(maxW, maxCropH * ar);
      } else if (dragMode.includes("n")) {
        const bottomEdgeInput = start.cropY + start.cropH;
        const bottomEdgeCanvas = start.placeY + start.placeHeight;
        const maxCropH = Math.min(bottomEdgeInput, bottomEdgeCanvas / sy);
        maxW = Math.min(maxW, maxCropH * ar);
      }

      newCropW = clamp(wantW, minW, Math.max(minW, maxW));
      newCropH = newCropW / ar;
      newCropX = dragMode.includes("w") ? start.cropX + start.cropW - newCropW : start.cropX;
      newCropY = dragMode.includes("n") ? start.cropY + start.cropH - newCropH : start.cropY;
    } else {
      newCropW = start.cropW;
      newCropX = start.cropX;
      if (dragMode.includes("e")) {
        const maxCropW = Math.min(inW - start.cropX, (canvasWidth - start.placeX) / sx);
        newCropW = clamp(start.cropW + dx / sx, minCropW, Math.max(minCropW, maxCropW));
        newCropX = start.cropX;
      } else if (dragMode.includes("w")) {
        const rightEdgeInput = start.cropX + start.cropW;
        const rightEdgeCanvas = start.placeX + start.placeWidth;
        const maxCropW = Math.min(rightEdgeInput, rightEdgeCanvas / sx);
        newCropW = clamp(start.cropW - dx / sx, minCropW, Math.max(minCropW, maxCropW));
        newCropX = rightEdgeInput - newCropW;
      }

      newCropH = start.cropH;
      newCropY = start.cropY;
      if (dragMode.includes("s")) {
        const maxCropH = Math.min(inH - start.cropY, (canvasHeight - start.placeY) / sy);
        newCropH = clamp(start.cropH + dy / sy, minCropH, Math.max(minCropH, maxCropH));
        newCropY = start.cropY;
      } else if (dragMode.includes("n")) {
        const bottomEdgeInput = start.cropY + start.cropH;
        const bottomEdgeCanvas = start.placeY + start.placeHeight;
        const maxCropH = Math.min(bottomEdgeInput, bottomEdgeCanvas / sy);
        newCropH = clamp(start.cropH - dy / sy, minCropH, Math.max(minCropH, maxCropH));
        newCropY = bottomEdgeInput - newCropH;
      }
    }

    // Place is derived FROM the (already fully clamped) crop at the locked
    // sx/sy scale -- guarantees place_w:place_h == crop_w:crop_h exactly.
    const newPlaceW = sx * newCropW;
    const newPlaceH = sy * newCropH;
    const newPlaceX = dragMode.includes("w") ? start.placeX + start.placeWidth - newPlaceW : start.placeX;
    const newPlaceY = dragMode.includes("n") ? start.placeY + start.placeHeight - newPlaceH : start.placeY;

    onChange({
      input_crop_x: Math.round(newCropX),
      input_crop_y: Math.round(newCropY),
      input_crop_w: Math.round(newCropW),
      input_crop_h: Math.round(newCropH),
      place_x: Math.round(newPlaceX),
      place_y: Math.round(newPlaceY),
      place_width: Math.round(newPlaceW),
      place_height: Math.round(newPlaceH),
    });
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    // A pinch pointer lifting always ENDS the whole pinch gesture (never
    // downgrades to a single-pointer drag with whichever finger remains) --
    // all fingers must be lifted before any new gesture can begin. An
    // ignored 3rd+ finger's pointerup (its id isn't one of the pinch's own
    // 2 pointerIds) intentionally falls through to the `!dragMode` bail
    // below and changes nothing.
    const pinchStart = pinchStartRef.current;
    const isPinchPointer = !!pinchStart && (e.pointerId === pinchStart.pointerIds[0] || e.pointerId === pinchStart.pointerIds[1]);
    activePointersRef.current.delete(e.pointerId);

    if (!dragMode && !isPinchPointer) return;
    // The ACTUAL applied kind for this drag/gesture (not just "which handle
    // was grabbed" -- dragMode alone can't distinguish Ctrl-resize from a
    // no-Ctrl crop-drag on the same handle, and a pinch gesture never sets
    // dragMode at all -- handlePinchMove sets this to "resize" on every
    // pinch move instead, so this release-time snap is shared verbatim).
    // null if pointerup fired without any pointermove (a plain click) --
    // nothing changed, so no snap-driven patch is needed either way.
    const kind = lastDragKindRef.current;
    setDragMode(null);
    dragStartRef.current = null;
    lastDragKindRef.current = null;
    if (isPinchPointer) {
      pinchStartRef.current = null;
    }
    if (snapEnabled && kind) {
      const patch: Partial<OutpaintPlacementParams> = {
        place_x: roundToN(placeX, 8),
        place_y: roundToN(placeY, 8),
      };
      // A plain move never touches size, so only snap position -- snapping
      // place_width/height here too (the old, pre-F3 behavior) could shift
      // the place:crop scale for no reason, since move never intended a
      // resize (a non-8-multiple Place Width/Height slider value plus any
      // move+release would silently nudge the resize scale).
      if (kind === "resize" || kind === "crop") {
        const snappedW = Math.max(MIN_PLACE_SIZE, roundToN(placeWidth, 8));
        const snappedH = Math.max(MIN_PLACE_SIZE, roundToN(placeHeight, 8));
        patch.place_width = snappedW;
        patch.place_height = snappedH;

        // Only a crop-drag (never a Ctrl-resize) re-derives input_crop_w/h:
        // re-deriving it after resize would overwrite/erode the crop on
        // every Ctrl-resize release (Ctrl-resize intentionally never
        // touches input_crop_* -- see onPointerMove). For a crop-drag, this
        // re-derives input_crop_w/h from the SAME (current, pre-snap)
        // place:crop scale so this cosmetic 8px snap can never reintroduce
        // a place_w:place_h != input_crop_w:input_crop_h mismatch --
        // crop_x/crop_y (position) are unaffected since only size ratios
        // can cause resize() distortion.
        if (kind === "crop") {
          const resolved = resolveCropRect(inputCropX, inputCropY, inputCropW, inputCropH, inputNativeW, inputNativeH);
          const curSx = resolved.w > 0 ? placeWidth / resolved.w : 0;
          const curSy = resolved.h > 0 ? placeHeight / resolved.h : 0;
          if (Number.isFinite(curSx) && curSx > 0) patch.input_crop_w = Math.max(1, Math.round(snappedW / curSx));
          if (Number.isFinite(curSy) && curSy > 0) patch.input_crop_h = Math.max(1, Math.round(snappedH / curSy));
        }
      }
      commitDrag(patch);
    }
  };

  const handleCenter = () => {
    onChange({
      place_x: Math.max(0, Math.round((canvasWidth - placeWidth) / 2)),
      place_y: Math.max(0, Math.round((canvasHeight - placeHeight) / 2)),
    });
  };

  const handleFit15x = () => {
    const baseW = inputImageSize?.width || placeWidth || 1024;
    const baseH = inputImageSize?.height || placeHeight || 1024;
    const newCanvasW = Math.max(64, roundToN(baseW * 1.5, 16));
    const newCanvasH = Math.max(64, roundToN(baseH * 1.5, 16));
    onChange({
      canvas_width: newCanvasW,
      canvas_height: newCanvasH,
      place_width: baseW,
      place_height: baseH,
      place_x: Math.max(0, Math.round((newCanvasW - baseW) / 2)),
      place_y: Math.max(0, Math.round((newCanvasH - baseH) / 2)),
    });
  };

  // Reuses handleFit15x's canvas/place math (native size, centered, 1.5x
  // canvas) but ALSO clears input_crop_* (a fresh "use full input" state)
  // and re-enables the aspect lock -- a single-click "start over" for the
  // whole placement, narrower in scope than the panel-level resetToDefault
  // (which also resets prompt/generation params; this never does).
  const handleResetPlacement = () => {
    const baseW = inputImageSize?.width || placeWidth || 1024;
    const baseH = inputImageSize?.height || placeHeight || 1024;
    const newCanvasW = Math.max(64, roundToN(baseW * 1.5, 16));
    const newCanvasH = Math.max(64, roundToN(baseH * 1.5, 16));
    onChange({
      canvas_width: newCanvasW,
      canvas_height: newCanvasH,
      place_width: baseW,
      place_height: baseH,
      place_x: Math.max(0, Math.round((newCanvasW - baseW) / 2)),
      place_y: Math.max(0, Math.round((newCanvasH - baseH) / 2)),
      input_crop_x: 0,
      input_crop_y: 0,
      input_crop_w: 0,
      input_crop_h: 0,
    });
    onMaintainAspectChange(true);
  };

  // Re-fits place_width:place_height to the CURRENT effective crop rect's
  // aspect ratio (without touching the crop itself), keeping place_width
  // and the rect's center fixed -- a one-click fix for a placement rect
  // that's drifted out of sync with its source crop (e.g. after unlocked
  // drags, or Place Width/Height slider edits). Touches ONLY place_* --
  // never prompt, crop, or the aspect-lock checkbox itself.
  const handleRestoreAspect = () => {
    const resolved = resolveCropRect(inputCropX, inputCropY, inputCropW, inputCropH, inputNativeW, inputNativeH);
    const ar = resolved.h > 0 ? resolved.w / resolved.h : 0;
    if (!Number.isFinite(ar) || ar <= 0) return;

    const cx = placeX + placeWidth / 2;
    const cy = placeY + placeHeight / 2;

    let newW = placeWidth;
    let newH = newW / ar;
    // Joint-clamp: if the height derived from the current place_width falls
    // outside the canvas, clamp height first, then RE-derive width from the
    // clamped height (via the same `ar`) so the ratio stays exact -- an
    // independent per-axis clamp here could silently distort it again.
    if (newH < MIN_PLACE_SIZE || newH > canvasHeight) {
      newH = clamp(newH, MIN_PLACE_SIZE, canvasHeight);
      newW = newH * ar;
    }

    const newX = clamp(Math.round(cx - newW / 2), 0, Math.max(0, canvasWidth - newW));
    const newY = clamp(Math.round(cy - newH / 2), 0, Math.max(0, canvasHeight - newH));

    onChange({
      place_x: newX,
      place_y: newY,
      place_width: Math.round(newW),
      place_height: Math.round(newH),
    });
  };

  // "Expand <direction>" pins the input against the opposite edge and grows
  // the canvas by a 50%-of-input margin on the requested side only (a bbox
  // preset, per the plan's directional-expand convenience buttons).
  const handleExpand = (direction: "left" | "right" | "top" | "bottom") => {
    const w = placeWidth || inputImageSize?.width || 1024;
    const h = placeHeight || inputImageSize?.height || 1024;
    if (direction === "left" || direction === "right") {
      const margin = Math.max(64, roundToN(w * 0.5, 16));
      const newCanvasW = roundToN(w + margin, 16);
      onChange({
        canvas_width: newCanvasW,
        canvas_height: roundToN(h, 16),
        place_width: w,
        place_height: h,
        place_x: direction === "right" ? 0 : newCanvasW - w,
        place_y: 0,
      });
    } else {
      const margin = Math.max(64, roundToN(h * 0.5, 16));
      const newCanvasH = roundToN(h + margin, 16);
      onChange({
        canvas_width: roundToN(w, 16),
        canvas_height: newCanvasH,
        place_width: w,
        place_height: h,
        place_x: 0,
        place_y: direction === "bottom" ? 0 : newCanvasH - h,
      });
    }
  };

  const handles: Array<{ key: "nw" | "ne" | "sw" | "se"; className: string; cursor: string }> = [
    { key: "nw", className: "-top-1.5 -left-1.5", cursor: "cursor-nwse-resize" },
    { key: "ne", className: "-top-1.5 -right-1.5", cursor: "cursor-nesw-resize" },
    { key: "sw", className: "-bottom-1.5 -left-1.5", cursor: "cursor-nesw-resize" },
    { key: "se", className: "-bottom-1.5 -right-1.5", cursor: "cursor-nwse-resize" },
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Slider
          label="Canvas Width"
          value={canvasWidth}
          min={64}
          max={8192}
          step={16}
          onChange={(e) => onChange({ canvas_width: Math.max(64, roundToN(parseInt(e.target.value || "0", 10), 16)) })}
        />
        <Slider
          label="Canvas Height"
          value={canvasHeight}
          min={64}
          max={8192}
          step={16}
          onChange={(e) => onChange({ canvas_height: Math.max(64, roundToN(parseInt(e.target.value || "0", 10), 16)) })}
        />
      </div>
      <p className="text-xs text-gray-500">
        Canvas dimensions are snapped to multiples of 16 (the loaded architecture's latent grid); the backend re-validates regardless.
      </p>

      <div className="flex flex-wrap gap-2">
        <Button onClick={handleCenter} variant="secondary" size="sm">Center</Button>
        <Button onClick={handleFit15x} variant="secondary" size="sm">Fit 1.5x</Button>
        <Button onClick={handleResetPlacement} variant="secondary" size="sm" disabled={!inputImageSize}>Reset Placement</Button>
        <Button onClick={handleRestoreAspect} variant="secondary" size="sm" disabled={!inputImageSize}>Restore Aspect</Button>
        <Button onClick={() => handleExpand("left")} variant="secondary" size="sm">Expand Left</Button>
        <Button onClick={() => handleExpand("right")} variant="secondary" size="sm">Expand Right</Button>
        <Button onClick={() => handleExpand("top")} variant="secondary" size="sm">Expand Top</Button>
        <Button onClick={() => handleExpand("bottom")} variant="secondary" size="sm">Expand Bottom</Button>
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="outpaint_snap8"
          checked={snapEnabled}
          onChange={(e) => setSnapEnabled(e.target.checked)}
          className="rounded"
        />
        <label htmlFor="outpaint_snap8" className="text-sm text-gray-300">
          Snap placement to 8px on release
        </label>
        <span className="text-xs text-gray-500">(cosmetic only -- the backend always guarantees exact placement)</span>
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="outpaint_maintain_aspect"
          checked={maintainAspect}
          onChange={(e) => onMaintainAspectChange(e.target.checked)}
          className="rounded"
        />
        <label htmlFor="outpaint_maintain_aspect" className="text-sm text-gray-300">
          Maintain aspect ratio when resizing
        </label>
        <span className="text-xs text-gray-500">
          (applies to both crop-drag and Ctrl+drag resize, and to pinch-resize on touch; Shift inverts it for the current drag)
        </span>
      </div>

      {/* Scaled visual preview: canvas box + draggable/resizable placed rect */}
      <div className="flex justify-center py-2">
        <div
          className="relative bg-gray-800 border border-gray-600 rounded overflow-hidden touch-none select-none"
          style={{ width: previewW, height: previewH }}
          onPointerDown={handleContainerPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          {/* Fill-mode backdrop hint */}
          <div className="absolute inset-0 bg-[repeating-linear-gradient(45deg,rgba(255,255,255,0.03)_0,rgba(255,255,255,0.03)_6px,transparent_6px,transparent_12px)]" />

          <div
            onPointerDown={startDrag("move")}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            className={`absolute border-2 border-blue-500 bg-blue-500/10 ${dragMode === "move" ? "cursor-grabbing" : "cursor-grab"}`}
            style={{
              left: placeX * scale,
              top: placeY * scale,
              width: Math.max(1, placeWidth * scale),
              height: Math.max(1, placeHeight * scale),
              ...previewBackgroundStyle,
            }}
            title="Drag to move. Drag edges/corners to CROP the input (native scale, never stretches). Hold Ctrl while dragging edges/corners to RESIZE/scale the placed input instead. 'Maintain aspect ratio' locks both drag modes to the current ratio (Shift inverts it for the current drag); on touch, pinch with two fingers to resize."
          >
            {handles.map((h) => (
              <div
                key={h.key}
                onPointerDown={startDrag(`resize-${h.key}` as DragMode)}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
                className={`absolute w-3 h-3 bg-blue-500 border border-white rounded-sm ${h.className} ${h.cursor}`}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Slider
          label="Place X"
          value={placeX}
          min={0}
          max={Math.max(0, canvasWidth - placeWidth)}
          step={1}
          onChange={(e) => onChange({ place_x: clamp(parseInt(e.target.value || "0", 10), 0, Math.max(0, canvasWidth - placeWidth)) })}
        />
        <Slider
          label="Place Y"
          value={placeY}
          min={0}
          max={Math.max(0, canvasHeight - placeHeight)}
          step={1}
          onChange={(e) => onChange({ place_y: clamp(parseInt(e.target.value || "0", 10), 0, Math.max(0, canvasHeight - placeHeight)) })}
        />
        <Slider
          label="Place Width"
          value={placeWidth}
          min={MIN_PLACE_SIZE}
          max={Math.max(MIN_PLACE_SIZE, canvasWidth - placeX)}
          step={1}
          onChange={(e) => onChange({ place_width: clamp(parseInt(e.target.value || "0", 10), MIN_PLACE_SIZE, Math.max(MIN_PLACE_SIZE, canvasWidth - placeX)) })}
        />
        <Slider
          label="Place Height"
          value={placeHeight}
          min={MIN_PLACE_SIZE}
          max={Math.max(MIN_PLACE_SIZE, canvasHeight - placeY)}
          step={1}
          onChange={(e) => onChange({ place_height: clamp(parseInt(e.target.value || "0", 10), MIN_PLACE_SIZE, Math.max(MIN_PLACE_SIZE, canvasHeight - placeY)) })}
        />
      </div>

      <details className="bg-gray-800/40 border border-gray-700 rounded-lg p-3">
        <summary className="text-sm font-semibold text-gray-300 cursor-pointer select-none">
          Input Crop (trim before placement)
        </summary>
        <p className="text-xs text-gray-500 mt-2 mb-2">0 width/height = no trim (use the full input).</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Slider
            label="Crop X"
            value={inputCropX}
            min={0}
            max={inputImageSize?.width ?? 8192}
            step={1}
            onChange={(e) => onChange({ input_crop_x: Math.max(0, parseInt(e.target.value || "0", 10)) })}
          />
          <Slider
            label="Crop Y"
            value={inputCropY}
            min={0}
            max={inputImageSize?.height ?? 8192}
            step={1}
            onChange={(e) => onChange({ input_crop_y: Math.max(0, parseInt(e.target.value || "0", 10)) })}
          />
          <Slider
            label="Crop Width (0 = full)"
            value={inputCropW}
            min={0}
            max={inputImageSize?.width ?? 8192}
            step={1}
            onChange={(e) => onChange({ input_crop_w: Math.max(0, parseInt(e.target.value || "0", 10)) })}
          />
          <Slider
            label="Crop Height (0 = full)"
            value={inputCropH}
            min={0}
            max={inputImageSize?.height ?? 8192}
            step={1}
            onChange={(e) => onChange({ input_crop_h: Math.max(0, parseInt(e.target.value || "0", 10)) })}
          />
        </div>
      </details>

      <Select
        label="Canvas Fill Mode"
        value={outpaintFillMode || "replicate"}
        onChange={(e) => onChange({ outpaint_fill_mode: e.target.value })}
        options={fillModeOptions}
      />

      <Slider
        label="Mask Blur (outward-only)"
        min={0}
        max={64}
        step={1}
        value={maskBlur}
        onChange={(e) => onChange({ mask_blur: parseInt(e.target.value) })}
      />
    </div>
  );
}
