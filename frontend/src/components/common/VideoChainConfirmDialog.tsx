"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArchCapabilities,
  VideoChainContinuationMode,
  VideoChainIssue,
  VideoChainManifest,
  VideoChainPlan,
  VideoChainPlanRequest,
  VideoChainPlanRequestSeedPolicy,
  VideoChainReferenceInput,
  chainContextCapability,
  chainContinuationOverlapLengths,
  planVideoChainRequest,
  validateVideoChainManifest,
} from "@/utils/api";

/** Everything POST /video-chain/plan needs that only the panel knows. Null
 *  when the panel cannot describe the request to the planner at all (no
 *  loaded architecture), which is a planner-unavailable case, not a licence
 *  to copy the root prompt into every segment. */
export interface VideoChainPlanInput {
  architecture: string;
  variant?: string | null;
  rootPrompt: string;
  negativePrompt?: string;
  targetFrames: number;
  fps: number;
  requestedSegmentFrames?: number | null;
  seedPolicy?: VideoChainPlanRequestSeedPolicy;
  rootSeed?: number;
  /** Reference inventory: kind and label only, in the order the model reads
   *  them (see videoChain.ts's `buildChainImageReferenceInventory`). */
  references?: VideoChainReferenceInput[];
}

export interface VideoChainConfirmDialogProps {
  isOpen: boolean;
  /** The length actually held in the frame control, in frames. */
  requestedFrames: number;
  /**
   * The length of clip ONE request in the plan produces, in frames -- the
   * user's `chain_segment_frames` when they set one, otherwise the loaded
   * architecture's own single-inference cap (see api.ts's `chainSegmentCap`).
   * Not necessarily a hard technical wall: on an architecture with no
   * `max_frames` at all, this is purely the user's own chosen segment size.
   */
  capFrames: number;
  /** Pre-formatted (e.g. `.toFixed(2)`) seconds readout, or null if unknown. */
  capSeconds: string | null;
  /** Pre-formatted seconds readout for `plan.finalFrames`, or null if unknown. */
  finalSeconds: string | null;
  plan: VideoChainPlan | null;
  planInput: VideoChainPlanInput | null;
  /**
   * Extra, caller-supplied disclosure lines specific to what THIS request
   * would drop or approximate by chaining -- e.g. which reference tracks stop
   * conditioning after segment 1, or what a keyframe anchor's `-1` ("pin to
   * end") placement resolves against. Rendered verbatim, one per paragraph;
   * omitted entirely when there is nothing to disclose.
   */
  notes?: string[];
  /**
   * True when the request that opened this dialog is conditioned on the image
   * references alone (MiniMax-H3 ref2va with no video/audio reference track),
   * so unbinding every reference from segment 1 in the editor would leave the
   * first request with nothing to reference. The editor refuses to start on
   * that binding rather than sending it; only the panel knows whether the
   * other reference tracks exist, so it is the panel that sets this.
   */
  requireSegmentZeroReference?: boolean;
  /**
   * The capability matrix, so the continuation-mode control offers exactly what
   * the loaded arch/variant advertises (`chain_context`) and exactly the
   * overlap lengths its video VAE can address (`latent_chunk_pattern`). Omit it
   * and the control is not rendered at all -- the plan then uses the default
   * `boundary_frame`, which is what every chain did before this control existed.
   */
  archCapabilities?: ArchCapabilities | null;
  onCancel: () => void;
  /** Default action: generate once, at the cap (snapped). */
  onGenerateAtCap: () => void;
  /**
   * Explicit, non-default action: enqueue the chain. The manifest is the plan
   * every segment's prompt and reference set is fixed from; `null` is the
   * legacy repeat mode, which the user has to choose by name.
   */
  onStartChain: (manifest: VideoChainManifest | null) => void;
}

type Phase = "planning" | "editor" | "unavailable";

// The backend (`_video_chain_warning_issues` in routes.py) converts every
// core planner warning -- plain strings internally -- into a `VideoChainIssue`
// object before it crosses the wire, so `/video-chain/plan` and
// `/video-chain/validate` responses always carry this one shape. This only
// guards against a missing/malformed field (e.g. an aborted request), not a
// string-vs-object split.
const normalizeIssues = (issues: VideoChainIssue[] | null | undefined): VideoChainIssue[] =>
  Array.isArray(issues) ? issues : [];

const issueText = (issue: VideoChainIssue) =>
  issue.code && issue.code !== "warning" ? `${issue.message} (${issue.code})` : issue.message;

/**
 * The choice CLAUDE.md's opt-in-chaining requirement mandates: a value held
 * in a video length control above the loaded architecture's single-inference
 * cap must never silently become either "clamp to the cap" or "chain
 * automatically" -- Generate has to force a deliberate pick between the two,
 * with the single-inference request as the DEFAULT.
 *
 * On top of that it is the Phase A plan editor (design §10.1): it asks the
 * backend planner for a Chain Manifest, shows one editable prompt per segment
 * and the per-segment reference binding, and starts the chain only while the
 * validator reports no hard error. When planning fails there is no silent
 * fallback to copying the root prompt into every segment -- the user picks
 * legacy repeat, a single request at the cap, or cancel, by name.
 *
 * Deliberately NOT here (Phase B): incoming/outgoing state, the event list,
 * visual-context and seed columns -- i.e. the full table editor.
 *
 * Built as its own dialog (same visual language as `common/ConfirmDialog`)
 * rather than reusing that component: it needs more than two actions, and
 * ConfirmDialog only ever renders two.
 */
export default function VideoChainConfirmDialog({
  isOpen,
  requestedFrames,
  capFrames,
  capSeconds,
  finalSeconds,
  plan,
  planInput,
  notes,
  requireSegmentZeroReference,
  archCapabilities,
  onCancel,
  onGenerateAtCap,
  onStartChain,
}: VideoChainConfirmDialogProps) {
  const [phase, setPhase] = useState<Phase>("planning");
  const [manifest, setManifest] = useState<VideoChainManifest | null>(null);
  const [errors, setErrors] = useState<VideoChainIssue[]>([]);
  const [warnings, setWarnings] = useState<VideoChainIssue[]>([]);
  const [planError, setPlanError] = useState<string | null>(null);
  // Set by any edit; cleared by a successful re-validation. The manifest's
  // `plan_hash` is computed by the backend alone, so an edited manifest has a
  // stale one until it has been sent back -- and that hash is the provenance
  // recorded on every generated segment.
  const [dirty, setDirty] = useState(false);
  const [validating, setValidating] = useState(false);
  const [attempt, setAttempt] = useState(0);
  // What each continuation is conditioned on. The DEFAULT stays
  // `boundary_frame` -- the richer modes are unmeasured, and this dialog is not
  // where an unmeasured default gets introduced.
  const [continuationMode, setContinuationMode] =
    useState<VideoChainContinuationMode>("boundary_frame");
  const [overlapFrames, setOverlapFrames] = useState<number>(0);

  // Offered modes and overlap lengths come from the backend's own tables, never
  // from a list in this file.
  const continuationModes =
    chainContextCapability(archCapabilities, planInput?.architecture, planInput?.variant)
      ?.chain_continuation_modes ?? [];
  const overlapLengths = chainContinuationOverlapLengths(
    archCapabilities, planInput?.architecture, planInput?.variant
  );
  const showContinuationControl = continuationModes.length > 1 && overlapLengths.length > 0;
  const effectiveMode = continuationModes.includes(continuationMode)
    ? continuationMode
    : "boundary_frame";
  const effectiveOverlap =
    effectiveMode === "pinned_tail"
      ? (overlapLengths.includes(overlapFrames) ? overlapFrames : overlapLengths[0] ?? 0)
      : 0;

  const planRequest: VideoChainPlanRequest | null = useMemo(() => {
    if (!planInput) return null;
    return {
      architecture: planInput.architecture,
      variant: planInput.variant ?? null,
      root_prompt: planInput.rootPrompt,
      negative_prompt: planInput.negativePrompt ?? "",
      target_frames: planInput.targetFrames,
      fps: planInput.fps,
      requested_segment_frames: planInput.requestedSegmentFrames ?? null,
      context_mode: "timeline",
      seed_policy: planInput.seedPolicy ?? "fixed",
      root_seed: planInput.rootSeed ?? -1,
      continuation_mode: effectiveMode,
      requested_overlap_frames: effectiveOverlap,
      references: planInput.references ?? [],
    };
  }, [planInput, effectiveMode, effectiveOverlap]);
  const planRequestKey = planRequest ? JSON.stringify(planRequest) : null;

  useEffect(() => {
    if (!isOpen) {
      setPhase("planning");
      setManifest(null);
      setErrors([]);
      setWarnings([]);
      setPlanError(null);
      setDirty(false);
      return;
    }
    if (!planRequestKey) {
      setPhase("unavailable");
      setPlanError(
        "The planner was not given enough information about this request (no loaded architecture)."
      );
      return;
    }
    let cancelled = false;
    setPhase("planning");
    setPlanError(null);
    planVideoChainRequest(JSON.parse(planRequestKey) as VideoChainPlanRequest)
      .then((response) => {
        if (cancelled) return;
        setManifest(response.manifest);
        setErrors(normalizeIssues(response.errors));
        setWarnings([
          ...normalizeIssues(response.warnings),
          ...normalizeIssues(response.manifest?.warnings),
        ]);
        setDirty(false);
        setPhase("editor");
      })
      .catch((error: any) => {
        if (cancelled) return;
        setManifest(null);
        setErrors([]);
        setWarnings([]);
        setPlanError(
          error?.response?.data?.error ??
            error?.response?.data?.detail ??
            error?.message ??
            "The planner did not answer."
        );
        setPhase("unavailable");
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen, planRequestKey, attempt]);

  const editSegmentPrompt = useCallback((index: number, prompt: string) => {
    setManifest((previous) =>
      previous == null
        ? previous
        : {
            ...previous,
            segments: previous.segments.map((segment) =>
              segment.index === index ? { ...segment, prompt } : segment
            ),
          }
    );
    setDirty(true);
  }, []);

  // One reference may be bound to several, non-contiguous segments, and one
  // segment may carry several references. `references[]` is authoritative;
  // `segments[].reference_ids` is its inverse and is recomputed here so the
  // per-segment count stays right, then normalized by the backend on validate.
  const toggleBinding = useCallback((referenceId: string, segmentIndex: number) => {
    setManifest((previous) => {
      if (previous == null) return previous;
      const references = (previous.references ?? []).map((reference) => {
        if (reference.id !== referenceId) return reference;
        const current = reference.segment_indices ?? [];
        const segment_indices = current.includes(segmentIndex)
          ? current.filter((i) => i !== segmentIndex)
          : [...current, segmentIndex].sort((a, b) => a - b);
        return { ...reference, segment_indices, binding_source: "explicit" as const };
      });
      return {
        ...previous,
        references,
        segments: previous.segments.map((segment) => ({
          ...segment,
          reference_ids: references
            .filter((reference) => (reference.segment_indices ?? []).includes(segment.index))
            .map((reference) => reference.id),
        })),
      };
    });
    setDirty(true);
  }, []);

  const revalidate = useCallback(async () => {
    if (manifest == null) return;
    setValidating(true);
    try {
      const response = await validateVideoChainManifest({
        manifest,
        recompute_plan_hash: true,
      });
      setErrors(normalizeIssues(response.errors));
      setWarnings(normalizeIssues(response.warnings));
      // `dirty` clears only when the edits actually came back with a hash to
      // carry. A validate that returns neither a manifest nor a `plan_hash`
      // leaves the edited manifest holding its PRE-edit hash, and that hash is
      // stamped onto every segment's provenance -- so it stays dirty and the
      // "Start chain" button stays disabled until a validate answers properly.
      if (response.manifest) {
        setManifest({
          ...response.manifest,
          plan_hash: response.plan_hash ?? response.manifest.plan_hash,
        });
        setDirty(false);
      } else if (response.plan_hash) {
        setManifest((previous) =>
          previous == null ? previous : { ...previous, plan_hash: response.plan_hash as string }
        );
        setDirty(false);
      } else {
        setErrors((previous) => [
          ...previous,
          {
            code: "validate_no_plan_hash",
            severity: "error",
            message:
              "The validator returned no recomputed plan hash, so these edits still carry the " +
              "previous plan's hash. Validate again before starting the chain.",
          },
        ]);
      }
    } catch (error: any) {
      setErrors([
        {
          code: "validate_request_failed",
          severity: "error",
          message:
            error?.response?.data?.error ??
            error?.message ??
            "The validator did not answer, so these edits are unchecked.",
        },
      ]);
    } finally {
      setValidating(false);
    }
  }, [manifest]);

  if (!isOpen) return null;

  const overshoot = plan != null ? plan.finalFrames - requestedFrames : 0;
  const segmentCount = manifest?.segments.length ?? 0;
  // An ABSENT `reference_ids` is the manifest's `default_all` binding (every
  // reference carries); an EMPTY one is an explicit "none", which for a
  // reference-only request leaves segment 1 with nothing to condition on.
  const segmentZeroBindings = manifest?.segments.find((s) => s.index === 0)?.reference_ids;
  const segmentZeroUnbound =
    requireSegmentZeroReference === true &&
    (manifest?.references?.length ?? 0) > 0 &&
    segmentZeroBindings != null &&
    segmentZeroBindings.length === 0;
  const canStartPlanned =
    phase === "editor" && manifest != null && errors.length === 0 && !dirty && !segmentZeroUnbound;
  const showFallbackChoices = phase === "unavailable" || errors.length > 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-gray-800 rounded-lg shadow-xl max-w-3xl w-full mx-4 border border-gray-700 max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">Length exceeds the current segment length</h3>
        </div>

        <div className="p-4 space-y-3 overflow-y-auto">
          <p className="text-sm text-gray-300">
            {requestedFrames} frames exceeds the current segment length of {capFrames} frames
            {capSeconds != null ? ` (${capSeconds}s)` : ""}.
          </p>
          {plan != null && (
            <p className="text-sm text-gray-300">
              Reaching {requestedFrames} frames takes {plan.segments} generation requests, chained via temporal
              outpaint. The chain actually reaches {plan.finalFrames} frames
              {finalSeconds != null ? ` (${finalSeconds}s)` : ""}
              {overshoot > 0 ? `, ${overshoot} more than requested (the arithmetic that lands each segment on the model's frame grid does not land exactly on the requested total)` : ""}.
              Segments after the first are conditioned on the boundary frame of the previous segment (and, for
              ref2va/ia2v, the image references bound to that segment and an automatic video reference derived
              from the end of the previous segment), not the rest of its content.
            </p>
          )}
          {notes != null && notes.map((note, index) => (
            <p key={index} className="text-sm text-gray-400">{note}</p>
          ))}

          {showContinuationControl && (
            <div className="flex flex-wrap items-center gap-3 rounded border border-gray-700 bg-gray-900/40 p-3">
              <label className="text-xs text-gray-300">
                Continuation context
                <select
                  value={effectiveMode}
                  onChange={(e) =>
                    setContinuationMode(e.target.value as VideoChainContinuationMode)
                  }
                  className="ml-2 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-white"
                >
                  {continuationModes.map((mode) => (
                    <option key={mode} value={mode}>
                      {mode === "boundary_frame" ? "Boundary frame (1 frame)" : mode}
                    </option>
                  ))}
                </select>
              </label>
              {effectiveMode === "pinned_tail" && (
                <label className="text-xs text-gray-300">
                  Pinned frames
                  <select
                    value={effectiveOverlap}
                    onChange={(e) => setOverlapFrames(Number(e.target.value))}
                    className="ml-2 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-white"
                  >
                    {overlapLengths.map((frames) => (
                      <option key={frames} value={frames}>{frames}</option>
                    ))}
                  </select>
                </label>
              )}
              <p className="w-full text-xs text-gray-400">
                {effectiveMode === "pinned_tail"
                  ? `Each continuation is conditioned on the last ${effectiveOverlap} frame(s) of the previous segment (and, when the input audio is preserved, that span's soundtrack) instead of on its final frame alone. Those frames are re-rendered and discarded, so the previous segment's pixels are unchanged; the generated span grows by the same amount and is rounded up to the model's frame grid, which is why the segment ranges below can add more frames per segment than the boundary-frame plan. Shorter pins are not offered: a single pinned frame is a motionless still, which the model can continue as a static scene — boundary frame is the one-frame option, and it is a different kind of conditioning rather than a smaller pin.`
                  : "Each continuation is conditioned on the previous segment's final frame alone."}
              </p>
            </div>
          )}

          {phase === "planning" && (
            <p className="text-sm text-gray-400">Planning the chain…</p>
          )}

          {phase === "unavailable" && (
            <div className="rounded border border-amber-700 bg-amber-950/40 p-3 space-y-2">
              <p className="text-sm text-amber-300">
                No per-segment plan was produced: {planError}
              </p>
              <p className="text-xs text-gray-300">
                Nothing is chained on this outcome by default. Choose below: retry planning, run the
                chain in legacy repeat mode (the same full-length prompt on every segment), generate a
                single request at the cap, or cancel.
              </p>
              <button
                onClick={() => setAttempt((n) => n + 1)}
                className="px-3 py-1.5 rounded text-xs font-medium bg-gray-700 hover:bg-gray-600 text-white"
              >
                Retry planning
              </button>
            </div>
          )}

          {phase === "editor" && manifest != null && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-400">
                <span>Segments: {segmentCount}</span>
                <span>Final length: {manifest.expected_final_frames} frames</span>
                <span>Mode: {manifest.context_mode}</span>
                <span>Seeds: {manifest.seed_policy}</span>
                <span className="font-mono">plan {manifest.plan_hash?.slice(0, 12) ?? "unknown"}</span>
              </div>

              {manifest.context_mode === "legacy_repeat" && (
                <p className="text-xs text-amber-400">
                  Legacy repeat: every segment is sent the same full-length prompt, so events in it can
                  be re-enacted in later segments.
                </p>
              )}

              {segmentZeroUnbound && (
                <p className="rounded border border-red-700 bg-red-950/40 p-3 text-xs text-red-300">
                  Segment 1 has no reference bound to it. This request generates from image references
                  only, so its first segment must carry at least one. Bind a reference to segment 1, or
                  cancel and generate a single request at the cap.
                </p>
              )}
              {errors.length > 0 && (
                <ul className="rounded border border-red-700 bg-red-950/40 p-3 space-y-1">
                  {errors.map((issue, index) => (
                    <li key={index} className="text-xs text-red-300">
                      {issue.segment_index != null ? `Segment ${issue.segment_index + 1}: ` : ""}
                      {issueText(issue)}
                    </li>
                  ))}
                </ul>
              )}
              {warnings.length > 0 && (
                <ul className="rounded border border-amber-700 bg-amber-950/30 p-3 space-y-1">
                  {warnings.map((issue, index) => (
                    <li key={index} className="text-xs text-amber-300">
                      {issue.segment_index != null ? `Segment ${issue.segment_index + 1}: ` : ""}
                      {issueText(issue)}
                    </li>
                  ))}
                </ul>
              )}

              {(manifest.references?.length ?? 0) > 0 && (
                <div className="rounded border border-gray-700 p-3 space-y-2">
                  <p className="text-xs text-gray-300">
                    Reference binding: which segments each reference is passed to. A reference can be
                    bound to any set of segments, and a segment can carry several. Reference rows are
                    carried through every denoise step of the segments they are bound to.
                  </p>
                  {manifest.references?.map((reference) => (
                    <div key={reference.id} className="flex flex-wrap items-center gap-2">
                      <span className="text-xs text-gray-200 w-56 truncate" title={reference.label}>
                        {reference.label || reference.id} ({reference.kind})
                      </span>
                      {manifest.segments.map((segment) => (
                        <label
                          key={segment.index}
                          className="flex items-center gap-1 text-xs text-gray-400"
                        >
                          <input
                            type="checkbox"
                            checked={(reference.segment_indices ?? []).includes(segment.index)}
                            onChange={() => toggleBinding(reference.id, segment.index)}
                          />
                          {segment.index + 1}
                        </label>
                      ))}
                    </div>
                  ))}
                </div>
              )}

              <div className="space-y-3">
                {manifest.segments.map((segment) => (
                  <div key={segment.index} className="rounded border border-gray-700 p-3 space-y-2">
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-400">
                      <span className="text-gray-200 font-medium">
                        Segment {segment.index + 1} / {segmentCount}
                      </span>
                      <span>
                        frames {segment.owned_start_frame}–{segment.owned_end_frame}
                      </span>
                      <span>generated {segment.generated_span_frames} frames</span>
                      <span>references {segment.reference_ids?.length ?? 0}</span>
                    </div>
                    <textarea
                      value={segment.prompt}
                      onChange={(event) => editSegmentPrompt(segment.index, event.target.value)}
                      rows={6}
                      className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-xs text-gray-100 font-mono"
                    />
                  </div>
                ))}
              </div>

              {dirty && (
                <p className="text-xs text-amber-400">
                  Edited. Validate the changes to recompute the plan hash before starting the chain.
                </p>
              )}
              <button
                onClick={revalidate}
                disabled={validating || !dirty}
                className={`px-3 py-1.5 rounded text-xs font-medium ${
                  validating || !dirty
                    ? "bg-gray-700 text-gray-500 cursor-not-allowed"
                    : "bg-gray-700 hover:bg-gray-600 text-white"
                }`}
              >
                {validating ? "Validating…" : "Validate changes"}
              </button>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-2 p-4 border-t border-gray-700">
          <button
            onClick={onGenerateAtCap}
            className="px-4 py-2 rounded text-sm font-medium transition-colors bg-blue-600 hover:bg-blue-500 text-white"
          >
            Generate at {capFrames} frames (single request)
          </button>
          <button
            onClick={() => manifest != null && onStartChain(manifest)}
            disabled={!canStartPlanned}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              canStartPlanned
                ? "bg-amber-700 hover:bg-amber-600 text-white"
                : "bg-gray-700 text-gray-500 cursor-not-allowed"
            }`}
          >
            {manifest != null
              ? `Start chain: ${segmentCount} segments, reaches ${manifest.expected_final_frames} frames`
              : "Start chain"}
          </button>
          {showFallbackChoices && (
            <button
              onClick={() => onStartChain(null)}
              className="px-4 py-2 rounded text-sm font-medium transition-colors bg-gray-700 hover:bg-gray-600 text-amber-300 border border-amber-700"
            >
              Start chain in legacy repeat mode
              {plan != null ? `: ${plan.segments} segments, reaches ${plan.finalFrames} frames` : ""}
              {" "}— the same full-length prompt on every segment
            </button>
          )}
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm font-medium transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
