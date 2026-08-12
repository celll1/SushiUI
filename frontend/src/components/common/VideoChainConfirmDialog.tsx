"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  VideoChainIssue,
  VideoChainManifest,
  VideoChainPlan,
  VideoChainPlanRequest,
  VideoChainReferenceInput,
  VideoChainSeedPolicy,
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
  seedPolicy?: VideoChainSeedPolicy;
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

// `openapi.yaml` types every finding as a VideoChainIssue object, but the
// planner module's own warnings are plain strings, so a manifest can carry
// either shape at this seam. Both are rendered rather than one of them coming
// out blank.
const normalizeIssues = (issues: unknown): VideoChainIssue[] =>
  Array.isArray(issues)
    ? issues.map((issue) =>
        typeof issue === "string"
          ? { code: "warning", severity: "warning" as const, message: issue }
          : (issue as VideoChainIssue)
      )
    : [];

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
      continuation_mode: "boundary_frame",
      references: planInput.references ?? [],
    };
  }, [planInput]);
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
      if (response.manifest) {
        setManifest({
          ...response.manifest,
          plan_hash: response.plan_hash ?? response.manifest.plan_hash,
        });
      } else if (response.plan_hash) {
        setManifest((previous) =>
          previous == null ? previous : { ...previous, plan_hash: response.plan_hash as string }
        );
      }
      setDirty(false);
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
  const canStartPlanned = phase === "editor" && manifest != null && errors.length === 0 && !dirty;
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
