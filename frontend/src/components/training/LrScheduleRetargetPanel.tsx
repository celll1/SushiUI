"use client";

import { useEffect, useMemo, useState } from "react";
import {
  LrRetargetOp,
  LrScheduleRetargetRequest,
  LrSchedulePreviewResult,
  LrScheduleStatusResponse,
  lrScheduleResultExplanation,
  previewLrScheduleEvents,
  queueLrScheduleRetarget,
} from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";
import { LR_SCHEDULER_OPTIONS } from "./TrainingConfig";

// §19.6/D27: four forms of one event. `scale`, `hold` and `undo` describe no
// curve of their own -- the server derives theirs from the one in force -- so
// they refuse every lr_* key and `anchor`.
const OPS: { value: LrRetargetOp; label: string; note: string }[] = [
  { value: "retarget", label: "Retarget",
    note: "Switch onto the schedule described below, blending out of the one the run is on." },
  // The factor is exact AT the anchor and nowhere else: the curve is
  // re-anchored, so later steps are not the old curve times the gain.
  { value: "scale", label: "Scale",
    note: "Re-anchor the schedule in force at this step and multiply it by the gain. It is exactly the gain at that step only — the curve restarts from there, so the rest of the run is not the old curve scaled. Its warmup is dropped, which re-anchored would mean ramping up from 0 again." },
  { value: "hold", label: "Hold",
    note: "Freeze the multiplier the run is at now. Entered on the spot unless a blend length is given." },
  { value: "undo", label: "Undo",
    note: "Back to the schedule in force before the last retarget on this curve. It is appended, never a rewrite of the event it reverses, and it does not cancel a future reservation." },
];

// Mirrors DECAY_SHAPE_NAMES / BLEND_SHAPE_NAMES, which are the same tuple.
const SHAPE_OPTIONS = ["cosine", "linear", "rex"];

const POSITION_SOURCE_TEXT: Record<string, string> = {
  run_row: "the run's own current step",
  status_file:
    "the last state the trainer published, which is rewritten only when the schedule changes",
  config: "a hypothetical schedule with no run behind it",
};

const CURVE_COLORS = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24"];

const APPLIED_COLOR = "#a78bfa";

// Event kinds an operator ordered. A refused command is stored as `noop`, and
// `total_steps` / `restart` are the trainer's own seams rather than commands.
const APPLIED_KINDS: Record<string, string> = {
  retarget: "retarget",
  decay: "decay",
  cancel: "cancel",
};

/** `[step, multiplier]` samples as an SVG path in a 0..100 box. The only
 *  arithmetic here is the scaling: every multiplier comes from the endpoint
 *  (D20), and no schedule is evaluated in this file. */
const svgPath = (points: [number, number][], lastStep: number): string =>
  points
    .map(([step, m], i) => {
      const x = (step / lastStep) * 100;
      const y = 100 - Math.max(0, Math.min(1, m)) * 100;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

interface Props {
  runId: number;
  status: LrScheduleStatusResponse | null;
  /** Refetch the run's schedule state after a retarget is queued. */
  onQueued: () => void;
}

export default function LrScheduleRetargetPanel({ runId, status, onQueued }: Props) {
  const [open, setOpen] = useState(false);
  const [op, setOp] = useState<LrRetargetOp>("retarget");
  // Every field is a STRING and starts empty. Empty is absent from the request
  // -- D44's "an omitted key keeps the value in force" only holds if nothing
  // is substituted on this side.
  const [fields, setFields] = useState<Record<string, string>>({});
  const [groups, setGroups] = useState<string[]>([]);
  const [preview, setPreview] = useState<LrSchedulePreviewResult | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewNonce, setPreviewNonce] = useState(0);
  const [applying, setApplying] = useState(false);
  const [applyError, setApplyError] = useState<string | null>(null);
  const [queued, setQueued] = useState<string | null>(null);

  const { lrRetargetDefaults: defaults, trainingDefaults } = useStartup();
  // What a blank box resolves to, stated rather than guessed. `length`, `at`
  // and `groups` are null on purpose: they resolve against the RUN, so only
  // their meaning can be named, never a number.
  const gainPlaceholder = defaults ? String(defaults.gain) : "default";
  const lengthPlaceholder = defaults
    ? (defaults.length === null ? "the run's warmup" : String(defaults.length))
    : "default";
  const anchorDefaultLabel = defaults ? `default (${defaults.anchor})` : "default";
  const shapeDefaultLabel = defaults ? `default (${defaults.shape})` : "default";

  const state = status?.status ?? null;
  // Both halves are omitted rather than approximated: the global total is
  // absent when the trainer could not name it, and the two axes coincide at
  // interval 1.
  const axisNote = [
    state && state.global_total_steps != null
      ? `This run ends at ${state.global_total_steps.toLocaleString()} global steps.`
      : null,
    state && state.advance_interval > 1
      ? `It advances its scheduler once every ${state.advance_interval} global `
        + `steps, so the boxes above are global steps while the preview below `
        + `is in optimizer steps.`
      : null,
  ].filter(Boolean).join(" ");
  const scheduler = fields.lr_scheduler ?? "";
  // D44: the decay and cycle keys are inherited only while the name stays the
  // same. Under a new name a blank takes the training default instead, and the
  // placeholder has to say which of the two it is.
  const sameSchedule = !!state && scheduler === state.scheduler;
  const keepOrDefault = (key: string): string => {
    if (sameSchedule) return "keep current";
    const value = trainingDefaults?.[key];
    return value === undefined || value === null
      ? "default" : `default (${String(value)})`;
  };

  // Invariant 17 keeps `op` off the event, so the only record of which form
  // was pressed is the result the trainer wrote, keyed by request_id.
  const opByRequest = useMemo(() => {
    const map: Record<string, string> = {};
    for (const r of status?.results ?? []) {
      if (r.op && r.request_id) map[String(r.request_id)] = String(r.op);
    }
    return map;
  }, [status]);

  const components = useMemo(() => {
    const names: string[] = [];
    for (const g of state?.groups ?? []) {
      const name = g.component;
      if (name && !names.includes(name)) names.push(name);
    }
    return names;
  }, [state]);

  const setField = (key: string, value: string) =>
    setFields((prev) => ({ ...prev, [key]: value }));

  const payload = useMemo((): LrScheduleRetargetRequest => {
    const out: LrScheduleRetargetRequest = { op };
    const num = (key: string): number | undefined => {
      const raw = (fields[key] ?? "").trim();
      if (raw === "") return undefined;
      const value = Number(raw);
      return Number.isFinite(value) ? value : undefined;
    };
    const text = (key: string): string | undefined => {
      const raw = (fields[key] ?? "").trim();
      return raw === "" ? undefined : raw;
    };

    if (op === "retarget") {
      out.lr_scheduler = text("lr_scheduler");
      out.lr_warmup_steps = num("lr_warmup_steps");
      out.lr_floor_ratio = num("lr_floor_ratio");
      out.lr_decay_start_ratio = num("lr_decay_start_ratio");
      out.lr_decay_start_step = num("lr_decay_start_step");
      out.lr_decay_steps = num("lr_decay_steps");
      out.lr_decay_shape = text("lr_decay_shape");
      out.lr_cycle_steps = num("lr_cycle_steps");
      out.lr_cycle_peak_decay = num("lr_cycle_peak_decay");
      out.anchor = text("anchor") as "restart" | "continue" | undefined;
      out.gain = num("gain");
    } else if (op === "scale") {
      // Required here, and refused for hold and undo, which have no factor.
      out.gain = num("gain");
    }
    out.at = num("at");
    out.length = num("length");
    out.shape = text("shape");
    // An empty selection is refused by the server rather than read as "all",
    // so it is sent as no key at all.
    if (groups.length > 0) out.groups = groups;

    for (const key of Object.keys(out) as (keyof LrScheduleRetargetRequest)[]) {
      if (out[key] === undefined) delete out[key];
    }
    return out;
  }, [op, fields, groups]);

  const incomplete =
    (op === "retarget" && !payload.lr_scheduler) ||
    (op === "scale" && payload.gain === undefined);

  // D29: the curve is drawn before anything is committed, from the same code
  // the trainer runs. Debounced because it refires on every keystroke.
  useEffect(() => {
    if (!open || incomplete || !state) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      previewLrScheduleEvents({ run_id: runId, events: [payload] })
        .then((data) => {
          if (cancelled) return;
          setPreview(data);
          setPreviewError(null);
        })
        .catch((err: any) => {
          if (cancelled) return;
          setPreview(null);
          setPreviewError(
            err?.response?.data?.detail || err?.message || "Could not draw the candidate"
          );
        });
    }, 350);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // written_at, not the state object: the trainer rewrites that file when the
    // timeline changes, and polling it does not move the published position.
  }, [open, incomplete, state?.written_at, runId, payload, previewNonce]);

  const handleApply = async () => {
    setApplying(true);
    setApplyError(null);
    setQueued(null);
    try {
      const accepted = await queueLrScheduleRetarget(runId, payload);
      setQueued(accepted.request_id);
      onQueued();
    } catch (err: any) {
      setApplyError(
        err?.response?.data?.detail || err?.message || "Failed to queue the retarget"
      );
    } finally {
      setApplying(false);
    }
  };

  // One grid for every curve: sample_curve spans timeline.current_total,
  // which is not per-group. A per-group span would stretch one curve onto
  // another's axis here.
  const lastStep = preview?.curves?.[0]?.points?.slice(-1)[0]?.[0] || 0;
  // Header value, so null when the curves are on different floors: then no
  // single line can be drawn and none is.
  const floorRatio = preview?.floor_ratio ?? 0;
  const candidateResult = preview?.results?.[0] ?? null;
  const refused = !!candidateResult && candidateResult.result.startsWith("rejected_");

  // §19.8's markers for the events already on the timeline. Both coordinates
  // are the server's: `at` is the event's own scheduler step, and the height is
  // the nearest baseline SAMPLE, picked rather than interpolated.
  const appliedMarkers = useMemo(() => {
    const baseline = preview?.curves?.[0]?.baseline_points ?? [];
    if (!state || baseline.length === 0 || lastStep <= 0) return [];
    return (state.events ?? [])
      .filter((e) => APPLIED_KINDS[String(e.kind)] !== undefined)
      .map((e) => {
        const at = Number(e.at);
        const nearest = baseline.reduce(
          (best, p) => (Math.abs(p[0] - at) < Math.abs(best[0] - at) ? p : best),
          baseline[0]);
        const label = opByRequest[String(e.request_id)]
          ?? APPLIED_KINDS[String(e.kind)];
        return { at, label, m: nearest[1] };
      })
      .filter((m) => Number.isFinite(m.at) && m.at >= 0 && m.at <= lastStep);
  }, [state, preview, lastStep, opByRequest]);

  const numberField = (
    key: string, label: string, placeholder: string, step?: string
  ) => (
    <label key={key} className="block">
      <span className="block text-gray-400">{label}</span>
      <input
        type="number"
        step={step}
        value={fields[key] ?? ""}
        placeholder={placeholder}
        onChange={(e) => setField(key, e.target.value)}
        className="w-full px-1.5 py-1 bg-gray-900 border border-gray-700 rounded text-xxs focus:outline-none focus:border-blue-500"
      />
    </label>
  );

  return (
    <div className="space-y-2 rounded border border-gray-700 bg-gray-900/60 p-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between text-xxs text-gray-300 hover:text-gray-100"
      >
        <span className="font-medium">Change the schedule</span>
        <span className="text-gray-500">{open ? "hide" : "show"}</span>
      </button>

      {open && (
        <div className="space-y-2 text-xxs">
          <div className="flex gap-1">
            {OPS.map((entry) => (
              <button
                key={entry.value}
                onClick={() => setOp(entry.value)}
                className={`flex-1 px-2 py-1 rounded transition-colors ${
                  op === entry.value
                    ? "bg-blue-700 text-white"
                    : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                }`}
              >
                {entry.label}
              </button>
            ))}
          </div>
          <p className="leading-relaxed text-gray-500">
            {OPS.find((e) => e.value === op)?.note}
          </p>

          {op === "retarget" && (
            <div className="grid grid-cols-2 gap-2">
              <label className="col-span-2 block">
                <span className="block text-gray-400">Schedule</span>
                <select
                  value={scheduler}
                  onChange={(e) => setField("lr_scheduler", e.target.value)}
                  className="w-full px-1.5 py-1 bg-gray-900 border border-gray-700 rounded text-xxs focus:outline-none focus:border-blue-500"
                >
                  <option value="">Pick the schedule to switch to</option>
                  {LR_SCHEDULER_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </label>
              {numberField("lr_warmup_steps", "Warmup steps", "keep current")}
              {numberField("lr_floor_ratio", "Floor ratio", "keep current", "any")}
              {scheduler === "plateau_cosine_floor" &&
                numberField("lr_decay_start_ratio", "Decay start ratio", keepOrDefault("lr_decay_start_ratio"), "any")}
              {scheduler === "wsd" && (
                <>
                  {numberField("lr_decay_start_step", "Decay start step", keepOrDefault("lr_decay_start_step"))}
                  {numberField("lr_decay_steps", "Decay length", keepOrDefault("lr_decay_steps"))}
                  <label className="block">
                    <span className="block text-gray-400">Decay shape</span>
                    <select
                      value={fields.lr_decay_shape ?? ""}
                      onChange={(e) => setField("lr_decay_shape", e.target.value)}
                      className="w-full px-1.5 py-1 bg-gray-900 border border-gray-700 rounded text-xxs focus:outline-none focus:border-blue-500"
                    >
                      <option value="">{keepOrDefault("lr_decay_shape")}</option>
                      {SHAPE_OPTIONS.map((name) => (
                        <option key={name} value={name}>{name}</option>
                      ))}
                    </select>
                  </label>
                </>
              )}
              {scheduler === "cosine_with_restarts" && (
                <>
                  {numberField("lr_cycle_steps", "Cycle length", keepOrDefault("lr_cycle_steps"))}
                  {numberField("lr_cycle_peak_decay", "Cycle peak decay", keepOrDefault("lr_cycle_peak_decay"), "any")}
                </>
              )}
              <label className="block">
                <span className="block text-gray-400">Anchor</span>
                <select
                  value={fields.anchor ?? ""}
                  onChange={(e) => setField("anchor", e.target.value)}
                  className="w-full px-1.5 py-1 bg-gray-900 border border-gray-700 rounded text-xxs focus:outline-none focus:border-blue-500"
                >
                  <option value="">{anchorDefaultLabel}</option>
                  <option value="restart">restart — start where the LR is now</option>
                  <option value="continue">continue — the new curve&apos;s own value here</option>
                </select>
              </label>
              {numberField("gain", "Gain", gainPlaceholder, "any")}
            </div>
          )}

          {op === "scale" && (
            <div className="grid grid-cols-2 gap-2">
              {numberField("gain", "Gain (required)", "required", "any")}
            </div>
          )}

          <div className="grid grid-cols-3 gap-2">
            {numberField(
              "at", "Apply at (global step)",
              state?.global_step != null ? `now (${state.global_step})` : "now")}
            {numberField("length", "Blend length (global steps)", lengthPlaceholder)}
            <label className="block">
              <span className="block text-gray-400">Blend shape</span>
              <select
                value={fields.shape ?? ""}
                onChange={(e) => setField("shape", e.target.value)}
                className="w-full px-1.5 py-1 bg-gray-900 border border-gray-700 rounded text-xxs focus:outline-none focus:border-blue-500"
              >
                <option value="">{shapeDefaultLabel}</option>
                {SHAPE_OPTIONS.map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </label>
          </div>

          {components.length > 0 && (
            <div>
              <span className="block text-gray-400">Groups</span>
              <div className="flex flex-wrap gap-x-3 gap-y-1 mt-0.5">
                {components.map((name) => (
                  <label key={name} className="flex items-center gap-1 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={groups.includes(name)}
                      onChange={(e) =>
                        setGroups((prev) =>
                          e.target.checked
                            ? [...prev, name]
                            : prev.filter((g) => g !== name))
                      }
                      className="w-3 h-3"
                    />
                    <span className="font-mono text-gray-300">{name}</span>
                  </label>
                ))}
                <span className="text-gray-500">
                  {groups.length === 0 ? "none ticked = every group" : ""}
                </span>
              </div>
            </div>
          )}

          {!!axisNote && (
            <p className="leading-relaxed text-gray-400">{axisNote}</p>
          )}

          <p className="leading-relaxed text-gray-500">
            A blank field keeps the value the run is already on — it is left out of the
            request rather than filled in here, so nothing moves that you did not name.
            The decay and cycle boxes only inherit while the schedule name stays the same;
            under a different name a blank one takes the training default. Step counts are
            global steps.
          </p>

          {/* D29/D20: drawn before anything is committed, and every number in
              it comes from the endpoint. */}
          <div className="rounded border border-gray-700 bg-gray-950/60 p-2">
            <div className="flex items-baseline justify-between">
              <span className="text-gray-400">Preview</span>
              <button
                onClick={() => setPreviewNonce((n) => n + 1)}
                disabled={incomplete || !state}
                className="text-gray-400 hover:text-gray-200 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                refresh
              </button>
            </div>
            {!state ? (
              <p className="mt-1 text-gray-500">
                The run has not published a schedule yet; it writes one at its first batch.
              </p>
            ) : incomplete ? (
              <p className="mt-1 text-gray-500">
                {op === "retarget"
                  ? "Pick a schedule to draw the candidate."
                  : "Enter the factor to scale by."}
              </p>
            ) : previewError ? (
              <p className="mt-1 text-red-400">{previewError}</p>
            ) : preview && lastStep > 0 ? (
              <>
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-24 mt-1">
                  <line x1="0" y1="0" x2="100" y2="0" stroke="#374151" strokeWidth="1"
                        vectorEffect="non-scaling-stroke" />
                  <line x1="0" y1="100" x2="100" y2="100" stroke="#374151" strokeWidth="1"
                        vectorEffect="non-scaling-stroke" />
                  {floorRatio > 0 && (
                    <line
                      x1="0" y1={100 - floorRatio * 100}
                      x2="100" y2={100 - floorRatio * 100}
                      stroke="#4b5563" strokeWidth="1" strokeDasharray="4 3"
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                  <line
                    x1={(preview.step / lastStep) * 100} y1="0"
                    x2={(preview.step / lastStep) * 100} y2="100"
                    stroke="#6b7280" strokeWidth="1" vectorEffect="non-scaling-stroke"
                  />
                  {candidateResult?.at != null && (
                    <line
                      x1={(candidateResult.at / lastStep) * 100} y1="0"
                      x2={(candidateResult.at / lastStep) * 100} y2="100"
                      stroke="#93c5fd" strokeWidth="1" strokeDasharray="3 3"
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                  {preview.curves.map((curve, i) => (
                    <path
                      key={`base-${curve.group ?? i}`}
                      d={svgPath(curve.baseline_points, lastStep)}
                      fill="none" stroke="#6b7280" strokeWidth="1" strokeDasharray="3 2"
                      vectorEffect="non-scaling-stroke"
                    />
                  ))}
                  {preview.curves.map((curve, i) => (
                    <path
                      key={`cand-${curve.group ?? i}`}
                      d={svgPath(curve.points, lastStep)}
                      fill="none" stroke={CURVE_COLORS[i % CURVE_COLORS.length]}
                      strokeWidth="1.5" vectorEffect="non-scaling-stroke"
                    />
                  ))}
                  {appliedMarkers.map((marker, i) => {
                    // A tick rather than a dot: the box is stretched to the
                    // panel's width, which turns a circle into an ellipse.
                    const y = 100 - Math.max(0, Math.min(1, marker.m)) * 100;
                    return (
                      <line
                        key={`applied-${marker.at}-${i}`}
                        x1={(marker.at / lastStep) * 100} y1={Math.max(0, y - 7)}
                        x2={(marker.at / lastStep) * 100} y2={Math.min(100, y + 7)}
                        stroke={APPLIED_COLOR} strokeWidth="1.5"
                        vectorEffect="non-scaling-stroke"
                      />
                    );
                  })}
                </svg>
                <p className="text-gray-500">
                  Dashed grey is the run as it stands, solid is the candidate. Multiplier on
                  the base LR, 0 at the bottom and 1 at the top, over{" "}
                  {preview.scheduler_total_steps.toLocaleString()} optimizer steps. The plain
                  vertical line is where the candidates are dated from, the dashed one is
                  where this one takes effect, and each short tick on the grey curve is an
                  event already applied.
                </p>
                {appliedMarkers.length > 0 && (
                  <p className="mt-1 leading-relaxed" style={{ color: APPLIED_COLOR }}>
                    Already on the timeline:{" "}
                    {appliedMarkers
                      .map((m) => `${m.label} at ${m.at.toLocaleString()}`)
                      .join(" · ")}
                    . A scale, hold or undo is one retarget event, named here from the
                    run&apos;s result records; one whose record has aged out reads as
                    &quot;retarget&quot;.
                  </p>
                )}
                <p className="mt-1 text-gray-500">
                  Dated from step {preview.step.toLocaleString()} —{" "}
                  {POSITION_SOURCE_TEXT[preview.position_source] ?? preview.position_source}.
                </p>
                {preview.source !== "run" && (
                  <p className="mt-1 leading-relaxed text-yellow-400">
                    This curve is a schedule with no run behind it, not a picture of this
                    run.
                  </p>
                )}
                {preview.curves.length === 1 ? (
                  <p className="mt-1 font-mono text-gray-500">
                    {preview.curves[0].description}
                  </p>
                ) : (
                  <>
                    {preview.lr_scheduler === null && (
                      <p className="mt-1 leading-relaxed text-gray-400">
                        These param groups are not all on the same schedule, so each
                        curve is named for itself below.
                      </p>
                    )}
                    {preview.curves.map((curve, i) => (
                      <p
                        key={curve.group ?? i}
                        className="mt-1 font-mono"
                        style={{ color: CURVE_COLORS[i % CURVE_COLORS.length] }}
                      >
                        {curve.group ?? "all groups"}: {curve.description}
                      </p>
                    ))}
                  </>
                )}
                {candidateResult && (
                  <p className={`mt-1 leading-relaxed ${refused ? "text-red-400" : "text-gray-400"}`}>
                    {lrScheduleResultExplanation(candidateResult.result)}
                    {candidateResult.detail ? ` ${candidateResult.detail}` : ""}
                  </p>
                )}
                {candidateResult?.result === "rejected_backdated"
                  && preview.position_source === "run_row" && (
                  <p className="mt-1 leading-relaxed text-yellow-400">
                    This refusal may be spurious: the position it was scored against came
                    from the run&apos;s own step, which is divided down to the scheduler
                    axis and can run ahead of the true position. The trainer scores this
                    candidate against its own position and may accept it.
                  </p>
                )}
                {preview.warnings.map((w, i) => (
                  <p key={`${w.code}-${i}`} className="mt-1 leading-relaxed text-yellow-400">
                    {w.message}
                  </p>
                ))}
              </>
            ) : (
              <p className="mt-1 text-gray-500">Drawing…</p>
            )}
          </div>

          <button
            onClick={handleApply}
            disabled={!status?.is_running || applying || incomplete}
            title={
              !status?.is_running
                ? "Only a running training run can be told to change its LR."
                : incomplete
                ? (op === "retarget"
                    ? "Pick the schedule to switch to."
                    : "Enter the factor to scale by.")
                : undefined
            }
            className="w-full px-2 py-1.5 bg-blue-700 hover:bg-blue-600 rounded text-xxs sm:text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {applying ? "Sending..." : `Apply ${op}`}
          </button>
          {queued && (
            <p className="leading-relaxed text-gray-400">
              Queued as {queued}. The trainer claims it at the head of its next batch and the
              outcome appears below.
            </p>
          )}
          {applyError && <p className="leading-relaxed text-red-400">{applyError}</p>}
        </div>
      )}
    </div>
  );
}
