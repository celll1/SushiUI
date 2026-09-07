"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  LrScheduleStatusResponse,
  LrScheduleTriggerRequest,
  LrScheduleTriggerState,
  LrScheduleTriggersResponse,
  LrTriggerCommand,
  LrTriggerPredicate,
  cancelLrScheduleTrigger,
  getLrScheduleTriggers,
  lrScheduleResultExplanation,
  queueLrScheduleTrigger,
} from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";
import {
  EMPTY_RETARGET_FORM,
  LR_FIELD_CLASS,
  LrRetargetFields,
  RetargetFormState,
  retargetIncomplete,
  retargetPayload,
} from "./LrScheduleRetargetPanel";

// Mirrors TRIGGER_SIGNALS. D45's two exclusions are absent rather than
// disabled: `learning_rate`, and the run's own `extra:lr` / `extra:lr_*`
// reporting, which reaches the same value through the extra channel.
const SIGNALS: { value: string; label: string }[] = [
  { value: "loss", label: "loss" },
  { value: "grad_norm", label: "grad_norm" },
];

// Mirrors TRIGGER_PREDICATES.
const PREDICATES: { value: LrTriggerPredicate; label: string; note: string }[] = [
  { value: "plateau", label: "Plateau",
    note: "Fires after this many consecutive observations with no improvement bigger than the minimum delta on the best observation since the last firing. Improvement means a DECREASE: every signal a trigger may watch is one that is minimised, and there is no way to ask for the opposite." },
  { value: "below", label: "Below",
    note: "Fires on the first observation below the threshold. The threshold is compared against the window mean, never against a single step." },
  { value: "above", label: "Above",
    note: "Fires on the first observation above the threshold. The threshold is compared against the window mean, never against a single step." },
];

// Mirrors TRIGGER_ACTION_COMMANDS (the two parameterless buttons, then the
// retarget family, whose own op picker covers scale / hold / undo).
const ACTION_COMMANDS: { value: LrTriggerCommand; label: string; note: string }[] = [
  { value: "start_decay", label: "Decay now",
    note: "Decays from the multiplier in force to the floor, over the run's own configured decay length and shape." },
  { value: "cancel_decay", label: "Cancel decay",
    note: "Returns to the configured curve, and voids a decay the config had scheduled." },
  { value: "retarget", label: "Change the schedule",
    note: "Issues the retarget below — including scale, hold and undo, which are the same event in another form." },
];

// Which boxes each predicate reads, mirroring lr_triggers._REQUIRED_FIELDS. A
// box the predicate does NOT read is refused rather than ignored (D46), so the
// ones that do not apply are not rendered and never reach the request.
const REQUIRED_FIELDS: Record<LrTriggerPredicate, string[]> = {
  plateau: ["interval", "patience", "min_delta"],
  below: ["interval", "threshold"],
  above: ["interval", "threshold"],
};

const numeric = (raw: string | undefined): number | undefined => {
  const text = (raw ?? "").trim();
  if (text === "") return undefined;
  const value = Number(text);
  return Number.isFinite(value) ? value : undefined;
};

/** What one registered trigger watches, in a sentence. */
const conditionText = (t: LrScheduleTriggerState): string => {
  const windowText = `every ${t.interval.toLocaleString()} global steps`;
  if (t.predicate === "plateau") {
    return `${t.signal} — no drop of more than ${t.min_delta} for `
      + `${t.patience} observations, averaged ${windowText}`;
  }
  return `${t.signal} — ${t.predicate} ${t.threshold}, averaged ${windowText}`;
};

/** What it will do when it fires. `op` is the retarget family's own label. */
const actionText = (action: Record<string, any>): string => {
  const command = String(action?.command ?? "retarget");
  if (command !== "retarget") {
    return ACTION_COMMANDS.find((c) => c.value === command)?.label ?? command;
  }
  const parts: string[] = [String(action?.op ?? "retarget")];
  if (action?.lr_scheduler) parts.push(String(action.lr_scheduler));
  if (action?.gain !== undefined) parts.push(`gain ${action.gain}`);
  if (Array.isArray(action?.groups) && action.groups.length > 0) {
    parts.push(`groups ${action.groups.join(", ")}`);
  }
  return parts.join(" · ");
};

interface Props {
  runId: number;
  /** The run's published schedule, for the retarget form the action reuses. */
  status: LrScheduleStatusResponse | null;
}

export default function LrScheduleTriggerPanel({ runId, status }: Props) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<LrScheduleTriggersResponse | null>(null);
  const [id, setId] = useState("");
  const [signal, setSignal] = useState<string>(SIGNALS[0].value);
  const [extraName, setExtraName] = useState("");
  const [predicate, setPredicate] = useState<LrTriggerPredicate>("plateau");
  // Every numeric box is a STRING and starts EMPTY. Blank is left out of the
  // request and refused by name (D46, invariant 20): this build has no default
  // for interval, patience, min_delta or threshold, and neither has this form.
  const [numbers, setNumbers] = useState<Record<string, string>>({});
  const [command, setCommand] = useState<LrTriggerCommand>("retarget");
  const [action, setAction] = useState<RetargetFormState>(EMPTY_RETARGET_FORM);
  const [registering, setRegistering] = useState(false);
  const [registerError, setRegisterError] = useState<string | null>(null);
  const [queued, setQueued] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState<string | null>(null);

  const { lrTriggerDefaults: defaults } = useStartup();

  const refresh = useCallback(async () => {
    try {
      setData(await getLrScheduleTriggers(runId));
    } catch {
      setData(null);
    }
  }, [runId]);

  // The observation counters move on their own between commands, so this is
  // polled while the run executes rather than fetched once.
  const parentRunning = !!status?.is_running;
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const next = await getLrScheduleTriggers(runId);
        if (!cancelled) setData(next);
      } catch {
        if (!cancelled) setData(null);
      }
    };
    load();
    if (!parentRunning) {
      return () => {
        cancelled = true;
      };
    }
    const interval = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [runId, parentRunning]);

  const running = !!data?.is_running;
  const triggers = data?.triggers ?? [];

  // `max_fires` is the ONE field with a default, and it comes from
  // /schema/lr-trigger-defaults. Undefined until that lands, which is why a
  // blank box does not resolve to a number here either.
  const maxFires = numeric(numbers.max_fires) ?? defaults?.max_fires;
  // D62: required above 1, refused at 1. Both directions are the same gate, so
  // the box is only shown — and only sent — when it is required.
  const needsCooldown = maxFires !== undefined && maxFires > 1;

  const composedSignal =
    signal === "extra"
      ? (extraName.trim() === "" ? "" : `extra:${extraName.trim()}`)
      : signal;

  const request = useMemo((): LrScheduleTriggerRequest => {
    const out: LrScheduleTriggerRequest = {};
    if (id.trim() !== "") out.id = id.trim();
    if (composedSignal !== "") out.signal = composedSignal;
    out.predicate = predicate;
    out.interval = numeric(numbers.interval);
    if (predicate === "plateau") {
      out.patience = numeric(numbers.patience);
      out.min_delta = numeric(numbers.min_delta);
    } else {
      out.threshold = numeric(numbers.threshold);
    }
    out.max_fires = numeric(numbers.max_fires);
    if (needsCooldown) out.cooldown = numeric(numbers.cooldown);
    out.action = command === "retarget"
      ? { command, ...retargetPayload(action, { includeAt: false }) }
      : { command };

    for (const key of Object.keys(out) as (keyof LrScheduleTriggerRequest)[]) {
      if (out[key] === undefined) delete out[key];
    }
    return out;
  }, [id, composedSignal, predicate, numbers, needsCooldown, command, action]);

  const missing = useMemo(() => {
    const out: string[] = [];
    if (composedSignal === "") out.push("signal");
    for (const key of REQUIRED_FIELDS[predicate]) {
      if ((request as Record<string, any>)[key] === undefined) out.push(key);
    }
    if (needsCooldown && request.cooldown === undefined) out.push("cooldown");
    if (command === "retarget" && retargetIncomplete(action)) {
      out.push(action.op === "scale" ? "the action's gain" : "the action's schedule");
    }
    return out;
  }, [composedSignal, predicate, request, needsCooldown, command, action]);

  const handleRegister = async () => {
    setRegistering(true);
    setRegisterError(null);
    setQueued(null);
    try {
      const accepted = await queueLrScheduleTrigger(runId, request);
      setQueued(accepted.trigger?.id ?? accepted.request_id);
      await refresh();
    } catch (err: any) {
      setRegisterError(
        err?.response?.data?.detail || err?.message || "Failed to queue the trigger"
      );
    } finally {
      setRegistering(false);
    }
  };

  const handleCancel = async (triggerId: string) => {
    setCancelling(triggerId);
    setRegisterError(null);
    try {
      await cancelLrScheduleTrigger(runId, triggerId);
      await refresh();
    } catch (err: any) {
      setRegisterError(
        err?.response?.data?.detail || err?.message || "Failed to queue the cancellation"
      );
    } finally {
      setCancelling(null);
    }
  };

  const numberField = (
    key: string, label: string, placeholder: string, step?: string
  ) => (
    <label key={key} className="block">
      <span className="block text-gray-400">{label}</span>
      <input
        type="number"
        step={step}
        value={numbers[key] ?? ""}
        placeholder={placeholder}
        onChange={(e) => setNumbers((prev) => ({ ...prev, [key]: e.target.value }))}
        className={LR_FIELD_CLASS}
      />
    </label>
  );

  return (
    <div className="space-y-2 rounded border border-gray-700 bg-gray-900/60 p-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between text-xxs text-gray-300 hover:text-gray-100"
      >
        <span className="font-medium">
          Conditions
          {triggers.length > 0 && (
            <span className="ml-1 text-gray-500">
              {triggers.filter((t) => t.armed).length} armed of {triggers.length}
            </span>
          )}
        </span>
        <span className="text-gray-500">{open ? "hide" : "show"}</span>
      </button>

      {/* D52: what each one watches, where its observation is and how much is
          left. Shown whether or not the form is open — an automation nobody
          can see coming is worse than watching the chart. */}
      {triggers.length > 0 && (
        <ul className="space-y-1 text-xxs">
          {triggers.map((t) => (
            <li
              key={t.id}
              className="rounded border border-gray-700 bg-gray-950/60 p-1.5"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <span className="font-mono text-gray-200">{t.id}</span>
                  <span className={`ml-1 ${t.armed ? "text-green-400" : "text-gray-500"}`}>
                    {t.armed ? "armed" : "disarmed"}
                  </span>
                  <p className="text-gray-400">{conditionText(t)}</p>
                  <p className="text-gray-400">
                    Fires: <span className="text-gray-300">{actionText(t.action)}</span>
                  </p>
                </div>
                <button
                  onClick={() => handleCancel(t.id)}
                  disabled={!running || cancelling !== null}
                  title={
                    !running
                      ? "Only a running training run can be told to drop a trigger."
                      : "Disarms and removes it. What it already fired stays on the schedule."
                  }
                  className="flex-shrink-0 px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {cancelling === t.id ? "Sending..." : "Cancel"}
                </button>
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-gray-400">
                <span>
                  Now{" "}
                  <span className="font-mono text-gray-100">
                    {t.observation == null ? "—" : t.observation}
                  </span>
                  {t.observation_step != null && ` at global step ${t.observation_step.toLocaleString()}`}
                </span>
                <span>{t.observations.toLocaleString()} observations</span>
                {t.created_step != null && (
                  <span>armed at global step {t.created_step.toLocaleString()}</span>
                )}
                {t.best != null && (
                  <span>best <span className="font-mono text-gray-100">{t.best}</span></span>
                )}
                {t.patience_used != null && t.patience != null && (
                  <span>
                    without improvement{" "}
                    <span className="font-mono text-gray-100">
                      {t.patience_used}/{t.patience}
                    </span>
                  </span>
                )}
                {t.observations_to_fire != null && (
                  <span>
                    {t.observations_to_fire} more would fire it
                  </span>
                )}
                <span>
                  fired {t.fires}/{t.max_fires}, {t.fires_left} left
                  {t.cooldown != null && `, ${t.cooldown} observations apart`}
                </span>
                {!!t.cooldown_reason && (
                  <span className="text-yellow-400">
                    {/* The number cannot say which: D64's debounce after a
                        REFUSED firing sets the same counter as the configured
                        cooldown after one that took effect. */}
                    {t.cooldown_reason === "refusal"
                      ? `standing off ${t.cooldown_left} more observation(s) after a refused firing`
                      : `cooling down for ${t.cooldown_left} more observations after firing`}
                  </span>
                )}
              </div>
              {!!t.refusals && (
                <p className="mt-0.5 leading-relaxed text-yellow-400">
                  Matched and was refused {t.refusals}{" "}
                  {t.refusals === 1 ? "time" : "times"}
                  {t.last_refusal_step != null
                    && `, last at global step ${t.last_refusal_step.toLocaleString()}`}
                  {t.last_refusal
                    ? `: ${lrScheduleResultExplanation(t.last_refusal)}`
                    : "."}{" "}
                  A refused firing is not a firing: it cost none of the fires above,
                  and the condition is still live.
                </p>
              )}
              {t.observation == null && (
                <p className="mt-0.5 text-gray-500">
                  No observation has closed yet
                  {t.created_step != null
                    && `: it was armed at global step ${t.created_step.toLocaleString()}`}
                  . The window in progress at registration was discarded rather than
                  averaged over part of itself, one closes at the end of each window of{" "}
                  {t.interval.toLocaleString()} global steps, and a window with no sample
                  in it produces none.
                </p>
              )}
            </li>
          ))}
        </ul>
      )}

      {open && (
        <div className="space-y-2 text-xxs">
          {triggers.length === 0 && (
            <p className="leading-relaxed text-gray-500">
              Nothing is registered. A trigger is not a new control: when its condition
              is met it presses one of the buttons above for you, and what lands on the
              schedule is the ordinary event that button makes.
            </p>
          )}

          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className="block text-gray-400">Watch</span>
              <select
                value={signal}
                onChange={(e) => setSignal(e.target.value)}
                className={LR_FIELD_CLASS}
              >
                {SIGNALS.map((s) => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
                <option value="extra">extra:… — a metric this run logs</option>
              </select>
            </label>
            {signal === "extra" ? (
              <label className="block">
                <span className="block text-gray-400">Metric name</span>
                <input
                  type="text"
                  value={extraName}
                  placeholder="required"
                  onChange={(e) => setExtraName(e.target.value)}
                  className={LR_FIELD_CLASS}
                />
              </label>
            ) : (
              <div />
            )}
          </div>
          <p className="leading-relaxed text-gray-500">
            Only what the trainer already computes every step. The learning rate is not
            on this list, in either its own name or the <span className="font-mono">extra:lr</span>{" "}
            reporting the run publishes: a trigger changes the learning rate, so a
            condition on it responds to its own effect. <span className="font-mono">grad_norm</span>{" "}
            exists only on optimizer-update steps, so under gradient accumulation a short
            window holds fewer samples than steps.
          </p>

          <div className="flex gap-1">
            {PREDICATES.map((entry) => (
              <button
                key={entry.value}
                onClick={() => setPredicate(entry.value)}
                className={`flex-1 px-2 py-1 rounded transition-colors ${
                  predicate === entry.value
                    ? "bg-blue-700 text-white"
                    : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                }`}
              >
                {entry.label}
              </button>
            ))}
          </div>
          <p className="leading-relaxed text-gray-500">
            {PREDICATES.find((e) => e.value === predicate)?.note}
          </p>

          <div className="grid grid-cols-2 gap-2">
            {numberField("interval", "Observation window (global steps, required)", "required")}
            {/* D46 as amended: a box the predicate does not read is refused,
                not ignored, so only its own are rendered or sent. */}
            {predicate === "plateau" ? (
              <>
                {numberField("patience", "Patience (observations, required)", "required")}
                {numberField("min_delta", "Minimum improvement (required)", "required", "any")}
              </>
            ) : (
              numberField("threshold", "Threshold (required)", "required", "any")
            )}
            {numberField(
              "max_fires",
              data?.max_trigger_fires != null
                ? `Max fires (up to ${data.max_trigger_fires})` : "Max fires",
              defaults ? `default (${defaults.max_fires})` : "default")}
            {needsCooldown &&
              numberField("cooldown", "Cooldown (observations, required)", "required")}
            <label className="block">
              <span className="block text-gray-400">Id</span>
              <input
                type="text"
                value={id}
                placeholder="generated"
                onChange={(e) => setId(e.target.value)}
                className={LR_FIELD_CLASS}
              />
            </label>
          </div>
          <p className="leading-relaxed text-gray-500">
            An empty required box is refused by name, not filled in: a usable window,
            patience, minimum improvement or threshold depends entirely on this run&apos;s
            own loss scale, and nothing here has measured it. Read the loss chart and
            choose them. One observation is the MEAN over its window and patience counts
            observations, not steps — the per-step loss of a diffusion run varies more
            between steps than it trends.
            {needsCooldown && " More than one firing needs a cooldown, or the same"
              + " plateau fires it again on the very next observation and spends every"
              + " fire at once."}
          </p>

          <div className="flex gap-1">
            {ACTION_COMMANDS.map((entry) => (
              <button
                key={entry.value}
                onClick={() => setCommand(entry.value)}
                className={`flex-1 px-2 py-1 rounded transition-colors ${
                  command === entry.value
                    ? "bg-blue-700 text-white"
                    : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                }`}
              >
                {entry.label}
              </button>
            ))}
          </div>
          <p className="leading-relaxed text-gray-500">
            {ACTION_COMMANDS.find((e) => e.value === command)?.note}
          </p>

          {/* D59: the same retarget form, with no "apply at" box. A firing
              takes effect where it fires, and a fixed step would be in the
              past for every firing after the first. */}
          {command === "retarget" && (
            <div className="rounded border border-gray-700 bg-gray-950/60 p-2">
              <LrRetargetFields
                form={action}
                onChange={setAction}
                state={status?.status ?? null}
                includeAt={false}
              />
            </div>
          )}

          <button
            onClick={handleRegister}
            disabled={!running || registering || missing.length > 0}
            title={
              !running
                ? "Only a running training run can arm a trigger."
                : missing.length > 0
                ? `Still needed: ${missing.join(", ")}. This build supplies no default for any of them.`
                : undefined
            }
            className="w-full px-2 py-1.5 bg-blue-700 hover:bg-blue-600 rounded text-xxs sm:text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {registering ? "Sending..." : "Register condition"}
          </button>
          {missing.length > 0 && (
            <p className="leading-relaxed text-gray-400">
              Still needed: {missing.join(", ")}.
            </p>
          )}
          {queued && (
            <p className="leading-relaxed text-gray-400">
              Queued as {queued}. The trainer arms it at the head of its next batch, and
              it appears above once it has. It is saved with the training state, so
              resuming from a checkpoint written before it was registered drops it —
              along with anything it had fired by then.
            </p>
          )}
          {registerError && (
            <p className="leading-relaxed text-red-400">{registerError}</p>
          )}
        </div>
      )}

      {!!data?.pending?.length && (
        <p className="text-xxs text-gray-300">
          Queued:{" "}
          {data.pending
            .map((p) => `${p.command}${p.trigger_id ? ` ${p.trigger_id}` : ""}`)
            .join(", ")}{" "}
          (max {data.max_pending})
        </p>
      )}
      {data?.results?.slice(0, 3).map((r) => (
        <p
          key={r.request_id}
          className={`text-xxs leading-relaxed ${
            r.result.startsWith("rejected_") || r.result === "error"
              ? "text-yellow-400"
              : "text-gray-400"
          }`}
        >
          {r.trigger_id ? `${r.trigger_id}: ` : ""}
          {lrScheduleResultExplanation(r.result)}
          {r.error ? ` (${r.error})` : ""}
        </p>
      ))}
      {open && data && (
        <p className="text-xxs leading-relaxed text-gray-500">
          {triggers.length} of {data.max_triggers} registered
          {data.max_trigger_fires != null
            && `, each able to fire up to ${data.max_trigger_fires} times`}
          . A firing the schedule refuses is not a firing: it costs no fire, and says so
          above.
        </p>
      )}
    </div>
  );
}
