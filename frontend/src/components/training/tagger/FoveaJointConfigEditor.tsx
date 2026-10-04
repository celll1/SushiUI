"use client";

import { useState } from "react";
import { editFoveaJointConfig, TaggerTrainingRun } from "@/utils/api";

const fields = [
  ["epochs", "Epochs"], ["batch_size", "Batch size"], ["workers", "Workers"],
  ["learning_rate", "Learning rate"], ["weight_decay", "Weight decay"], ["warmup_steps", "Warmup steps"],
  ["candidate_count", "Candidates"], ["selection_budget", "Focus budget K"], ["random_count", "Random exploration slots"],
  ["oracle_interval", "Oracle interval"], ["oracle_chunk_size", "Oracle chunk size"], ["router_weight", "Router loss weight"],
  ["checkpoint_every_steps", "Checkpoint interval"], ["log_every", "Log interval"], ["review_steps", "Bounded-start review steps"],
] as const;

export default function FoveaJointConfigEditor({ run, active, onSaved, onClose }: {
  run: TaggerTrainingRun; active: boolean; onSaved: (run: TaggerTrainingRun) => void; onClose: () => void;
}) {
  const [draft, setDraft] = useState<Record<string, number | boolean>>(() => {
    const plan = run.config.joint_review as Record<string, number | boolean>;
    return Object.fromEntries([...fields.map(([key]) => [key, plan[key]]), ["external_flash", plan.external_flash]]);
  });
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const save = async () => {
    setPending(true); setError("");
    try { onSaved(await editFoveaJointConfig(run.run_id, draft)); onClose(); }
    catch { setError("保存に失敗しました。停止状態・値・0 ≤ random < K ≤ candidates を確認してください。"); }
    finally { setPending(false); }
  };
  return <section className="bg-gray-900 rounded-lg p-3 space-y-3">
    <h3 className="text-sm text-gray-200">Edit Config · FoveaTag joint training</h3>
    <p className="text-xs text-gray-400">変更は停止後に保存し、次回Startから反映します。Adamの状態とRNGは保持します。LR・warmup・epoch・batch変更時は保存stepを基準にスケジュールを再計算します。データ・モデル構造・入力方式はこのRunでは変更できません。</p>
    {active && <p className="text-xs text-yellow-400">編集中も学習は継続します。保存する前に下のStopで停止してください。</p>}
    <div className="grid grid-cols-2 gap-2">
      {fields.map(([key, label]) => <label key={key} className="text-xs text-gray-300 flex flex-col gap-1">
        {label}
        <input type="number" step="any" value={String(draft[key])} disabled={pending}
          onChange={event => setDraft({ ...draft, [key]: Number(event.target.value) })}
          className="bg-gray-800 rounded p-1" />
      </label>)}
    </div>
    <label className="text-xs text-gray-300 flex gap-2">
      <input type="checkbox" checked={Boolean(draft.external_flash)} disabled={pending}
        onChange={event => setDraft({ ...draft, external_flash: event.target.checked })} />External FlashAttention
    </label>
    {error && <p className="text-xs text-red-400">{error}</p>}
    <div className="flex gap-2">
      <button onClick={() => void save()} disabled={active || pending || fields.some(([key]) => !Number.isFinite(Number(draft[key])))}
        className="text-xs bg-gray-700 rounded px-2 py-1 disabled:opacity-40">{pending ? "保存中…" : "設定を保存"}</button>
      <button onClick={onClose} disabled={pending} className="text-xs bg-gray-700 rounded px-2 py-1">閉じる</button>
    </div>
  </section>;
}
