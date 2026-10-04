"use client";

import { useEffect, useRef, useState } from "react";
import {
  getTaggerDiagnostics, setTaggerDiagnostics, createTaggerDiagnostic,
  getTaggerDiagnosticImage, TaggerDiagnosticStatus, TaggerDiagnosticSettings,
} from "@/utils/api";

export default function TaggerDiagnosticsPanel({ runId, active, legacyComparison = false }: { runId: string; active: boolean; legacyComparison?: boolean }) {
  const [state, setState] = useState<TaggerDiagnosticStatus | null>(null);
  const [draft, setDraft] = useState<TaggerDiagnosticSettings | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null);
  const generation = useRef(0);
  const validSettings = !!draft && Number.isInteger(draft.interval_steps) && draft.interval_steps >= 1000 && draft.interval_steps <= 1000000
    && Number.isFinite(draft.threshold) && draft.threshold >= 0 && draft.threshold <= 1
    && Number.isInteger(draft.min_samples_for_per_tag) && draft.min_samples_for_per_tag >= 1
    && Number.isFinite(draft.min_best_thr) && draft.min_best_thr >= 0 && draft.min_best_thr <= 1;
  const selectedIndex = Math.max(0, state?.items.findIndex(item => item.filename === selectedFilename) ?? 0);
  const selected = state?.items[selectedIndex];
  const filename = selected?.filename;
  const skippedAttempts = state?.attempts?.filter(attempt => attempt.status === "skipped") ?? [];
  useEffect(() => {
    let cancelled = false;
    let ownedUrl: string | undefined;
    setImageUrl(null);
    if (filename) void getTaggerDiagnosticImage(runId, filename).then(blob => {
      if (!cancelled) {
        ownedUrl = URL.createObjectURL(blob);
        setImageUrl(ownedUrl);
      }
    }).catch(() => { if (!cancelled) setError("診断画像を取得できませんでした。"); });
    return () => { cancelled = true; if (ownedUrl) URL.revokeObjectURL(ownedUrl); };
  }, [runId, filename]);
  useEffect(() => {
    let cancelled = false;
    generation.current += 1;
    const refresh = async () => {
      try {
        const data = await getTaggerDiagnostics(runId);
        if (!cancelled) {
          setState(data);
          setDraft(previous => previous ?? data.settings);
        }
      } catch { if (!cancelled) setError("診断情報を取得できませんでした。"); }
    };
    setState(null); setDraft(null); setError(""); setSelectedFilename(null); setPending(false);
    void refresh();
    const timer = setInterval(refresh, 10000);
    return () => { cancelled = true; generation.current += 1; clearInterval(timer); };
  }, [runId]);
  const update = async (generate: boolean) => {
    if (!draft || !validSettings) return;
    const requestGeneration = generation.current;
    setPending(true); setError("");
    try {
      const saved = await setTaggerDiagnostics(runId, draft);
      if (generation.current !== requestGeneration) return;
      const nextState = generate ? await createTaggerDiagnostic(runId, selected?.snapshot_filename ? selected.filename : undefined) : saved;
      if (generation.current === requestGeneration) setState(nextState);
    } catch { if (generation.current === requestGeneration) setError("診断設定・生成に失敗しました。学習は変更されていません。"); }
    finally { if (generation.current === requestGeneration) setPending(false); }
  };
  return <section className="bg-gray-900 rounded-lg p-3 space-y-2">
    <h3 className="text-sm text-gray-200">診断画像（任意）</h3>
    {draft && <>
      <label className="flex gap-2 text-xs text-gray-300">
        <input type="checkbox" checked={draft.enabled} disabled={pending}
          onChange={e => setDraft({ ...draft, enabled: e.target.checked })} />
        学習stepの画像から定期作成（デフォルトOFF）
      </label>
      <label className="flex gap-2 items-center text-xs text-gray-300">
        判定閾値
        <select value={draft.threshold_mode} disabled={pending}
          onChange={e => setDraft({...draft, threshold_mode: e.target.value as TaggerDiagnosticSettings["threshold_mode"]})}
          className="bg-gray-800 rounded p-1">
          <option value="training">保存時の学習共通閾値</option>
          <option value="global">指定した共通閾値</option>
          <option value="per_tag">タグ別最適閾値（不足時は共通値）</option>
        </select>
      </label>
      {draft.threshold_mode !== "training" && <label className="flex gap-2 items-center text-xs text-gray-300">
        共通値 / fallback
        <input type="number" min={0} max={1} step={0.01} value={draft.threshold} disabled={pending}
          onChange={e => setDraft({...draft, threshold: Number(e.target.value)})} className="bg-gray-800 rounded p-1 w-24" />
      </label>}
      {draft.threshold_mode === "per_tag" && <div className="flex gap-2 items-center text-xs text-gray-300">
        <label>最少positive数 <input type="number" min={1} step={1} value={draft.min_samples_for_per_tag} disabled={pending}
          onChange={e => setDraft({...draft, min_samples_for_per_tag: Number(e.target.value)})} className="bg-gray-800 rounded p-1 w-20" /></label>
        <label>閾値下限 <input type="number" min={0} max={1} step={0.01} value={draft.min_best_thr} disabled={pending}
          onChange={e => setDraft({...draft, min_best_thr: Number(e.target.value)})} className="bg-gray-800 rounded p-1 w-20" /></label>
      </div>}
      <label className="flex gap-2 items-center text-xs text-gray-300">
        作成間隔（step）
        <input type="number" min={1000} max={1000000} step={1}
          value={draft.interval_steps} disabled={pending}
          onChange={e => setDraft({ ...draft, interval_steps: Number(e.target.value) })}
          className="bg-gray-800 rounded p-1 w-24" />
      </label>
      {legacyComparison && <label className="flex gap-2 text-xs text-gray-300">
        <input type="checkbox" checked={draft.focus_preview} disabled={pending}
          onChange={e => setDraft({ ...draft, focus_preview: e.target.checked })} />
        旧A/B診断のみ：C方式のプレビューを併記（router Runには適用しない）
      </label>}
      <div className="flex gap-2">
        <button disabled={pending || !validSettings}
          onClick={() => void update(false)} className="text-xs bg-gray-700 rounded px-2 py-1 disabled:opacity-40">設定を保存</button>
        <button disabled={pending || state?.busy || (legacyComparison && active) || !validSettings}
          onClick={() => void update(true)} className="text-xs bg-gray-700 rounded px-2 py-1 disabled:opacity-40">現在の閾値で再描画（保存済みforward）</button>
      </div>
    </>}
    <p className="text-xs text-gray-400">SigLIP2 / Foveaとも実際のforwardを保存します（Rating制限なし）。学習・共通指定・タグ別閾値でTP/FP/FNを判定します。タグ別閾値は保存時の統計を使用し、不足時はfallbackへ戻ります。再描画はCPUのみで、予測値・学習・focusを変更しません。Router枠は実線、ランダム探索は破線。旧比較の固定画像と匿名表示は履歴に残ります。</p>
    {state?.busy && <p className="text-xs text-gray-400">診断画像を作成中…</p>}
    {(error || state?.error) && <p className="text-xs text-red-400">{error || state?.error}</p>}
    {!!skippedAttempts.length && <p className="text-xs text-gray-400">未作成step: {skippedAttempts.map(attempt => `${attempt.step.toLocaleString()} (${attempt.reason === "no_source_metadata_in_step_batch" ? "入力情報なし" : "診断処理中"})`).join(" / ")}</p>}
    {selected && <>
      <div className="flex flex-wrap gap-2 items-center text-xs">
        <button disabled={selectedIndex >= (state?.items.length ?? 0) - 1}
          onClick={() => setSelectedFilename(state?.items[selectedIndex + 1]?.filename ?? null)}
          className="bg-gray-700 rounded px-2 py-1 disabled:opacity-40">古い画像へ</button>
        <select value={filename} onChange={e => setSelectedFilename(e.target.value)} className="bg-gray-800 rounded p-1" aria-label="診断画像の履歴">
          {state?.items.map(item => <option key={item.filename} value={item.filename}>step {item.step.toLocaleString()} · {item.mode} · {item.threshold_mode ?? "旧設定"}</option>)}
        </select>
        <button disabled={selectedIndex === 0} onClick={() => setSelectedFilename(state?.items[selectedIndex - 1]?.filename ?? null)}
          className="bg-gray-700 rounded px-2 py-1 disabled:opacity-40">新しい画像へ</button>
        <button onClick={() => setSelectedFilename(null)} className="bg-gray-700 rounded px-2 py-1">最新を表示</button>
        <span className="text-gray-400">{selectedIndex + 1} / {state?.items.length}</span>
      </div>
      <p className="text-xs text-gray-300">{selected.mode} · step {selected.step.toLocaleString()} · {selected.tag_labels === "vocabulary" ? "タグ名表示" : "匿名表示（旧画像）"} · TP {selected.counts.TP} / FP {selected.counts.FP} / FN {selected.counts.FN} / TN {selected.counts.TN}</p>
      {selected.threshold_mode && <p className="text-xs text-gray-400">閾値方式: {selected.threshold_mode}{selected.threshold_mode === "per_tag" ? ` · タグ別 ${selected.per_tag_threshold_count ?? 0}件 / fallback ${selected.threshold_fallback_count ?? 0}件` : ""}。区間集計はその区間に記載の閾値で、画像の再描画では再計算しません。</p>}
      {selected.training_window && <div className="text-xs text-gray-300 space-y-1">
        <p>区間集計（学習ラベル基準・検証精度ではありません）: step {selected.training_window.start_step.toLocaleString()}–{selected.training_window.end_step.toLocaleString()} · {selected.training_window.samples.toLocaleString()}枚 · threshold {selected.training_window.threshold}</p>
        <p>Precision {selected.training_window.precision == null ? "—" : `${(selected.training_window.precision * 100).toFixed(2)}%`} / Recall {selected.training_window.recall == null ? "—" : `${(selected.training_window.recall * 100).toFixed(2)}%`} / FP率（既知negative中） {selected.training_window.fp_rate == null ? "—" : `${(selected.training_window.fp_rate * 100).toFixed(4)}%`} / FP/画像 {selected.training_window.fp_per_image?.toFixed(2) ?? "—"}</p>
        <p>TP {selected.training_window.tp.toLocaleString()} / FP {selected.training_window.fp.toLocaleString()} / FN {selected.training_window.fn.toLocaleString()}。再開・閾値変更後の区間は短くなる場合があります。</p>
      </div>}
      {imageUrl && <a href={imageUrl} target="_blank" rel="noreferrer">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={imageUrl} alt="TP/FP/FN診断とfocus入力範囲" className="w-full rounded" />
      </a>}
    </>}
  </section>;
}
