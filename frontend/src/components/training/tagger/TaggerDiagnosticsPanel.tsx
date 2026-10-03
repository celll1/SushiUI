"use client";

import { useEffect, useState } from "react";
import {
  getTaggerDiagnostics, setTaggerDiagnostics, createTaggerDiagnostic,
  getTaggerDiagnosticImage, TaggerDiagnosticStatus, TaggerDiagnosticSettings,
} from "@/utils/api";

export default function TaggerDiagnosticsPanel({ runId, active }: { runId: string; active: boolean }) {
  const [state, setState] = useState<TaggerDiagnosticStatus | null>(null);
  const [draft, setDraft] = useState<TaggerDiagnosticSettings | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const filename = state?.items[0]?.filename;
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
    const refresh = async () => {
      try {
        const data = await getTaggerDiagnostics(runId);
        if (!cancelled) {
          setState(data);
          setDraft(previous => previous ?? data.settings);
        }
      } catch { if (!cancelled) setError("診断情報を取得できませんでした。"); }
    };
    setState(null); setDraft(null); setError("");
    void refresh();
    const timer = setInterval(refresh, 10000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [runId]);
  const update = async (generate: boolean) => {
    if (!draft) return;
    setPending(true); setError("");
    try {
      const saved = await setTaggerDiagnostics(runId, draft);
      setState(generate ? await createTaggerDiagnostic(runId) : saved);
    } catch { setError("診断設定・生成に失敗しました。学習は変更されていません。"); }
    finally { setPending(false); }
  };
  const latest = state?.items[0];
  return <section className="bg-gray-900 rounded-lg p-3 space-y-2">
    <h3 className="text-sm text-gray-200">診断画像（任意・個別タグ名なし）</h3>
    {draft && <>
      <label className="flex gap-2 text-xs text-gray-300">
        <input type="checkbox" checked={draft.enabled} disabled={pending}
          onChange={e => setDraft({ ...draft, enabled: e.target.checked })} />
        保存checkpointから定期作成（デフォルトOFF）
      </label>
      <label className="flex gap-2 items-center text-xs text-gray-300">
        最小間隔（step）
        <input type="number" min={1000} max={1000000} step={1000}
          value={draft.interval_steps} disabled={pending}
          onChange={e => setDraft({ ...draft, interval_steps: Number(e.target.value) })}
          className="bg-gray-800 rounded p-1 w-24" />
      </label>
      <label className="flex gap-2 text-xs text-gray-300">
        <input type="checkbox" checked={draft.focus_preview} disabled={pending}
          onChange={e => setDraft({ ...draft, focus_preview: e.target.checked })} />
        旧A/B診断のみ：C方式のプレビューを併記（router Runには適用しない）
      </label>
      <div className="flex gap-2">
        <button disabled={pending || !Number.isInteger(draft.interval_steps) || draft.interval_steps < 1000 || draft.interval_steps > 1000000}
          onClick={() => void update(false)} className="text-xs bg-gray-700 rounded px-2 py-1 disabled:opacity-40">設定を保存</button>
        <button disabled={pending || state?.busy || active || !Number.isInteger(draft.interval_steps) || draft.interval_steps < 1000 || draft.interval_steps > 1000000}
          onClick={() => void update(true)} className="text-xs bg-gray-700 rounded px-2 py-1 disabled:opacity-40">今すぐ作成（停止中のみ）</button>
      </div>
    </>}
    <p className="text-xs text-gray-400">Generalの固定画像。TP/FP/FNは各上位100件、TNは件数のみ。CPUで別実行。router Runでは学習中routerが実際に選択した原画像領域を表示します。枠だけでタグへの寄与や選択の正しさを保証するものではなく、目視レビュー待ちです。</p>
    {state?.busy && <p className="text-xs text-gray-400">診断画像を作成中…</p>}
    {(error || state?.error) && <p className="text-xs text-red-400">{error || state?.error}</p>}
    {latest && <>
      <p className="text-xs text-gray-300">{latest.mode} · step {latest.step.toLocaleString()} · TP {latest.counts.TP} / FP {latest.counts.FP} / FN {latest.counts.FN} / TN {latest.counts.TN}</p>
      {imageUrl && <a href={imageUrl} target="_blank" rel="noreferrer">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={imageUrl} alt="匿名TP/FP/FN診断とfocus入力範囲" className="w-full rounded" />
      </a>}
    </>}
  </section>;
}
