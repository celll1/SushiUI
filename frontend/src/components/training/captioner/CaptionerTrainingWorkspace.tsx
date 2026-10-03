"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import CaptionerTrainingPanel from "./CaptionerTrainingPanel";
import { CaptionerPreview, getCaptionerDefaults, saveCaptionerPreview } from "@/utils/api";

export default function CaptionerTrainingWorkspace() {
  const [bundlePath, setBundlePath] = useState("");
  const [preview, setPreview] = useState<CaptionerPreview | null>(null);
  const [captionType, setCaptionType] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const sampleEpoch = useRef(0);
  const saving = useRef(false);
  const input = "rounded border border-gray-700 bg-gray-900 p-2 text-sm";

  useEffect(() => {
    let mounted = true;
    getCaptionerDefaults().then(defaults => {
      if (mounted) setCaptionType(defaults.save_caption_type);
    }).catch(e => {if (mounted) setError(String(e.message ?? e));});
    return () => {mounted = false;};
  }, []);

  const save = async () => {
    if (!preview || !captionType || saving.current) return;
    const epoch = sampleEpoch.current;
    saving.current = true;
    setBusy(true); setError("");
    try {
      await saveCaptionerPreview(preview, captionType);
      if (sampleEpoch.current === epoch) setSaved(true);
    } catch (e) {
      const value = e as {response?: {data?: {message?: unknown; detail?: unknown}}; message?: string};
      const detail = value.response?.data?.message || value.response?.data?.detail || value.message;
      if (sampleEpoch.current === epoch) setError(typeof detail === "string" ? detail : JSON.stringify(detail) ?? "保存に失敗しました");
    } finally {saving.current = false; setBusy(false);}
  };

  return <div className="min-w-0 flex-1 overflow-y-auto p-3 sm:p-4 space-y-4 text-gray-200">
    <CaptionerTrainingPanel onSelectBundle={setBundlePath} onSample={sample => {
      sampleEpoch.current += 1;
      setPreview(sample); setSaved(false); setError("");
    }} />
    {bundlePath && <section className="rounded border border-gray-700 p-3 space-y-2">
      <p className="text-sm">選択bundle / checkpoint</p>
      <code className="block break-all text-xs">{bundlePath}</code>
      <Link className="text-sm text-violet-300 underline" href="/tagger">Tagger画面のCaptionerでこのパスをロードして推論</Link>
    </section>}
    {error && <p role="alert" className="text-red-400">{error}</p>}
    {preview && <section className="rounded border border-gray-700 p-3 space-y-2">
      <h3>Checkpoint sample {saved && "· 保存済み"}</h3>
      <p className="break-all text-xs">{preview.run_id} · {preview.checkpoint} · item #{preview.item_id}</p>
      <textarea aria-label="Caption training sample" className={`${input} w-full`} rows={4}
        disabled={busy || saved} value={preview.caption}
        onChange={e => setPreview({...preview, caption: e.target.value})} />
      <p className="text-xs text-gray-400">{preview.termination} · {preview.token_count} tokens · {preview.warnings.join(" / ")}</p>
      <details><summary>使用タグ（未校正score）</summary><pre className="overflow-auto text-xs">{JSON.stringify(preview.used_tags, null, 2)}</pre></details>
      <label>保存先caption type <input className={input} disabled={busy || saved || captionType === null}
        value={captionType ?? ""} onChange={e => setCaptionType(e.target.value)} /></label>
      <button className={input} disabled={busy || saved || !captionType?.trim() || !preview.caption.trim()}
        onClick={() => void save()}>確認済みsampleを保存</button>
      <p className="text-xs text-gray-400">自動保存はしません。保存はJSON自然文列だけを変更し、TXT tagsを保持します。</p>
    </section>}
  </div>;
}
