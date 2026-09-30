"use client";

import { useEffect, useRef, useState } from "react";
import CaptionerTrainingPanel from "./CaptionerTrainingPanel";
import {
  CaptionerDefaults, CaptionerOptions, CaptionerPreview, CaptionerStatus, Dataset,
  getCaptionerDefaults, getCaptionerStatus, loadCaptioner, unloadCaptioner,
  predictCaptioner, predictCaptionerItem, cancelCaptionerRequest, saveCaptionerPreview, listDatasets,
} from "@/utils/api";

type Result = { key: string; preview?: CaptionerPreview; error?: string; saved?: boolean };
const message = (error: unknown) => {
  const value = error as {response?: {data?: {message?: unknown; detail?: unknown}}; message?: string};
  const detail = value.response?.data?.message || value.response?.data?.detail || value.message || "処理に失敗しました";
  return typeof detail === "string" ? detail : JSON.stringify(detail);
};

export default function CaptionerPanel() {
  const [defaults, setDefaults] = useState<CaptionerDefaults | null>(null);
  const [status, setStatus] = useState<CaptionerStatus>({loaded: false});
  const [options, setOptions] = useState<CaptionerOptions | null>(null);
  const [path, setPath] = useState("");
  const [tags, setTags] = useState("");
  const [captionType, setCaptionType] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [itemIds, setItemIds] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const active = useRef<string | null>(null);
  const cancelled = useRef(false);

  useEffect(() => {
    let mounted = true;
    Promise.all([getCaptionerDefaults(), getCaptionerStatus(), listDatasets()]).then(([d, s, data]) => {
      if (!mounted) return;
      setDefaults(d); setStatus(s);
      setOptions({tag_source: d.tag_source, tags: d.tags, device: d.device,
        max_new_tokens: Math.min(d.max_new_tokens, s.max_new_tokens ?? d.max_new_tokens)});
      setTags(d.tags.join("\n")); setCaptionType(d.save_caption_type); setDatasets(data.datasets);
    }).catch(e => {if (mounted) setError(message(e));});
    return () => {
      mounted = false; cancelled.current = true;
      if (active.current) void cancelCaptionerRequest(active.current).catch(() => {});
    };
  }, []);

  const modelAction = async (load: boolean) => {
    setBusy(true); setError("");
    try {
      const next = load ? await loadCaptioner(path) : await unloadCaptioner();
      setStatus(next);
      if (next.max_new_tokens) setOptions(old => old && ({...old, max_new_tokens: Math.min(old.max_new_tokens, next.max_new_tokens!)}));
    } catch (e) {setError(message(e));}
    finally {setBusy(false);}
  };
  const run = async (fromDataset: boolean) => {
    if (!options || !defaults) return;
    const ids = [...new Set(itemIds.split(/[\s,]+/).filter(Boolean).map(Number))];
    if (fromDataset && (!Number(datasetId) || !ids.length || ids.some(id => !Number.isSafeInteger(id) || id < 1))) {
      setError("datasetと正しいitem IDを選択してください"); return;
    }
    if (!fromDataset && (!file || file.size > defaults.max_image_bytes)) {
      setError("画像ファイルを選択し、容量上限を確認してください"); return;
    }
    const requestOptions = {...options, tags: options.tag_source === "predicted" ? [] : tags.split("\n").map(t => t.trim()).filter(Boolean)};
    setBusy(true); setError(""); setResults([]); cancelled.current = false;
    try {
      for (const id of fromDataset ? ids : [null]) {
        if (cancelled.current) break;
        const requestId = crypto.randomUUID(); active.current = requestId;
        const key = id === null ? file!.name : String(id);
        try {
          const preview = id === null ? await predictCaptioner(file!, requestId, requestOptions)
            : await predictCaptionerItem(Number(datasetId), id, requestId, requestOptions);
          setResults(old => [...old, {key, preview}]);
        } catch (e) {setResults(old => [...old, {key, error: message(e)}]);}
        finally {active.current = null;}
      }
    } finally {setBusy(false);}
  };
  const cancel = async () => {
    cancelled.current = true;
    if (active.current) {
      try {await cancelCaptionerRequest(active.current);} catch (e) {setError(message(e));}
    }
  };
  const save = async (index: number) => {
    const preview = results[index].preview;
    if (!preview) return;
    setBusy(true); setError("");
    try {
      const saved = await saveCaptionerPreview(preview, captionType);
      setResults(old => old.map((row, i) => ({...row, saved: i === index || row.saved,
        preview: row.preview && row.preview.dataset_id === preview.dataset_id && row.preview.dataset_revision === preview.dataset_revision
          ? {...row.preview, dataset_revision: saved.dataset_revision} : row.preview})));
    } catch (e) {setError(message(e));}
    finally {setBusy(false);}
  };

  const input = "rounded border border-gray-700 bg-gray-900 p-2 text-sm";
  return <div className="h-full overflow-y-auto p-4 space-y-4 text-gray-200">
    <h2 className="text-lg">Compact captioner</h2>
    <p className="text-sm text-gray-400">左のPyTorch SigLIP2 taggerを共有します。推論deviceはtaggerと一致させてください。GPU実行は他ワークツリーの学習と調整してください。</p>
    {error && <p role="alert" className="text-red-400">{error}</p>}
    <div className="flex flex-wrap gap-2">
      <input aria-label="Caption bundle directory" className={`${input} flex-1`} value={path} disabled={busy} onChange={e => setPath(e.target.value)} placeholder="captioner bundle directory" />
      <button className={input} disabled={busy || !path.trim()} onClick={() => void modelAction(true)}>ロード</button>
      <button className={input} disabled={busy || !status.loaded} onClick={() => void modelAction(false)}>解放</button>
    </div>
    <p className="text-sm">{status.loaded ? `ロード済み · ${status.language} · 上限 ${status.max_new_tokens} tokens` : "captioner未ロード"}</p>
    {options && <div className="flex flex-wrap items-end gap-3">
      <label>Tag source <select className={`${input} block`} disabled={busy} value={options.tag_source} onChange={e => setOptions({...options, tag_source: e.target.value as CaptionerOptions["tag_source"]})}>
        <option value="predicted">predicted</option><option value="provided">provided</option><option value="merged">merged</option>
      </select></label>
      <label>Device <select className={`${input} block`} disabled={busy} value={options.device} onChange={e => setOptions({...options, device: e.target.value as CaptionerOptions["device"]})}>
        <option value="cpu">cpu</option><option value="cuda:0">cuda:0</option>
      </select></label>
      <label>Output tokens <input className={`${input} block w-28`} type="number" min={1} max={status.max_new_tokens} disabled={busy} value={options.max_new_tokens} onChange={e => setOptions({...options, max_new_tokens: Number(e.target.value)})} /></label>
    </div>}
    {options?.tag_source !== "predicted" && <label className="block">入力タグ（1行に1タグ）<textarea className={`${input} block w-full`} value={tags} disabled={busy} onChange={e => setTags(e.target.value)} /></label>}
    <div className="flex flex-wrap gap-2">
      <input aria-label="Caption image" type="file" accept="image/*" disabled={busy} onChange={e => setFile(e.target.files?.[0] ?? null)} />
      <button className={input} disabled={busy || !status.loaded || !options || !file} onClick={() => void run(false)}>画像をpreview</button>
    </div>
    <div className="flex flex-wrap gap-2">
      <select aria-label="Caption dataset" className={input} disabled={busy} value={datasetId} onChange={e => setDatasetId(e.target.value)}>
        <option value="">datasetを選択</option>{datasets.map(d => <option key={d.id} value={d.id}>{d.name} (#{d.id})</option>)}
      </select>
      <input aria-label="Caption item ids" className={`${input} flex-1`} disabled={busy} value={itemIds} onChange={e => setItemIds(e.target.value)} placeholder="item IDs（カンマ／改行区切り）" />
      <button className={input} disabled={busy || !status.loaded || !options} onClick={() => void run(true)}>datasetをpreview</button>
      <button className={input} disabled={!busy || !active.current} onClick={() => void cancel()}>キャンセル</button>
    </div>
    <label className="block">保存先caption type（別sourceの既存列は上書きしません）<input className={`${input} ml-2`} value={captionType} disabled={busy} onChange={e => setCaptionType(e.target.value)} /></label>
    <p className="text-xs text-gray-500">previewは自動保存しません。自然文はJSON専用fieldへ保存し、TXT tagsを保持します。</p>
    {results.map((row, index) => <section key={row.key} className="rounded border border-gray-700 p-3 space-y-2">
      <p>{row.key} {row.saved && "· 保存済み"}</p>
      {row.error && <p className="text-red-400">{row.error}</p>}
      {row.preview && <>
        <textarea aria-label={`Caption ${row.key}`} className={`${input} w-full`} rows={3} value={row.preview.caption} disabled={busy || row.saved} onChange={e => setResults(old => old.map((r, i) => i === index && r.preview ? {...r, preview: {...r.preview, caption: e.target.value}} : r))} />
        <p className="text-xs text-gray-400">{row.preview.termination} · {row.preview.token_count} tokens · {row.preview.timings_ms.total.toFixed(0)} ms · {row.preview.warnings.join(" / ")}</p>
        <details><summary className="text-xs">使用タグ（未校正score）</summary><pre className="overflow-x-auto text-xs">{JSON.stringify(row.preview.used_tags, null, 2)}</pre></details>
        <button className={input} disabled={busy || row.saved || !row.preview.caption.trim() || row.preview.dataset_id === undefined || !captionType.trim()} onClick={() => void save(index)}>確認済みcaptionを保存</button>
      </>}
    </section>)}
    <CaptionerTrainingPanel onSelectBundle={setPath} />
  </div>;
}
