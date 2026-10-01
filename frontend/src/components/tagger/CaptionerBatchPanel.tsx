"use client";

import { useEffect, useRef, useState } from "react";
import {
  CaptionerBatch, CaptionerBatchDefaults, CaptionerBatchItem, CaptionerOptions, Dataset,
  createCaptionerBatch, getCaptionerBatch, getCaptionerBatchDefaults, getCaptionerBatchResults,
  listCaptionerBatches, saveCaptionerBatchItem, startCaptionerBatch, stopCaptionerBatch,
} from "@/utils/api";

const input = "rounded border border-gray-700 bg-gray-900 p-2 text-sm";
const message = (error: unknown) => {
  const value = error as {response?: {data?: {message?: unknown; detail?: unknown}}; message?: string};
  const detail = value.response?.data?.message || value.response?.data?.detail || value.message || "処理に失敗しました";
  return typeof detail === "string" ? detail : JSON.stringify(detail) ?? "処理に失敗しました";
};

export default function CaptionerBatchPanel({bundlePath, options, datasets, captionType}: {
  bundlePath: string; options: CaptionerOptions; datasets: Dataset[]; captionType: string;
}) {
  const [defaults, setDefaults] = useState<CaptionerBatchDefaults | null>(null);
  const [jobs, setJobs] = useState<CaptionerBatch[]>([]);
  const [jobId, setJobId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [itemIds, setItemIds] = useState("");
  const [offset, setOffset] = useState<number | null>(null);
  const [items, setItems] = useState<CaptionerBatchItem[]>([]);
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const busyRef = useRef(false);
  const epoch = useRef(0);
  const selected = jobs.find(job => job.job_id === jobId);
  const active = selected?.status === "running" || selected?.status === "stopping";

  useEffect(() => {
    let mounted = true;
    Promise.all([getCaptionerBatchDefaults(), listCaptionerBatches()]).then(([d, data]) => {
      if (!mounted) return;
      setDefaults(d); setOffset(d.offset); setJobs(data.batches);
    }).catch(e => {if (mounted) setError(message(e));});
    return () => {mounted = false;};
  }, []);

  useEffect(() => {setDrafts({});}, [jobId]);

  useEffect(() => {
    setItems([]);
    if (!jobId || !defaults || offset === null) return;
    let mounted = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      const version = epoch.current;
      try {
        if (!busyRef.current) {
          const [job, data] = await Promise.all([getCaptionerBatch(jobId), getCaptionerBatchResults(jobId, offset, defaults.limit)]);
          if (mounted && !busyRef.current && epoch.current === version) {
            setJobs(old => old.map(row => row.job_id === job.job_id ? job : row));
            setItems(data.items);
          }
        }
      } catch (e) {if (mounted) setError(message(e));}
      finally {if (mounted) timer = setTimeout(() => void poll(), 2000);}
    };
    void poll();
    return () => {mounted = false; if (timer !== undefined) clearTimeout(timer);};
  }, [jobId, defaults, offset]);

  const update = (job: CaptionerBatch) => setJobs(old => old.some(row => row.job_id === job.job_id)
    ? old.map(row => row.job_id === job.job_id ? job : row) : [job, ...old]);
  const action = async (work: () => Promise<void>) => {
    busyRef.current = true; epoch.current += 1;
    setBusy(true); setError("");
    try {await work();} catch (e) {setError(message(e));}
    finally {epoch.current += 1; busyRef.current = false; setBusy(false);}
  };
  const create = async () => {
    if (!defaults) return;
    const ids = itemIds.split(/[\s,]+/).filter(Boolean).map(Number);
    if (!ids.length || ids.length > defaults.max_items || ids.some(id => !Number.isSafeInteger(id) || id < 1)
      || new Set(ids).size !== ids.length || !Number(datasetId)) throw new Error("datasetと重複のない正しいitem IDを選択してください");
    const job = await createCaptionerBatch({...options, bundle_path: bundlePath, dataset_id: Number(datasetId), item_ids: ids});
    update(job); setDrafts({}); setJobId(job.job_id); setOffset(defaults.offset);
  };
  const save = async (row: CaptionerBatchItem) => {
    if (!row.preview || !defaults || offset === null) return;
    await saveCaptionerBatchItem(jobId, row.item_id, row.preview,
      row.pending_save ? row.reviewed_caption! : drafts[row.item_id] ?? row.reviewed_caption ?? row.preview.caption,
      row.pending_save ? row.reviewed_caption_type! : captionType);
    const [job, data] = await Promise.all([getCaptionerBatch(jobId), getCaptionerBatchResults(jobId, offset, defaults.limit)]);
    update(job); setItems(data.items);
    setDrafts(old => {const next = {...old}; delete next[row.item_id]; return next;});
  };

  return <section className="rounded border border-gray-700 p-3 space-y-3">
    <h3>永続batch preview</h3>
    <p className="text-xs text-gray-400">上のbundle pathと推論設定を固定して作成します。画面を閉じても実行を継続し、結果はDBに保存します。sidecarは確認後の明示保存のみです。</p>
    {error && <p role="alert" className="text-red-400">{error}</p>}
    <div className="flex flex-wrap gap-2">
      <select aria-label="Persistent caption batch dataset" className={input} disabled={busy} value={datasetId} onChange={e => setDatasetId(e.target.value)}>
        <option value="">datasetを選択</option>{datasets.map(ds => <option key={ds.id} value={ds.id}>{ds.name} (#{ds.id})</option>)}
      </select>
      <input aria-label="Persistent caption batch item ids" className={`${input} flex-1`} disabled={busy} value={itemIds} onChange={e => setItemIds(e.target.value)} placeholder="item IDs（カンマ／改行区切り）" />
      <button className={input} disabled={busy || !defaults || !bundlePath.trim() || !datasetId || !itemIds.trim()} onClick={() => void action(create)}>batch作成（まだ実行しません）</button>
    </div>
    <div className="flex flex-wrap gap-2">
      <select aria-label="Persistent caption batch" className={`${input} flex-1`} disabled={busy} value={jobId} onChange={e => {setDrafts({}); setJobId(e.target.value); if (defaults) setOffset(defaults.offset);}}>
        <option value="">保存済みbatchを選択</option>{jobs.map(job => <option key={job.job_id} value={job.job_id}>#{job.dataset_id} · {job.status} · {job.job_id}</option>)}
      </select>
      <button className={input} disabled={busy} onClick={() => void action(async () => setJobs((await listCaptionerBatches()).batches))}>一覧更新</button>
    </div>
    {selected && <>
      <p className="text-sm">{selected.status} · 完了 {selected.counts.completed ?? 0} / {selected.total} · 失敗 {selected.counts.failed ?? 0}</p>
      {selected.error && <p className="text-red-400">{selected.error_code}: {selected.error}</p>}
      <details><summary className="text-xs">固定されたbundleと推論設定</summary><pre className="overflow-auto text-xs">{JSON.stringify({bundle_path: selected.bundle_path, options: selected.options}, null, 2)}</pre></details>
      <div className="flex flex-wrap gap-2">
        <button className={input} disabled={busy || active || !defaults || !(selected.counts.pending ?? 0)} onClick={() => void action(async () => update(await startCaptionerBatch(jobId, defaults!.retry_failed)))}>未完了分を開始／再開</button>
        <button className={input} disabled={busy || active || !(selected.counts.failed ?? 0)} onClick={() => void action(async () => update(await startCaptionerBatch(jobId, true)))}>失敗itemも再試行</button>
        <button className={input} disabled={busy || !active || selected.status === "stopping"} onClick={() => void action(async () => update(await stopCaptionerBatch(jobId)))}>停止</button>
      </div>
      <p className="text-xs text-gray-400">編集途中の本文はこの画面内だけに保持します。保存成功した本文とreceiptはDBに残ります。保存先: {captionType || "未選択"}。batchの停止／完了後に保存できます。</p>
      {items.map(row => <div key={row.item_id} className="rounded border border-gray-800 p-2 space-y-2">
        <p className="text-xs">Item #{row.item_id} · {row.status} · attempts {row.attempts} {row.saved && `· 保存済み (${row.saved.caption_type})`}</p>
        {row.error && <p className="text-red-400 text-xs">{row.error_code}: {row.error}</p>}
        {row.preview && <>
          <textarea aria-label={`Batch caption ${row.item_id}`} className={`${input} w-full`} rows={3} disabled={busy || active || !!row.saved || row.pending_save} value={row.pending_save ? row.reviewed_caption ?? row.preview.caption : drafts[row.item_id] ?? row.reviewed_caption ?? row.preview.caption} onChange={e => setDrafts(old => ({...old, [row.item_id]: e.target.value}))} />
          <p className="text-xs text-gray-400">{row.preview.termination} · {row.preview.token_count} tokens · {row.preview.timings_ms.total.toFixed(0)} ms · {row.preview.warnings.join(" / ")}</p>
          <details><summary className="text-xs">使用タグ</summary><pre className="overflow-auto text-xs">{JSON.stringify(row.preview.used_tags, null, 2)}</pre></details>
          {row.pending_save && <p className="text-xs text-amber-400">前の保存結果を確認中です。同じ本文・保存先 ({row.reviewed_caption_type}) で再試行してください。</p>}
          <button className={input} disabled={busy || active || !!row.saved || !captionType.trim() || !(drafts[row.item_id] ?? row.reviewed_caption ?? row.preview.caption).trim()} onClick={() => void action(() => save(row))}>{row.pending_save ? "保存結果を再確認" : "確認済みcaptionを保存"}</button>
        </>}
      </div>)}
      {defaults && offset !== null && <div className="flex gap-2 items-center text-xs">
        <button className={input} disabled={busy || offset <= defaults.offset} onClick={() => setOffset(Math.max(defaults.offset, offset - defaults.limit))}>前のpage</button>
        <span>{offset + 1}–{Math.min(offset + defaults.limit, selected.total)} / {selected.total}</span>
        <button className={input} disabled={busy || offset + defaults.limit >= selected.total} onClick={() => setOffset(offset + defaults.limit)}>次のpage</button>
      </div>}
    </>}
  </section>;
}
