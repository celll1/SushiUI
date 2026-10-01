"use client";

import { useEffect, useRef, useState } from "react";
import {
  CaptionerDatasetConfig, CaptionerRunDefaults, CaptionerTaggerConfig, CaptionerTrainingConfig,
  CaptionerTrainingEvent, CaptionerTrainingRun, CaptionTypeInfo, Dataset,
  CaptionerOptions, CaptionerPreview, cancelCaptionerRequest, getCaptionerDefaults, sampleCaptionerRun,
  createCaptionerRun, exportCaptionerRun, getCaptionerRun, getCaptionerRunDefaults,
  getCaptionerRunEvents, getCaptionerTrainingDefaults, getDatasetCaptionTypes,
  listCaptionerRuns, listDatasets, startCaptionerRun, stopCaptionerRun,
} from "@/utils/api";

const input = "rounded border border-gray-700 bg-gray-900 p-2 text-sm";
const message = (error: unknown) => {
  const value = error as {response?: {data?: {message?: unknown; detail?: unknown}}; message?: string};
  const detail = value.response?.data?.message || value.response?.data?.detail || value.message || "処理に失敗しました";
  return typeof detail === "string" ? detail : JSON.stringify(detail) ?? "処理に失敗しました";
};
const activeStatus = (status: string) => ["starting", "running", "stopping"].includes(status);

export default function CaptionerTrainingPanel({onSelectBundle, onSample}: {
  onSelectBundle: (path: string) => void; onSample: (preview: CaptionerPreview) => void;
}) {
  const [defaults, setDefaults] = useState<CaptionerRunDefaults | null>(null);
  const [tagger, setTagger] = useState<CaptionerTaggerConfig | null>(null);
  const [configText, setConfigText] = useState("");
  const [runName, setRunName] = useState("");
  const [output, setOutput] = useState("");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [columns, setColumns] = useState<CaptionTypeInfo[]>([]);
  const [tagTypes, setTagTypes] = useState<string[]>([]);
  const [targetTypes, setTargetTypes] = useState<string[]>([]);
  const [datasetConfigs, setDatasetConfigs] = useState<CaptionerDatasetConfig[]>([]);
  const [filterError, setFilterError] = useState("");
  const [runs, setRuns] = useState<CaptionerTrainingRun[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [events, setEvents] = useState<CaptionerTrainingEvent[]>([]);
  const [checkpoint, setCheckpoint] = useState("");
  const [exported, setExported] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sampleOptions, setSampleOptions] = useState<CaptionerOptions | null>(null);
  const [sampleDataset, setSampleDataset] = useState("");
  const [sampleItem, setSampleItem] = useState("");
  const [sampleTags, setSampleTags] = useState("");
  const sampleRequest = useRef<string | null>(null);
  const busyRef = useRef(false);
  const epoch = useRef(0);
  const selected = runs.find(run => run.run_id === selectedId);
  const previewCheckpoint = selected && (checkpoint
    ? selected.checkpoints.find(path => path.split(/[\\/]/).pop() === checkpoint)
    : selected.checkpoints[selected.checkpoints.length - 1]);

  useEffect(() => {
    let mounted = true;
    Promise.all([getCaptionerRunDefaults(), getCaptionerTrainingDefaults(), listCaptionerRuns(), listDatasets(), getCaptionerDefaults()])
      .then(([d, cfg, data, ds, inference]) => {
        if (!mounted) return;
        setDefaults(d); setRunName(d.run_name); setOutput(d.output_dir ?? "");
        setTagger({checkpoint_path: "", vocab_path: "", vision_encoder_path: d.vision_encoder_path,
          lora_rank: d.lora_rank, lora_alpha: d.lora_alpha});
        setConfigText(JSON.stringify(cfg, null, 2)); setRuns(data.runs); setDatasets(ds.datasets);
        setCheckpoint(d.checkpoint);
        setSampleOptions({tag_source: inference.tag_source, tags: inference.tags,
          max_new_tokens: inference.max_new_tokens, device: inference.device});
        setSampleTags(inference.tags.join("\n"));
      }).catch(e => {if (mounted) setError(message(e));});
    return () => {
      mounted = false;
      if (sampleRequest.current) void cancelCaptionerRequest(sampleRequest.current).catch(() => {});
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    setColumns([]); setTagTypes([]); setTargetTypes([]);
    if (datasetId) getDatasetCaptionTypes(Number(datasetId)).then(data => {
      if (mounted) setColumns(data.caption_types.filter(c => c.field_category === "training"));
    }).catch(e => {if (mounted) setError(message(e));});
    return () => {mounted = false;};
  }, [datasetId]);

  useEffect(() => {
    setEvents([]); setExported("");
    if (defaults) setCheckpoint(defaults.checkpoint);
    if (!selectedId || !defaults) return;
    let mounted = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let cursor = defaults.after_id;
    const poll = async () => {
      const version = epoch.current;
      try {
        if (busyRef.current) return;
        const [run, data] = await Promise.all([getCaptionerRun(selectedId),
          getCaptionerRunEvents(selectedId, cursor, defaults.event_limit)]);
        if (!mounted || busyRef.current || epoch.current !== version) return;
        setRuns(old => old.map(row => row.run_id === run.run_id ? run : row));
        if (data.events.length) {
          cursor = data.events[data.events.length - 1].id;
          setEvents(old => [...old, ...data.events].slice(-200));
        }
      } catch (e) {
        if (mounted && !busyRef.current && epoch.current === version) setError(message(e));
      }
      finally {if (mounted) timer = setTimeout(() => void poll(), 2000);}
    };
    void poll();
    return () => {mounted = false; if (timer !== undefined) clearTimeout(timer);};
  }, [selectedId, defaults]);

  const updateRun = (run: CaptionerTrainingRun) => setRuns(old =>
    old.some(row => row.run_id === run.run_id) ? old.map(row => row.run_id === run.run_id ? run : row) : [run, ...old]);
  const action = async (work: () => Promise<void>) => {
    if (busyRef.current) return;
    busyRef.current = true; epoch.current += 1;
    setBusy(true); setError("");
    try {await work();} catch (e) {setError(message(e));}
    finally {epoch.current += 1; busyRef.current = false; setBusy(false);}
  };
  const create = async () => {
    if (!tagger) return;
    const config = JSON.parse(configText) as CaptionerTrainingConfig;
    const run = await createCaptionerRun({run_name: runName, output_dir: output.trim() || null,
      tagger, dataset_configs: datasetConfigs, config});
    updateRun(run); setSelectedId(run.run_id);
  };
  const sample = async () => {
    if (!selected || !sampleOptions) return;
    const item = Number(sampleItem), dataset = Number(sampleDataset);
    if (!Number.isSafeInteger(item) || item < 1 || !dataset) throw new Error("正しいdatasetとitem IDを選択してください");
    const id = crypto.randomUUID();
    sampleRequest.current = id;
    try {
      const preview = await sampleCaptionerRun(selected.run_id, checkpoint, dataset, item, id,
        {...sampleOptions, max_new_tokens: Math.min(sampleOptions.max_new_tokens, selected.config.max_length),
          tags: sampleOptions.tag_source === "predicted" ? [] : sampleTags.split("\n").map(t => t.trim()).filter(Boolean)});
      onSample(preview);
    } finally {sampleRequest.current = null;}
  };
  const toggle = (values: string[], value: string) => values.includes(value)
    ? values.filter(v => v !== value) : [...values, value];
  const addDataset = () => {
    if (!datasetId || !targetTypes.length) return;
    const item: CaptionerDatasetConfig = {dataset_id: Number(datasetId), input_tag_caption_types: tagTypes,
      target_caption_types: targetTypes, filters: {}};
    setDatasetConfigs(old => [...old.filter(row => row.dataset_id !== item.dataset_id), item]);
  };
  const setBasic = (key: string, value: string | number) => {
    try {setConfigText(JSON.stringify({...JSON.parse(configText), [key]: value}, null, 2)); setError("");}
    catch {setError("詳細設定JSONを修正してください");}
  };
  let draft: Partial<CaptionerTrainingConfig> = {};
  try {
    const parsed = JSON.parse(configText);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) draft = parsed;
  } catch {}

  return <section className="rounded border border-gray-700 p-3 space-y-3">
    <h3 className="text-base">Captioner学習</h3>
    <p className="text-xs text-gray-400">private SigLIP2 encoderは固定し、caption decoderのみ学習します。GPUを選ぶ前に他ワークツリーと調整してください。画面を閉じてもrunは継続します。</p>
    {error && <p role="alert" className="text-red-400">{error}</p>}
    {filterError && <p role="alert" className="text-red-400">{filterError}</p>}
    <details>
      <summary>新しいrunを作成</summary>
      <div className="space-y-3 pt-3">
        <div className="flex flex-wrap gap-2">
          <label>Run名 <input className={input} disabled={busy || !defaults} value={runName} onChange={e => setRunName(e.target.value)} /></label>
          <label className="flex-1">新しいoutput directory（空欄は自動）<input className={`${input} w-full`} disabled={busy || !defaults} value={output} onChange={e => setOutput(e.target.value)} /></label>
        </div>
        {tagger && <>
          {(["checkpoint_path", "vocab_path", "vision_encoder_path"] as const).map(key => <label className="block" key={key}>{key}{key === "vision_encoder_path" && "（LoRA taggerの場合に必要）"}
            <input className={`${input} block w-full`} disabled={busy} value={tagger[key]} onChange={e => setTagger({...tagger, [key]: e.target.value})} /></label>)}
          <div className="flex gap-2">{(["lora_rank", "lora_alpha"] as const).map(key => <label key={key}>{key}<input type="number" className={`${input} block w-28`} disabled={busy} value={tagger[key]} onChange={e => setTagger({...tagger, [key]: Number(e.target.value)})} /></label>)}</div>
        </>}
        <div className="rounded border border-gray-800 p-2 space-y-2">
          <select aria-label="Training caption dataset" className={input} disabled={busy} value={datasetId} onChange={e => setDatasetId(e.target.value)}>
            <option value="">datasetを選択</option>{datasets.map(ds => <option key={ds.id} value={ds.id}>{ds.name} (#{ds.id})</option>)}
          </select>
          <p className="text-xs">入力tags列（省略可）</p>
          <div className="flex flex-wrap gap-3">{columns.filter(c => c.is_tags_format).map(c => <label key={c.caption_type}><input type="checkbox" disabled={busy} checked={tagTypes.includes(c.caption_type)} onChange={() => setTagTypes(toggle(tagTypes, c.caption_type))} /> {c.caption_type}</label>)}</div>
          <p className="text-xs">正解caption列（自然文のみ・必須）</p>
          <div className="flex flex-wrap gap-3">{columns.filter(c => !c.is_tags_format).map(c => <label key={c.caption_type}><input type="checkbox" disabled={busy} checked={targetTypes.includes(c.caption_type)} onChange={() => setTargetTypes(toggle(targetTypes, c.caption_type))} /> {c.caption_type}</label>)}</div>
          <button className={input} disabled={busy || !datasetId || !targetTypes.length} onClick={addDataset}>datasetを追加／更新</button>
          {datasetConfigs.map(row => <p key={row.dataset_id} className="text-xs">#{row.dataset_id} · tags: {row.input_tag_caption_types.join(", ") || "なし"} → caption: {row.target_caption_types.join(", ")}
            <button className={`${input} ml-2`} disabled={busy} onClick={() => setDatasetConfigs(old => old.filter(c => c.dataset_id !== row.dataset_id))}>除外</button></p>)}
          <details><summary className="text-xs">dataset filters（JSON）</summary>{datasetConfigs.map(row => <label className="block text-xs" key={row.dataset_id}>#{row.dataset_id}
            <input className={`${input} block w-full`} disabled={busy} defaultValue={JSON.stringify(row.filters)} key={JSON.stringify(row)} onBlur={e => {
              try {
                const filters = JSON.parse(e.target.value);
                if (!filters || typeof filters !== "object" || Array.isArray(filters)) throw new Error();
                setDatasetConfigs(old => old.map(c => c.dataset_id === row.dataset_id ? {...c, filters} : c));
                setFilterError("");
              } catch {setFilterError("filter JSONをobjectに修正してください");}
            }} /></label>)}</details>
        </div>
        <div className="flex flex-wrap gap-2">
          <label>Device <select className={`${input} block`} disabled={busy || !configText} value={draft.device ?? ""} onChange={e => setBasic("device", e.target.value)}><option value="cpu">cpu</option><option value="cuda:0">cuda:0</option></select></label>
          {(["epochs", "batch_size", "learning_rate", "max_num_patches"] as const).map(key => <label key={key}>{key}<input className={`${input} block w-32`} type="number" step="any" disabled={busy || !configText} value={draft[key] ?? ""} onChange={e => setBasic(key, Number(e.target.value))} /></label>)}
        </div>
        <details><summary>詳細学習設定（API defaultsから取得）</summary><textarea aria-label="Caption training configuration JSON" className={`${input} w-full font-mono`} rows={16} disabled={busy} value={configText} onChange={e => setConfigText(e.target.value)} /></details>
        <button className={input} disabled={busy || !!filterError || !defaults || !tagger?.checkpoint_path.trim() || !tagger?.vocab_path.trim() || !datasetConfigs.length || !runName.trim()} onClick={() => void action(create)}>runを作成（まだ学習しません）</button>
      </div>
    </details>
    <div className="flex flex-wrap gap-2">
      <select aria-label="Caption training run" className={`${input} flex-1`} disabled={busy} value={selectedId} onChange={e => setSelectedId(e.target.value)}>
        <option value="">runを選択</option>{runs.map(run => <option key={run.run_id} value={run.run_id}>{run.run_name} · {run.status} · {run.run_id}</option>)}
      </select>
      <button className={input} disabled={busy} onClick={() => void action(async () => setRuns((await listCaptionerRuns()).runs))}>一覧更新</button>
    </div>
    {selected && <div className="space-y-2">
      <p className="text-sm">{selected.status} · resume #{selected.resume_seq} · aliases {selected.alias_count}</p>
      <p className="break-all text-xs text-gray-400">{selected.output_dir}</p>
      {selected.error && <p className="text-red-400">{selected.error}</p>}
      <div className="flex flex-wrap gap-2">
        <select aria-label="Caption checkpoint" className={`${input} flex-1`} disabled={busy || activeStatus(selected.status)} value={checkpoint} onChange={e => setCheckpoint(e.target.value)}>
          <option value={defaults?.checkpoint ?? ""}>latest</option>{selected.checkpoints.map(path => {const name = path.split(/[\\/]/).pop()!; return <option key={path} value={name}>{name}</option>;})}
        </select>
        <button className={input} disabled={busy || activeStatus(selected.status) || selected.checkpoints.length > 0} onClick={() => void action(async () => updateRun(await startCaptionerRun(selected.run_id, false, defaults!.checkpoint)))}>開始</button>
        <button className={input} disabled={busy || activeStatus(selected.status) || !selected.checkpoints.length} onClick={() => void action(async () => updateRun(await startCaptionerRun(selected.run_id, true, checkpoint)))}>再開</button>
        <button className={input} disabled={busy || !activeStatus(selected.status) || selected.status === "stopping"} onClick={() => void action(async () => updateRun(await stopCaptionerRun(selected.run_id)))}>停止してcheckpoint保存</button>
        <button className={input} disabled={busy || !selected.checkpoints.length} onClick={() => void action(async () => setExported((await exportCaptionerRun(selected.run_id, checkpoint)).path))}>推論bundleをexport</button>
        <button className={input} disabled={!previewCheckpoint || busy} onClick={() => {if (previewCheckpoint) onSelectBundle(previewCheckpoint);}}>checkpointパスを表示</button>
      </div>
      {exported && <p className="text-xs break-all">Export: {exported} <button className={input} onClick={() => onSelectBundle(exported)}>bundleパスを表示</button></p>}
      {sampleOptions && <div className="rounded border border-gray-800 p-2 space-y-2">
        <p className="text-xs">選択checkpointからsampleを生成します。Tagger画面で同じPyTorch taggerをロードし、deviceを一致させてください。preview用captionerは差し替えません。</p>
        <div className="flex flex-wrap gap-2">
          <select aria-label="Sample dataset" className={input} disabled={busy} value={sampleDataset} onChange={e => setSampleDataset(e.target.value)}>
            <option value="">datasetを選択</option>{datasets.map(ds => <option key={ds.id} value={ds.id}>{ds.name} (#{ds.id})</option>)}
          </select>
          <input aria-label="Sample item id" className={`${input} w-28`} disabled={busy} value={sampleItem} onChange={e => setSampleItem(e.target.value)} placeholder="item ID" />
          <select aria-label="Sample tag source" className={input} disabled={busy} value={sampleOptions.tag_source} onChange={e => setSampleOptions({...sampleOptions, tag_source: e.target.value as CaptionerOptions["tag_source"]})}>
            <option value="predicted">predicted</option><option value="provided">provided</option><option value="merged">merged</option>
          </select>
          <select aria-label="Sample device" className={input} disabled={busy} value={sampleOptions.device} onChange={e => setSampleOptions({...sampleOptions, device: e.target.value as CaptionerOptions["device"]})}>
            <option value="cpu">cpu</option><option value="cuda:0">cuda:0</option>
          </select>
          <label>Tokens <input className={`${input} w-24`} type="number" min={1} max={selected.config.max_length} disabled={busy} value={Math.min(sampleOptions.max_new_tokens, selected.config.max_length)} onChange={e => setSampleOptions({...sampleOptions, max_new_tokens: Number(e.target.value)})} /></label>
          <button className={input} disabled={busy || !selected.checkpoints.length || !sampleDataset || !sampleItem} onClick={() => void action(sample)}>sampleを生成</button>
          <button className={input} disabled={!sampleRequest.current} onClick={() => {if (sampleRequest.current) void cancelCaptionerRequest(sampleRequest.current).catch(e => setError(message(e)));}}>sampleをキャンセル</button>
        </div>
        {sampleOptions.tag_source !== "predicted" && <textarea aria-label="Sample tags one per line" className={`${input} w-full`} disabled={busy} value={sampleTags} onChange={e => setSampleTags(e.target.value)} placeholder="入力タグ（1行に1タグ）" />}
        <p className="text-xs text-gray-400">生成結果はこの画面のsample欄に表示し、run eventsにも保存します。sidecarへの保存は確認後に行ってください。</p>
      </div>}
      <details><summary>固定された設定</summary><pre className="overflow-x-auto text-xs">{JSON.stringify({tagger: selected.tagger, dataset_configs: selected.dataset_configs, config: selected.config}, null, 2)}</pre></details>
      <p className="text-xs">最新進捗（2秒ごとに取得）</p>
      <pre className="overflow-x-auto text-xs">{JSON.stringify(selected.progress, null, 2)}</pre>
      <details><summary>保存されたevents（最新200件）</summary><pre className="max-h-80 overflow-auto text-xs">{events.map(event => `${event.id} · resume ${event.resume_seq} · ${event.timestamp} · ${JSON.stringify(event.payload)}`).join("\n")}</pre></details>
    </div>}
  </section>;
}
