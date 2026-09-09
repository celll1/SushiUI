"use client";

import { useEffect, useMemo, useState, type DragEvent } from "react";
import Button from "../common/Button";
import Card from "../common/Card";
import GenerationQueue from "../common/GenerationQueue";
import Input from "../common/Input";
import NumberInput from "../common/NumberInput";
import Select from "../common/Select";
import LoRASelector from "../common/LoRASelector";
import { useGenerationQueue } from "@/contexts/GenerationQueueContext";
import { useStartup } from "@/contexts/StartupContext";
import { Img2TxtParams, Img2TxtTask } from "@/utils/api";

type Img2TxtSettings = Omit<Img2TxtParams, "image">;

const FALLBACK_SETTINGS: Img2TxtSettings = {
  task: "caption",
  instruction: "",
  hint_tags: [],
  max_new_tokens: 1024,
  do_sample: false,
  temperature: 0.7,
  top_p: 0.9,
  top_k: null,
  repetition_penalty: null,
  seed: -1,
  prompt_template_version: 1,
  loras: [],
};

const TASK_OPTIONS = [
  { value: "caption", label: "Caption" },
  { value: "caption_tags", label: "Caption + tags" },
  { value: "tags", label: "Tags (secondary to Tagger)" },
  { value: "custom", label: "Custom instruction" },
];

function saveDownload(filename: string, body: string, type: string) {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export default function Img2TxtPanel() {
  const { generationDefaults, modelInfo } = useStartup();
  const { addToQueue, currentItem, progressSnapshot, completedResults } = useGenerationQueue();
  const [settings, setSettings] = useState<Img2TxtSettings>(FALLBACK_SETTINGS);
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [hintDraft, setHintDraft] = useState("");
  const [editedText, setEditedText] = useState("");

  useEffect(() => {
    const defaults = generationDefaults?.img2txt;
    if (defaults) setSettings((previous) => ({ ...previous, ...defaults }));
  }, [generationDefaults]);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const completed = completedResults.img2txt;
  const result = completed?.kind === "text"
    ? completed
    : null;

  useEffect(() => {
    if (result) setEditedText(result.rawText);
  }, [result?.revision]);

  const ownsCurrent = currentItem?.type === "img2txt";
  const progress = ownsCurrent && progressSnapshot?.itemId === currentItem.id
    ? progressSnapshot
    : null;
  const progressPercent = progress?.totalSteps
    ? Math.min(100, (progress.step / progress.totalSteps) * 100)
    : 0;

  const hints = useMemo(
    () => hintDraft.split(/[\n,]/).map((tag) => tag.trim()).filter(Boolean),
    [hintDraft],
  );

  const selectImage = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      alert("Please select an image file.");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const clearImage = () => {
    if (preview) URL.revokeObjectURL(preview);
    setImage(null);
    setPreview(null);
  };

  const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    selectImage(event.dataTransfer.files?.[0]);
  };

  const enqueue = (
    params: Img2TxtParams,
    identity: { type: string; source: string } | null | undefined = modelInfo,
  ) => {
    addToQueue({
      panel: "img2txt",
      type: "img2txt",
      params,
      prompt: params.instruction.trim() || params.task,
      modelIdentity: identity ? { type: identity.type, source: identity.source } : undefined,
    });
  };

  const handleEnqueue = () => {
    if (!image) {
      alert("Please select one reference image.");
      return;
    }
    if (settings.task === "custom" && !settings.instruction.trim()) {
      alert("Custom instruction is required.");
      return;
    }
    enqueue({ ...settings, image, hint_tags: hints });
  };

  const requeueResult = () => {
    if (!result) return;
    enqueue(result.params, result.model.source
      ? { type: result.model.type, source: result.model.source }
      : undefined);
  };

  return (
    <div className="grid min-h-full grid-cols-1 gap-3 p-3 xl:grid-cols-[minmax(0,1fr)_minmax(320px,0.75fr)_280px]">
      <div className="space-y-3">
        <Card title="Reference image">
          <label
            className={`block cursor-pointer rounded-md border-2 border-dashed p-4 text-center text-sm text-gray-400 transition-colors hover:border-violet-500 ${
              isDragging ? "border-violet-500 bg-violet-500/10" : "border-gray-700"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {preview ? (
              <img src={preview} alt="img2txt reference" className="mx-auto max-h-72 rounded object-contain" />
            ) : (
              <span className="block py-12">{isDragging ? "Drop image here" : "Drop image here or click to upload"}</span>
            )}
            <input
              className="hidden"
              type="file"
              accept="image/*"
              onChange={(event) => {
                selectImage(event.target.files?.[0]);
                event.currentTarget.value = "";
              }}
            />
          </label>
          {image && (
            <div className="flex items-center justify-between gap-2">
              <p className="min-w-0 flex-1 truncate text-xs text-gray-500">{image.name}</p>
              <Button size="sm" variant="secondary" onClick={clearImage}>Clear</Button>
            </div>
          )}
        </Card>

        <Card title="Task and instruction">
          <Select
            label="Task"
            options={TASK_OPTIONS}
            value={settings.task}
            onChange={(event) => setSettings({ ...settings, task: event.target.value as Img2TxtTask })}
          />
          <label className="block text-xs font-medium text-gray-400">Instruction</label>
          <textarea
            className="min-h-28 w-full resize-y rounded-md border border-gray-700 bg-gray-800 p-2.5 text-sm text-gray-100 focus:border-violet-500 focus:outline-none"
            value={settings.instruction}
            maxLength={16384}
            placeholder={settings.task === "custom" ? "Required" : "Leave empty to use the versioned backend preset"}
            onChange={(event) => setSettings({ ...settings, instruction: event.target.value })}
          />
          <label className="block text-xs font-medium text-gray-400">Hint tags (optional, comma or newline separated)</label>
          <textarea
            className="min-h-16 w-full resize-y rounded-md border border-gray-700 bg-gray-800 p-2.5 text-sm text-gray-100 focus:border-violet-500 focus:outline-none"
            value={hintDraft}
            onChange={(event) => setHintDraft(event.target.value)}
          />
          <p className="text-xs text-gray-500">Hints are input context only and are not copied into the result.</p>
        </Card>
      </div>

      <div className="space-y-3">
        <Card title="Text generation">
          <label className="flex items-center gap-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={settings.do_sample}
              onChange={(event) => setSettings({ ...settings, do_sample: event.target.checked })}
            />
            Sampling
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-xs text-gray-400">Maximum new tokens
              <NumberInput label="Maximum new tokens" value={settings.max_new_tokens} min={1} max={8192} parse="int" onCommit={(value) => setSettings({ ...settings, max_new_tokens: value })} />
            </label>
            <Input type="number" label="Seed (-1 random)" min={-1} max={2147483647} value={settings.seed} onChange={(event) => setSettings({ ...settings, seed: Number(event.target.value) })} />
            <Input type="number" label="Temperature" min={0.01} max={5} step="any" disabled={!settings.do_sample} value={settings.temperature} onChange={(event) => setSettings({ ...settings, temperature: Number(event.target.value) })} />
            <Input type="number" label="Top-p" min={0.01} max={1} step="any" disabled={!settings.do_sample} value={settings.top_p} onChange={(event) => setSettings({ ...settings, top_p: Number(event.target.value) })} />
            <Input type="number" label="Top-k (blank = unset)" min={1} disabled={!settings.do_sample} value={settings.top_k ?? ""} onChange={(event) => setSettings({ ...settings, top_k: event.target.value === "" ? null : Number(event.target.value) })} />
            <Input type="number" label="Repetition penalty" min={0.01} max={10} step="any" value={settings.repetition_penalty ?? ""} onChange={(event) => setSettings({ ...settings, repetition_penalty: event.target.value === "" ? null : Number(event.target.value) })} />
          </div>
          <Button className="w-full" disabled={!image || ownsCurrent} onClick={handleEnqueue}>Add img2txt to queue</Button>
          {progress && (
            <div className="space-y-1">
              <div className="h-1.5 overflow-hidden rounded bg-gray-800"><div className="h-full bg-violet-500" style={{ width: `${progressPercent}%` }} /></div>
              <p className="text-xs text-gray-400">{progress.message} · {progress.step}/{progress.totalSteps}</p>
            </div>
          )}
        </Card>

        <LoRASelector
          value={settings.loras}
          onChange={(loras) => setSettings({ ...settings, loras })}
          disabled={ownsCurrent}
          storageKey="img2txt_lora_collapsed"
          simpleMode
          loadedArch="sensenova"
        />

        <Card title="Result">
          {result ? (
            <div className="space-y-2">
              {result.parseWarning && <p className="rounded border border-amber-700 bg-amber-950/40 p-2 text-xs text-amber-300">{result.parseWarning}</p>}
              <textarea
                aria-label="Generated text"
                className="min-h-52 w-full resize-y rounded-md border border-gray-700 bg-gray-950 p-2.5 text-sm text-gray-100"
                value={editedText}
                onChange={(event) => setEditedText(event.target.value)}
              />
              {result.structured?.caption && (
                <div className="rounded border border-gray-800 bg-gray-900 p-2.5">
                  <p className="mb-1 text-xs font-medium text-gray-400">Parsed caption</p>
                  <p className="whitespace-pre-wrap text-sm text-gray-200">{result.structured.caption}</p>
                </div>
              )}
              {result.structured?.tags && <p className="text-xs text-gray-400">Tags: {result.structured.tags.join(", ")}</p>}
              <p className="text-xs text-gray-500">Seed {result.seed} · {(result.timing.generation_seconds).toFixed(2)}s · template v{result.promptTemplateVersion}</p>
              {result.warnings?.map((warning, index) => <p key={index} className="text-xs text-amber-300">{warning}</p>)}
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="secondary" onClick={() => navigator.clipboard.writeText(editedText)}>Copy</Button>
                <Button size="sm" variant="secondary" onClick={() => saveDownload("sensenova-img2txt.txt", editedText, "text/plain;charset=utf-8")}>Download .txt</Button>
                <Button size="sm" variant="secondary" onClick={() => saveDownload("sensenova-img2txt.json", JSON.stringify({ structured: result.structured, raw_text: editedText }, null, 2), "application/json")}>Download .json</Button>
                <Button size="sm" onClick={requeueResult}>Requeue frozen request</Button>
              </div>
              <details className="text-xs text-gray-500"><summary>Effective instruction</summary><pre className="mt-1 whitespace-pre-wrap">{result.effectiveInstruction}</pre></details>
            </div>
          ) : <p className="py-10 text-center text-sm text-gray-500">No text result in this session.</p>}
        </Card>
      </div>

      <div className="min-h-80 overflow-hidden rounded-md border border-gray-800"><GenerationQueue /></div>
    </div>
  );
}
