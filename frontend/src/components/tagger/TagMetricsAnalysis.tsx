"use client";

import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { TagMetricsData } from "@/utils/api";

// ── Category colors (matches TaggerTrainingMonitor + TagResultsChart) ─────────
const CATEGORY_COLORS: Record<string, string> = {
  General:   "#16a34a",
  Character: "#3b82f6",
  Copyright: "#a855f7",
  Meta:      "#9ca3af",
  Quality:   "#eab308",
  Rating:    "#f97316",
  Artist:    "#ec4899",
  Unknown:   "#4b5563",
};

const ALL_CATEGORIES = ["General", "Character", "Copyright", "Meta", "Quality", "Rating", "Artist", "Unknown"];

// ── Glob-style tag query → RegExp ─────────────────────────────────────────────
function buildTagRegex(query: string): RegExp | null {
  const q = query.trim();
  if (!q) return null;
  if (q.includes("*") || q.includes("?")) {
    const escaped = q.replace(/[.+^${}()|[\]\\]/g, "\\$&")
                     .replace(/\*/g, ".*")
                     .replace(/\?/g, ".");
    return new RegExp(escaped, "i");
  }
  return new RegExp(q.replace(/[.+^${}()|[\]\\]/g, "\\$&"), "i");
}

// ── Null-safe formatter helpers ────────────────────────────────────────────────
const fmt = (v: number | null, digits = 3) =>
  v === null ? "—" : v.toFixed(digits);
const fmtPct = (v: number | null, digits = 1) =>
  v === null ? "—" : (v * 100).toFixed(digits) + "%";
const fmtInt = (v: number | null) =>
  v === null ? "—" : Math.round(v).toLocaleString();

// ── Sort key extractor ─────────────────────────────────────────────────────────
type SortKey = "tag" | "category" | "n_pos" | "global_freq" | "hard_rate" | "fp_rate_50" | "fn_rate_50" | "best_f1" | "best_thr";

function getSortValue(data: TagMetricsData, idx: number, key: SortKey): number | string {
  switch (key) {
    case "tag":       return data.tag_names[idx] ?? "";
    case "category":  return data.categories[idx] ?? "";
    case "n_pos":     return data.n_pos[idx] ?? -1;
    case "global_freq": return data.global_freq[idx] ?? -1;
    case "hard_rate": return data.hard_rate[idx] ?? -1;
    case "fp_rate_50": return data.fp_rate_50[idx] ?? -1;
    case "fn_rate_50": return data.fn_rate_50[idx] ?? -1;
    case "best_f1":   return data.best_f1[idx] ?? -1;
    case "best_thr":  return data.best_thr[idx] ?? -1;
  }
}

// ── Props ──────────────────────────────────────────────────────────────────────
interface TagMetricsAnalysisProps {
  data: TagMetricsData | null;
  loading: boolean;
  error: string | null;
}

// ═════════════════════════════════════════════════════════════════════════════
// Main component
// ═════════════════════════════════════════════════════════════════════════════
export default function TagMetricsAnalysis({ data, loading, error }: TagMetricsAnalysisProps) {
  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
        メトリクス読み込み中…
      </div>
    );
  }
  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center text-red-400 text-sm">
        {error}
      </div>
    );
  }
  if (!data) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
        データがありません
      </div>
    );
  }

  return <AnalysisContent data={data} />;
}

// ═════════════════════════════════════════════════════════════════════════════
// Inner content (data guaranteed non-null)
// ═════════════════════════════════════════════════════════════════════════════
function AnalysisContent({ data }: { data: TagMetricsData }) {
  // ── Filter state ─────────────────────────────────────────────────────────
  const [tagQuery,        setTagQuery]        = useState("");
  const [debouncedQuery,  setDebouncedQuery]  = useState("");
  const [selCategories,   setSelCategories]   = useState<Set<string>>(new Set());
  const [minNpos,         setMinNpos]         = useState(0);
  const [hideNaN,         setHideNaN]         = useState(false);

  // View mode: table | charts
  const [viewMode, setViewMode] = useState<"table" | "charts">("table");

  // Sort state (table only)
  const [sortKey, setSortKey]   = useState<SortKey>("n_pos");
  const [sortAsc, setSortAsc]   = useState(false); // desc by default

  // Debounce tag query
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handleQueryChange = (v: string) => {
    setTagQuery(v);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => setDebouncedQuery(v), 200);
  };

  // ── Filtered + sorted indices ─────────────────────────────────────────────
  const filteredIndices = useMemo(() => {
    const re = buildTagRegex(debouncedQuery);
    const indices: number[] = [];
    for (let i = 0; i < data.n_tags; i++) {
      if (selCategories.size > 0 && !selCategories.has(data.categories[i])) continue;
      if (minNpos > 0 && (data.n_pos[i] ?? 0) < minNpos) continue;
      if (hideNaN && data.best_f1[i] === null) continue;
      if (re && !re.test(data.tag_names[i])) continue;
      indices.push(i);
    }
    return indices;
  }, [data, debouncedQuery, selCategories, minNpos, hideNaN]);

  const sortedIndices = useMemo(() => {
    if (viewMode !== "table") return filteredIndices;
    return [...filteredIndices].sort((a, b) => {
      const va = getSortValue(data, a, sortKey);
      const vb = getSortValue(data, b, sortKey);
      let cmp = 0;
      if (typeof va === "string" && typeof vb === "string") {
        cmp = va.localeCompare(vb);
      } else {
        cmp = (va as number) - (vb as number);
      }
      return sortAsc ? cmp : -cmp;
    });
  }, [filteredIndices, viewMode, sortKey, sortAsc, data]);

  const handleSort = (key: SortKey) => {
    if (key === sortKey) setSortAsc((v) => !v);
    else { setSortKey(key); setSortAsc(false); }
  };

  const toggleCategory = (cat: string) => {
    setSelCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat); else next.add(cat);
      return next;
    });
  };

  return (
    <div className="flex flex-col flex-1 min-h-0 gap-2">
      {/* ── Summary ── */}
      <div className="flex items-center gap-4 text-xs text-gray-500 flex-shrink-0">
        <span>総タグ数: <span className="text-gray-300">{data.n_tags.toLocaleString()}</span></span>
        <span>学習画像数: <span className="text-gray-300">{data.total_images.toLocaleString()}</span></span>
        <span>表示中: <span className="text-blue-300">{filteredIndices.length.toLocaleString()}</span> / {data.n_tags.toLocaleString()}</span>
      </div>

      {/* ── Filter bar ── */}
      <div className="flex flex-wrap gap-2 items-center flex-shrink-0">
        {/* Tag search */}
        <input
          type="text"
          value={tagQuery}
          onChange={(e) => handleQueryChange(e.target.value)}
          placeholder="タグ検索 (* ワイルドカード対応)"
          className="px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs focus:outline-none focus:border-blue-500 w-52"
        />

        {/* Category checkboxes */}
        <div className="flex flex-wrap gap-1">
          <button
            onClick={() => setSelCategories(new Set())}
            className={`px-2 py-0.5 text-xs rounded border transition-colors ${
              selCategories.size === 0
                ? "bg-gray-600 border-gray-500 text-white"
                : "border-gray-600 text-gray-400 hover:text-gray-200"
            }`}
          >
            すべて
          </button>
          {ALL_CATEGORIES.map((cat) => (
            <button
              key={cat}
              onClick={() => toggleCategory(cat)}
              className={`px-2 py-0.5 text-xs rounded border transition-colors ${
                selCategories.has(cat)
                  ? "border-transparent text-white"
                  : "border-gray-600 text-gray-500 hover:text-gray-300"
              }`}
              style={selCategories.has(cat) ? { backgroundColor: CATEGORY_COLORS[cat] + "40", borderColor: CATEGORY_COLORS[cat] } : {}}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* n_pos filter */}
        <select
          value={minNpos}
          onChange={(e) => setMinNpos(Number(e.target.value))}
          className="px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs text-gray-300"
        >
          <option value={0}>n_pos ≥ 0（全件）</option>
          <option value={10}>n_pos ≥ 10</option>
          <option value={50}>n_pos ≥ 50</option>
          <option value={100}>n_pos ≥ 100</option>
          <option value={500}>n_pos ≥ 500</option>
        </select>

        {/* Hide NaN toggle */}
        <label className="flex items-center gap-1 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={hideNaN}
            onChange={(e) => setHideNaN(e.target.checked)}
            className="w-3 h-3 accent-blue-500"
          />
          <span className="text-xs text-gray-400">F1なし非表示</span>
        </label>

        {/* View toggle */}
        <div className="flex rounded overflow-hidden border border-gray-600 text-xs ml-auto">
          <button
            onClick={() => setViewMode("table")}
            className={`px-3 py-1 ${viewMode === "table" ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
          >
            テーブル
          </button>
          <button
            onClick={() => setViewMode("charts")}
            className={`px-3 py-1 ${viewMode === "charts" ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
          >
            チャート
          </button>
        </div>
      </div>

      {/* ── Content ── */}
      <div className="flex-1 min-h-0">
        {viewMode === "table" ? (
          <MetricsTable data={data} indices={sortedIndices} sortKey={sortKey} sortAsc={sortAsc} onSort={handleSort} />
        ) : (
          <MetricsCharts data={data} indices={filteredIndices} />
        )}
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// Virtual-scroll table
// ═════════════════════════════════════════════════════════════════════════════
const ROW_H = 30;
const BUFFER = 40;

const COLUMNS: { key: SortKey; label: string; width: string }[] = [
  { key: "tag",        label: "タグ",      width: "min-w-[200px] flex-1" },
  { key: "n_pos",      label: "n_pos",     width: "w-20 shrink-0" },
  { key: "global_freq", label: "頻度",     width: "w-20 shrink-0" },
  { key: "hard_rate",  label: "hard_rate", width: "w-24 shrink-0" },
  { key: "fp_rate_50", label: "FP@0.5",   width: "w-20 shrink-0" },
  { key: "fn_rate_50", label: "FN@0.5",   width: "w-20 shrink-0" },
  { key: "best_f1",    label: "best_F1",  width: "w-24 shrink-0" },
  { key: "best_thr",   label: "best_thr", width: "w-20 shrink-0" },
];

interface MetricsTableProps {
  data: TagMetricsData;
  indices: number[];
  sortKey: SortKey;
  sortAsc: boolean;
  onSort: (key: SortKey) => void;
}

function MetricsTable({ data, indices, sortKey, sortAsc, onSort }: MetricsTableProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop]         = useState(0);
  const [viewportH, setViewportH]         = useState(400);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setViewportH(el.clientHeight));
    observer.observe(el);
    setViewportH(el.clientHeight);
    return () => observer.disconnect();
  }, []);

  const total      = indices.length;
  const totalH     = total * ROW_H;
  const viewRows   = Math.ceil(viewportH / ROW_H);
  const startIdx   = Math.max(0, Math.floor(scrollTop / ROW_H) - BUFFER);
  const endIdx     = Math.min(total, startIdx + viewRows + 2 * BUFFER);
  const topSpace   = startIdx * ROW_H;
  const bottomSpace = Math.max(0, totalH - endIdx * ROW_H);

  const sortIndicator = (key: SortKey) => {
    if (key !== sortKey) return <span className="text-gray-600 ml-1">↕</span>;
    return <span className="text-blue-400 ml-1">{sortAsc ? "↑" : "↓"}</span>;
  };

  return (
    <div className="flex flex-col h-full border border-gray-700 rounded overflow-hidden">
      {/* Header */}
      <div className="flex bg-gray-800 border-b border-gray-700 text-xs text-gray-400 flex-shrink-0">
        {COLUMNS.map((col) => (
          <button
            key={col.key}
            onClick={() => onSort(col.key)}
            className={`${col.width} px-2 py-2 text-left hover:text-gray-200 hover:bg-gray-700 transition-colors flex items-center`}
          >
            {col.label}{sortIndicator(col.key)}
          </button>
        ))}
      </div>

      {/* Scrollable body */}
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto"
        onScroll={(e) => setScrollTop((e.target as HTMLDivElement).scrollTop)}
      >
        <div style={{ height: totalH, position: "relative" }}>
          <div style={{ height: topSpace }} />
          {indices.slice(startIdx, endIdx).map((dataIdx) => (
            <TableRow key={dataIdx} data={data} dataIdx={dataIdx} />
          ))}
          <div style={{ height: bottomSpace }} />
        </div>
      </div>
    </div>
  );
}

function TableRow({ data, dataIdx }: { data: TagMetricsData; dataIdx: number }) {
  const cat   = data.categories[dataIdx] ?? "Unknown";
  const color = CATEGORY_COLORS[cat] ?? "#6b7280";
  const hr    = data.hard_rate[dataIdx];
  const f1    = data.best_f1[dataIdx];

  return (
    <div className="flex text-xs border-b border-gray-800 hover:bg-gray-800/50 items-center" style={{ height: ROW_H }}>
      {/* Tag + category badge */}
      <div className="min-w-[200px] flex-1 px-2 flex items-center gap-1.5 overflow-hidden">
        <span
          className="inline-block px-1 rounded text-[10px] font-medium flex-shrink-0"
          style={{ backgroundColor: color + "30", color }}
        >
          {cat[0]}
        </span>
        <span className="text-gray-200 truncate">{data.tag_names[dataIdx]}</span>
      </div>

      {/* n_pos */}
      <div className="w-20 shrink-0 px-2 text-gray-300 text-right">
        {fmtInt(data.n_pos[dataIdx])}
      </div>

      {/* global_freq */}
      <div className="w-20 shrink-0 px-2 text-gray-400 text-right">
        {fmtPct(data.global_freq[dataIdx], 3)}
      </div>

      {/* hard_rate with bar */}
      <div className="w-24 shrink-0 px-2 relative">
        {hr !== null && (
          <div
            className="absolute inset-0 opacity-20 rounded"
            style={{ width: `${Math.min(hr * 100, 100)}%`, backgroundColor: "#ef4444" }}
          />
        )}
        <span className="relative text-gray-300 text-right block">{fmtPct(hr)}</span>
      </div>

      {/* fp_rate_50 */}
      <div className="w-20 shrink-0 px-2 text-gray-400 text-right">
        {fmtPct(data.fp_rate_50[dataIdx])}
      </div>

      {/* fn_rate_50 */}
      <div className="w-20 shrink-0 px-2 text-gray-400 text-right">
        {fmtPct(data.fn_rate_50[dataIdx])}
      </div>

      {/* best_f1 with bar */}
      <div className="w-24 shrink-0 px-2 relative">
        {f1 !== null && (
          <div
            className="absolute inset-0 opacity-20 rounded"
            style={{ width: `${Math.min(f1 * 100, 100)}%`, backgroundColor: "#22c55e" }}
          />
        )}
        <span className="relative text-gray-200 text-right block">{fmt(f1)}</span>
      </div>

      {/* best_thr */}
      <div className="w-20 shrink-0 px-2 text-gray-400 text-right">
        {fmt(data.best_thr[dataIdx], 2)}
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// Charts view — 2×2 grid
// ═════════════════════════════════════════════════════════════════════════════
function MetricsCharts({ data, indices }: { data: TagMetricsData; indices: number[] }) {
  // Sub-filter for scatter: n_pos >= 20
  const scatterIndices = useMemo(
    () => indices.filter((i) => (data.n_pos[i] ?? 0) >= 20),
    [data, indices],
  );

  // Random sample if too many points
  const sampleIndices = useMemo(() => {
    if (scatterIndices.length <= 5000) return scatterIndices;
    const sampled: number[] = [];
    const step = scatterIndices.length / 5000;
    for (let j = 0; j < 5000; j++) sampled.push(scatterIndices[Math.floor(j * step)]);
    return sampled;
  }, [scatterIndices]);

  return (
    <div className="overflow-y-auto h-full">
      <div className="grid grid-cols-2 gap-4 p-2">
        <FpFnScatterCanvas data={data} indices={sampleIndices} title="FP vs FN @ thr=0.5" />
        <FreqF1ScatterCanvas data={data} indices={sampleIndices} title="頻度 vs best_F1" />
        <BestF1Histogram data={data} indices={indices} />
        <CategoryF1BarChart data={data} indices={indices} />
      </div>
    </div>
  );
}

// ─── Canvas scatter: FP/FN ────────────────────────────────────────────────────
function FpFnScatterCanvas({
  data, indices, title,
}: { data: TagMetricsData; indices: number[]; title: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string } | null>(null);

  const W = 280, H = 280;
  const ML = 40, MR = 16, MT = 28, MB = 36;
  const PW = W - ML - MR, PH = H - MT - MB;

  type Point = { cx: number; cy: number; tag: string; cat: string };
  const pointsRef = useRef<Point[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#1f2937";
    ctx.lineWidth = 1;
    for (let t = 0; t <= 1; t += 0.25) {
      const x = ML + t * PW;
      const y = MT + t * PH;
      ctx.beginPath(); ctx.moveTo(x, MT); ctx.lineTo(x, MT + PH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ML, y); ctx.lineTo(ML + PW, y); ctx.stroke();
    }

    // Reference lines
    ctx.strokeStyle = "#374151";
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    const cx50 = ML + 0.5 * PW;
    const cy50 = MT + 0.5 * PH;
    ctx.beginPath(); ctx.moveTo(cx50, MT); ctx.lineTo(cx50, MT + PH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ML, cy50); ctx.lineTo(ML + PW, cy50); ctx.stroke();
    ctx.setLineDash([]);

    // Points
    const pts: Point[] = [];
    for (const i of indices) {
      const fp = data.fp_rate_50[i];
      const fn = data.fn_rate_50[i];
      if (fp === null || fn === null || isNaN(fp) || isNaN(fn)) continue;
      const cx = ML + fp * PW;
      const cy = MT + fn * PH;
      const cat = data.categories[i] ?? "Unknown";
      const color = CATEGORY_COLORS[cat] ?? "#6b7280";
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
      ctx.fillStyle = color + "cc";
      ctx.fill();
      pts.push({ cx, cy, tag: data.tag_names[i], cat });
    }
    pointsRef.current = pts;

    // Axes
    ctx.strokeStyle = "#374151"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ML, MT); ctx.lineTo(ML, MT + PH); ctx.lineTo(ML + PW, MT + PH); ctx.stroke();

    // Labels
    ctx.fillStyle = "#9ca3af"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      ctx.fillText(t.toFixed(2), ML + t * PW, MT + PH + 14);
    }
    ctx.textAlign = "right";
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      ctx.fillText(t.toFixed(2), ML - 4, MT + t * PH + 3);
    }
    ctx.fillStyle = "#6b7280"; ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("FP rate", ML + PW / 2, MT + PH + 28);
    ctx.save();
    ctx.translate(12, MT + PH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("FN rate", 0, 0);
    ctx.restore();

    // Title
    ctx.fillStyle = "#d1d5db"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(title, ML + PW / 2, 16);
    ctx.fillStyle = "#4b5563"; ctx.font = "10px sans-serif";
    ctx.fillText(`${pts.length} タグ`, ML + PW / 2, MT + PH + 42);
  }, [data, indices, title]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top)  * (H / rect.height);
    let best: Point | null = null;
    let bestDist = 15;
    for (const p of pointsRef.current) {
      const d = Math.hypot(p.cx - mx, p.cy - my);
      if (d < bestDist) { bestDist = d; best = p; }
    }
    setTooltip(best ? { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY, label: best.tag } : null);
  }, []);

  return (
    <div className="bg-gray-900 rounded-lg p-2 relative">
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        className="w-full"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      />
      {tooltip && (
        <div
          className="absolute z-10 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs text-gray-200 pointer-events-none max-w-[200px] truncate"
          style={{ left: tooltip.x + 8, top: tooltip.y - 24 }}
        >
          {tooltip.label}
        </div>
      )}
    </div>
  );
}

// ─── Canvas scatter: global_freq vs best_F1 ───────────────────────────────────
function FreqF1ScatterCanvas({
  data, indices, title,
}: { data: TagMetricsData; indices: number[]; title: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string } | null>(null);

  const W = 280, H = 280;
  const ML = 46, MR = 16, MT = 28, MB = 36;
  const PW = W - ML - MR, PH = H - MT - MB;

  const LOG_MIN = -5; // 1e-5
  const LOG_MAX = 0;  // 1.0

  type Point = { cx: number; cy: number; tag: string };
  const pointsRef = useRef<Point[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#1f2937"; ctx.lineWidth = 1;
    for (let t = 0; t <= 1; t += 0.25) {
      const y = MT + t * PH;
      ctx.beginPath(); ctx.moveTo(ML, y); ctx.lineTo(ML + PW, y); ctx.stroke();
    }
    for (let logv = LOG_MIN; logv <= LOG_MAX; logv++) {
      const t = (logv - LOG_MIN) / (LOG_MAX - LOG_MIN);
      const x = ML + t * PW;
      ctx.beginPath(); ctx.moveTo(x, MT); ctx.lineTo(x, MT + PH); ctx.stroke();
    }

    // Points
    const pts: Point[] = [];
    for (const i of indices) {
      const freq = data.global_freq[i];
      const f1   = data.best_f1[i];
      if (freq === null || f1 === null || freq <= 0) continue;
      const logFreq = Math.log10(freq);
      if (logFreq < LOG_MIN || logFreq > LOG_MAX) continue;
      const tx = (logFreq - LOG_MIN) / (LOG_MAX - LOG_MIN);
      const cx = ML + tx * PW;
      const cy = MT + (1 - f1) * PH;
      const cat = data.categories[i] ?? "Unknown";
      const color = CATEGORY_COLORS[cat] ?? "#6b7280";
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
      ctx.fillStyle = color + "cc";
      ctx.fill();
      pts.push({ cx, cy, tag: data.tag_names[i] });
    }
    pointsRef.current = pts;

    // Axes
    ctx.strokeStyle = "#374151"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ML, MT); ctx.lineTo(ML, MT + PH); ctx.lineTo(ML + PW, MT + PH); ctx.stroke();

    // X-axis labels (log scale)
    ctx.fillStyle = "#9ca3af"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
    for (let logv = LOG_MIN; logv <= LOG_MAX; logv++) {
      const t = (logv - LOG_MIN) / (LOG_MAX - LOG_MIN);
      const x = ML + t * PW;
      ctx.fillText(`1e${logv}`, x, MT + PH + 14);
    }
    // Y-axis labels
    ctx.textAlign = "right";
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      ctx.fillText((1 - t).toFixed(2), ML - 4, MT + t * PH + 3);
    }
    ctx.fillStyle = "#6b7280"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("頻度 (log)", ML + PW / 2, MT + PH + 28);
    ctx.save();
    ctx.translate(12, MT + PH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("best_F1", 0, 0);
    ctx.restore();

    ctx.fillStyle = "#d1d5db"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(title, ML + PW / 2, 16);
    ctx.fillStyle = "#4b5563"; ctx.font = "10px sans-serif";
    ctx.fillText(`${pts.length} タグ`, ML + PW / 2, MT + PH + 42);
  }, [data, indices, title]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const my = (e.clientY - rect.top)  * (H / rect.height);
    let best: Point | null = null, bestDist = 15;
    for (const p of pointsRef.current) {
      const d = Math.hypot(p.cx - mx, p.cy - my);
      if (d < bestDist) { bestDist = d; best = p; }
    }
    setTooltip(best ? { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY, label: best.tag } : null);
  }, []);

  return (
    <div className="bg-gray-900 rounded-lg p-2 relative">
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        className="w-full"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      />
      {tooltip && (
        <div
          className="absolute z-10 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-xs text-gray-200 pointer-events-none max-w-[200px] truncate"
          style={{ left: tooltip.x + 8, top: tooltip.y - 24 }}
        >
          {tooltip.label}
        </div>
      )}
    </div>
  );
}

// ─── SVG histogram: best_F1 distribution ─────────────────────────────────────
function BestF1Histogram({ data, indices }: { data: TagMetricsData; indices: number[] }) {
  const W = 280, H = 200;
  const ML = 40, MR = 12, MT = 24, MB = 32;
  const PW = W - ML - MR, PH = H - MT - MB;
  const N_BINS = 10;

  const counts = useMemo(() => {
    const bins = new Array(N_BINS).fill(0);
    let nullCount = 0;
    for (const i of indices) {
      const v = data.best_f1[i];
      if (v === null) { nullCount++; continue; }
      const b = Math.min(Math.floor(v * N_BINS), N_BINS - 1);
      bins[b]++;
    }
    return { bins, nullCount };
  }, [data, indices]);

  const maxCount = Math.max(...counts.bins, 1);

  return (
    <div className="bg-gray-900 rounded-lg p-2">
      <svg width={W} height={H}>
        <text x={ML + PW / 2} y={16} textAnchor="middle" fill="#d1d5db" fontSize={11}>
          best_F1 分布
        </text>

        {counts.bins.map((c, b) => {
          const bw = PW / N_BINS;
          const bh = (c / maxCount) * PH;
          const x  = ML + b * bw + 1;
          const y  = MT + PH - bh;
          return (
            <g key={b}>
              <rect x={x} y={y} width={bw - 2} height={bh} fill="#3b82f6" opacity={0.7} />
              {c > 0 && bh > 12 && (
                <text x={x + (bw - 2) / 2} y={y + 10} textAnchor="middle" fill="#e5e7eb" fontSize={8}>
                  {c}
                </text>
              )}
            </g>
          );
        })}

        {/* Axes */}
        <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#374151" />
        <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#374151" />

        {/* X ticks */}
        {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((t) => (
          <text key={t} x={ML + t * PW} y={MT + PH + 14} textAnchor="middle" fill="#9ca3af" fontSize={9}>
            {t.toFixed(1)}
          </text>
        ))}
        <text x={ML + PW / 2} y={H - 4} textAnchor="middle" fill="#6b7280" fontSize={10}>
          best_F1
        </text>

        {/* Y tick max */}
        <text x={ML - 4} y={MT + 4} textAnchor="end" fill="#9ca3af" fontSize={9}>
          {maxCount}
        </text>

        {/* Null count note */}
        <text x={ML + PW} y={MT + 10} textAnchor="end" fill="#4b5563" fontSize={9}>
          NaN: {counts.nullCount}
        </text>
      </svg>
    </div>
  );
}

// ─── SVG bar chart: category avg F1 ──────────────────────────────────────────
function CategoryF1BarChart({ data, indices }: { data: TagMetricsData; indices: number[] }) {
  const W = 280, H = 200;
  const ML = 40, MR = 12, MT = 24, MB = 50;
  const PW = W - ML - MR, PH = H - MT - MB;

  const catStats = useMemo(() => {
    const acc: Record<string, { sum: number; count: number }> = {};
    for (const cat of ALL_CATEGORIES) acc[cat] = { sum: 0, count: 0 };
    for (const i of indices) {
      if ((data.n_pos[i] ?? 0) < 20) continue;
      const f1  = data.best_f1[i];
      if (f1 === null) continue;
      const cat = data.categories[i] ?? "Unknown";
      if (!acc[cat]) acc[cat] = { sum: 0, count: 0 };
      acc[cat].sum   += f1;
      acc[cat].count += 1;
    }
    return Object.entries(acc)
      .filter(([, v]) => v.count > 0)
      .map(([cat, v]) => ({ cat, avg: v.sum / v.count, count: v.count }));
  }, [data, indices]);

  if (catStats.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg p-2 flex items-center justify-center h-[200px]">
        <span className="text-gray-500 text-xs">n_pos≥20のタグなし</span>
      </div>
    );
  }

  const bw = PW / catStats.length;

  return (
    <div className="bg-gray-900 rounded-lg p-2">
      <svg width={W} height={H}>
        <text x={ML + PW / 2} y={16} textAnchor="middle" fill="#d1d5db" fontSize={11}>
          カテゴリ別 平均 best_F1 (n_pos≥20)
        </text>

        {catStats.map(({ cat, avg, count }, idx) => {
          const bh = avg * PH;
          const x  = ML + idx * bw + 2;
          const y  = MT + PH - bh;
          const color = CATEGORY_COLORS[cat] ?? "#6b7280";
          return (
            <g key={cat}>
              <rect x={x} y={y} width={bw - 4} height={bh} fill={color} opacity={0.7} />
              <text x={x + (bw - 4) / 2} y={MT + PH + 10} textAnchor="middle" fill={color} fontSize={8}
                transform={`rotate(-45, ${x + (bw - 4) / 2}, ${MT + PH + 10})`}>
                {cat.slice(0, 5)}
              </text>
              <text x={x + (bw - 4) / 2} y={y - 2} textAnchor="middle" fill="#d1d5db" fontSize={8}>
                {avg.toFixed(2)}
              </text>
              <text x={x + (bw - 4) / 2} y={MT + PH + 22} textAnchor="middle" fill="#4b5563" fontSize={7}>
                {count}
              </text>
            </g>
          );
        })}

        {/* Axes */}
        <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#374151" />
        <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#374151" />

        {/* Y ticks */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((t) => (
          <text key={t} x={ML - 4} y={MT + (1 - t) * PH + 3} textAnchor="end" fill="#9ca3af" fontSize={8}>
            {t.toFixed(2)}
          </text>
        ))}
      </svg>
    </div>
  );
}
