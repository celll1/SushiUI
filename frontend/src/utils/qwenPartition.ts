import type { QwenPartitionConfig, QwenPartitionSplit } from "./api";

export type Box = { top: number; left: number; bottom: number; right: number };
export type Region = { core: Box; input: Box; path: string };
export type Divider = { box: Box; axis: "x" | "y"; cut: number; path: string };
export const area = (b: Box) => (b.bottom - b.top) * (b.right - b.left);

function split(b: Box, axis: "x" | "y", ratio: number): [Box, Box, number] {
  const length = axis === "x" ? b.right - b.left : b.bottom - b.top;
  const offset = Math.floor(length * ratio / 2 + 0.5) * 2;
  if (offset < 2 || offset > length - 2) throw new Error("Boundary collapses after 32-pixel snapping.");
  const cut = offset + (axis === "x" ? b.left : b.top);
  return axis === "x" ? [{ ...b, right: cut }, { ...b, left: cut }, cut]
    : [{ ...b, bottom: cut }, { ...b, top: cut }, cut];
}

export function partitionPreview(config: QwenPartitionConfig, height: number, width: number) {
  if (!Number.isInteger(config.halo_tokens) || config.halo_tokens < 0 || config.halo_tokens % 2) throw new Error("Halo must be a nonnegative even integer.");
  const full: Box = { top: 0, left: 0, bottom: height, right: width };
  const expand = (b: Box): Box => ({ top: Math.max(0, b.top - config.halo_tokens),
    left: Math.max(0, b.left - config.halo_tokens), bottom: Math.min(height, b.bottom + config.halo_tokens),
    right: Math.min(width, b.right + config.halo_tokens) });
  const regions: Region[] = [];
  const dividers: Divider[] = [];
  const visit = (b: Box, node: QwenPartitionSplit | null, path: string, depth: number) => {
    if (!node) { regions.push({ core: b, input: expand(b), path }); return; }
    if (depth > 8 || dividers.length >= 63) throw new Error("Layout exceeds 64 regions / depth 8.");
    const [first, second, cut] = split(b, node.axis, node.ratio);
    dividers.push({ box: b, axis: node.axis, cut, path });
    visit(first, node.first, path + "0", depth + 1);
    visit(second, node.second, path + "1", depth + 1);
  };
  if (config.mode === "manual") visit(full, config.layout, "", 0);
  else {
    regions.push({ core: full, input: expand(full), path: "" });
    if (config.mode === "auto") {
      if (!Number.isInteger(config.max_target_tokens) || config.max_target_tokens < 1) throw new Error("Token ceiling must be positive.");
      for (;;) {
        let index = 0;
        regions.forEach((r, i) => { if (area(r.input) > area(regions[index].input)) index = i; });
        const region = regions[index];
        if (area(region.input) <= config.max_target_tokens) break;
        if (regions.length >= 64) throw new Error("Auto exceeds 64 regions.");
        const b = region.core;
        const axis = b.right - b.left >= b.bottom - b.top ? "x" : "y";
        let pair: [Box, Box, number];
        try { pair = split(b, axis, 0.5); }
        catch { pair = split(b, axis === "x" ? "y" : "x", 0.5); }
        regions.splice(index, 1, ...pair.slice(0, 2).map((core, i) => ({ core: core as Box,
          input: expand(core as Box), path: region.path + i })));
      }
    }
  }
  return { regions, dividers };
}

export function replaceNode(root: QwenPartitionSplit | null, path: string,
  update: (node: QwenPartitionSplit | null) => QwenPartitionSplit | null): QwenPartitionSplit | null {
  if (!path) return update(root);
  if (!root) return root;
  const key = path[0] === "0" ? "first" : "second";
  return { ...root, [key]: replaceNode(root[key], path.slice(1), update) };
}

export function strips(count: number, axis: "x" | "y"): QwenPartitionSplit | null {
  return count < 2 ? null : { axis, ratio: 1 / count, first: null, second: strips(count - 1, axis) };
}
