const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('../node_modules/typescript');
function load(name) {
  const source = fs.readFileSync(path.join(__dirname, '../src/utils', name), 'utf8');
  const context = { exports: {} };
  // Strip types for function tests; this performs no project build/type-check.
  vm.runInNewContext(ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020,
  } }).outputText, context);
  return context.exports;
}
const math = load('videoChainMath.ts');
const overlay = load('videoCodecGeometry.ts').overlayLoadedVideoConstraints;
const plain = value => JSON.parse(JSON.stringify(value));
test('native chunks retain their nonuniform repeated boundaries', () => {
  assert.deepEqual(plain(math.latentGroupSpans([1,4,4,4,4], 22)).map(row => row[1]), [1,5,9,13,17,18,22]);
});
test('causal first frame does not recur inside the clip', () => {
  const codec = { latent_frame_policy: 'causal_first_frame_v1', temporal_scale: 2 };
  const spans = plain(math.latentGroupSpans([], 25, codec));
  assert.equal(spans.length, 13);
  assert.deepEqual(spans.map(row => row[1]), [1,3,5,7,9,11,13,15,17,19,21,23,25]);
  assert.deepEqual(plain(math.snapRangeToLatentGroups(spans, 2, 4)), { start: 1, end: 5 });
  const caps = { video_constraints: { minimax_h3: { ...codec, latent_chunk_pattern: [] } },
    chain_context: { minimax_h3: { chain_context_min_frames: 5, chain_context_max_frames: 17 } } };
  assert.deepEqual(plain(math.chainContinuationOverlapLengths(caps, 'minimax_h3')), [5,7,9,11,13,15,17]);
});
test('independent frame codecs expose each frame and invalid widths cannot loop', () => {
  const spans = plain(math.latentGroupSpans([], 5, { latent_frame_policy: 'independent_frames_v1' }));
  assert.deepEqual(spans, [[0,1],[1,2],[2,3],[3,4],[4,5]]);
  assert.deepEqual(plain(math.latentGroupSpans([0], 5)), []);
  assert.deepEqual(plain(math.latentGroupSpans([1], Infinity)), []);
});
test('loaded overlay is immutable and removed on unload or native-model change', () => {
  const native = { frame_multiple: 17, frame_offset: 5 };
  const other = { frame_multiple: 8, frame_offset: 1 };
  const base = { video_constraints: { minimax_h3: native, ltx2: other }, unsupported: {} };
  const geometry = { frame_multiple: 2, frame_offset: 1 };
  const effective = overlay(base, { type: 'minimax_h3', loaded_video_constraints: geometry });
  assert.equal(effective.video_constraints.minimax_h3, geometry);
  assert.equal(effective.video_constraints.ltx2, other);
  assert.equal(base.video_constraints.minimax_h3, native);
  assert.equal(overlay(base, null), base);
  assert.equal(overlay(base, { type: 'minimax_h3' }), base);
  assert.equal(overlay(null, { type: 'minimax_h3', loaded_video_constraints: geometry }), null);
});
test('changed TS and TSX wiring has valid syntax without invoking a type checker', () => {
  for (const name of ['contexts/StartupContext.tsx', 'components/generation/InpaintPanel.tsx',
    'components/generation/VideoInpaintTimeline.tsx', 'utils/api.ts', 'utils/videoChainMath.ts',
    'utils/videoCodecGeometry.ts']) {
    const file = path.join(__dirname, '../src', name);
    const source = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true,
      name.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
    assert.equal(source.parseDiagnostics.length, 0, name);
  }
});
