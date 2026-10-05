const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('../node_modules/typescript');
const root = path.resolve(__dirname, '../src/components/training');
const formText = fs.readFileSync(path.join(root, 'TrainingConfig.tsx'), 'utf8');
const form = ts.createSourceFile('TrainingConfig.tsx', formText, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const definitions = ts.createSourceFile('definitions.tsx', fs.readFileSync(path.join(root, 'trainingConfigDefinitions.tsx'), 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
function declaration(source, name) {
  let found;
  function walk(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === name) found = node;
    ts.forEachChild(node, walk);
  }
  walk(source);
  assert.ok(found, name);
  return found.initializer;
}
function transpile(text) {
  return ts.transpileModule(text, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
}
const moduleContext = { exports: {} };
vm.runInNewContext(transpile(fs.readFileSync(path.join(root, 'trainingParams.ts'), 'utf8')), moduleContext);
const plumbing = moduleContext.exports;
const restore = declaration(form, 'applyParamsToState').arguments[0];
function restored(incoming) {
  let state = { full_finetune_dequantize_int8_base: false, training_export_format: 'none', sample_prompts: [{ prompt: 'default' }] };
  const context = { PARAM_KEYS: plumbing.PARAM_KEYS };
  for (const name of new Set(restore.getText(form).match(/\bset[A-Z]\w*/g))) context[name] = () => {};
  for (const name of new Set(restore.getText(form).match(/\b\w+Ref\b/g))) context[name] = { current: null };
  context.setParams = update => { state = update(state); };
  context.incoming = incoming;
  vm.runInNewContext(transpile(`const restore = ${restore.getText(form)}; restore(incoming);`), context);
  return JSON.parse(JSON.stringify(state));
}

test('all defaulted form fields have a restore path', () => {
  const defaults = declaration(definitions, 'DEFAULT_PARAMS');
  const explicit = new Set(Array.from(restore.getText(form).matchAll(/incoming\.([a-z0-9_]+)/g), m => m[1]));
  const missing = defaults.properties.map(p => p.name?.getText(definitions)).filter(name => name && !plumbing.PARAM_KEYS.includes(name) && !explicit.has(name));
  assert.deepEqual(missing, []);
});

test('the full API request surface has a restore path or explicit separate owner', () => {
  const api = fs.readFileSync(path.resolve(root, '../../utils/api.ts'), 'utf8').split('export interface TrainingRunCreateRequest')[1].split('\n}')[0];
  const explicit = new Set(Array.from(restore.getText(form).matchAll(/incoming\.([a-z0-9_]+)/g), m => m[1]));
  const separate = new Set(['dataset_id', 'network_type', 'vae_config']);
  const missing = Array.from(api.matchAll(/^  ([a-z0-9_]+)\??:/gm), m => m[1]).filter(key =>
    !plumbing.PARAM_KEYS.includes(key) && !explicit.has(key) && !separate.has(key)
    && !(key.startsWith('qwen_partition_') && plumbing.PARAM_KEYS.includes(key.replace('qwen_partition_', 'dit_partition_'))));
  assert.deepEqual(missing, []);
});

test('edit restoration and preset transport preserve nondefault Full and Chimera fields', () => {
  const incoming = {
    full_finetune_dequantize_int8_base: true, training_export_format: 'int8_convrot',
    chimera_flow_version: 'v2', chimera_v2_parameterization: 'centered',
    chimera_v2_latent_mean: [0.1, 0.2], chimera_v2_latent_centered_second_moment: [1.1, 1.2],
    chimera_v2_calibration: { source: 'saved' }, qwen_convrot_training_forward: 'dequant',
  };
  for (const [key, value] of Object.entries(incoming)) {
    const state = restored(incoming);
    assert.deepEqual(state[key], value, key);
    assert.deepEqual(JSON.parse(JSON.stringify(plumbing.passThroughParams(state)[key])), value, key);
    assert.deepEqual(JSON.parse(JSON.stringify(plumbing.presetConfigToParams(incoming)[key])), value, key);
  }
});

test('explicit empty sample prompts do not turn into default prompts', () => {
  assert.deepEqual(restored({ sample_prompts: [] }).sample_prompts, []);
  assert.deepEqual(restored({}).sample_prompts, [{ prompt: 'default' }]);
});

test('restore marks bundle default ownership before delayed schema effects', () => {
  const context = { PARAM_KEYS: plumbing.PARAM_KEYS, incoming: { base_model_path: 'saved-model' } };
  for (const name of new Set(restore.getText(form).match(/\bset[A-Z]\w*/g))) context[name] = () => {};
  for (const name of new Set(restore.getText(form).match(/\b\w+Ref\b/g))) context[name] = { current: null };
  vm.runInNewContext(transpile(`const restore = ${restore.getText(form)}; restore(incoming);`), context);
  assert.equal(context.lastBundleVaeModelRef.current, 'saved-model');
});

test('late capability and optimizer effects cannot rewrite a restored recipe', () => {
  const effects = [];
  function walk(node) {
    if (ts.isCallExpression(node) && node.expression.getText(form) === 'useEffect') {
      const callback = node.arguments[0];
      if (/unsupportedTrainingMethod\(trainingMethod\)|!adapterAlgorithmChoices\.includes|if \(blockSwapUnsupported\)|if \(isSDOrSDXLModel\(baseModelPath\)\)|const startsNewContract|const config = OPTIMIZER_CONFIGS/.test(callback.getText(form))) effects.push(callback);
    }
    ts.forEachChild(node, walk);
  }
  walk(form);
  assert.equal(effects.length, 6);
  for (const effect of effects) {
    const context = {
      preservingRestoredRecipe: true, capabilitiesReady: true,
      restoringFromYAMLRef: { current: false },
      restoredRecipeRef: { current: { optimizer: 'adamw' } }, optimizer: 'adamw',
      updateParam: () => assert.fail('saved parameter was rewritten'),
    };
    vm.runInNewContext(transpile(`const effect = ${effect.getText(form)}; effect();`), context);
  }
});
