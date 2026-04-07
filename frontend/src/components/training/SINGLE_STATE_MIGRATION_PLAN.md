# TrainingConfig.tsx 単一state object移行計画書

## Context

`frontend/src/components/training/TrainingConfig.tsx`（3919行）には現在約120個の`useState`フックが存在し、トレーニングパラメータごとに個別のstate変数を管理している。Phase 3で`getRequestData()` / `applyParamsToState()`ヘルパーを導入してサイレントドロップは防止済みだが、以下の問題が残っている:

### 残存する問題
1. **冗長な宣言**: `useState`宣言ブロックが約200行を占める
2. **2箇所更新の必要性**: 新パラメータ追加時、`useState`宣言 + `getRequestData`/`applyParamsToState`の両方を更新する必要がある
3. **依存配列の肥大化**: `getRequestData`の`useCallback`依存配列が90+項目で保守不可能
4. **型不整合**: 一部のstateが`string`（`learningRate: "1e-5"`）、一部は`number`（`batchSize: 4`）と統一されていない
5. **再レンダリング非効率**: 1つのフィールドを変更すると、全`useCallback`が再生成される

### 目指す状態
- **単一の `params: TrainingRunCreateRequest` state**で全パラメータを管理
- 各UI入力要素は `params.field_name` を読み、`updateParam("field_name", value)` で更新
- `getRequestData()` / `applyParamsToState()` は不要になり削除
- 新パラメータ追加: `TrainingRunCreateRequest` 型 + UI要素のみ（実質1箇所）

---

## アーキテクチャ設計

### 新しい state 構造

```typescript
import { TrainingRunCreateRequest } from "@/utils/api";

// すべてのデフォルト値を1箇所に集約
const DEFAULT_PARAMS: TrainingRunCreateRequest = {
  training_method: "lora",
  base_model_path: "",
  total_steps: 1000,
  epochs: undefined,
  batch_size: 4,
  learning_rate: 1e-5,
  lr_scheduler: "constant",
  lr_warmup_steps: 0,
  optimizer: "adamw8bit",
  // ... 全96フィールド
};

const [params, setParams] = useState<TrainingRunCreateRequest>(DEFAULT_PARAMS);

// ジェネリックな更新ヘルパー（型安全）
const updateParam = useCallback(
  <K extends keyof TrainingRunCreateRequest>(
    key: K,
    value: TrainingRunCreateRequest[K]
  ) => {
    setParams(prev => ({ ...prev, [key]: value }));
  },
  []
);
```

### UI 要素の書き換えパターン

#### Pattern A: 単純な数値入力
**Before:**
```tsx
const [batchSize, setBatchSize] = useState(4);
// ...
<input
  type="number"
  value={batchSize}
  onChange={(e) => setBatchSize(parseInt(e.target.value))}
/>
```

**After:**
```tsx
<input
  type="number"
  value={params.batch_size}
  onChange={(e) => updateParam("batch_size", parseInt(e.target.value))}
/>
```

#### Pattern B: 文字列入力（数値だが空文字列許容）
学習率やbeta1などは文字列で保持し、submit時にparseFloat:

**Before:**
```tsx
const [learningRate, setLearningRate] = useState<string>("1e-5");
```

**After:**
- ローカルに表示用文字列を残しつつ、`params.learning_rate`は数値で同期
- もしくは型を `number | null` に変更し、`String(params.learning_rate)` で表示

**推奨**: ローカル文字列stateを残す（科学記法の入力中状態を保持するため）。`params`は数値、`localLrText`は表示用。`onBlur`時に同期。

```tsx
const [localLrText, setLocalLrText] = useState(String(params.learning_rate ?? ""));

// Edit Config時の同期
useEffect(() => {
  setLocalLrText(String(params.learning_rate ?? ""));
}, [params.learning_rate]);

<input
  type="text"
  value={localLrText}
  onChange={(e) => setLocalLrText(e.target.value)}
  onBlur={(e) => {
    const v = parseFloat(e.target.value);
    if (!isNaN(v)) updateParam("learning_rate", v);
  }}
/>
```

#### Pattern C: Select / Dropdown
```tsx
<select
  value={params.optimizer}
  onChange={(e) => updateParam("optimizer", e.target.value)}
>
  ...
</select>
```

#### Pattern D: チェックボックス
```tsx
<input
  type="checkbox"
  checked={params.optimizer_is_paged ?? false}
  onChange={(e) => updateParam("optimizer_is_paged", e.target.checked)}
/>
```

#### Pattern E: ネストされたオブジェクト（timestep_sampling）
```tsx
const updateTimestepSampling = (key: string, value: any) => {
  setParams(prev => ({
    ...prev,
    timestep_sampling: { ...(prev.timestep_sampling || {}), [key]: value },
  }));
};

<input
  value={params.timestep_sampling?.distribution || "uniform"}
  onChange={(e) => updateTimestepSampling("distribution", e.target.value)}
/>
```

#### Pattern F: 派生state（UIのみ、API送信しない）
- `useEpochs`, `loading`, `error`, `samplers`, `presets`, `availableModels` などのUI/データロード state は **`params`に統合せず、useStateのまま残す**
- 残す対象: `runName`（trim処理あり）, `useEpochs`（ラジオボタン状態）, `priorityEnabled`/`priorityText`/`priorityMultiplier`/`priorityExpanded`（複合UI）, `timestepDistribution`/`timestepMin`/`timestepMax`/`timestepMean`/`timestepStd`/`timestepAlpha`/`timestepBeta`（UI用、submit時に`timestep_sampling`オブジェクトに集約）

---

## カテゴリ別移行マップ

`useState` → `params.field_name` のマッピング表。`params`に統合する対象のみリスト。

| カテゴリ | 旧 useState | 新 params field |
|---------|------------|----------------|
| **Core training** | `totalSteps` | `params.total_steps` |
|  | `epochs` | `params.epochs` |
|  | `batchSize` | `params.batch_size` |
|  | `learningRate` (string) | `params.learning_rate` (number, +localLrText) |
|  | `lrScheduler` | `params.lr_scheduler` |
|  | `lrWarmupSteps` | `params.lr_warmup_steps` |
|  | `optimizer` | `params.optimizer` |
| **Optimizer** | `optimizerIsPaged` | `params.optimizer_is_paged` |
|  | `optimizerCautious` | `params.optimizer_cautious` |
|  | `optimizerBeta1/2/Epsilon/WeightDecay` | `params.optimizer_beta1/2/...` (+localText) |
|  | `optimizerScheduleFree*` | `params.optimizer_schedule_free*` |
|  | `optimizerUseRadam` | `params.optimizer_use_radam` |
|  | `optimizerStochasticRounding` | `params.optimizer_stochastic_rounding` |
| **LoRA** | `loraRank/Alpha/Dtype` | `params.lora_rank/alpha/dtype` |
| **ReLoRA** | `reloraMergeEvery` 他5個 | `params.relora_merge_every` 他 |
| **Save/Sample** | `saveEvery/SaveEveryUnit/SampleEvery` | `params.save_every/...` |
|  | `resumeFromCheckpoint` | `params.resume_from_checkpoint` |
|  | `samplePrompts` | `params.sample_prompts` |
|  | `sampleWidth/Height/Steps/CfgScale/Sampler/ScheduleType/Seed` | `params.sample_*` |
| **Debug** | `debugLatents/DebugLatentsEvery` | `params.debug_latents/...` |
| **Bucketing** | `enableBucketing/baseResolutions/bucketStrategy/multiResolutionMode` | `params.enable_bucketing/...` |
|  | `cacheLatentsToDisk/forceRecache` | `params.cache_latents_to_disk/...` |
| **Component training** | `trainUnet/TrainTextEncoder/TrainImageEncoder` | `params.train_unet/...` |
|  | `unetLr/textEncoderLr/textEncoder1Lr/textEncoder2Lr/imageEncoderLr` | `params.unet_lr/...` (+localText) |
| **Precision** | `weightDtype/trainingDtype/outputDtype/vaeDtype` | `params.weight_dtype/...` |
|  | `mixedPrecision/useFlashAttention/minSnrGamma/reconstructionLossWeight` | `params.mixed_precision/...` |
| **Encoding** | `textEncodingMode/Interval` | `params.text_encoding_mode/...` |
|  | `latentEncodingMode/Interval` | `params.latent_encoding_mode/...` |
| **Block Swap** | `blocksToSwap/usePinnedMemory/numOptimizerGroups` | `params.blocks_to_swap/...` |
| **MNT** | `multiNoiseTimesteps/Mode/trajectoryBlendAlpha` | `params.multi_noise_timesteps/...` |
| **Regularization** | `regularizationType/snr*/energy*` | `params.regularization_type/...` |
| **Unified Framework** | `noiseProcess/predictionTarget/strictValidation` | `params.noise_process/...` |
| **Vision Encoder** | `useReferenceImages/visionEncoderPath/trainVisionEncoder/visionEncoderLr/gradientRoutingVE` | `params.use_reference_images/...` |
| **Param Tracking** | `paramTracking/paramTrackingInterval` | `params.param_tracking/...` |
| **ControlNet** | `controlnetType/PretrainedPath/InitFromUnet/llliteConditioningChannels/llliteRank/conditionPreprocessors/conditionCacheMode` | `params.controlnet_*` |
| **Datasets** | `datasetConfigs` | `params.dataset_configs` |
| **Run name** | `runName` | `params.run_name` (trim時のみ手動) |
| **Training method** | `trainingMethod` | `params.training_method` |
| **Base model path** | `baseModelPath` | `params.base_model_path` (trim時のみ手動) |

### 残すuseState（UIのみ、API送信しない）

| useState | 理由 |
|---------|------|
| `availableModels`, `availableControlNets`, `availableCheckpoints`, `samplers`, `scheduleTypes`, `presets`, `datasets` | API取得結果（リクエストとは無関係） |
| `useEpochs` | total_steps/epochsの排他選択用UI |
| `loading`, `error` | ローディング/エラー表示 |
| `showPresetDialog`, `showLoadPresetDialog`, `presetName`, `presetDescription` | プリセットダイアログUI |
| `showSD15`, `showSDXL`, `showZImage`, `showFlux2` | モデルフィルタUI |
| `priorityEnabled`, `priorityText`, `priorityMultiplier`, `priorityExpanded` | priority_trainingオブジェクトの構築前UI |
| `timestepDistribution`, `timestepMin`, `timestepMax`, `timestepMean`, `timestepStd`, `timestepAlpha`, `timestepBeta` | timestep_samplingオブジェクトの構築前UI |
| `conditionImagePreviews`, `referenceImagePreviews` | プレビュー画像URL |
| `localLrText`, `localBeta1Text` 等（**新規追加**） | 数値入力中の表示用 |
| `dtypeExplicitlySetRef`, `restoringFromYAMLRef` | useEffect副作用の制御フラグ |

---

## 段階的移行戦略

3919行のコンポーネントを一度に書き換えるのは危険なため、**カテゴリ単位**で段階的に移行する。各段階でビルド・動作確認を行い、問題があれば即座にロールバック可能。

### Phase 3a: 基盤整備
1. `params: TrainingRunCreateRequest` state を追加（`DEFAULT_PARAMS`を定義）
2. `updateParam<K>(key, value)` ヘルパーを定義
3. `getRequestData()` を `params` 直接参照に書き換え（簡素化）
4. `applyParamsToState()` を `setParams(data)` 1行に書き換え
5. **既存の useState は残したまま**、`params` と useState を `useEffect` で双方向同期する暫定状態にする
6. ビルド & 動作確認

### Phase 3b〜3l: カテゴリ別移行（11コミット推奨）
各カテゴリごとに以下の作業を行う:

1. 該当 useState を削除
2. 対応する UI 入力要素を `params.x` / `updateParam("x", v)` に書き換え
3. `getRequestData()` / `applyParamsToState()` から該当行を削除
4. ビルド & 動作確認
5. コミット

| Phase | カテゴリ | useState数 |
|-------|---------|----------|
| 3b | Core training (steps/epochs/batch/lr/scheduler/warmup/optimizer) | 8 |
| 3c | Optimizer hyperparams (beta/epsilon/weight_decay/schedule_free/...) | 11 |
| 3d | LoRA + ReLoRA (rank/alpha/dtype/relora_*) | 8 |
| 3e | Save/Resume/Sample (save_every/sample_*/...) | 14 |
| 3f | Debug + Bucketing + Cache | 8 |
| 3g | Component training (train_unet/te/ie + LRs) | 8 |
| 3h | Precision/dtype settings | 8 |
| 3i | Encoding + Block Swap + MNT | 11 |
| 3j | Regularization + Unified Framework | 11 |
| 3k | Vision Encoder + Param Tracking | 7 |
| 3l | ControlNet (controlnet_type/lllite_*/condition_*) | 7 |

各Phaseで7〜14個のuseStateを移行。各Phase完了時にコミット。

### Phase 3m: クリーンアップ
1. 双方向同期useEffectを削除
2. 残存する不要useStateを削除
3. `getRequestData()` / `applyParamsToState()` を削除（または最小化）
4. 依存配列を簡素化
5. 最終コミット

---

## 主要な技術的課題と対策

### 課題1: 文字列ベースの数値入力（localText pattern）

`learningRate`, `optimizerBeta1`, `unetLr` などは、ユーザーが「`1e-5`」を入力途中の状態（「`1e`」「`1e-`」）を保持するため文字列で保持されている。

**対策**: ローカル文字列stateを残し、`params`との同期は `onBlur` で行う:
```typescript
const [localLrText, setLocalLrText] = useState(String(params.learning_rate ?? "1e-5"));

// Edit Config時の同期
useEffect(() => {
  if (params.learning_rate !== undefined && params.learning_rate !== null) {
    setLocalLrText(String(params.learning_rate));
  }
}, [params.learning_rate]);
```

対象フィールド（局所文字列を残す）:
- `learning_rate`, `unet_lr`, `text_encoder_lr`, `text_encoder_1_lr`, `text_encoder_2_lr`, `image_encoder_lr`, `vision_encoder_lr`
- `optimizer_beta1`, `optimizer_beta2`, `optimizer_epsilon`, `optimizer_weight_decay`
- `optimizer_schedule_free_r`, `optimizer_schedule_free_weight_lr_power`

合計: 約13個のlocalText state

### 課題2: 排他的フィールド（total_steps vs epochs）

`useEpochs` ラジオボタンに応じて、submit時に片方を `undefined` にする必要がある。

**対策**: `useEpochs` はuseStateのまま残し、`params.total_steps`/`params.epochs` 両方を保持。submit時のみ条件付きで送信:
```typescript
const dataToSubmit = {
  ...params,
  total_steps: useEpochs ? undefined : params.total_steps,
  epochs: useEpochs ? params.epochs : undefined,
};
```

### 課題3: 型 union のキャスト

`controlnet_type: "standard" | "lllite"` のような型のため、`updateParam` の型推論が効かない場合がある。

**対策**: 入力時にキャスト:
```typescript
updateParam("controlnet_type", e.target.value as "standard" | "lllite");
```

### 課題4: 派生フィールド（priority_training, timestep_sampling）

UIは複数のstateを使うが、API送信時は1つのオブジェクトに集約される。

**対策**: UI用stateは別途保持し、`params`に集約するのはsubmit直前。または `useMemo` で派生:
```typescript
const priorityTrainingObj = useMemo(() => {
  if (!priorityEnabled || !priorityText.trim()) return undefined;
  return {
    entries: priorityText.trim().split("\n").map(line => line.trim()).filter(Boolean),
    multiplier: priorityMultiplier,
  };
}, [priorityEnabled, priorityText, priorityMultiplier]);

// Submit時に param に注入
const dataToSubmit = { ...params, priority_training: priorityTrainingObj };
```

### 課題5: useEffect 副作用との競合

既存の `dtypeExplicitlySetRef` / `restoringFromYAMLRef` などのフラグは、特定のuseEffectがstate更新と競合しないように制御している。`params` 統合時もこの仕組みは維持する必要がある。

**対策**: フラグは現状維持。`setParams(prev => ...)` の関数形式を使うことで、useEffectの依存値による不整合を回避。

### 課題6: useCallback 依存配列の管理

`getRequestData()` の依存配列は90+項目で爆発している。`params` 統合後は依存が`[params]`の1つになるため、この問題は自動的に解決する。

---

## 検証戦略

各Phase完了時に以下を確認:

### ビルドチェック
```bash
cd frontend && npm run build  # ユーザーが実施
```

### 機能テスト
1. **新規training run作成** (LoRA/ReLoRA/Full FT/ControlNet 4方式)
2. **Edit Config**: 既存training runを開き、全フィールドが正しく復元されることを確認
3. **設定変更→保存→再度開く**: 値が保持されることを確認
4. **プリセットの保存/読み込み**: パラメータが正しく永続化されることを確認

### 回帰テスト用チェックリスト
| カテゴリ | テスト項目 |
|---------|----------|
| 数値入力 | learning_rate に `1e-5` を入力→ submit → YAMLで `1.0e-05` を確認 |
| 文字列入力 | run_name に空白付き文字列 → trim される |
| チェックボックス | optimizer_is_paged ON → YAML に反映 |
| ラジオ | useEpochs 切替 → total_steps/epochs が正しく切り替わる |
| 派生 | priority_training enabled + entries → YAMLにオブジェクトとして反映 |
| ネスト | timestep_sampling distribution = "beta" → mean/std が省略される |
| 排他フィールド | training_method = controlnet → lora_* が undefined |
| Edit Config | 全フィールドが復元され、編集→再保存で値が保持される |

---

## リスクと対策

| リスク | 影響度 | 対策 |
|-------|-------|------|
| 移行中の暫定状態でビルドエラー | 高 | カテゴリ単位の小コミット、各コミットでビルド確認 |
| 既存useStateとparamsの双方向同期で無限ループ | 高 | useEffectの依存配列を厳密管理。`prev`関数形式の活用 |
| localText stateとparamsの不整合（科学記法入力中） | 中 | onBlur同期パターンで吸収 |
| 型推論が効かないunion型フィールド | 低 | 必要箇所で `as` キャスト |
| プリセット保存/読み込みのフォーマット変更 | 中 | 既存形式と互換を保つ。プリセット読み込み時に`applyParamsToState`を経由 |
| dtype自動設定useEffectとの競合 | 中 | `dtypeExplicitlySetRef` を維持 |
| 大量のUI要素更新による typo | 中 | カテゴリごとに視覚レビュー、段階的なテスト |

---

## 完了条件

- [ ] `params: TrainingRunCreateRequest` state が導入され、全主要パラメータを保持
- [ ] `useState` 宣言が約120個 → 約20個（UI用のみ）に削減
- [ ] 全UI入力要素が `params.x` / `updateParam("x", v)` パターンに統一
- [ ] `getRequestData()` が `{ ...params, /* 派生フィールド */ }` のシンプルな形に
- [ ] `applyParamsToState()` が `setParams(data)` の1行に
- [ ] 4学習方式すべてで新規作成・Edit Config・保存・再オープンが正常動作
- [ ] フロントエンドビルドがエラーなく完了
- [ ] `frontend/src/components/training/SINGLE_STATE_MIGRATION_PLAN.md` を削除（または「完了」マークを追加）

---

## 推定工数

- Phase 3a (基盤整備): 30〜60分
- Phase 3b〜3l (11カテゴリ移行): 各15〜30分 = 3〜5時間
- Phase 3m (クリーンアップ): 30〜60分
- テスト・修正: 1〜2時間

**合計: 6〜10時間**（複数セッションに分割推奨）

---

## 参考: 改善後の追加手順

新しい top-level training param を追加する手順（移行完了後）:

1. **`backend/api/routes.py`** の `TrainingRunCreateRequest` にフィールド追加
2. **`frontend/src/utils/api.ts`** の `TrainingRunCreateRequest` インターフェース に追加
3. **`frontend/src/components/training/TrainingConfig.tsx`** の `DEFAULT_PARAMS` にデフォルト値追加 + UI入力要素を追加（`params.x` / `updateParam("x", v)`）
4. **trainer/adapter** で `self.config.get("x")` で消費

**従来の `getRequestData()` / `applyParamsToState()` への追加は不要**。`useState` 追加も不要。

---

## 関連ドキュメント

- [TRAINING_PARAMS_GUIDE.md](../../../../backend/core/training/TRAINING_PARAMS_GUIDE.md) - パラメータ追加の総合ガイド
- Phase 1〜3 のコミット履歴を参照（`git log --oneline -- backend/core/training/training_config.py`）
