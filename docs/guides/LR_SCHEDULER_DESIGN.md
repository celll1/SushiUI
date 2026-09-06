# LR スケジューラ拡張設計（WSD / 実行時減衰と取り消し / 床 / restart / REX / LLRD / 集約）

Status: **P0 / P1 / P2 / P3 / P4 / P5 / P6 実装済み（§18。P0 の軸変換の取りこぼしは §18.5、P3 の config キー分は §18.6、ReLoRA の統合は §18.7、LLRD とグループ別スケジュールは §18.8）。残るは P7（VAE トレーナーの語彙統合）のみ。** 本書は §14 のフェーズ単位で実装・検証・コミットする前提で
書かれており、各フェーズの受け入れ条件を持つ。既存挙動の記述は全て `file:line` を付す。
引用のない記述は設計上の決定であり、「要検証」と付したものは実装前に確認が必要な事実主張である。
一次資料は 2 本の read-only 調査を統合したブリーフ（本書執筆時点の作業ファイル）で、
本書はその引用を再確認したうえで書いている。

**本書はいかなるスケジュールについても学習品質の主張をしない。** 品質は所有者が実データで
測るものであり、本書の決定は全て「機構が config どおりに動き、再開・延長で形が保たれ、
状態が観測できる」ことだけを根拠にしている。

要求（リポジトリ所有者）:

1. 減衰開始を実行時に決める warmup-stable-decay（WSD）。**開始後に「やっぱりなし」で
   元の LR に戻せる**こと（取り消しは要件）。
2. 全スケジュール共通の床（`lr_floor_ratio` の一般化）。
3. パラメータグループごとに別スケジュール。**既定オフ**。
4. 延長耐性（`total_steps` 変更で形が変わらない）。**明示設定ではなく暗黙**に効かせる。
5. restart 付き cosine を 1 と組み合わせる。annealing（ピーク減衰）と床を持つ。
   `cosine_with_restarts` の UI 未露出も解消する。
6. REX。「1 の減衰率を強めれば実現できる気がする」を式で検証する。
7. LLRD（深さ方向 LR 減衰）。細かい設定は持たせず「よきに計らう」プリセットとして。
8. **スケジューラ機能をどこに集約すれば効率がいいか**（明示の問い）。

---

## 0. 決定の一覧

| # | 論点 | 決定 |
|---|---|---|
| D1 | 集約先 | 新規モジュール **`backend/core/training/lr_schedules.py`** に「スケジュール定義（純関数）・レジストリ・構築・実行時タイムライン・直列化」を置く。`lr_utils.py` は再表明専用のまま残し（docstring `lr_utils.py:1-9` の役割分担どおり）、不変条件 `:44-53` を真に戻す。`base_trainer.py` 内の 2 構築点（`:6001-6012`, `:6506-6522`）と `_build_plateau_cosine_floor_scheduler`（`:6547-6588`）は `lr_schedules.build_lr_scheduler()` 呼び出しに置き換える。**`base_trainer.py` にスケジュール式を残さない** |
| D2 | `LambdaLR` 不変条件 | 本書の対象（BaseTrainer / ReLoRA / P7 の VAE）が作る LR スケジューラは**例外なく `torch.optim.lr_scheduler.LambdaLR`**。ReLoRA の `CosineWithMultipleWarmups`（`relora_scheduler.py:33`、`_LRScheduler` 直系）はレジストリの `LambdaLR` 実装に置き換え、`_fast_forward_one_lr_scheduler` の O(step) リプレイ（`base_trainer.py:3908-3911`）と re-warmup の無言スキップ（`:5536-5541`）を到達不能にする |
| D3 | lambda の純粋性 | lambda は「**step 整数**と、**seam でしか書き換わらないタイムライン**」の純関数。評価は状態を変えない（順不同・複数回評価可、`lr_utils.py:174` / `base_trainer.py:3896` / `:5572` の 3 経路要件）。タイムラインを書き換えてよい seam は (a) 構築時、(b) resume の状態読み込み直後、(c) 制御コマンドの適用点、(d) ReLoRA merge の 3+1 箇所のみ（§5.2） |
| D4 | 実行時状態の置き場 | **`save_training_state` の新キー `lr_schedule_events`**（`base_trainer.py:3793-3811` に追加、`at <= step` に切り詰めて保存）。別 artifact（VAE の `lr_scheduler.pt` 方式）は採らない: refusal matrix（`vae_trainer.py:1355-1378`）のコストと、チェックポイント step より後の事象を resume で捨てなければならない要件（`_cleanup_future_metrics` `:13085` と同じ意味論）を state.json の切り詰めが自然に満たすため。config への書き戻しも採らない（実行時事象は config ではない） |
| D5 | 取り消しの形 | **base curve へ線形に復帰**する。復帰長 `R = lr_warmup_steps`、`R = 0` なら不連続復帰。逆向き補間（減衰に費やした step 数で戻す）は採らない: 所要 step 数が無制限で「元の LR で続けたい」という意図と逆行する。「現在値で保持」は取り消しではないので採らない（§5.4） |
| D6 | 制御経路 | **sample RPC と同形の新モジュール `training_control_rpc.py`**（ファイル RPC、run スコープ、claim-delete）。sample RPC のペイロード流用はしない（schema・上限 3・「1 バッチ 1 件」・stop 時に claim しない `:10255` が全て sample 専用の意味論）。共有プリミティブ（atomic write / read / owns / age sort）は `training_file_rpc.py` に移して両者が import する。config 経由（stop→edit→resume）は**併存**するが主経路ではない（§6） |
| D7 | 延長耐性の暗黙化 | **時間軸の warp**。最初に構築されたときの `total_steps` を名目軸とし、以後の `total_steps` 変化は `(anchor=resume step, new_total)` 事象として state.json に記録、resume 点より前の形は不変、残りの区間を新しい残り step 数に線形写像する。**再構築の引数は変わらず、形は再構築のたびに事象列から再導出される**。設定キーは増やさない。旧挙動（無言の再伸長 §5 の危険）は消える。すでに減衰中の run の扱いは §7.3 |
| D8 | 軸の定義 | 「絶対 step で書かれたものは実軸、`total_steps` 相対で書かれたものは名目軸（warp 後）」。warmup `W`、`lr_decay_steps`、`lr_cycle_steps`、復帰長 `R`、事象の `at` は実軸。**実軸/名目軸（どの時計で読むか）と global_step/scheduler 軸（何を 1 と数えるか）は独立**で、config の step キーには両方が掛かる（§18.6）。`cosine`/`linear`/`polynomial` の進行度、「終端まで」の減衰、「1 サイクル＝全体」は名目軸 |
| D9 | スケジューラ軸と `gradient_accumulation_steps` | スケジューラの位置は**更新境界での advance 数**で数える（保存契約は §17.1）。現状は構築 `T` が `global_step` 単位（`:12557-12561` → `:6011`）、`scheduler.step()` は optimizer step ごと（`:15933`, `:15987-15992`）、resume は `last_epoch = global_step`（`:3901`）で、`gradient_accumulation_steps > 1` では三者が一致しない（§1-3）。`T_sched = floor(T / gas)`、**`W` も同じ規則で `floor(W / gas)`**（片方だけ変換すると `W/T` が gas 倍ずれる。§18.5）、fast-forward は保存した scheduler_step に統一する（旧状態のみ `global_step // gas` 推定）。P0 で直す（先に直さないとタイムラインの `at` が壊れる） |
| D10 | 床の一般化 | `lr_floor_ratio` を**レジストリの全スケジュールに適用**（`m = warmup(step) · (F + (1−F)·shape)`）。`constant` の床は効かないが、warmup は §4.2 の挙動変更対象。既定値は `param_defaults.py:2193` の `0.25` のまま。YAML には**無条件に書く**（`rewarmup_on_optimizer_reset` と同じ、`training_config.py:198-202`）。**YAML にキーが無い旧 run は `plateau_cosine_floor` のみ 0.25、他は 0.0** と読む（polynomial を除く旧床の保存（§4.2）。§12.2） |
| D11 | `plateau_cosine_floor` | 名前は残し、レジストリでは **`wsd` の別名**（`decay_start = round(ratio·T_sched)`、長さ「終端まで」、形 `cosine`）。同一 `T` で現行実装と bit 同一の乗数（P0 の回帰条件）。別名化により D7 の延長耐性と §6 のコマンドが自動で効く |
| D12 | 新スケジュール名 | `wsd`（設定で減衰開始 step を持てる WSD）、`rex`（`wsd` の別名: 減衰開始 = warmup 終了、形 `rex`）。`cosine_with_restarts` は**名前を保ち in-house 実装に置換**（絶対サイクル長 `lr_cycle_steps`、annealing `lr_cycle_peak_decay`、床）。現状この名前は `num_cycles` を渡していないため（`:6007-6012`、diffusers 既定 `1`、`optimization.py:294`）**単一 cosine と同一**であり、`lr_cycle_steps = 0`（既定）を「1 サイクル＝全体」と定義すれば既存 run の形は変わらない |
| D13 | 減衰オーバーレイの適用範囲 | 「今から減衰」「取り消し」は **レジストリの全スケジュールに対する共通オーバーレイ**（base curve の上に乗る）。`wsd` は「config で開始 step を書ける」点だけが特別。`constant` + 「今から減衰」は手動 WSD そのものになる |
| D14 | REX | **「1 の減衰率を強める」では出ない。** cosine 系はどんな指数を掛けても減衰開始の傾きが 0 だが REX は `−1/2`（§8）。ただし **1 の減衰形を選択式にすれば 1 の特殊形として出る**（`decay_shape = rex`、開始 = warmup 終了、床 0）。よって別型は作らず `lr_decay_shape ∈ {cosine, linear, rex}` と別名 `rex` で提供する |
| D15 | restart + 1 の合成 | restart 曲線を base curve とし、D13 のオーバーレイで「今から床へ減衰（restart 停止）」「取り消しで restart 曲線へ復帰」を得る。annealing はサイクル `i` のピーク `peak_i = decay^i`、床は D10 と同じ `F` |
| D16 | グループ別スケジュール | **既定オフ**。`lr_group_schedules`（コンポーネント名 → スケジュール**名**のみ。数値パラメータは run 共通）。`LambdaLR` の `lr_lambdas` リストで実現し、`lr_utils.py:173` の長さ契約と `:3895-3900` の zip はそのまま満たす。**fused optimizer groups とは併用拒否**（`create_optimizer_groups` がコンポーネント境界を捨てる `optimizers/fused_optimizer_groups.py:222-256`） |
| D17 | LLRD の置き場 | スケジューラではなく**グループ構築の後処理**。`setup_optimizer` がアダプタのグループ（契約 `base_trainer.py:3553`）を受けた直後に、`ArchHandler.depth_blocks(trainer)`（新フック、block swap の列挙 `arch/base_arch.py:666` と同じ出所）から作る `id(param) → depth` で各グループを深さ別に分割し `lr · d^(n−1−depth)` を書く。キーは `lr_layer_decay`（float、`1.0` = オフ）1 つ。**プリセット値は本書では決めない**（未測定の数値を書かない）。fused optimizer groups とは併用拒否 |
| D18 | 語彙統一 | 正典は `lr_schedules.LR_SCHEDULER_NAMES`。拡散側 Pydantic（`routes.py:15308`、素の `str`）に validator を付け、openapi（`:19772`）に `enum` を付ける。VAE 側の `VALID_LR_SCHEDULERS`（`vae_config.py:87-88`）は VAE がレジストリ builder を採用するフェーズ（P7）で同じタプルを import する。tagger は対象外（§15） |
| D19 | 観測 | 減衰状態は `log_extra_metric("lr_decay_state", …)`（`:16954`）で毎 step emit（0 = base, 1 = decaying, 2 = floor, 3 = recovering）。永続記録は state.json（D4）。表示用に `<output_dir>/.lr_schedule.json` を事象適用のたびに書く（読み取り専用、resume の根拠にしない） |
| D20 | プレビュー | スケジュール曲線のプレビューは**サーバ側**エンドポイント（`GET /training/lr-schedule/preview`）が同じ Python 実装で標本点を返す。TS に式を二重実装しない |

---

## 1. 前提の訂正

1. **`cosine_with_restarts` は今日、`cosine` と同じ曲線である。** 両構築点とも `num_cycles` を渡さず
   （`base_trainer.py:6007-6012`, `:6516-6521`）、diffusers の既定は `1`（`optimization.py:294`）。
   hard-restart の式 `0.5·(1+cos(π·((num_cycles·p) mod 1)))`（`:217`）は `num_cycles = 1` で
   `cosine` の式（`:182`、`num_cycles = 0.5` 既定）と一致する。UI 未露出（`TrainingConfig.tsx:4063-4068`）
   なので実害は無いが、「露出すれば restart が使える」わけではない。⇒ D12。
2. **`lr_utils.py:44-53` の不変条件「作るスケジューラは全て `LambdaLR`」は現在偽。** ReLoRA が
   `_LRScheduler` 直系（`relora_scheduler.py:33`）で、`lr_lambdas` を持たず独自カウンタ
   `_relora_step`（`:73`, `:149`）で動く。3 つの汎用機構がそれぞれ劣化している: fast-forward は
   `for _ in range(global_step): scheduler.step()`（`base_trainer.py:3908-3911`）、re-warmup は
   `skipped` 扱い（`:5536-5541`）、再表明は乗数 1.0（`lr_utils.py:171-179`）。⇒ D2。
3. **`gradient_accumulation_steps > 1` でスケジューラ軸がずれている。** `global_step` はマイクロステップ
   ごとに増え（`:15716`）、optimizer と scheduler は `global_step % gas == 0` のときだけ進む
   （`:15933`, `:15941-15949`, `:15987-15992`）。一方、構築時の `num_training_steps` は
   `actual_total_steps`（`global_step` 単位、`:12557-12561` → `:6011`）、resume の位置決めは
   `last_epoch = global_step`（`:3901`）。同一 run 内で scheduler は `T/gas` 回しか進まないのに、
   曲線は `T` で定義され、resume すると位置が `gas` 倍に跳ぶ。本書のタイムラインは `at` を
   scheduler 軸で記録するので、この不一致を先に解消しないと事象の位置が session ごとに変わる。⇒ D9。
4. **`total_steps` の変更は完全に無言**（ブリーフ §5）。前回値はどこにも無く（`:3793-3811` の
   state.json に無い、DB 行はコールバック `train_runner.py:2789-2792` で上書き）、resume 時は
   新しい `T` で作り直して古い `global_step` に fast-forward する。10k→20k で
   `plateau_cosine_floor` の `D` が 8500→17000 に動き、減衰中の run がプラトーへ戻る。
   隣接する警告（dataset 構造変化 `:13019-13024`、MNT 変化 `:13070-13080`）はこの契機を覆わない。⇒ D7。
5. **warmup はスケジュール固有のノブではない。** `self.optimizer_warmup_steps`（`:1290`, `:1579`）が
   diffusers の `num_warmup_steps`（`:6010`）、`plateau_cosine_floor` の `W`（`:6567`）、ReLoRA の
   `initial_warmup_steps`（`relora_trainer.py:160`）、resume 時 re-warmup のランプ長（`:5523`）を兼ねる。
   本書は新しい warmup 意味論を導入せず、復帰長 `R` にも同じ `W` を使う（D5）。
6. **グループが持てるのは基準 LR だけ**で、スケジュールはその上の単一乗数（ブリーフ §7）。
   全構築点が optimizer 全体に 1 つの lambda/型を渡し、`LambdaLR` が全グループへ複製する。
   グループ名は `g.get("name") or f"group{i}"`（`:5229`）だが、アダプタは `name` を書かない
   （`adapters/base_adapter.py:480`、`adapters/zimage_adapter.py:254, 259`）。⇒ §10, §11 の前提。

---

## 2. 現状の構築と再開経路（要約）

| 箇所 | 何をするか | 出典 |
|---|---|---|
| メイン経路 | `plateau_cosine_floor` なら in-house、他は `diffusers.get_scheduler(name, warmup, T)` | `base_trainer.py:6001-6012` |
| fused optimizer groups | 同じ分岐を optimizer ごとに N 回。`self.lr_scheduler = lr_schedulers[0]`、全体は `self.lr_schedulers` | `:6504-6528` |
| fused backward pass | スケジューラを作らずメイン経路の 1 つを使う | `:6077` |
| ReLoRA | 親の構築を走らせた**あと破棄**して `CosineWithMultipleWarmups` に差し替え、`lr_scheduler` 設定を無視 | `relora_trainer.py:147-181` |
| VAE トレーナー | diffusers `get_scheduler`、失敗時 `None`（定数 LR）。唯一 `lr_scheduler.pt` を保存・復元 | `vae/vae_trainer.py:663-674`, `:1032-1033`, `:1802-1813` |
| tagger | `SequentialLR(LinearLR, CosineAnnealingLR)` を手組み、`lr_scheduler` 設定なし | `core/tagger/tagger_trainer.py:1001-1009` |
| resume 順序 | fast-forward → optimizer 復元 → （失敗時）re-warmup → 再表明 → EMA | `:12884-12911`, `:12958-12981` |
| fast-forward | `LambdaLR` なら `base_lr · λ(global_step)` を代入して `last_epoch`/`_step_count`/`_last_lr` を書く | `:3888-3906` |
| 保存される状態 | `global_step, epoch, batch_idx, multi_noise_timesteps, random_state, dataset_fingerprint, batches_per_epoch, crop_plan_fingerprint` のみ | `:3793-3811` |
| `scheduler.step()` | optimizer step ごとに 1 回、fused では N 個を一括 | `:15987-15992` |
| 観測 | `extra_metrics["lr"]` = 適用値（`param_groups[0]["lr"]`）、TensorBoard/DB = `get_last_lr()[0]` | `:15746-15748`, `:15728`, `:15823`, `:15902` |

`ops/*_ops.py` / `controlnet_trainer.py` の `get_scheduler` は拡散ノイズスケジューラであり LR ではない
（`ops/sd_sdxl_ops.py:1129,1179,1220`、`controlnet_trainer.py:842,884,1057,1110`）。本書は触らない。

---

## 3. 集約先（D1）

### 3.1 モジュール構成

```
core/training/
  lr_schedules.py        [新規] スケジュール定義・レジストリ・構築・タイムライン・直列化（§4, §5, §7）
  lr_utils.py            変更なし（再表明）。docstring :44-53 の不変条件を「レジストリが保証する」と書き換え
  training_file_rpc.py   [新規] training_sample_rpc.py から _atomic_write_json / _read_json /
                         _sorted_by_age / owns を移動（純移動、挙動不変）
  training_control_rpc.py[新規] 実行時コマンド（§6）
  base_trainer.py        setup_optimizer / _setup_fused_optimizer_groups の構築分岐を
                         build_lr_scheduler() 呼び出しに置換、_build_plateau_cosine_floor_scheduler 削除、
                         save/load_training_state に lr_schedule_events、resume 順序に timeline.load、
                         ポーリング seam、MNT 再計算のフック
  relora_trainer.py      CosineWithMultipleWarmups の差し替えを廃し、merge を timeline 事象に
  relora_scheduler.py    削除（P4）
  arch/base_arch.py      ArchHandler.depth_blocks(trainer)（§11）
api/arch_capabilities.py TRAINING_FEATURE_PARAMS["lr_layer_decay"], 非対応 arch の登録（§11.4）
api/param_defaults.py    新キー（§12.1）
api/routes.py            validator、/training/runs/{id}/lr-schedule、/training/lr-schedule/preview
openapi.yaml             enum、新スキーマ、新パス
```

### 3.2 `lr_schedules.py` の公開 API

```python
LR_SCHEDULER_NAMES: tuple[str, ...]        # 正典の語彙（D18）

@dataclass(frozen=True)
class ScheduleSpec:                        # config から解決した不変の入力
    name: str
    warmup_steps: int                      # W_sched（optimizer step 単位、D9）
    total_steps: int                       # T_sched（optimizer step 単位、D9）
    floor_ratio: float                     # F（D10）
    decay_start_step: int                  # wsd: 0 = 手動
    decay_steps: int                       # 0 = 終端まで
    decay_shape: str                       # cosine | linear | rex
    cycle_steps: int                       # cosine_with_restarts: 0 = 1 サイクル＝全体
    cycle_peak_decay: float                # 1.0 = annealing なし
    relora_restart_warmup_steps: int       # relora のみ
    legacy_decay_start_ratio: float        # plateau_cosine_floor 別名の解決にだけ使う

class ScheduleTimeline:                    # 実行時状態（§5）。seam でのみ変更
    events: list[dict]
    def load(self, events) -> None
    def dump(self, upto_step: int) -> list[dict]
    def add(self, kind: str, at: int, **payload) -> str   # 戻り値は結果コード
    def clock(self, step: int) -> float                    # 実軸 → 名目軸（§7）
    def state_at(self, step: int) -> tuple[int, float]     # (状態コード, 復帰/減衰の開始乗数)

def to_scheduler_axis(steps: int, interval: int) -> int   # floor(steps/gas)。W も T も config の step キーもこれを通る
def resolve_spec(config: dict, *, warmup_steps: int, total_steps: int,
                 name: str, advance_interval: int = 1) -> ScheduleSpec
def make_lambda(spec: ScheduleSpec, timeline: ScheduleTimeline) -> Callable[[int], float]
def build_lr_scheduler(optimizer, spec, timeline,
                       group_names: Sequence[str] | None = None,
                       group_schedules: Mapping[str, str] | None = None) -> LambdaLR
def sample_curve(spec, timeline, n_points: int) -> list[tuple[int, float]]   # プレビュー（D20）
```

`build_lr_scheduler` は常に `LambdaLR(optimizer, lr_lambda=<list>)` を返す。`group_schedules` が
無ければ全グループ同一の lambda を N 個並べる（`LambdaLR` の自然な形。`:3895-3900` の zip、
`lr_utils.py:173` の `len(lambdas) == n_groups` を満たす）。

### 3.3 移行

| 現行 | 移行後 |
|---|---|
| `:6001-6012` | `spec = resolve_spec(self.config, warmup_steps=self.optimizer_warmup_steps, total_steps=T_sched, name=lr_scheduler_type)`; `self.lr_timeline = ScheduleTimeline()`; `self.lr_scheduler = build_lr_scheduler(self.optimizer, spec, self.lr_timeline, names)` |
| `:6506-6522` | 同じ `spec` と**同じ `self.lr_timeline` オブジェクト**で optimizer ごとに build。N 個が 1 つのタイムラインを共有するので非同期化しない（ブリーフ §9 制約 8） |
| `_build_plateau_cosine_floor_scheduler` `:6547-6588` | 削除。`plateau_cosine_floor` は `resolve_spec` 内で `wsd` に正規化（D11） |
| `relora_trainer.py:147-181` の差し替え | 削除。`lr_scheduler_type` を `"relora"` に正規化して親の構築に任せる（`lr_scheduler` 設定は従来どおり無視されるが、無視することを `resolve_spec` が 1 行ログに出す） |
| `relora_trainer.py:325-338` `_add_lr_restart` | `self.lr_timeline.add("restart", at=last_epoch)` |
| `relora_trainer.py:395-421` `_restore_scheduler_restarts` | 事象が state.json から戻るので原則不要。事象キーの無い旧チェックポイントに対する後方互換としてのみ残す（`steps` 単位の再計算 `:408`） |
| `vae_trainer.py:663-674` | P7 で `build_lr_scheduler` に置換。`lr_scheduler.pt` の方式を維持し、timeline の保存と constant + warmup 拒否の撤去を行う（`LambdaLR.state_dict()` は `last_epoch`/`base_lrs` を持ち、通常の関数 closure の内容は保存されない。タイムラインは別途保存・復元が必要（§17）） |
| `tagger_trainer.py:1001-1009` | 対象外（§15） |

`lr_utils.py` に構築を置かない理由: docstring `:1-9` が「再表明のためのモジュール、両トレーナーが import し
何にも依存しない」と役割を限定しており、`BaseTrainer` と `VaeTrainer` の両方から import される
（`:3-4`）。構築・タイムライン・直列化を足すとファイルの責務が 2 つになる。同じ理由で
`lr_schedules.py` も `core.training` 内の他モジュールに依存しない（`torch.optim.lr_scheduler.LambdaLR` のみ）。

---

## 4. スケジュール定義（純関数）

記法: `s` = scheduler 軸の step（optimizer step、D9）、`W` = warmup、`T` = `T_sched`、`F` = 床、
`τ(s)` = 名目軸への写像（§7、事象が無ければ恒等）、`T` は最初の total_steps 事象の名目終端、`p = clamp((τ(s) − τ(W)) / max(1, T − τ(W)), 0, 1)` を名目進行度とする。

### 4.1 共通形

```
m(s) = ramp(s) · ( F + (1 − F) · shape(s) )          # D10
ramp(s) = s / W   (W > 0 かつ s < W)、それ以外 1
```

warmup は床の**外**に掛ける（0 から立ち上がる）。これは現行 `plateau_cosine_floor` の
「warmup 0→1、減衰は床まで」（`:6575-6584`）と同じ配置であり、`constant`（`shape ≡ 1`）では
`m = ramp` となる。現行 `constant_with_warmup` と一致し、`constant` は §4.2 の変更となる。warmup 中は shape = 1 を先に返し、減衰式を評価しない。

### 4.2 base curve `shape(s)` の表

| 名前 | `shape(s)`（`s ≥ W`） | 現行との関係 |
|---|---|---|
| `constant` | `1` | `optimization.py:52`（`lambda _: 1`）。`W > 0` でも warmup しない現行 diffusers の挙動（`:323-324`、`constant` 分岐は `num_warmup_steps` を受け取らない）は**本書で変える**: `constant` + `W > 0` は warmup する。VAE 側がこの組み合わせを拒否している理由（`vae_config.py:819-824`「YAML は warmup と言うのに走らない」）が消えるので、P7 でその拒否も外す |
| `constant_with_warmup` | `1` | 上と同一になる。名前は互換のため残す |
| `linear` | `max(0, 1 − p)` | `optimization.py:143-146` と同式（`F = 0` のとき） |
| `cosine` | `max(0, ½(1 + cos(π p)))` | `:178-182`（`num_cycles = 0.5`）と同式 |
| `polynomial` | `(1 − p)^1` | `:222-271`。`power` は渡されていない（`:6007-6012`）ので `1.0` 固定、`lr_end` は `F` に置き換わる（`lr_end / lr_init` の役割が `F`） |
| `cosine_with_restarts` | サイクル `i`（開始 `c_i`、長さ `C`）で `peak_i · ½(1 + cos(π q))`、`q = (s − c_i)/C`、`peak_i = decay^i` | D12。`C = lr_cycle_steps`（実軸）、`0` なら `C = T − W`（名目軸で 1 サイクル）→ `cosine` と一致 |
| `wsd` | `s < D` で `1`、`D ≤ s < D + L` で `k(q)`、以後 `0`。`q = (s − D)/L`、`k` は `decay_shape` | `D = lr_decay_start_step`（`0` = 手動、config では減衰しない）。`L = lr_decay_steps`、`0` なら終端まで（名目軸） |
| `plateau_cosine_floor` | `wsd` の別名: `D = clamp(round(ratio·T), W, T)`、`L = T − D`、`k = cosine` | `:6571-6573` と同じクランプ。同一 `T` で `:6575-6584` と bit 同一（P0 回帰条件） |
| `rex` | `wsd` の別名: `D = W`、`L = T − W`、`k = rex`、床は `F` | §8 |
| `relora` | サイクル `i`（開始 `r_i` = 事象 `restart` の `at`、`r_0 = 0`）で `r_i ≤ s < r_i + W_r` は `(s − r_i)/W_r`、以後 cosine でその時点の `T` まで（未来 restart を参照しない、§17.3） | `relora_scheduler.py:90-139` と同式。`W_r = restart_warmup_steps`（初回は `W`）。`min_lr_ratio = 0.0` 固定（`relora_trainer.py:162, 175`）は `F` に置き換わる（旧 YAML は D10 の規則で `0.0`） |

減衰形 `k(q)`（`q ∈ [0,1]`、`k(0) = 1`、`k(1) = 0`）:

| `decay_shape` | `k(q)` |
|---|---|
| `cosine` | `½(1 + cos(π q))` |
| `linear` | `1 − q` |
| `rex` | `(1 − q) / (1 − q/2)` |

`shape ∈ [0, 1]` を全て満たすので `m ∈ [0, 1]`、`s ≥ T` では終端値を保持する（現行 `plateau_cosine_floor` の
「床を維持」`:6584` と同じ。`cosine`/`linear` は `F` を保持する。旧挙動の「0 を保持」は `F = 0` のとき）。

### 4.3 純粋性の契約

`make_lambda` が返す関数は `(s, timeline.events)` の関数であり、呼び出しで何も書かない。
`timeline.events` は §5.2 の seam でしか変わらない。よって

- `_fast_forward_one_lr_scheduler`（`:3894-3902`）の `λ(global_step)` 評価、
- `reassert_config_lr` の `λ(last_epoch)` 評価（`lr_utils.py:174`）、
- `_compose_warmup_lambda` の合成（`:5560-5574`）、

はどの順で何度呼ばれても同じ値を返す。この契約を `backend/tests/lr_schedules_test.py` が
「同一事象列に対し全 step を昇順・降順・乱順で評価して一致」で固定する。

---

## 5. 実行時タイムライン（要求 1 の状態機械）

### 5.1 事象

```json
{"kind": "total_steps", "at": 0,    "value": 10000}            // §7。最初の構築で必ず 1 件
{"kind": "total_steps", "at": 9000, "value": 20000}            // resume で T が変わった
{"kind": "decay",       "at": 9137, "length": null, "shape": "cosine"}   // 今から減衰（length null = 終端まで）
{"kind": "cancel",      "at": 9800}                            // 取り消し
{"kind": "restart",     "at": 500}                             // ReLoRA merge
```

`at` は scheduler 軸（D9）。事象列は `(at, seq)` 昇順に保ち、同じ `at` の事象も許可する
（単調な `seq` と request_id で順序と再送の冪等性を保証する）。`decay` の `length`/`shape` は事象に**焼き込む**
（あとで config が変わっても、開始済みの減衰の形は変わらない）。

### 5.2 書き換えてよい seam

| seam | 何を書くか | 位置 |
|---|---|---|
| (a) 構築 | `total_steps(at=0)`（事象列が空のときだけ） | `setup_optimizer` 内、build 直前 |
| (b) resume | `timeline.load(state["lr_schedule_events"])`、次いで `T` 比較で `total_steps(at=resume_step)`（§7） | `load_training_state` の直後、**`_fast_forward_lr_schedulers` より前**（`:12885` / `:12959` の直前）。fast-forward が lambda を評価するので、その前に事象が揃っていなければならない |
| (c) コマンド | `decay` / `cancel` | バッチ先頭のポーリング（§6.3） |
| (d) ReLoRA merge | `restart` | `_add_lr_restart`（`relora_trainer.py:325-338`） |

(b) が `setup_optimizer`（`:12557`）より後になる点が、ブリーフ §9 制約 3（構築時に resume step を知る
手段が無い）への答えである: **構築は暫定 total_steps(at=0) で行い、resume で保存済み事象列に置換する**。lambda は
タイムライン**オブジェクト**を閉じ込めているので、注入後の評価は注入後の値を返す。

### 5.3 状態機械

状態は `BASE` / `DECAYING` / `FLOOR` / `RECOVERING`。`timeline.state_at(s)` が `s` 以下の事象を
走査して決める（純関数）。

| 現状態 | 事象 | 遷移 | 結果コード |
|---|---|---|---|
| `BASE`（config の減衰が未到来） | `decay` | `DECAYING`（開始乗数 `m_start = m_base(at)`） | `applied` |
| `BASE`（config の減衰が未到来） | `cancel` | `BASE`、ただし config の `D` を**無効化**（以後 base curve は減衰しない） | `disarmed_scheduled_decay` |
| `BASE`（config の減衰なし） | `cancel` | 変化なし | `ignored_no_active_decay` |
| `DECAYING` / `FLOOR` | `decay` | 変化なし | `ignored_already_decaying` |
| `DECAYING` / `FLOOR` | `cancel` | `RECOVERING`（開始乗数 `m_c = m(at)`、長さ `R = W`） | `applied` |
| `RECOVERING` | `decay` | `DECAYING`（`m_start = m(at)`、復帰途中の値から） | `applied` |
| `RECOVERING` | `cancel` | 変化なし | `ignored_already_recovering` |
| `DECAYING` かつ `q ≥ 1` | （時間経過） | `FLOOR` | — |
| `RECOVERING` かつ `s ≥ at + R` | （時間経過） | `BASE` | — |

乗数:

```
DECAYING:   m(s) = F + (m_start − F) · k(q)   # start は warmup 終了以後のみ受理
FLOOR:      m(s) = F
RECOVERING: m(s) = m_c + (m_base(s) − m_c) · (s − at) / R        (R > 0)
            m(s) = m_base(s)                                      (R = 0)
BASE:       m(s) = m_base(s)                                      # §4 の曲線（config の D を含む）
```

`m_base` は `wsd` なら（config 減衰が無効化されていなければ）config の `D` を含む曲線、
`cosine_with_restarts` なら restart 曲線（D15）、`constant` なら `ramp`。**復帰先は常に base curve**
なので、`cosine` の途中で減衰→取り消しをしても cosine の「その時点の値」へ戻る（元の LR ＝ config の
スケジュールが指す LR、と定義する）。`wsd` では cancel 時に config 減衰も無効化する。warmup 後の `wsd`/`constant` では base curve が `1` なので所有者の言う
「元の LR」に一致する。

`decay` の `length = null` は `q = clamp((τ(s) − τ(at))/(T_nominal − τ(at)), 0, 1)`、明示長 L は `q = clamp((s − at)/L, 0, 1)`（D8）。分母が 0 以下なら開始要求を拒否する。warmup 中の start は `rejected_during_warmup` とし、ゼロ除算と「減衰なのに LR が増える」挙動を避ける。cancel には復帰長 R を焼き込む。開始乗数と状態はグループごとに評価する。

### 5.4 取り消しの形（D5）

線形復帰を選ぶ理由:

- 本コードベースで LR が**上がる**経路は 3 つあり、全て線形ランプである: 初期 warmup（`:6577`、
  `optimization.py:145`）、ReLoRA の restart warmup（`relora_scheduler.py:124-126`）、resume 時の
  re-warmup（`:5560-5574`）。取り消しに第 4 の形を持ち込まない。
- ランプ長を `W` に固定するのは、`W` が既に「この run が LR を上げるときに使う長さ」として
  re-warmup（`:5523`）に流用されているため。新しいノブは増やさない。`W = 0` の run は
  不連続に戻る — その run は元々 warmup 無しで最大 LR から始めており、config が「ランプ不要」と
  言っている。
- 逆向き補間（減衰に費やした step 数で同じ曲線を逆走）は、取り消しが遅いほど復帰が遅く、
  「元の LR で続ける」意図と逆行するので採らない。

### 5.5 永続化（D4）

- `save_training_state`（`:3793-3811`）に `"lr_schedule_events": self.lr_timeline.dump(upto_step=<scheduler step>)`
  を追加。`dump` は `at <= step` の事象だけを返す。step 9000 のチェックポイントには step 9137 の
  `decay` は入らない。**だから step 9000 から再開すると減衰は無かったことになる** — これは
  `_cleanup_future_metrics(global_step)`（`:13085`）が resume 点より後のメトリクスを消すのと同じ意味論
  であり、意図した挙動である。§6.5 で UI に明示する。
- 読み側は欠落キーを「事象なし」と読む（`batches_per_epoch` の後方互換コメント `:3804-3806` と同じ規則）。
  事象が無い旧チェックポイントの resume は (a) の `total_steps(at=0, value=現在の T)` から始まる。
  つまり**旧チェックポイントの最初の resume では、旧挙動どおり現在の `T` が名目軸になる**（それ以前の
  `T` を知る術が無いため）。以後は D7 が効く。
- `state.json` が無い resume（`load_training_state` が `None`、`:3855`）は同上。
- 表示用の `<output_dir>/.lr_schedule.json`（D19）は事象適用のたびに atomic write。resume の根拠にしない。

### 5.6 fused optimizer groups / re-warmup / 再表明との整合

- N 個の `LambdaLR` が 1 つの `ScheduleTimeline` を参照する（§3.3）。`.step()` 以外で進む状態が無いので、
  N 個は常に同じ値を返す（ブリーフ §9 制約 8）。
- `_compose_warmup_lambda`（`:5560-5574`）は我々の lambda を `inner` として包むだけなので、そのまま動く。
- `reassert_config_lr`（`lr_utils.py:171-179`）は `len(lambdas) == n_groups` で乗数を評価する。
  `build_lr_scheduler` は常にリストを渡すので条件は常に真になる。
- コマンド適用後の即時反映は `_fast_forward_one_lr_scheduler(scheduler, scheduler.last_epoch)`
  （`:3889-3906`）を**位置を変えずに**呼ぶことで行う（`base_lr · λ(last_epoch)` を各グループへ書く）。
  ループは `optimizer.step()` → `scheduler.step()` の順（`:15987-15992`）なので、書き換えなければ
  次の optimizer step は前回 `scheduler.step()` が書いた旧値で走る。fused groups では hook が backward
  中に `optimizers[i].step()` を呼ぶ（`optimizers/fused_optimizer_groups.py:106-109`）ため、forward の
  前に書き換える必要がある（§6.3 のポーリング位置の根拠）。

---

## 6. 制御経路（D6）

### 6.1 なぜ sample RPC を流用しないか

sample RPC（`training_sample_rpc.py`）は機構としては汎用（任意 JSON、run スコープの `owns` `:109`、
atomic write `:83-86`、claim-delete `:183-203`）だが、次が全て sample 固有である:

- 上限 `MAX_PENDING_REQUESTS = 3`（`:38`）と「1 バッチ 1 件」（`base_trainer.py:16167-16172`）— 生成を
  伴うから。コマンドは軽く、到着順の状態遷移を全件評価する（同一 step の減衰→取り消し→減衰も順序どおり評価する）。
- stop 要求中は claim しない（`:10255`）— 生成が stop を遅らせるから。コマンドは stop 中でも適用してよい
  （直後の保存に事象が乗る）。
- claim 位置がサンプリングブロック内（`:16167-16172`）— 減衰は forward 前に効かせたい（§5.6）。
- エンドポイント `POST /training/runs/{id}/sample`（`openapi.yaml:6033`）の 202/400/409/429 の意味論と
  `TrainingSampleQueueResponse`。

ペイロードに `kind` を足して分岐させると、上の 4 点全てに `if kind == ...` が入る。別モジュールにして
プリミティブだけ共有する。

### 6.2 `training_control_rpc.py`

- 要求: `<output_dir>/.control_request_<id>.json`、`{"request_id", "run_id", "command": "start_decay"|"cancel_decay", "queued_at"}`。
- 結果: `.control_result_<id>.json`、`{"request_id", "run_id", "command", "result": <§5.3 の結果コード>, "at": <scheduler step>, "global_step", "completed_at"}`。
- `claim_all(output_dir, run_id)`: 自 run の要求を**全件**古い順に claim-delete して返す（coalesce は
  trainer 側で事象列に順に `add` することで自然に起こる。`ignored_*` はそれぞれ結果に残る）。
- `clear_all(output_dir)`: spawn 前に呼ぶ（`training_process.py:256-257` の sample 用と並べる）。
  stop→resume を跨いだ「減衰しろ」は適用しない。適用済みの事象は state.json に居る。
- 上限: pending 20 件で 429（コマンドは軽いが無制限にはしない）。

### 6.3 trainer 側

- ポーリング位置: バッチループの**先頭**（forward の前）、`.stop_training` 検査の隣。sample の claim
  （`:16167`）とは別。1 バッチにつき 1 回、`claim_all` → 各コマンドを `timeline.add(kind, at=last_epoch, ...)`
  → 1 件でも `applied`/`disarmed_*` なら全スケジューラに §5.6 の即時反映 → 各結果を書く →
  `.lr_schedule.json` を書く → `emit_training_event("info", code="lr_schedule_command")`。
- `at` は `all_lr_schedulers(self)[0].last_epoch`（`:281`）。保存した scheduler_step を根拠とする（§17）。単純な global_step の除算は旧状態の移行に限る。
- 例外を投げない（sample の `_claim_on_demand_sample_request` `:10248-10260` と同じ方針）。

### 6.4 API（openapi 先行）

| メソッド/パス | 内容 | 応答 |
|---|---|---|
| `POST /training/runs/{id}/lr-schedule` | body `{"command": "start_decay" \| "cancel_decay"}`。要求ファイルを書いて 202 | 202 `LrScheduleCommandAccepted`、404、409（run が実行中でない: `status not in ("running","starting")`）、429（pending 上限）、500 |
| `GET /training/runs/{id}/lr-schedule` | `.lr_schedule.json`（あれば）＋ pending ＋ 最近の結果 | 200 `LrScheduleStatus`（`state`, `events`, `nominal_total_steps`, `effective_total_steps`, `pending`, `results`）。停止中の run でも読める（ファイル） |
| `GET /training/lr-schedule/preview` | query に `lr_scheduler`, `total_steps`, `lr_warmup_steps`, 各キー。`sample_curve()` の標本 | 200 `LrSchedulePreview`（`points: [[step, multiplier], ...]`、`n_points ≤ 512`） |

`PATCH /training/runs/{id}/config` が実行中を拒否する（`routes.py:17323-17339`）のは変えない。
config 経由の減衰開始（`lr_decay_start_step` の編集）は stop→edit→resume で、`wsd` の base curve の
`D` として効く。実行中に効かせたいならコマンドを使う。両者の優先は §5.3 の表（事象が勝つ）。

### 6.5 UI

- 実行中 run のパネルに「Decay now」「Cancel decay」の 2 ボタンと、`GET lr-schedule` の状態表示
  （`state`、減衰開始 step、`effective_total_steps`）。
- 表示文言に「チェックポイント step 以前に戻して再開すると、その後に出したコマンドは無かったことになる」
  を 1 文で書く（§5.5）。主観的表現は書かない。

---

## 7. 延長耐性の暗黙化（D7）

### 7.1 何が暗黙になるか

ユーザーが触るのは今までどおり `total_steps`（create `routes.py:15949-15973`、update `:16641-16657`）
だけ。新しい設定キーは無い。変わるのは resume 時の 3 点:

1. `load_training_state` 直後に、事象列の最後の `total_steps.value` と今回の `T_sched` を比較する。
2. 異なれば `total_steps(at=resume_step, value=T_sched)` を追加し、`emit_training_warning(code="lr_schedule_total_steps_changed")`
   で「step N から先の残り区間を旧 (T_old − N) から新 (T_new − N) に写像した」と 1 行出す。
   現在の**無言**（§1-4）はここで消える。
3. MNT 変化による再計算（`:13039-13064`）は `actual_total_steps` を更新しながらスケジューラには渡していない
   （`:13068`、警告 `:13070-13080`）。ここで `self.lr_timeline.add("total_steps", at=resume_step, value=new_T_sched)`
   を呼び、即時反映（§5.6）する。`:13077-13080` の 3 行の WARNING は不要になるので削除する。

### 7.2 時間軸の写像 `τ`

事象 `total_steps` を `(a_0 = 0, T_0), (a_1, T_1), …, (a_n, T_n)` とする（`a` 昇順）。`i ≥ 1` について

```
τ_i(s) = s                                         (s < a_i)
τ_i(s) = a_i + (s − a_i) · (T_{i−1} − a_i) / (T_i − a_i)   (s ≥ a_i, T_i > a_i)
τ_i(s) = T_{i−1}                                   (s ≥ a_i, T_i ≤ a_i)   # 縮小で既に終端: 終端値を即時保持
τ(s)   = τ_1(τ_2(…τ_n(s)))
```

`a_i < T_{i−1}` かつ `a_i < T_i` の変更では `τ` は連続・単調非減少。合成後は `τ_new(a_i) = τ_old(a_i)` であり、複数回変更後に `τ(a_i) = a_i` とは限らない。名目軸で定義された曲線 `shape(τ(s))` は
**anchor より前では以前と bit 同一**（新しい写像だけが恒等）であり、anchor で連続、`s = T_n` で名目 `T_0` に達する。
例: `T_0 = 10000`、step 9000 で 20000 に延長 → `τ_1(s) = 9000 + (s − 9000)/11`、`τ_1(20000) = 10000`。
さらに 15000 で 30000 に延長 → `τ_2(s) = 15000 + (s − 15000)/3`、`τ_1(τ_2(30000)) = τ_1(20000) = 10000`。

### 7.3 すでに減衰中の run の `total_steps` が変わったら

`plateau_cosine_floor`（= `wsd` 別名、長さ「終端まで」）で step 9000（`D = 8500`）まで走った run を
20000 に延長して再開する場合:

- 旧挙動: `D` が 17000 に動き、**プラトーへ戻る**（§1-4）。
- 新挙動: `D = 8500` は名目軸上にあり `τ(8500) = 8500`（anchor 9000 より前）なので過去は不変。
  `s = 9000` で `τ = 9000`、乗数は連続。残りの減衰 `q = (τ(s) − 8500)/(10000 − 8500)` は実 step
  9000→20000 で `τ` が 9000→10000 に進むので、**残りの減衰が 11000 step に引き伸ばされ、20000 で床に着く**。
  プラトーには戻らない。
- 縮小（20000 → 9500）: `τ` の傾きが 1 より大きくなり、9500 で床に着き以後保持。
- 明示長さの減衰（`lr_decay_steps > 0`、または `decay` 事象の `length` 非 null）は**実軸**なので延長で
  伸びない: 予定どおりの step で床に着き、延長分は床で走る。これは「長さを指定した」意図をそのまま守る。
  `emit_training_warning(code="lr_schedule_extension_on_floor")` で「延長分は床 F で走る」と 1 行出す。
- `constant`: `τ` に依存しないので何も変わらない（警告も出ない）。

### 7.4 保証できないこと

- 旧チェックポイント（事象キー無し）の**最初の** resume は、以前の `T` を知る術が無いので現在の `T` を
  名目軸とする（§5.5）。この 1 回だけは旧挙動と同じ形になる。以後は保護される。
- `total_steps` を変えずに `gradient_accumulation_steps` を変えると `T_sched` が変わる。D9 の帰結であり
  過去の scheduler 軸を再換算してはならない。保存済み scheduler_step を anchor とし、残り更新回数から新終端を求めて warp する（§17）。警告に `gas` の変化を含める。

---

## 8. REX の検証（要求 6、D14）

REX（Chen, Wolfe, Kyrillidis 2021, [原典 §4.1](https://arxiv.org/html/2107.04197#S4.SS1) の式を監査時に確認済み）の乗数は
進行度 `p ∈ [0, 1]` に対して

```
k_rex(p) = (1 − p) / (1 − p/2) = 2(1 − p) / (2 − p)
```

比較対象の cosine 減衰 `k_cos(p) = ½(1 + cos(π p))` と、「減衰率を強める」の最も自然な形である
指数付き cosine `k_cos(p)^n`（`n > 0`）について:

| 量 | `k_cos` | `k_cos^n` | `k_rex` | `k_lin = 1 − p` |
|---|---|---|---|---|
| `k(0)` | 1 | 1 | 1 | 1 |
| `k(½)` | 0.5 | `0.5^n` | 2/3 ≈ 0.667 | 0.5 |
| `k(1)` | 0 | 0 | 0 | 0 |
| `k'(0)` | 0 | 0（`n·k^{n−1}·k'`、`k'(0) = 0`） | −1/2 | −1 |
| `k'(1)` | 0 | 0（`n > 1/2`）/ −π/2（`n = 1/2`）/ −∞（`0 < n < 1/2`） | −2 | −1 |

導出: `k_rex'(p) = −2 / (2 − p)²`。`p = 0` で `−1/2`、`p = 1` で `−2`。

結論:

1. **cosine の減衰率をどう強めても REX にはならない。** cosine 系は減衰開始点で傾き 0（水平に入る）、
   REX は有限の負の傾きで入り、終端で cosine の 0 に対し `−2` で落ちる。中点では REX の方が高い
   （0.667 対 0.5）。「減衰率を強める」は中盤では逆方向、終盤でだけ同方向である。
2. **形を選択式にすれば、REX は 1（WSD）の特殊形として出る。** `wsd` で `D = W`（プラトー無し）、
   `L = T − W`、`k = rex`、`F = 0` とすれば原典の REX である。原典に無いプラトーと床は、そのまま
   `D > W`、`F > 0` として一般化される。
3. よって REX を別型として実装しない。`lr_decay_shape ∈ {cosine, linear, rex}` を `wsd` のパラメータに
   し、`rex` という名前は `wsd` の別名として登録する（`resolve_spec` が `D = W, L = T − W, k = rex` に
   正規化。床は `F` をそのまま使う）。同じ機構でコマンドによる開始（`constant` + `start_decay` に
   `shape = rex` を持たせる: `decay` 事象の `shape` は config の `lr_decay_shape` を焼き込む）も得られる。

---

## 9. restart 付き cosine（要求 5、D12 / D15）

### 9.1 base curve

`cosine_with_restarts`（名前は diffusers 互換のまま、実装は in-house）:

```
C      = lr_cycle_steps  (> 0: 実軸)  |  0: C = T − W（名目軸、1 サイクル＝全体）
i(s)   = floor((s − W) / C)            サイクル番号
c_i    = W + i·C                       サイクル開始
peak_i = lr_cycle_peak_decay ^ i       annealing（1.0 = 無し）
shape(s) = peak_i · ½(1 + cos(π · (s − c_i)/C))
m(s)   = ramp(s) · (F + (1 − F) · shape(s))
```

- 各サイクルは床 `F` まで下がり、次サイクル開始で `F + (1−F)·peak_i` まで**不連続に**上がる（hard restart。
  diffusers の `cosine_with_restarts` も不連続、`optimization.py:217`）。サイクル先頭の再 warmup は
  持たせない（それは `relora` の形。ReLoRA を使わないなら warmup 無しの hard restart が diffusers 互換）。
- `C = 0` のとき `i ≡ 0`、`peak_0 = 1` なので `cosine` と一致し、現行の `cosine_with_restarts`（§1-1）と
  同じ曲線になる。既存 YAML の resume は形を変えない。
- `lr_cycle_peak_decay < 1` で `peak_i → 0` に向かうが、床 `F` があるので `m ≥ ramp·F`。

### 9.2 1 との合成

D13 のオーバーレイがそのまま効く。`start_decay` は `m_start = m(at)`（サイクル途中の値）から `F` へ
`k` で降り、以後 restart しない（`FLOOR` 保持）。`cancel_decay` は `RECOVERING` で restart 曲線の
「その時点の値」へ `R` step で戻る。restart 曲線が不連続な区間（サイクル境界）を復帰中に跨ぐと
`m_base(s)` が跳ぶので復帰値も跳ぶ — base curve の不連続をそのまま踏襲する（隠さない）。

### 9.3 UI 露出

`TrainingConfig.tsx:4063-4068` の `<option>` に `cosine_with_restarts` を追加し、選択時に
`lr_cycle_steps` / `lr_cycle_peak_decay` を表示する（`plateau_cosine_floor` の条件表示 `:4090-4118` と同じ形）。

---

## 10. グループ別スケジュール（要求 3、D16。既定オフ）

### 10.1 仕様

- キー `lr_group_schedules`: `null`（既定 = オフ）または `{ "<component>": "<schedule name>" }`。
  `<component>` は `LORA_COMPONENT_ORDER` の名前（`unet`, `te1`, `te2`, `ve`, …。要検証: 実定義は
  `adapters/base_adapter.py` 上部の `LORA_COMPONENT_ORDER`）。指定の無いコンポーネントは run の
  `lr_scheduler`。**数値パラメータ（`W`, `F`, `lr_decay_*`, `lr_cycle_*`）は run 共通。** タイムラインも
  1 つ（コマンドは全グループに効く）。
- なぜ名前だけか: 所有者が「複雑だから既定オフ」と言った機能に、コンポーネント × パラメータの行列を
  与えると、設定面・YAML・PARAM_KEYS・プリセットの全てが行列になる。「TE は `constant`、backbone は
  `wsd`」のような**種類の違い**だけを許し、量の違いは基準 LR（`unet_lr` 等）で表す。

### 10.2 単一スケジュール前提 6 箇所との整合

| 箇所 | 現状 | 対応 |
|---|---|---|
| `:6588` 単一 lambda | `LambdaLR(optimizer, lr_lambda=fn)` | `build_lr_scheduler` が `group_names` と `group_schedules` からグループごとの lambda リストを作る。名前の無いグループは run 既定 |
| `:6007` / `:6516` 単一型 | 同一 `name` | グループごとの名前で resolve_spec を呼び、alias の開始軸・終端種別も再解決する（§17.3） |
| `lr_utils.py:173-179` | `len(lambdas) == n_groups` | `LambdaLR` はリスト長 = グループ数を要求するので常に真 |
| `:3895-3900` zip | `base_lrs` と `lr_lambdas` | 長さが一致するので不変 |
| `:5542-5544` re-warmup | 全 lambda に同一 anchor/warmup | グループ別でも「resume 時の再 warmup は全員同じ」で正しい。不変 |
| `:15748` / `:15728` | `param_groups[0]` / `get_last_lr()[0]` | 不変。グループ別の系列は `:15760-15765` が既に emit する |

### 10.3 グループ名の必須化

`_name_configured_groups`（`:5229`）は `g.get("name")` を読むがアダプタは書かない（§1-6）。
`component_param_groups`（`adapters/base_adapter.py:480`）に `"name": component` を足す。フル FT
アダプタ（例 `zimage_adapter.py:254, 259`）も同様に `"name"` を書く（対象アダプタ数は要検証。
無名グループが残る arch では `lr_group_schedules` を `component_lr_resume_unavailable` と同じ
系統の警告 `lr_group_schedules_unnamed_groups` で無視する — 黙って別グループに当てない）。

### 10.4 fused optimizer groups との併用拒否

`create_optimizer_groups` は全パラメータを平坦化してパラメータ**数**で等分し、単一 `learning_rate`
で optimizer を作る（`optimizers/fused_optimizer_groups.py:222-256`）。コンポーネント境界が無いので
グループ別スケジュールは定義できない。`lr_group_schedules` ≠ `null` かつ `num_optimizer_groups > 0` は
`setup_optimizer` で `ValueError`（8-bit optimizer と Block Swap の既存拒否と同じ場所・同じ文体。
CLAUDE.md「Block Swap + Optimizer 互換性」節）。

---

## 11. LLRD（要求 7、D17）

### 11.1 何であって何でないか

LLRD はグループの**基準 LR の係数**であり、スケジュール（乗数の時間関数）ではない。置き場は
`lr_schedules.py` ではなく `setup_optimizer` のグループ後処理と、深さの出所である `ArchHandler`。

### 11.2 機構

1. `ArchHandler.depth_blocks(trainer) -> Sequence[nn.Module] | None`（新フック、`arch/base_arch.py`）。
   forward 順のブロック列を返す。block swap がブロック列を列挙している `setup_block_swap`（`:666`）と
   同じ出所を使う（各 arch の属性名は要検証。double-stream/single-stream を持つ arch は forward 順に連結）。
   `None` = この arch では未定義。
2. `setup_optimizer` がアダプタのグループ列を受けた直後（契約 `base_trainer.py:3553`。挿入行は要検証）に
   `lr_schedules.apply_layer_decay(groups, depth_of, factor)`:
   - `depth_of = {id(p): depth}` を `depth_blocks` から作る（ブロック `j` の `parameters()` 全て → `j`。
     LoRA が注入したモジュールがブロックの子モジュールである限り、その params も同じ辞書に入る — 要検証）。
   - 各グループの params を `depth` でバケットし、`lr_g · factor^(n − 1 − depth)` のグループに分割。
     どのブロックにも属さない params（embedder、final layer、TE など）は `depth = n − 1`（係数 1.0）。
   - 分割後グループの `name` は `f"{name}.d{depth:02d}"`。
3. `_record_configured_group_lrs`（`:5172-5214`）は分割後の各グループの基準 LR をそのまま記録するので、
   resume 再表明（`:5576-5639`）は同じ分割構造なら index で戻る。component 名は name と別に保持し、.dNN 追加後もグループ別スケジュールへ正しく対応させる。深さ・trainable params・LLRD の変更で optimizer のグループ構造が変わる resume は状態復元を拒否して既存の optimizer reset 経路へ送る。

### 11.3 プリセット

キーは `lr_layer_decay`（float、`1.0` = オフ）1 つ。UI は「Layer-wise LR decay」の 1 入力（`1.0` なら
均一、という事実のみを助言文に書く）。**本書は係数の値を定めない。** 「よきに計らう」の値は所有者が
一度決めて `param_defaults.py` に置くか、プリセットに保存する（プリセットは全 LR キーを運ぶ。
`PRESET_EXCLUDED_KEYS` `trainingParams.ts:189-199` に LR キーは無い）。未測定の数値を設計に書かない
リポジトリ規則に従う。

### 11.4 制約と登録

- **fused optimizer groups と併用拒否**（§10.4 と同じ理由。`component_lr_flattened` `:5291-5301` が
  出る状況ではそもそも係数が消える）。
- **capability**: `TRAINING_FEATURE_PARAMS["lr_layer_decay"] = ["lr_layer_decay"]`（`api/arch_capabilities.py:254`）。
  `depth_blocks` を実装しない arch は `_add_training_feature_unsupported(arch, "lr_layer_decay", reason)`
  （`:322`）で登録し、UI は `trainingFeatureUnsupportedReason`（`TrainingConfig.tsx:859` と同じ経路）で隠す。
  第 1 波は DiT 系（`_DIT_ARCHS` `:187`）。sd15/sdxl は U-Net の skip 接続で「深さ」の全順序が定義
  できないので unsupported として登録する（理由文は構造の事実のみ）。
- **メトリクス**: `:15760-15765` は `len(param_groups) > 1` で全グループを emit するが、名前は
  `_build_component_lr_list` 由来で、長さ不一致なら `g{i}` になる。LLRD で 30 グループになると
  `lr_g0..lr_g29` が出る。`_configured_group_names` を使い、`.dNN` 付きは `d(n−1)`（最終深度、係数 1.0 のグループ）
  だけ emit するよう変更する（`lr_<component>` の系列は現状の意味を保つ）。

---

## 12. 設定・API・UI 面

### 12.1 新キーと既定（`param_defaults.py` の `TRAINING_DEFAULTS` に追加。唯一の情報源）

| キー | 型 | 既定 | 適用 | YAML 書き込み |
|---|---|---|---|---|
| `lr_scheduler` | str（validator 付き） | `"constant"`（既存 `:2187`） | 全 | 既存（`training_config.py:180`） |
| `lr_floor_ratio` | float | `0.25`（既存 `:2193`） | 全（D10） | **無条件に変更**（現状は pcf のみ `:192-197`） |
| `lr_decay_start_ratio` | float | `0.85`（既存 `:2192`） | `plateau_cosine_floor` のみ | 現状どおり pcf のときのみ |
| `lr_decay_start_step` | int | `0` | `wsd` | 無条件 |
| `lr_decay_steps` | int | `0` | `wsd`、コマンドの `decay` | 無条件 |
| `lr_decay_shape` | str | `"cosine"` | `wsd`、`rex`、コマンドの `decay` | 無条件 |
| `lr_cycle_steps` | int | `0` | `cosine_with_restarts` | 無条件 |
| `lr_cycle_peak_decay` | float | `1.0` | `cosine_with_restarts` | 無条件 |
| `lr_group_schedules` | dict \| null | `None` | 全（D16） | `null` でない時のみ（`null` は「無し」で既定と同義） |
| `lr_layer_decay` | float | `1.0` | 全（D17） | 無条件 |

無条件に書く理由: 「既定が無害でない新キーは `rewarmup_on_optimizer_reset` と同じく無条件に書く」
（`training_config.py:198-202` の方針）。`lr_decay_*` / `lr_cycle_*` は既定が「効かない」値なので
無害だが、床は既定 0.25 が無害でない（D10 の後方互換規則を YAML 上で明示するため）。

### 12.2 床の後方互換規則（D10）

`resolve_spec` は `config` に `lr_floor_ratio` が**存在しない**とき（= 本書以前に書かれた YAML）、
`name == "plateau_cosine_floor"` なら `0.25`（現行 `:6570` の既定）、それ以外は `0.0`（現行 diffusers 名は
0 まで落ちる、ReLoRA は `min_lr_ratio = 0.0` `relora_trainer.py:162, 175`）を使う。存在するときは値を
使う。**この規則は「YAML に無い」で判定し、Pydantic 既定は関与しない**（Pydantic 既定 0.25 は新規 run
と UI のためのもの）。所有者への明示事項: 本書以前に保存された**プリセット**は `lr_floor_ratio = 0.25`
を運ぶ（`PARAM_KEYS` に入っている `trainingParams.ts:33`）ので、`cosine` プリセットから作る新 run は
床 0.25 になる。UI に値が表示されるので無言ではないが、旧挙動（0）ではない。

### 12.3 登録先チェックリスト（新キー 1 つにつき全て）

1. `param_defaults.py` `TRAINING_DEFAULTS`
2. `openapi.yaml` `TrainingRunCreateRequest`（`:19625`）: フィールド + description + example。`lr_scheduler` に `enum`
3. `routes.py:15307-15316` Pydantic。`lr_scheduler` に `field_validator`（`LR_SCHEDULER_NAMES` 照合）
4. `training_config.py:184-202` YAML 書き込み（12.1 の列に従う）
5. `train` 節なので `_YAML_FIELD_LOCATIONS`（`routes.py:16314`）は不要
6. trainer 読み出しは `resolve_spec(self.config, ...)` の 1 箇所（`self.config = dict(train_config)` `:1750`）
7. `api.ts:6463-6467` 型
8. `TrainingConfig.tsx` DEFAULT_PARAMS + UI（条件表示は `lr_scheduler` 値で）
9. **`trainingParams.ts:32-33` の `PARAM_KEYS`**（漏れると編集保存のたびに既定へ戻る）
10. `PRESET_EXCLUDED_KEYS`（`:189-199`）には入れない（LR キーはプリセットが運ぶ）
11. ガードテスト `backend/tests/training_preset_payload_test.py`、`training_edit_restore_coverage_test.py`

### 12.4 語彙（D18）

- `LR_SCHEDULER_NAMES = ("constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial", "plateau_cosine_floor", "wsd", "rex")`。
  `relora` は内部名（`resolve_spec` が ReLoRA トレーナーからのみ受ける）で公開語彙に入れない。
- 拡散側: Pydantic validator + openapi enum。**破壊的変更**: 今まで任意文字列が通っていた
  （`routes.py:15308`）。ただし未知名は diffusers の `SchedulerType(name)`（`optimization.py:321`）で
  run 時に落ちていたので、動いていた run を壊すことはない。`piecewise_constant` は `step_rules` を
  渡していない（`:6007-6012`）ので今も動かない → 語彙に入れない。
- フロント `<select>`: TS 側にミラー配列（VAE 側の慣行 `VaeTrainingConfig.tsx:161-168`）。
  ミラーのずれは openapi enum と TSX を突き合わせるテストで固定する（既存の TSX 走査テストと同じ手法）。
- VAE 側: P7 で `VALID_LR_SCHEDULERS` を `LR_SCHEDULER_NAMES` の import に置換し、`constant` + warmup の
  拒否（`vae_config.py:819-824`）を外す（§4.2 `constant` の行）。VAE の `lr_scheduler.pt` 方式は不変。
- tagger: 対象外。

### 12.5 UI 変更点（`TrainingConfig.tsx`）

- `<select>` `:4063-4068`: 全公開語彙。`constant_with_warmup` は `constant` と同曲線になる（§4.2）ので
  **表示しない**（受理はする）。
- `lr_floor_ratio` `:4107-4117`: `constant` 以外で表示。
- `plateau_cosine_floor`: `lr_decay_start_ratio` のみ（現状どおり）。
- `wsd`: `lr_decay_start_step`（0 = manual）、`lr_decay_steps`（0 = to end）、`lr_decay_shape`。
- `rex`: 追加項目なし（床は共通）。
- `cosine_with_restarts`: `lr_cycle_steps`、`lr_cycle_peak_decay`。
- LLRD: `lr_layer_decay`（capability で隠す）。
- Advanced 折り畳み: `lr_group_schedules`（コンポーネントごとの select）。既定は閉じていて `null`。
- プレビュー: `GET /training/lr-schedule/preview` の標本を折れ線で描く（D20）。`archCapabilities` による
  ゲートは LLRD 以外に無い（スケジュールは arch 非依存）。

---

## 13. 観測と警告コード

| 種別 | 名前 | 出所 |
|---|---|---|
| extra metric | `lr`（既存 `:15746-15748`）、`lr_<component>`（既存 `:15760-15765`、§11.4 の変更） | 毎 step |
| extra metric | `lr_decay_state`（0 base / 1 decaying / 2 floor / 3 recovering） | 毎 step。`metric_registry.py` に登録（`lr` の隣 `:151`、右軸ではなく段階値なので専用軸） |
| 表示ファイル | `<output_dir>/.lr_schedule.json` | 事象適用時 |
| warning | `lr_schedule_total_steps_changed` | §7.1-2 |
| warning | `lr_schedule_extension_on_floor` | §7.3 |
| warning | `lr_schedule_state_missing` | 事象キー無し／state.json 無しで resume（§5.5） |
| warning | `lr_group_schedules_unnamed_groups` | §10.3 |
| info | `lr_schedule_command` | §6.3 |
| 起動ログ | `resolve_spec` が解決後の `spec` を 1 行で出す（現行 `:6586` の形式を全スケジュールへ） | 構築時 |

`_cleanup_future_metrics`（`:13085`）が resume 点より後の行を消すので、メトリクスは永続記録ではない。
永続記録は state.json の事象列だけ（D4）。

---

## 14. 実装フェーズ

各フェーズは独立に検証・コミット可能で、前フェーズの上に積む。**既存 run の同一 `T` での乗数が
bit 同一であること**は互換条件を満たすケースだけの回帰条件とする。constant の warmup、gas 軸修正、polynomial の床、新しい床指定、終端後の cosine 固定は意図した変更として別テストにする（§17）。

| Phase | 内容 | 検証 | リスク |
|---|---|---|---|
| **P0 集約と等価移植（実装済み、§18）** | `lr_schedules.py`（`ScheduleSpec`, `resolve_spec`, 6 つの diffusers 名 + `plateau_cosine_floor` 別名, `build_lr_scheduler`、タイムラインは空実装で `total_steps(at=0)` のみ）。`:6001-6012` / `:6506-6522` を置換、`_build_plateau_cosine_floor_scheduler` 削除。D9 の軸統一（`T_sched = floor(T/gas)`、fast-forward は保存した scheduler_step、旧状態のみ除算推定、詳細は §17.1）。`lr_utils.py:44-53` の docstring 更新 | `backend/tests/lr_schedules_test.py`: 互換条件（§17）を満たす名前と床、`0 ≤ W < T`、`0 ≤ s ≤ T` で全 step、diffusers `get_scheduler` の `lr_lambdas[0]` と `abs diff == 0`。`plateau_cosine_floor` は旧実装を test 内に写して bit 同一。純粋性（昇順・降順・乱順）。既存 4 テスト（`test_lr_resume_override.py`, `rewarmup_on_optimizer_reset_test.py`, `fused_optimizer_group_resume_test.py`, `component_lr_resume_alignment_test.py`）が通る。`gas > 1` の resume 位置が `global_step // gas` になるテスト | 中。`gas > 1` の run は resume 後の位置が変わる（§1-3 の不一致の解消。CHANGELOG に書く）。`constant` + `W > 0` が warmup するようになる（挙動変更、明示） |
| **P1 タイムラインと延長耐性（実装済み、§18.2）** | `ScheduleTimeline` 本体、`τ`、`save/load_training_state` の `lr_schedule_events`、resume seam (b)、`total_steps` 比較と警告、MNT 再計算フック、`.lr_schedule.json`、`lr_schedule_state_missing` | テスト: 延長 10k→20k を step 9000 で行い、`s < 9000` で bit 同一、`s = 9000` で連続、`20000` で床。二重延長の合成。縮小。事象キー無し state.json の後方互換。`dump(upto_step)` の切り詰め。fused N 個が同値 | 中。resume 順序に 1 行挿入（`:12885` / `:12959` の前）。state.json のキー追加は後方互換 |
| **P2 実行時コマンド（実装済み、§18.3。UI のみ P3 へ）** | `training_file_rpc.py`（純移動）、`training_control_rpc.py`、ポーリング seam、§5.3 の状態機械と §5.4 の復帰、即時反映、`lr_decay_state` メトリクス、openapi → `POST/GET /training/runs/{id}/lr-schedule`、UI ボタンと状態表示、spawn 前 `clear_all` | テスト: 状態遷移表の全行（結果コード）。`decay` → `cancel` → `decay` の連続と乗数の連続性（`R > 0`）。`R = 0` の不連続。step 9000 のチェックポイントから resume すると 9137 の `decay` が消えること。sample RPC のテストが純移動後も通る。API は 202/404/409/429 | 中。ポーリングはバッチごとの `Path.glob` 1 回（sample と同じコスト） |
| **P3 新スケジュールと床の一般化（実装済み、§18.4）** | `wsd` / `rex` / in-house `cosine_with_restarts`（annealing）/ `polynomial` の床、`lr_decay_*` / `lr_cycle_*` / `lr_floor_ratio` 無条件書き込み、D10 の後方互換規則、validator と enum、UI 項目、プレビューエンドポイント、`PARAM_KEYS`、ガードテスト | テスト: §8 の表の数値（`k_rex(½) = 2/3`、傾き）。`lr_cycle_steps = 0` が `cosine` と bit 同一。YAML にキー無し × 各名で床が 0.25/0.0 に解決。preview の標本が `make_lambda` と一致。ガードテスト 2 本 | 中。プリセット経由の床 0.25（§12.2、所有者に明示） |
| **P4 ReLoRA 統合（実装済み、§18.7）** | `relora` を `LambdaLR` 化（`restart` 事象）、`relora_trainer.py:147-181` / `:395-421` の整理、`relora_scheduler.py` 削除。`:3908-3911` と `:5536-5541` のフォールバックはガードとして残すがテストで到達不能を確認 | テスト: 旧 `CosineWithMultipleWarmups` を test 内に写し、同じ restart 列で全 step bit 同一（`min_lr_ratio = 0` ⇔ `F = 0`）。restart は全件を先に渡さず当時の到着順で投入し、過去が変化しないことも検証する（§17）。`epochs` 単位の restart が resume 後も残る（旧実装では失われた `:421`） | 中。ReLoRA の resume が re-warmup / 再表明の対象になる（改善だが挙動変更） |
| **P5 LLRD** | `ArchHandler.depth_blocks`（DiT 第 1 波）、`apply_layer_decay`、グループ `name` 必須化（§10.3、LLRD でも必要）、メトリクス emit の変更、capability 登録、UI | テスト: 合成モデルで `depth_of` の被覆（全 trainable param がどこかに入る）、係数の等比、非ブロック param が 1.0、`_record_configured_group_lrs` が分割後を記録、fused 併用が `ValueError`。実 arch は 3 step smoke（有限 loss）で足りる | 中。arch ごとのブロック属性（要検証）。sd15/sdxl は unsupported |
| **P6 グループ別スケジュール** | `lr_group_schedules`、`build_lr_scheduler` の lambda リスト、fused 併用拒否、Advanced UI | テスト: 2 グループで異なる名前、`reassert_config_lr` と fast-forward が各グループの lambda を使う、無名グループの警告 | 低〜中。既定オフ |
| **P7 VAE トレーナー語彙統合** | `vae_trainer.py:663-674` を `build_lr_scheduler` に、`VALID_LR_SCHEDULERS` を import に、`constant` + warmup 拒否の撤去、openapi `:21653` の enum 更新 | 既存 VAE テストが通る。`lr_scheduler.pt` の新旧形式復元と timeline の延長耐性、constant + warmup 以外の refusal matrix を検証（§17.4） | 低。任意（後回し可） |

各フェーズのコミット前に `git diff --cached` の比較レビュー（CLAUDE.md 大規模変更手順）と、
`py_compile` に加えて実 import（`python -c "import core.training.lr_schedules"` 等）を行う。
フロントのビルドは所有者が実施する。学習の収束確認は行わない（3 step smoke で有限 loss を見るまで）。

---

## 15. 範囲外

- **tagger トレーナー**（`tagger_trainer.py:1001-1009`）: 別トレーナー、別既定（`param_defaults.py:3094` の
  `warmup_steps`）、`lr_scheduler` キー無し。語彙統一の対象にしない。
- **Schedule-Free optimizer との相互作用**: `optimizer_warmup_steps` が optimizer 側にも渡る
  （`train_runner.py:2587`, `:3479-3489`）。スケジューラを掛ける意味は optimizer 実装依存で、本書は触れない（要検証）。
- **U-Net の LLRD**: 深さの全順序が定義できないので第 1 波から外す（§11.4）。
- **グループ別の数値パラメータ**: 名前のみ（§10.1）。
- **TS 側のスケジュール式の再実装**: プレビューはサーバ側（D20）。
- **VAE の `lr_scheduler.pt` 廃止**: しない。VAE は復元、拡散側は再導出、という 2 方式のまま。
- **`total_steps` の専用「延長」ルート**: 作らない。編集 + resume のままで D7 が効く。
- **`constant_with_warmup` の削除**: 受理は残す（YAML 互換）。UI から消すだけ。

---

## 16. 不変条件（実装者向け）

1. 本書の対象（tagger を除く）の LR スケジューラは全て `LambdaLR`。`lr_schedules.build_lr_scheduler` 以外で作らない。
2. lambda は `(step, timeline.events)` の純関数。評価で書かない。書くのは §5.2 の 4 seam のみ。
3. `total_steps` 相対の量は名目軸 `τ(s)`、絶対 step の量は実軸（D8）。新しい量を足すときはどちらかを明記する。
4. scheduler 軸は更新境界での advance 数（D9 / §17.1）。`at`、`last_epoch`、`T_sched` は全てこの軸。
5. 事象の焼き込み: `decay` は `length`/`shape` を持つ。config の後変更で開始済みの減衰を変えない。
6. state.json への保存は `at <= step` に切り詰める。表示ファイルを resume の根拠にしない。
7. 既定値は `api/param_defaults.py` にのみ置く。Pydantic は参照する。
8. API 変更は `openapi.yaml` を先に更新する。
9. capability 判定は `api/arch_capabilities.py` と `core/adapters/capability.py` のみ。スケジュールは arch 非依存
   なので登録しない。登録するのは LLRD だけ。
10. 新キーは `PARAM_KEYS`（`trainingParams.ts:32-33`）に必ず入れる。
11. fused optimizer groups と `lr_group_schedules` / `lr_layer_decay` は併用拒否（警告ではなく `ValueError`）。
12. UI・コミットメッセージ・本書に主観的形容詞と未測定の数値を書かない。LLRD のプリセット値も同様。
13. スケジュールの品質を主張しない。主張してよいのは「config どおりの乗数」「再開・延長で形が保たれる」
    「状態が観測できる」の 3 点だけ。

## 17. 監査で確定した実装契約（2026-09-06）

以下は D1–D20 の補足条件であり、各フェーズの受け入れゲートに含める。

### 17.1 scheduler 軸は保存する（P0 / P1）

現行ループは `global_step % gas == 0` のときだけ scheduler を進める
（`base_trainer.py:15933-15947`, `:15987-15992`）。末尾の端数を flush しないので
通常経路の新規 run は `T_sched = floor(T/gas)`。`T=10, gas=4` は 2 回であり 3 回ではない。
`T < gas` は更新が一度も無いので開始前に拒否する。scheduler は更新後に進み、最後の更新が
使用する位置は `T_sched−1`、位置 `T_sched` の床はその更新後の値である。

ただし global_step はスキップでも増える（`:14990`, `:15198`, `:15242`）。CUDA 復旧では
optimizer 更新無しに scheduler を進める（`:15935-15947`）。従って「成功した optimizer 更新数」
と同義にはしない。state.json に version、`scheduler_step`、`gradient_accumulation_steps`、
有効な更新間隔を保存し、実際の scheduler advance 数を再開位置とする。通常の同一 gas で
スキップ無しの場合だけ `global_step//gas` と一致する。旧状態からの推定は警告する。
fused backward/groups は蓄積を無視する（`:6275-6314`）ので有効間隔は 1 とし、
scheduler も各 backward の共通 seam で 1 回進める。パラメータ hook ごとには進めない。

gas 変更後は過去の事象位置を変えず、保存位置 S に残りの更新回数を足した値を新終端とする。
残り回数は実ループの境界条件から求める（通常の modulo 継続なら `floor(T/gas)−floor(global_step/gas)`）。
部分蓄積の勾配を保存していない再開、スキップ、MNT、fused を個別にテストする。
resume 直後の re-warmup は anchor **とランプ長の両方**を scheduler 軸に変換する
（lambda の引数が scheduler 軸なので、長さだけ global step のままだと gas 倍のランプになる）。
D9 は除算 2 箇所だけの変更ではない。

### 17.2 曲線の境界と互換条件（P0 / P3）

- 新規設定は `0 ≤ W < T_sched`、`0 ≤ F ≤ 1`、`0 < cycle_peak_decay ≤ 1`、長さは非負整数。
  手動を表す外部 `D=0` は内部では None に解決する。REX の `W=0` の自動開始と区別する。
  旧 plateau の `D=T` は旧式の `max(1, T−D)` を保つ互換ケースとして個別検証する。
- polynomial の旧床は 0 ではなく `1e-7 / optimizer.defaults['lr']`
  （diffusers `optimization.py:226`, `:252-270`）。P0 はそのまま移植し、P3 の欠落床 0 への変更は
  CHANGELOG と警告対象にする。明示 F への移行後は linear と同形になる。
- cosine は現行式では T を越えると再上昇する（diffusers `optimization.py:178-182`）。
  本書の終端保持は変更点。未開始の手動 WSD / constant は終端でも 1 を保持し、
  明示長 WSD は指定区間に従う。「全スケジュールが T で床」とはしない。
- `C=0` は modulo を使わず名目進行度 p の単一 cosine とする。`C>0` のみ実軸のサイクル式を使う。
  alias の導出値は名目 T から解決する。再開時の新 T を分母や D に再投入しない。内部 spec に start_axis と end_kind を持たせる。plateau の D は名目軸（τ(s) と比較）、rex の D=W は実軸。両 alias は end_kind=nominal_total / length=None とし、T−D を明示実軸長として格納しない。§4 / §8 の L=T−D は名目区間の説明である。
  明示実軸 D から終端への WSD は `(τ(s)−τ(D))/(T_nominal−τ(D))` を使う。
- `T_new ≤ anchor` の縮小は意図的な不連続終端処理。旧終端に到達済みの場合は床を保持し、
  負の傾きを作る warp は追加しない。warmup 中の延長、二重延長、終了後の延長も検証する。

### 17.3 事象の因果性とグループ（P1 / P2 / P4 / P6）

同一 step の total_steps / restart / command を拒否すると、再開時の MNT 再計算や取り消しが
失われる。保存順 seq で適用し、コマンド結果を各 request_id に返す。取り消しは config WSD の
予定・進行中の減衰も無効化し、再減衰は新しい start 事象から始める。内部状態はグループ別に持ち、
共有するのは事象列だけ。`state_at` は spec/group を引数に受け、開始乗数・復帰先を各曲線から求める。
`lr_group_schedules` は名前だけ replace した未解決 spec ではなく、名前ごとに alias を再解決する。
単一状態の UI / metric は代表グループを明示し、API にはグループ別状態も返す。

ReLoRA の現行 `get_lr` は登録済みの未来 restart を次の終端に使う
（`relora_scheduler.py:111-117`）。実行中に後着した restart を再開時に全件渡すと、
過去の cosine が後から短くなる。新式は「評価位置 s 以下の restart だけ」を選び、各区間の
減衰終端はその時点の total とする。未来 restart を参照しない。従来との比較も逐次到着で行う。
ReLoRA の再 warmup は共通 ramp と二重に掛けず、区間式として直接乗数を返す。
F > 0 での再 warmup は F から 1 と定義する（初回のみ 0 から 1）。

`.lr_schedule.json` を事象時だけ更新しても時間経過で DECAYING→FLOOR になる。
状態遷移時にも更新し、GET が古い状態を返さないことをテストする。

### 17.4 VAE と再開の保存範囲（P7）

通常の lambda closure が閉じ込めた timeline は `LambdaLR.state_dict()` に保存されない。
VAE の `lr_scheduler.pt` 方式は維持し、その中に version / events / scheduler_step を明示的に
保存する。復元は timeline→scheduler state→LR 再表明の順とし、旧形式は移行する。
拒否行列のうち constant + warmup の拒否だけは §4.2 の新仕様に合わせて外す。
P7 を語彙の import 変更だけで済ませず、resume と延長の実経路テストを必要条件とする。

---

## 18. P0 で出荷した挙動変更（2026-09-06）

P0 は `backend/core/training/lr_schedules.py` を新設し、`base_trainer.py` の 2 構築点を
`build_lr_scheduler()` に置き換え、`_build_plateau_cosine_floor_scheduler` を削除した。
互換ケース（§17.2）の乗数は diffusers `get_scheduler` の `lr_lambdas[0]` と bit 同一で、
`plateau_cosine_floor` は削除した実装と bit 同一（`backend/tests/lr_schedules_test.py`）。
以下は**意図した挙動変更**であり、bit 同一の対象外である。

| 変更 | 旧 | 新 | 影響を受ける run |
|---|---|---|---|
| `constant` + `lr_warmup_steps > 0` | warmup しない（diffusers の `constant` 分岐は `num_warmup_steps` を受け取らない、`optimization.py:323-324`） | `constant_with_warmup` と同一曲線で warmup する | `lr_scheduler: constant` かつ `lr_warmup_steps > 0`。既定は `constant` / `0` なので既定 run は不変 |
| スケジューラ軸 | 構築 `T` は global_step 単位、resume は `last_epoch = global_step`。`gas > 1` では scheduler が `T/gas` 回しか進まないのに曲線は `T` で定義されていた（§1-3） | `T_sched = floor(T/gas)`。resume 位置は state.json の `scheduler_step`（無い旧 state のみ `global_step // gas` 推定＋警告 `lr_schedule_position_estimated`） | `gradient_accumulation_steps > 1` の run。resume 後の位置と減衰の速さが変わる |
| `total_steps < gradient_accumulation_steps` | optimizer が一度も step せず、無言で「何も学習しない run」になる | `setup_optimizer` で `ValueError`（開始前） | 該当設定のみ |
| `cosine` の `s > T` | 余弦が再上昇する（`optimization.py:178-182`） | 終端値 0 を保持 | 延長や MNT 再計算で `T_sched` を越えて走る run |
| `polynomial` の床 | `1e-7 / optimizer.defaults['lr']`（実装の帰結で、設計上の選択ではなかった） | **同じ値をそのまま移植**（変更ではない。明示床への移行は P3） | なし |

state.json に 4 キーを追加した（読み側は欠落を後方互換に扱う）:
`lr_schedule_version`, `scheduler_step`, `gradient_accumulation_steps`,
`lr_scheduler_advance_interval`。新しい警告コードは
`lr_schedule_position_estimated` と `lr_schedule_accumulation_changed`。

P0 が**やっていない**こと: 実行時タイムライン（`ScheduleTimeline` は
`total_steps(at=0)` だけの stub で、2 件目の `total_steps` は `NotImplementedError`）、
延長耐性、床の一般化、`wsd`/`rex`/in-house restart、ReLoRA の統合、API・UI・YAML の変更。
`lr_scheduler` の語彙検証（D18）も P3 のままなので、未知名は `resolve_spec` の
`ValueError` として構築時に落ちる（従来は diffusers の `SchedulerType` が落としていた）。

### 18.1 §17 が実コードと合わなかった点

- **fused backward / fused optimizer groups の有効間隔は 1 ではない。** §17.1 は
  「蓄積を無視するので有効間隔は 1、scheduler も各 backward で 1 回進める」としているが、
  実際の `scheduler.step()` は fused でも `global_step % gas == 0` の seam の中にある
  （`base_trainer.py` の optimizer step 分岐）。既存の警告文（`_warn_gradient_accumulation_ignored_under_fused`）も
  「LR schedule still advances once per {accum} backward passes」と明言している。
  したがって `T_sched = floor(T/gas)` は fused 経路でも同じであり、P0 は間隔を
  1 に変えていない。変えるなら `scheduler.step()` の呼び出し位置を動かす別の挙動変更になる。
- **`global_step // gas` は「移行推定」であって旧挙動そのものではない。** 旧 fast-forward は
  `last_epoch = global_step` を書いていたので、`gas > 1` の旧チェックポイントは
  推定値でも旧値でもない位置から再開する。これは §1-3 の不一致の解消であり、
  上表 2 行目に含まれる。
- **P1: `plateau_cosine_floor` の `D` は spec に焼き込めない。** §17.2 は
  「alias の導出値は名目 T から解決する。再開時の新 T を分母や D に再投入しない」と
  書いているが、P0 の `resolve_spec` は `D = round(ratio·T_sched)` を `ScheduleSpec` に
  焼き込んでいた。10000 → 20000 の延長を再開すると `T_sched` が新しい値になるため
  `D` が 8500 から 17000 へ動き、§7.3 が防ぐはずの「プラトーへ戻る」が warp を入れても
  そのまま起きる。P1 で `ScheduleSpec.decay_start_ratio` を足し、`D` を評価時に
  `T_nominal` から再導出する。事象が無ければ `T_nominal == spec.total_steps` なので
  P0 の bit 同一は保たれる（`lr_schedules_test.py` は全件通る）。
- **P1: `set_total_steps` の `NotImplementedError` は残した。** §18 はこれを P0 stub の
  印として書いているが、同時に構築 seam (a) の不変条件でもある（構築は run につき 1 回）。
  P1 は再アンカーを `add("total_steps", at=<resume step>, value=…)` という別の seam に置き、
  `set_total_steps` の 2 度目の拒否をそのまま残した。P0 のガードテストは意味を変えずに通る。
- **P1: `state_at` は `(int, float)` を返さない。** §3.2 の署名では復帰長 `R`、明示長 `L`、
  焼き込んだ形、config 減衰の無効化フラグが表現できない。§17.3 の「spec/group を引数に
  受ける」に従い `state_at(spec, step) -> OverlayState`（frozen dataclass）とした。
  `code` と `start_multiplier` が §3.2 の 2 要素にあたる。
- **P1: 結果コードは代表グループのもの、状態はグループごと。** §5.3 の遷移表は単一状態を
  前提に書かれているが §17.3 は状態をグループ別に持てと言う。両立のため事象列は「命令ログ」
  とし、`result` は要求者に返した記録として事象に残すだけで、状態機械は読まない
  （グループごとに再導出する）。拒否された要求だけは `kind="noop"` として残し、
  どのグループでも効かないようにしたうえで `request_id` の再送に同じ答えを返す。
- **P1: §17.1 の「保存位置 S に残りの更新回数を足した値を新終端とする」は未実装。**
  実装したのは §14 P1 行どおりの `T_sched` の単純比較である。S + 残り回数にすると、
  スキップのあった run は `S < global_step//gas` なので終端が必ず変わり、`total_steps` を
  一切触っていない通常の再開でも毎回 warp と `lr_schedule_total_steps_changed` が出る。
  スキップの無い run では両者は一致する。実装するには別の警告コードと、fused / MNT /
  部分蓄積の各経路ごとの残り回数導出（§17.1 自身が「実ループの境界条件から求める」と
  書いている部分）が要る。
- **P1: `.lr_schedule.json` は書いていない。** §14 の P1 行にあるが、読み手
  （`GET /training/runs/{id}/lr-schedule`）は P2 で、§17.3 が「事象時だけ更新すると
  DECAYING→FLOOR の時間遷移で古くなる」と明示している。既知で古くなる artifact を
  先に出荷しない。読み手・状態遷移時の更新・`lr_decay_state` メトリクスと一緒に P2 で入れる。
- **P1: `lr_schedule_extension_on_floor`（§7.3）は発火しない。** 明示長の減衰は config キー
  `lr_decay_steps`（P3）か `decay` 事象の `length`（P2 のコマンド）からしか生まれず、
  P1 の config 経由の減衰は全て「終端まで」である。機構（実軸の長さは warp で伸びない）は
  実装済みでテストが固定している。警告は長さを作れるフェーズと同時に入れる。
- **P2: §6.4 の 409 条件は DB の `status` では判定していない。** §6.4 は
  `status not in ("running","starting")` と書いているが、コマンドファイルを claim
  するのは生きたサブプロセスだけであり、DB 行は落ちたプロセスに対しても
  `running` のままになりうる。実装は sample の POST と同じ
  `training_process_manager.processes[id].is_running` を使う。同じ理由の同じ判定を
  2 つ持たない。
- **P2: `GET` の応答は §6.4 の 6 フィールドを平坦に返さない。** `state` /
  `events` / `nominal_total_steps` / `effective_total_steps` はトレーナーが書いた
  ファイル由来で「まだ無い」ことがあり、`pending` / `results` は API 側が毎回
  数えるものである。平坦に並べると前者のせいで全フィールドが nullable になるので、
  ファイル由来を nullable な `status` オブジェクトに入れ、`pending` / `results` /
  `run_id` / `is_running` / `max_pending` を必須にした（`LrScheduleStatusResponse`）。
- **P2: `clear_all` は `.lr_schedule.json` を消さない。** §6.2 は spawn 前の
  `clear_all` を「stop→resume を跨いだ『減衰しろ』を適用しない」ためと定義しており、
  それは要求・結果ファイルだけで達成できる。一方 §6.4 は「停止中の run でも読める」
  ことを要求しており、停止中にそれを答えられるのは表示ファイルだけである。よって
  消すのは `.control_request_*` / `.control_result_*` の 2 prefix のみ。
- **P2: 即時反映（§5.6）が実際に値を変えるのは不連続な取り消しだけである。**
  `decay`・`R > 0` の `cancel`・`disarmed_scheduled_decay` はいずれも適用 step で
  連続なので、`base_lr · λ_new(at) == base_lr · λ_old(at)` であり、
  `_fast_forward_one_lr_scheduler` は同じ値を書き直す。値が跳ぶのは
  `W = 0` の `cancel`（§5.4 の不連続復帰）だけで、この 1 ケースだけが
  「forward の前でなければこのバッチの更新に間に合わない」に該当する。
  ポーリング位置がバッチ先頭であるべき理由はもう 1 つあり、そちらは全ケースに効く:
  `at = last_epoch` は optimizer step の後では 1 進んでいる。
- **P2: `rejected_unknown_command` は §5.3 の表に無い結果コードである。**
  タイムラインは知らない `kind` を受け取らないので、未知のコマンド文字列は
  `add()` まで到達しない。API は `queue_request` で 400 として弾くが、
  別ビルドが書いた古い要求ファイルは弾けないので、トレーナーはそれを黙って捨てず
  この結果コードで記録する。openapi の `LrScheduleCommandResult` に載せた。
- **P2: コマンドは長さも形も運ばない。** §6.2 の要求スキーマどおり body は
  `command` だけで、`decay` 事象の `length` は `lr_decay_steps`、`shape` は
  `lr_decay_shape`（どちらも P3 の config キー）から来る。したがって P2 の
  コマンド由来の減衰は全て「名目終端まで」であり、P1 の記載どおり
  `lr_schedule_extension_on_floor`（§7.3）はまだ発火しない。
- **P2: 表示ファイルが古くなりうる窓は「バッチ境界の外」だけである。** §17.3 は
  「時間経過で DECAYING→FLOOR になる」ことを問題にしているが、状態が変わる条件は
  事象の追加か `last_epoch` の前進のみで、後者は `scheduler.step()`＝バッチループの
  中でしか起きない。ポーリングは同じループの先頭で毎バッチ状態を再評価するので、
  scheduler が進んだことによる遷移は必ず次のバッチで書き直される。残る窓は
  「バッチが 1 つも走っていない間」— resume 直後の fast-forward からその run の
  最初のバッチまで、およびデータセット走査・latent キャッシュ・チェックポイント
  書き込み中で、この間 `written_at` / `step` は最後のバッチのものである。
  openapi の `GET` の description に書いた。
- **P2: グループ名は DiT アーキテクチャでは `group{i}` になる。**
  `lr_schedule_group_states` は `_build_component_lr_list()` の名前を使うが、
  この関数は DiT 系で空を返す（`component_lr_resume_alignment_test.py` が
  固定している既知の性質）。長さが param group 数と一致しないときは
  index 由来の名前に落とす。状態・乗数・LR はどのアーキテクチャでも正しい。
- **P2: UI は入れていない（§14 の P2 行から外した）。** §14 の P2 行は
  「UI ボタンと状態表示」を含むが、本フェーズはバックエンドと API までとし、
  フロントは P3 の UI 作業とまとめる。API は openapi に載っているので
  `POST` / `GET` はそのまま呼べる。

- **P3: §12.3 のチェックリストには `PARAM_KEYS` を守るガードが無い。** 9 番の
  「漏れると編集保存のたびに既定へ戻る」は正しいが、それを固定するはずの
  `training_edit_restore_coverage_test.py` は pass-through キーでは無力である:
  そのテストの「送っているキー」集合自体が `PARAM_KEYS` から作られるので、
  エントリを消すと送信側と復元側の両方から同時に消え、差が出ない
  （実際に消して確認した）。リテラルに名前が出る computed キーだけが守られている。
  P3 は「`TrainingRunCreateRequest` の `lr_*` フィールド全件が `PARAM_KEYS` に居る」
  という assertion を `lr_schedule_vocabulary_test.py` に置いた。
- **P3: 無条件書き込みをループで書くと `train_section_key_vocabulary()` が見落とす。**
  この関数は `_build_train_section` の AST から `train["literal"] = ...` の
  添字リテラルを拾う（`training_config.py`）。`for key in (...): train[key] = ...`
  では語彙に入らず、`preserve_unmodelled_train_keys` が将来そのキーを
  「config チャンネル専用」と誤認する土台になる。1 キー 1 行で書いた。
- **P3: `routes.py` から `lr_schedules` を module-level import すると循環になる。**
  `lr_schedules` → `api.param_defaults` → `api/__init__.py` → `api.routes` →
  `lr_schedules`（初期化途中）で `ImportError` になり、**API を経由しない
  `core.training.base_trainer` の import が壊れる**（＝トレーナーのサブプロセスが
  起動しない）。D18 の「Pydantic に validator を付ける」には import 位置の制約が
  付く: validator の本体で import する。AGENTS.md の実 import チェックが捕捉した。
- **P3: `W` は scheduler 軸に換算されていない。** §17.1 は `T` と resume 位置を
  直したが、`optimizer_warmup_steps` は `gradient_accumulation_steps` で割られずに
  そのまま `resolve_spec` の `W` になる（`base_trainer.py` の
  `resolve_lr_schedule_spec` と再 warmup のランプ長）。`gas > 1` では warmup が
  `T_sched` に対して `gas` 倍の長さを占める。P3 はこれを**変えていない**（全 run の
  形が変わる挙動変更で、P3 の範囲外）。プレビューも同じ換算をしないことで
  トレーナーと一致させてある。直すなら独立したフェーズと警告が要る。
- **P3: `lr_decay_steps` は 2 つの意味を持つ。** §12.1 は「`wsd`、コマンドの `decay`」と
  1 行に書いているが、前者は base curve の長さ（`wsd` のみ、`plateau_cosine_floor` と
  `rex` は「名目終端まで」で固定）、後者はどの名前でも効くコマンドの長さである。
  `ScheduleSpec` は `decay_length`（curve 用）と `command_decay_length`（コマンド用）に
  分けた。同じキーを別名の base curve に効かせると D11 の bit 同一が壊れる。
- **P3: polynomial は明示床でも diffusers と bit 同一にはならない。** 式の結合が
  違う（`((lr_init−lr_end)·p + lr_end)/lr_init` 対 `F + (1−F)·p`）ので、
  `F = lr_end/lr_init` を与えても丸め一致まで。§17.2 の「明示 F への移行後は
  linear と同形になる」は形の話であり、bit 同一の主張ではない。P0 テストの
  polynomial 2 件は P3 の契約に書き換えた（`_COMPATIBLE` からも外した）。

### 18.2 P1 で出荷した挙動変更（2026-09-06）

P1 は `ScheduleTimeline` を本体にした: `(at, seq)` 順の事象列、§7.2 の時間軸 warp、
§5.3 の BASE / DECAYING / FLOOR / RECOVERING（式は §17.3 の訂正版）、spec ごとの状態、
state.json への直列化。制御 RPC と HTTP 面（P2）、ReLoRA merge（P4）、config キー（P3）は
入っていないので、外から事象を作れるのは resume の `total_steps` 比較と MNT 再計算だけである。
事象が 1 件（構築時の `total_steps(at=0)`）のときの乗数は P0 と bit 同一（P0 テスト 101 件が通る）。

| 変更 | 旧 | 新 | 影響を受ける run |
|---|---|---|---|
| resume 時の `total_steps` 変更 | 無言。新しい `T` で作り直して古い位置へ fast-forward するので、減衰中の run がプラトーへ戻る（§1-4） | anchor = 再開位置の `total_steps` 事象を記録し、anchor より前の形は不変、残りを新しい残り区間へ写像。`lr_schedule_total_steps_changed` を 1 行出す | 再開時に `steps` を編集した run |
| MNT 変化による `total_steps` 再計算 | 「LR scheduler was initialized with old total_steps / LR decay curve may be affected」の 3 行 WARNING を出して何もしない | 同じ warp を現在位置で適用し、`_fast_forward_one_lr_scheduler` で全グループに即時反映（§5.6）。3 行の WARNING は削除 | resume で MNT を変えた run |
| 事象キーの無いチェックポイント | — | 最初の resume だけ現在の `T` を名目軸とし、`lr_schedule_state_missing` を 1 行出す（§5.5/§7.4）。以後は保護される | P1 より前のチェックポイント |

state.json に 1 キーを追加した（読み側は欠落を「事象なし」と読む）: `lr_schedule_events`。
`dump` は `at <= scheduler_step` に切り詰め、`load` も `upto_step` で切り詰めるので、
step 9000 のチェックポイントから再開すると step 9137 に出した命令は無かったことになる
（`_cleanup_future_metrics` と同じ意味論）。新しい警告コードは
`lr_schedule_total_steps_changed` と `lr_schedule_state_missing`。

テストは `backend/tests/lr_schedule_timeline_test.py`（50 件）。P0 の
`backend/tests/lr_schedules_test.py`（101 件）は 1 件も書き換えずに通る。

### 18.3 P2 で出荷したもの（2026-09-06）

P2 はタイムラインに外から事象を入れる経路を付けた。sample RPC の transport を
`training_file_rpc.py` に**純移動**（atomic write / read / age sort / `owns` /
`make_request_id` と、それらの上の一覧・結果書き込み・prefix 一括削除）し、
`training_sample_rpc.py` はその薄い包みになった（挙動不変、既存テスト 140 件が通る）。
新しい queue は `training_control_rpc.py`:

- 要求 `<output_dir>/.control_request_<id>.json`、結果 `.control_result_<id>.json`、
  表示 `.lr_schedule.json`。
- `claim_all` は自 run の要求を**全件**古い順に claim-delete する。coalesce は
  トレーナー側で `timeline.add` を順に呼ぶことで自然に起きる。
- 上限 20（sample の 3 と別。コマンドは dict 追記なのでディレクトリの上限であって
  スループットの上限ではない）、保存する結果 40（1 バーストぶんが全部残る）。
- stop 中でも claim する（sample は stop を遅らせるので claim しない）。
- spawn 前の `clear_all` は要求と結果のみ（§18.1）。

トレーナー側は `base_trainer.py` にモジュール関数 4 つ:
`poll_lr_schedule_commands`（seam (c)）、`lr_schedule_status`、
`refresh_lr_schedule_status`、`lr_decay_state_code`。ポーリングはバッチループの
**先頭**、`.stop_training` 検査の直後・forward の前で、sample の claim
（サンプリングブロック内）とは別の位置である。1 バッチ 1 回、`claim_all` →
各コマンドを `timeline.add(kind, at=live_scheduler_step(), request_id=…)` →
1 件でも `applied` / `disarmed_scheduled_decay` なら全スケジューラへ即時反映 →
`request_id` ごとに結果を書く → `.lr_schedule.json` → `lr_schedule_command`。
例外はループに投げない。

`.lr_schedule.json` は**状態が変わったときに書く**。署名は
（状態コード, 状態の開始 step, 減衰無効化フラグ, 事象数, 実効 total）で、事象が
無くても毎バッチ再評価するので DECAYING→FLOOR と RECOVERING→BASE の時間遷移が
そのまま反映される（§17.3）。代表グループの状態に加え、param group ごとの
状態・乗数・LR を `groups` に入れる（P6 でグループごとに spec が分かれても
形は変わらない）。`lr_decay_state`（0/1/2/3）を毎 step emit し、
`metric_registry.py` に専用スケールで登録した。

API（openapi 先行、同一変更内）:

| メソッド/パス | 応答 |
|---|---|
| `POST /training/runs/{id}/lr-schedule` | 202 `LrScheduleCommandAccepted` / 400 未知コマンド / 404 / 409 サブプロセス不在 / 429 上限 / 500 |
| `GET /training/runs/{id}/lr-schedule` | 200 `LrScheduleStatusResponse`（`status` は nullable、停止中の run でも読める）/ 404 |

P2 が**やっていない**こと: config キー（P3）、`wsd`/`rex`/床の一般化（P3）、
UI（P3）、ReLoRA の `restart` 事象（P4）、グループ別スケジュール（P6）、
VAE トレーナー（P7）。テストは
`backend/tests/lr_schedule_control_rpc_test.py`（38 件）。P0 の
`lr_schedules_test.py`、P1 の `lr_schedule_timeline_test.py`、
sample RPC の既存テストは 1 件も書き換えずに通る。

### 18.4 P3 で出荷したもの（2026-09-06）

P3 は語彙を開いた。`wsd` と `rex`（どちらも 1 つの curve の別名）、実軸のサイクル長と
ピーク annealing を持つ in-house `cosine_with_restarts`、そして D10 の床
—— 全 curve が `m = ramp · (F + (1 − F) · shape)` になった。`F = 0` では
`1 − 0 == 1`、`1 · x == x`、`0 + x == x` がいずれも厳密なので、移植済み曲線は
P0 と bit 同一のまま（`lr_schedules_test.py` の diffusers 対照が通る）。
config キーは §12.3 のチェックリスト全段（`param_defaults` → `openapi.yaml` →
Pydantic → YAML 書き込み → `resolve_spec` → `api.ts` → `DEFAULT_PARAMS` →
コントロール → `PARAM_KEYS`）に載せた。

新キー 5 本（`TRAINING_DEFAULTS` が唯一の情報源）:
`lr_decay_start_step` (0)、`lr_decay_steps` (0)、`lr_decay_shape` ("cosine")、
`lr_cycle_steps` (0)、`lr_cycle_peak_decay` (1.0)。
`lr_floor_ratio` と合わせて 6 本を `_build_train_section` で**無条件に**書く
（`lr_decay_start_ratio` だけは §12.1 どおり `plateau_cosine_floor` のときだけ）。

| 変更 | 旧 | 新 | 影響を受ける run |
|---|---|---|---|
| `polynomial` の床 | `1e-7 / optimizer.defaults['lr']`（diffusers の `lr_end/lr_init`。P0 はそのまま移植した） | `lr_floor_ratio`。YAML にキーが無ければ **0.0** | `lr_scheduler: polynomial` の全 run。構築時に `lr_schedule_polynomial_floor_changed` を 1 行出す |
| `polynomial` の構築拒否 | `lr_end > lr_init` で `ValueError`（`lr < 1e-7` の run が開始前に落ちた） | 床は比率なので基準 LR を超えられない。拒否そのものが消えた | 基準 LR が `1e-7` 未満の run |
| `linear` / `cosine` / `cosine_with_restarts` の床 | 常に 0 | `lr_floor_ratio`（YAML にキーが無ければ 0.0 = 旧挙動） | 床を明示した新規 run のみ |
| `lr_scheduler` の未知名 | 任意文字列が API を通り、構築時に `ValueError`（P0 以降） | `TrainingRunCreateRequest` の validator が 422 で弾き、小文字に正規化する | 未知名を送っていた呼び出し元（動く run は無い） |
| `lr_floor_ratio` / `lr_decay_*` / `lr_cycle_*` の範囲 | 無検証 | Pydantic で `0 ≤ F ≤ 1`、`0 < peak ≤ 1`、step 系は `≥ 0`。`resolve_spec` も同じ境界で `ValueError` | 範囲外を送っていた呼び出し元 |
| 旧 run の編集保存 | 床キーの無い YAML はそのまま | `GET /params` が Pydantic 既定 0.25 を返し、保存で YAML に入る。`cosine` の run を編集保存すると床が 0 → 0.25 になる | 編集された旧 run。UI に値が出るので無言ではない（§12.2 のプリセット経由と同じ帰結） |
| `lr_schedule_extension_on_floor` | 発火しなかった（明示長を作れるキーが無かった、§18.1） | `lr_decay_steps > 0` か明示長の `decay` 事象を持つ run を延長再開したときに出る | 明示長を設定した run |

新しい警告コード: `lr_schedule_polynomial_floor_changed`、および P1 から用意されていた
`lr_schedule_extension_on_floor` の実発火。

API（openapi 先行、同一変更内）:

| メソッド/パス | 応答 |
|---|---|
| `GET /training/lr-schedule/preview` | 200 `LrSchedulePreview`（`sample_curve()` の標本、`n_points ≤ 512`）/ 400 未知名・未知 shape・`total_steps < gas` |

UI（§14 の P2 行から繰り越した分を含む）:

- `TrainingConfig.tsx` の `<select>` はレジストリ全語彙（`constant_with_warmup` は
  受理のみ・非表示。ただし run が既にその値を持つときだけ選択肢を出す。空表示を避けるため）。
  条件表示は `wsd` → 開始 step / 長さ / 形、`cosine_with_restarts` → サイクル長 / ピーク減衰、
  `plateau_cosine_floor` → 開始比、`constant` 以外 → 床。
- スケジュールプレビューは `GET /training/lr-schedule/preview` の標本を折れ線で描く（D20）。
  TS 側に式は無い。
- `TrainingMonitor.tsx` に「Decay now」「Cancel decay」と
  `GET /training/runs/{id}/lr-schedule` の状態表示（代表グループの state・乗数・位置、
  pending、直近の結果コード）。§6.5 の「チェックポイントより前に戻すとコマンドは無かったことになる」
  を 1 文で書いた。

テストは `backend/tests/lr_schedule_vocabulary_test.py`（83 件）。P0 の
`lr_schedules_test.py` は polynomial の 2 件だけ P3 の契約に書き換え（bit 同一の対象から外れたため）、
残りと P1 / P2 のテストは 1 件も書き換えずに通る。

P3 が**やっていない**こと: ReLoRA 統合（P4）、LLRD（P5）、`lr_group_schedules`（P6）、
VAE トレーナーの語彙統合（P7、`vae_config.py` の `VALID_LR_SCHEDULERS` は未変更で
`LR_SCHEDULER_NAMES` の真部分集合のまま）。

### 18.5 warmup を scheduler 軸に載せた（P0 の取りこぼし、2026-09-06）

P0 は `T` だけを `scheduler_total_steps()` で `floor(T/gas)` に変換し、`W`
（`optimizer_warmup_steps`）を global step のまま `resolve_spec` に渡していた。
両者は同じ軸に居なければ `W/T` が gas 倍ずれる。**変換前（P0〜P3）より、片方だけ
変換されていない状態の方が比率としては悪い**（P0 以前は両方 global step だった）。
規則は `lr_schedules.to_scheduler_axis(steps, interval)` 1 箇所が持ち、
`scheduler_total_steps` / 新設 `scheduler_warmup_steps` / プレビュー
（`GET /training/lr-schedule/preview`）がすべてそれを通る。

| 変更 | 旧 | 新 | 影響を受ける run |
|---|---|---|---|
| warmup の軸 | `W` は global step のまま `resolve_spec` へ。`gas` 倍の区間を warmup が占め、`lr_warmup_steps` が指す step では ramp が `1/gas` にしか達しない | `W_sched = floor(W/gas)`。ramp が 1.0 に達するのは設定どおり `lr_warmup_steps` 学習 step 目 | `gradient_accumulation_steps > 1` の run（`gas == 1` は全語彙で bit 同一） |
| re-warmup（`_rearm_warmup_after_optimizer_reset`）のランプ長 | anchor だけ scheduler 軸で、長さは global step。ランプが gas 倍長かった | 長さも `floor(W/gas)`。`W < gas` は「1 回も LR 更新が無い長さ」なので再 warmup せず、その旨を 1 行出す | 同上（optimizer state を復元できずに再開した run） |
| プレビューの `warmup_steps` | 分母だけ scheduler 軸で、warmup は未変換（コメントで明示していた） | トレーナーと同じ変換。応答の `warmup_steps` は `scheduler_total_steps` と同じ軸 | `gas > 1` でプレビューを見ていた UI |

同じ軸に載っていない **既知の残り**（この修正の対象外。どれも P0 の変換とは
独立で、それぞれの中では自己整合している）:

- ReLoRA の `CosineWithMultipleWarmups`（`relora_trainer.py:157-180`）は `total_steps` も
  `initial_warmup_steps` も global step で渡すので比率は正しい。レジストリへの統合は P4。
  **`optimizer_warmup_steps` 属性そのものを割ってはならない**（ReLoRA の分母が未変換のため）。
- VAE トレーナー（`vae_trainer.py:663-674`）は diffusers に両方 global step で渡す。P7。
- schedule-free 系 optimizer（`_ringbuffer_optimizer_kwargs` の `warmup_steps`）の内部カウンタ
  `k` は `optimizer.step()` ごとに増える（`adamw8bit_ringbuffer.py:1049`）ので実質 scheduler 軸だが、
  受け取る値は global step のまま。P0 以前からの状態で、LR スケジューラの ramp と乗算で
  二重に掛かる点も含めて所有者の判断待ち。
- `lr_decay_start_step` / `lr_decay_steps` / `lr_cycle_steps`（P3 の実軸キー）も未変換で、
  `gas > 1` では config の数値の gas 倍の位置・長さになる。W と同じ種類のずれだが、
  こちらは P3 の決定事項なので本修正では触っていない → **§18.6 で修正済み**。

テストは `lr_schedules_test.py`（W の floor、gas 1/2/4 で `W_sched/T_sched` が一定、
ramp が `W/gas` で 1.0 に達する、`gas == 1` は全レジストリ名で spec も曲線も同一）と
`rewarmup_on_optimizer_reset_test.py`（ランプ長、`W < gas`）に追加。既存テストは 1 件も
書き換えていない。

### 18.6 config の step キーも scheduler 軸に載せた（P3 の取りこぼし、2026-09-06）

§18.5 が記録だけして残した最後の 1 件。`lr_decay_start_step` / `lr_decay_steps` /
`lr_cycle_steps` はユーザーが `total_steps` や `lr_warmup_steps` と全く同じ意味で入力する
step 数だが、`resolve_spec` が config から未変換のまま読んでいたため、比較相手の
`T_sched` に対して `gas` 倍の位置・長さになっていた。P3（`437d8404`）の出荷から 1 時間なので
互換性の問題は無い。

規則は §18.5 と同じ 1 箇所（`to_scheduler_axis`）が持ち、変換は **spec を解決する seam**
（`resolve_spec` に渡す新引数 `advance_interval`）で行う。保存済みの属性は変換しない。
`base_trainer.resolve_lr_schedule_spec` とプレビュー（`GET /training/lr-schedule/preview`）の
両方が同じ値を渡すので、プレビューは run と一致し続ける。

| 変更 | 旧 | 新 |
|---|---|---|
| `lr_decay_start_step`（wsd の減衰開始、実軸の**位置**） | config の数値をそのまま `decay_start_step` へ。`gas=4` なら設定の 4 倍の位置で減衰開始 | `floor(D/gas)`。0（= 手動）の判定は**変換前の設定値**で行うので、1 蓄積窓より小さい開始値が「手動」に化けることはない（scheduler step 0 になる） |
| `lr_decay_steps`（減衰の**長さ**。config WSD と `start_decay` コマンドの両方） | 未変換。`gas` 倍の長さ | `max(1, floor(L/gas))`。0 は変換前に「未設定 = 終端まで」へ解決済み。1 未満に落とさないのは、0 が別曲線（終端まで）の sentinel であり、かつ除数だから |
| `lr_cycle_steps`（cosine_with_restarts のサイクル長） | 未変換。`gas` 倍のサイクル | `max(1, floor(C/gas))`。0（= 全体で 1 サイクル）の扱いは同上 |

軸が 2 つあることを混同しない: `lr_decay_start_step` は**実軸の位置**（timeline の warp で
動かない、§17.2）であると同時に**global_step 単位で入力される**（§17.1）。今回変わるのは
単位だけで、時計は変わらない。`lr_decay_start_ratio` は無次元の比で、掛ける相手の `T` が
既に `T_sched` なので変換不要（`plateau_cosine_floor` は今回も bit 同一）。

未変換のまま**正しい**と確認したもの: コマンド経路の `at`（`live_scheduler_step()` = scheduler
軸）、コマンドの長さ（`spec.command_decay_length` 経由なので今回の変換が効く）、`cancel` の
復帰長（`spec.warmup_steps` = §18.5 で変換済み）、`total_steps` 事象の `at` と値、
`lr_cycle_peak_decay` / `lr_floor_ratio` / `lr_decay_shape`（無次元・文字列）。
API のコマンドは `command` 以外のペイロードを持たない（`training_control_rpc.queue_request`）ので
外から step 値が入る経路は無い。

テストは `lr_schedules_test.py` に追加（gas 1/2/4 で減衰開始とサイクル長が schedule の同じ
**割合**に載る、長さが 0 に落ちない、`gas == 1` は全レジストリ名で修正前の式と spec も曲線も
同一）。既存テストは 1 件も書き換えていない。

### 18.7 P4 で出荷したもの（2026-09-06）

P4 は ReLoRA をレジストリに入れた。`relora` は 1 つの curve になり、merge は
timeline の `restart` 事象になった。`relora_scheduler.py`（`CosineWithMultipleWarmups`、
`_LRScheduler` 直系）は削除し、`relora_trainer.setup_optimizer` の差し替え（旧 `:147-181`）も
廃した。ReLoRA も `build_lr_scheduler` の `LambdaLR` を使う——D2 の不変条件と
`lr_utils.py:44-53` の docstring がこれで真になった。

旧実装との比較は**到着順**（その step までに登録済みの restart だけを渡す＝実行中の
スケジューラが実際に持っていた列）で行い、全 step bit 同一（`min_lr_ratio = 0` ⇔ `F = 0`）。
テストは `backend/tests/lr_schedule_relora_test.py`（35 件）で、旧 `get_lr` は
import ではなく**写し**である（モジュールごと消えているため）。

| 変更 | 旧 | 新 | 影響を受ける run |
|---|---|---|---|
| 未来 restart の参照（§17.3 の本題） | `get_lr` が登録済み**全** restart から次の終端を採る（`relora_scheduler.py:111-117`）。resume が全件を再登録するので、**過去の cosine が後から短くなる** | 評価位置 `s` 以下の restart だけを読み、終端は常にその時点の total | 1 回でも merge した run の resume 全部。W=200 / W_r=100 / T=2000、merge 500・1000・1500 の run では step 900 の乗数が 0.8909（実走時）→ 0.1464（再開後）で、全 step の 1/4 以上が変化した |
| 軸 | `total_steps` / `initial_warmup_steps` / `restart_warmup_steps` は global step、位置カウンタ `_relora_step` は scheduler 軸（§18.5 は「比率は正しい」と書いたが、**位置は正しくない**: `gas > 1` では曲線が run の到達しない位置に定義され、`add_restart(global_step)` も gas 倍先を指した） | 3 つとも `to_scheduler_axis`（floor）、`at` は `live_scheduler_step()` | `gradient_accumulation_steps > 1` の ReLoRA run。`gas = 4` なら曲線は 1/4 に縮み（＝ run が cosine を最後まで走る）、restart は設定どおりの merge 位置で効く。`gas == 1` は完全に不変 |
| 床 | `min_lr_ratio = 0.0` 固定（`relora_trainer.py:162, 175`） | `lr_floor_ratio`（D10 / §4.2）。キーの無い旧 YAML は 0.0（§12.2） | **新規 run は既定 0.25**（`_build_train_section` が無条件に書く）ので cosine が 0.25·lr で止まる。UI はこの入力を `lr_scheduler === "constant"`（ReLoRA の既定であり ReLoRA が無視する値）で隠していたので、表示条件に `training_method === "relora"` を足した |
| restart の反映時点 | merge フックは `scheduler.step()` の**後**なので、restart は次の更新（1 step 後）から効いた。reinit 直後の 1 step が restart 前の LR で走る | merge seam でも §5.6 の即時反映を行う（`reapply_lr_schedule_position`） | ReLoRA 全部。1 step 分の LR |
| 3 つのフォールバック | `_fast_forward_one_lr_scheduler` の O(global_step) リプレイ、`_rearm_warmup_after_optimizer_reset` の無言スキップ、`reassert_config_lr` の乗数 1.0 | ガードとして残置。ReLoRA からは到達しない（テストで固定） | ReLoRA の resume が O(1) になり、optimizer state を復元できなかった再開で再 warmup が掛かり、config の LR がスケジュール位置込みで再表明される（挙動変更、改善） |
| `epochs` 単位 merge の restart | `_restore_scheduler_restarts` は `i * merge_every`（steps 単位）しか再現できず、epochs では**全部失われた**（`:421`）。再開すると restart の無い cosine に戻る | 事象なので位置ごと state.json に残る | epoch 単位 merge の ReLoRA run |

事象の無い旧チェックポイントは `_restore_legacy_lr_restarts`（`install_lr_schedule_events` の
末尾から呼ぶ）が `merge_count` から steps 単位のみ再構成し、epochs では再構成できないことを
1 行出す。`at` 重複は `ignored_duplicate_restart` で弾くので、事象を持つチェックポイントで
二重登録は起きない。

§17.3 が実コードと合わなかった点:

- **「各区間の減衰終端はその時点の total」の *total* は名目 total とした。** 実 total
  （`current_total`）を分母にすると `total_steps` を変えた再開のたびに過去の減衰率まで
  変わり、§7.2 の「anchor より前は不変」を破る。`nominal_total` + `clock` にすれば warp が
  そのまま効き、事象が構築時の 1 件だけなら両者は一致するので bit 同一性にも影響しない。
- **`lr_scheduler` 設定を無視した旨の 1 行は `resolve_spec` が出せない**（§3.3 はそう書く）。
  `resolve_spec` は正規化後の `"relora"` しか受け取らず、無視した名前を知らない。
  `ReLoRATrainer.setup_optimizer` が出す。
- **`restart_warmup_steps` は config から解決できない。** §3.2 は
  `ScheduleSpec.relora_restart_warmup_steps` を config 解決の結果として書いているが、この値は
  YAML の `network.relora` にあり、トレーナーの `self.config`（`train` セクション）には無い。
  `resolve_spec` の引数として渡す（`resolve_lr_schedule_spec` が
  `getattr(trainer, "restart_warmup_steps", None)`、既定は `TRAINING_DEFAULTS`）。
- **`relora` は D18 の語彙に入れない。** `LR_SCHEDULER_NAMES` に足すと restart を持たない run が
  「restart 付きの曲線」を選べてしまう（実体は warmup → 単一 cosine）。
  `INTERNAL_SCHEDULER_NAMES` を分け、`resolve_spec` だけが受理する。validator・openapi の
  enum・UI の `<select>` は変わらない。
- **旧チェックポイントの restart 復元は `_restore_relora_state` からは呼べない。** これは
  `install_lr_schedule_events` より**前**に走り、`timeline.load()` が事象列を丸ごと置換するので、
  そこで足した restart は消える。復元は seam (b) の直後に移した。
- **再 warmup の 0/F の別は「初回か否か」で決まり、`step < W` では決まらない。** §17.3 の
  「初回のみ 0 から 1」を共通 ramp（`base_multiplier` の先頭）で表すと、初回 warmup 中に
  起きた merge の re-warmup が共通 ramp と二重に掛かる。`relora` だけ `base_multiplier` の
  先頭で自前の乗数を返し、区間が初回かどうかで 0→1 と F→1 を分ける。

P4 が**やっていない**こと: LLRD（P5）、`lr_group_schedules`（P6）、VAE（P7）。
`GET /training/lr-schedule/preview` は `relora` を受け付けない（語彙外）ので、ReLoRA run の
UI プレビューは選択中の（無視される）スケジュールを描く——プレビューの下に無視される旨を
1 行書いた。restart 位置を持つプレビューは本フェーズの範囲外。

### 18.8 P5 / P6 で出荷したもの（2026-09-06）

**§14 の表と実装の番号が入れ替わっている。** §14 は P5 = LLRD、P6 = グループ別スケジュールと
書いているが、実装は 1 コミットで両方入れた（LLRD がグループ構造を変え、グループ別スケジュールが
その構造の上に乗るので、片方だけ出荷すると `.dNN` とコンポーネント名の分離（§17.3）が
検証できない）。以下は機能名で書く。

**グループ別スケジュール（D16、既定オフ）**: `lr_group_schedules`（`null` = オフ）は
コンポーネント名 → スケジュール名の写像で、`build_lr_scheduler` が param group ごとに別の
lambda を持つ `LambdaLR` を返す。数値パラメータ・タイムライン・コマンドは run 共通のまま
（事象列だけが共有、状態は spec ごと。P1 の実装がそのまま効く）。

**LLRD（D17）**: `lr_layer_decay`（`1.0` = オフ）1 キー。`setup_optimizer` がアダプタの
グループを受けた直後、**optimizer を作る前**に `apply_layer_lr_decay` で深さ別に分割し、
深さ `d`（全 `n`）を `lr · factor^(n−1−d)` にする。深さの出所は新フック
`ArchHandler.depth_blocks(trainer)`。**プリセット値は本書どおり決めていない**（既定は
オフの 1.0）。

| 変更 | 旧 | 新 | 影響を受ける run |
|---|---|---|---|
| param group の `name` / `component` | アダプタは書かない（§1-6）ので `_name_configured_groups` は `_build_component_lr_list` か `group{i}` に落ちていた | LoRA は `component_param_groups` の 1 箇所、フル FT・ControlNet・VE は各アダプタが `name`（細分ラベル）と `component`（`LORA_COMPONENT_*`）の**両方**を書く | 全 run。resume の LR 再表明ログのラベルが `U-Net` → `unet` 等に変わる。`.lr_schedule.json` の `groups[].name` も同様 |
| `lr_<component>` メトリクス | `_build_component_lr_list` 由来、長さ不一致で `g{i}` | 同じ優先順を**先頭に残した**うえで、一致しないときだけ `_configured_group_names`、最後に `g{i}`。`.dNN` 付きは最終深度（係数 1.0）だけ emit（§11.4） | LLRD を使う run のみ。使わない run の系列名は 1 つも変わらない |
| optimizer state の復元 | グループ数が変われば prefix remap が index で寄せる | 深さ分割の署名（`name` と param 数）が食い違う resume は**復元を拒否して既存の fresh optimizer 経路へ**（§17.3）。`.dNN` がどちらにも無ければ判定自体しない | LLRD を切り替えた／ブロック数か trainable 集合が変わった resume |

§10 / §11 / §17 が実コードと合わなかった点:

- **`build_lr_scheduler` は `group_schedules: Mapping[str, str]` を受け取れない。** §3.2 / §10.2 の
  署名は名前の写像だが、名前から spec を作るには config・`gradient_accumulation_steps`・
  ReLoRA の restart warmup が要り、この関数はそのどれも持たない（`lr_schedules.py` は
  `core.training` の他モジュールに依存しない、という D1 の制約でもある）。実装は
  **解決済みの `group_specs: Sequence[ScheduleSpec]`（param group と同順）** を受け取り、
  名前の解決は `base_trainer.resolve_lr_group_specs` が `resolve_spec` を名前ごとに
  呼んで行う（§17.3 の「alias を再解決する」はそこで満たされる）。使わない引数
  `group_names` は削除した。
- **`depth_blocks` の 1 要素は 1 ブロックとは限らない。** §11.2 は
  `Sequence[nn.Module] | None` だが、Ideogram 4 は cond / uncond の 2 本の**並列**スタックを
  持ち（`ideogram4_train_uncond` で uncond にも LoRA が入る）、層 `j` は両者で同じ深さである。
  連結すると uncond 側が「より深い後半」に見えるので、要素は**モジュール 1 個または同じ深さを
  共有するモジュールの列**とした。他の arch は 1 要素 1 ブロックのまま。
- **fused optimizer groups の拒否条件は `num_optimizer_groups > 0` だけではない。** §10.4 は
  そう書くが、`create_optimizer_groups` を呼ぶのは `setup_optimizer` の
  `blocks_to_swap > 0` の枝の中だけで、block swap 無しの `num_optimizer_groups` は何も
  平坦化しない。§10.4 自身が「8-bit optimizer と Block Swap の既存拒否と同じ場所・同じ文体」と
  言っており、その既存拒否は `blocks_to_swap > 0 and num_optimizer_groups > 0` である。
  よって拒否は**その条件**（`fused_optimizer_groups_active`）とした。動く設定を拒否しない。
  ただし LLRD は optimizer を作る前に走るので、拒否の**位置**は既存の枝より前になる。
- **LLRD は 1 ブロックのスタックを拒否する。** `n = 1` では全係数が `factor^0 = 1.0` になり、
  設定が無言で無効になる。`n_depths < 2` は「深さの無い arch」と同じ拒否に落とす。
- **ReLoRA は `lr_group_schedules` も無視する。** §10 は触れていないが、group 別に別 curve を
  当てると、その group だけ merge の restart を失う。`lr_scheduler` を無視するのと同じ
  1 行を出して無視する（P4 の慣行）。
- **写像に載っているのに該当グループが無いコンポーネントは警告**（`lr_group_schedules_unknown_component`）で、
  拒否ではない。`train_text_encoder` を切った run が写像を持ったまま起動できなくなるのを避ける。
  名前の無いグループが 1 つでもある場合は §10.3 どおり写像ごと無視
  （`lr_group_schedules_unnamed_groups`）。

`depth_blocks` を実装した arch（= capability が `lr_layer_decay` を提供する arch）:
zimage（`transformer_original.layers`）、flux2（`transformer_blocks` + `single_transformer_blocks`）、
krea2 / lens / ltx2 / minimax_h3（`transformer.transformer_blocks`）、anima（`transformer.blocks`）、
acestep（`transformer.decoder.layers`）、ideogram4（`transformer.layers`、uncond と深さ共有）、
minit2i（`txt_preamble_blocks` + `double_blocks`）、sensenova
（`transformer.language_model.model.layers`、MoT の 2 半分は同じ層の中なので深さ軸は 1 本）。
**辞退**: sd15 / sdxl（U-Net の skip 接続で深さの全順序が定義できない。`ArchHandler.depth_blocks`
の既定 `None` のまま、capability に理由付きで登録）。

新しい警告コード: `lr_group_schedules_unnamed_groups`、`lr_group_schedules_unknown_component`。

テストは `backend/tests/lr_group_schedules_and_layer_decay_test.py`（81 件）。既定オフの
bit 同一性は、直前のコミットの `lr_schedules.py` を **git から取り出して import し**、
全レジストリ名 × `gas ∈ {1,4}` × 全 step で乗数を突き合わせて固定している（式を書き写さない）。
P0〜P4 のテスト（`lr_schedules_test.py` / `lr_schedule_timeline_test.py` /
`lr_schedule_vocabulary_test.py` / `lr_schedule_control_rpc_test.py` /
`lr_schedule_relora_test.py`、計 348 件）と resume 系 5 本（144 件）は 1 件も書き換えずに通る。

やっていないこと: P7（VAE トレーナーの語彙統合）。グループ別スケジュールの**プレビュー**も
入れていない（`GET /training/lr-schedule/preview` は run の代表スケジュールだけを描く。
UI にその旨を 1 行書いた）。
