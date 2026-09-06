# 収束加速の検討: 補助損失と潜在空間 critic

**状態: 検討のみ（未実装・未決定）**  日付: 2026-09-06

---

## 1. 背景と課題

VAE 差し替えなど、重みの一部が初期化された状態からのトレーニングが増えている。
ほぼフルスクラッチに近く、収束が極めて遅い。

### 症状 (*1)

単一 step の推論（training debug latent/pixel）ではそれなりの結果が出ているのに、
通しの軌跡では次の 2 つが起きる。

| 症状 | 推定原因 | 性質 |
|---|---|---|
| **A. 全体が真っ白／グレー** | exposure bias（誤差蓄積）。DC・低周波成分の系統誤差 | **軌跡でのみ現れる大域現象**。1発 x̂₀ には現れない |
| **B. latent セル単位のブロックノイズ**（SDXL なら 8x8px） | latent セルごとに誤差が独立。VAE 差し替えで in/out projection が実質ランダム初期化 | **1発 x̂₀ にも現れる局所現象** |

**A と B は原因も対処も別物**である。両方に効く単一の仕組みを狙わないこと。

### 本当の痛点

最終的には長期トレーニングしかないが、**結実するか不明な段階で長期に流すのが辛い**。
つまり求めているものは 2 つある。

1. 収束を速める仕組み（loss 側）
2. **早期に見切る／続行判断する指標**（メトリクス側）

2 の方が緊急度が高く、かつ圧倒的に安い。

---

## 2. 既存資産（重要）

新規作成コストの見積もりが大きく変わるため、先に確認された事実。

| 資産 | 場所 | 意味 |
|---|---|---|
| REPA 実装済み | `base_trainer.py:2722` / `core/training/repa.py` | エンコーダは **tagger または SigLIP2 選択式**。「REPA のモデルが古い」は既に解消済み |
| frozen latent critic を loss に足す前例 | `core/training/aesthetic_loss.py` | `LatentCNN`（約 50K params、VRAM < 5MB）を latent に直接適用。**提案手法の雛形そのもの** |
| critic の学習基盤 | `subapps/aesthetic_scorer/backend/core/` | `aesthetic_model.py` / `aesthetic_trainer.py` / `latent_generator.py` |
| **decoder に勾配を通すクロップ学習** | `core/training/vae/vae_trainer.py` + `VaeEpochCropSampler` | 既に MSE + **LPIPS-VGG 0.1** + YCbCr Charbonnier で稼働。LPIPS は既存依存 |
| メトリクスチャネル | `metric_registry.py` + `log_extra_metric()` | DB カラム追加なしで新指標を出せる |

### REPA の現状コストと既知の歪み

```python
img = flatten_to_rgb(img).resize((S, S), Image.BICUBIC)   # 正方形に潰す
```
（`base_trainer.py:2828`）

- 毎ステップ **PIL open → decode → 384² リサイズ**（LRU 4096 で部分償却のみ、CPU 側）
- 384² のエンコーダ forward
- **正方形スカッシュ**。アスペクト比バケツ使用時、特徴を DiT グリッドに interpolate し直しても
  **特徴自体が歪んだ画像から計算されている**。REPA はトークン単位の空間対応を前提とするため、
  これは効きを鈍らせる方向の実害。

---

## 3. 当初提案の評価

> latent 状態で、完成品の latent (t=1) と、t=[0,1) から 1 発生成した latent の集合から
> **t を予測するモデル**を作り、予測 t を 1 に近づける loss を足す。

### 正しい点

- **勾配パスが安い**。`x̂₀ = (z_t − σ·ε̂)/α` は訓練ステップ内で既に得られるので追加サンプリング不要。
  critic の fwd+bwd のみ（`LatentCNN` 級なら 1〜3% オーバーヘッド）。REPA より軽いのは事実。
- **REPA が \*1 を見ていないという指摘は妥当**。tagger も SigLIP2 も実画像のみで学習されており、
  ブロックノイズ・グレー化は表現空間で近傍に潰れる可能性が高い。

### 致命的になりうる点

**(A) frozen scalar critic は必ずハックされる。**
凍結判別器のスコアを最大化する勾配降下は、定義上 adversarial example 生成器である。
REPA が安全なのは「固定表現に *合わせる*（GT 由来のターゲットがある）」からで、
「固定スカラーを *上げる*」のとは全く別物。`aesthetic_loss` が成立しているのは
弱い正則化として小さい重みで使っているからであり、収束加速のため重みを上げると同じ壁に当たる。

回避策: (i) critic をオンライン共学習＝GAN 化 / (ii) N ステップごとに現行モデル出力で再 fit
（lagged discriminator）/ (iii) loss にせずメトリクス専用にする。

**(B) t ラベルの意味がドリフトする。**
「t=0.7 から 1 発生成した latent」のラベル 0.7 は、実は *その時点のモデルの t=0.7 での下手さ* である。
モデルが上達すると同じ t でも綺麗な出力になり、ラベルは陳腐化する。
critic は「古いモデルらしさ」の検出器となり、訓練中モデルを過去に引き戻す方向に働きうる。

**(C) 射程が症状 B に限られる。**
1 発 x̂₀ を入力とする critic では、**症状 A（グレー化）は原理的に見えにくい**。
1 発 x̂₀ 自体は綺麗だから。

**(D) 目標が well-posed でない。**
同じ latent が異なる t ラベルを持ちうる。後述の案 1 はこの点で構造的に優れる。

---

## 4. 設計案の比較

| # | 設計 | 入力 | 教師 | 対 B ブロック | 対 A グレー | ハック耐性 | コスト |
|---|---|---|---|---|---|---|---|
| 0 | **当初案**: unpaired t 回帰 | x̂₀ (1発) | 生成時の t | ◎ | △ | ✕（要再 fit） | 小 |
| 1 | **paired critic（潜在版知覚距離）** | (x̂₀, x₀) ペア | オフライン計測値 | ◎ | ○ | ◎ | 小 |
| **1'** | **critic を廃し小クロップ decode** | latent クロップ | —（直接計測） | ◎ | ✕ | ◎ | 小〜中 |
| 2 | patch-wise t マップ | x̂₀ | t（パッチ単位） | ◎◎ | △ | △ | 小 |
| 3 | lagged/online discriminator（LADD 相当） | x̂₀ | real/fake | ◎ | ○ | ○（共学習） | 中 |
| **4** | **統計モーメント整合（モデル不要）** | rollout x̂₀ | データセット統計 | ✕ | ◎◎ | ◎ | 極小 |
| 5 | trajectory critic | 数 step rollout | t or real/fake | ○ | ◎ | △ | 大 |
| 6 | decoder-aware critic | VAE encoder 中間特徴 | real/fake | ◎◎ | ○ | ○ | 中 |

### 案 1: paired critic

critic に GT latent `x₀` も入力する `d(x̂₀, x₀) → スカラー`。

```
オフライン（1回だけ）:  (x̂₀, x₀) → decode → 指標 → ラベル s
訓練中（毎ステップ）:   (x̂₀, x₀) → critic(latent 2枚) → ŝ   ← decode 不要
```

- **GT が入力に入るので「ノイズを盛って critic を騙す」ができず、(A) も (B) も原理的に消える。**
- decode は決定論的関数なので `指標(dec(a), dec(b))` は `(a,b)` の関数として一意に定まる
  → **回帰問題として well-posed**（案 0 との決定的な差）。
- 「latent L2 では等価だが decode するとブロックが出る誤差」を選択的に罰する。

**制約（重要）**: ラベル関数はモデル非依存だが、**critic の入力分布カバレッジはモデル依存**。
合成ノイズのみで作ったペアで学習すると「合成ノイズ距離」を覚え、実モデル誤差に外挿しない。
複数チェックポイントの実 1 発予測＋合成劣化を混ぜて広くカバーする必要がある。
案 0 のような *ラベル* ドリフトは起きないが、*入力* ドリフトは残る。

### 案 1': critic を廃す

`vae_trainer.py` に既にクロップ単位で decoder に勾配を通す実装（`VaeEpochCropSampler`）がある。
拡散側でも **サンプルごとに小クロップ（例: latent 8x8 = 64x64px）だけ decode** し、
画素側指標を直接かければよい。

- 蒸留誤差ゼロ、critic 学習コストゼロ、ハック余地ゼロ、ブラックボックスゼロ
- 症状 B は**局所・セル単位**の欠陥なので、小クロップでも統計的に捉えられる
- コストは tiny crop の decoder fwd+bwd（gradient checkpointing 併用）
- 症状 A（大域）は見えない → 案 4 と分業

### 案 4: 統計モーメント整合

判別モデルすら不要。数百ステップに一度、短い rollout（4〜8 step）を回し、
得られた latent のチャネル別 mean/std/低周波パワーをデータセット統計と比較する。

- `log_extra_metric()` に流せば **「このまま流して収束するのか」の早期判定が即座に得られる**
- loss 側に低周波モーメント整合項（重み小）を足すのも安全
- グレー化＝DC 成分の系統ずれなので、学習型 critic より素直で堅牢

---

## 5. 教師信号の選択（LPIPS の位置づけ）

### よくある誤解

案 1 で LPIPS が出るのは **オフラインのラベル生成器としてだけ**。LPIPS は latent を評価しない。
問うべきは別々の 2 問である。

- **Q1**: その指標は自分が気にする画素空間の劣化を正しく測るか？
- **Q2**: その劣化量は latent ペアから予測可能か？ → 上述のとおり理論上 well-posed

### LPIPS のブラックボックス性は、ここでは実害が小さい

LPIPS の悪評（直接最適化すると騙せる）は、**直接最適化しないため該当しない**。
LPIPS はラベルを吐くだけで、モデルが押し返す相手は latent しか見ない小さな回帰器。
LPIPS の高周波な脆弱部分は latent ペアからの平滑な回帰では再現されず、蒸留がローパスとして働く。

残る実際の懸念:
- **ドメイン不一致**: AlexNet/VGG + ImageNet + BAPPS(2018)。アニメ/イラスト分布での整合は無保証
- **見えないもの**: 大域的な色/輝度シフトに鈍い（→ 案 4 の担当なので分業可能）

一方 **ブロックノイズには比較的強い**。BAPPS の traditional distortion に JPEG 圧縮が含まれ、
8x8 グリッドのブロッキングはまさにそれ。SDXL の latent セル = 8x8px は JPEG グリッドと一致する。
加えて `vae_trainer.py` が既に LPIPS-VGG を使っており、既存の画質判断基準とも整合する。

### 教師は LPIPS である必要はない

案 1 の本質は「**GT に固定されたラベルをオフラインで作る**」ことであり、LPIPS は一実装にすぎない。

| 教師 | 性質 |
|---|---|
| LPIPS-VGG | 既存依存、ブロックに強い、ドメイン不一致あり |
| DISTS | テクスチャ再合成に寛容。ボケ/リサンプル系に妥当 |
| 自前 tagger の特徴距離 | ドメイン一致（danbooru）、AlexNet より遥かに新しい |
| **透明な物理量ベクトル** | **ブラックボックスゼロ（推奨）** |

**推奨は 4 番目**。critic に 1 個のスカラーではなく、解釈可能な少数ベクトルを回帰させる。

- latent セル周波数（8px 周期）の DCT/FFT パワー ← ブロックノイズそのもの。モデル不要で測れる
- GT 比の高周波パワー比 ← 眠さ／ボケ
- チャネル別 mean/std のずれ ← グレー化
- 局所分散マップの相関 ← 質感の潰れ

こうすると critic は「**decode して測る、という手続きの微分可能な代理**」になる。
教師側にブラックボックスがなく、loss が動いたときにどの物理量が動いたか読める。
critic の忠実度も held-out R² で直接検証できる。

---

## 6. latent tagger の検討

> tagger は pixel（を特定 pixel に縮小して学習）なので latent に適さないかもしれない。
> tagger 自体は 2〜3 日で学習できるので、latent space で新たに学習するのはアリか？

### 役割を分けること

| 役割 | いつ動く | 入力 | pixel tagger は？ |
|---|---|---|---|
| **T: 教師（ラベル生成器）** | オフライン 1 回 | decode 済み画素 | **問題なし。むしろ最適** |
| **C: critic（loss 項）** | 毎ステップ | latent | 不可（decode が要る） |
| **R: REPA ターゲット** | 毎ステップ | 現状 画素 384² | 動くが**高い＆歪む** |

「pixel なので latent に適さない」は **R と C については当たり、T については外れる**。

### R（REPA）に対しては明確に有効

- 入力が既にキャッシュ済みの latent → **画像ロードもリサイズも消滅**
- **ネイティブ latent 解像度 = DiT のトークングリッドと 1 対 1** → スカッシュ由来の歪みが消える
- エンコーダ本体も小さくできる（16ch×(H/8)² は 3ch×384² よりずっと軽い）

*既存機能のコスト削減 + 品質バグの解消* が同時に取れる。
「特定 pixel に縮小している」という懸念は REPA 用途では**本物の欠点**であり、latent 化はその直球の解決。

### しかし C（\*1 の critic）には向かない

**tagging は意味論タスクであり、学習は低次アーティファクトへの *不変性* を積極的に獲得する。**
ブロックノイズが乗ってもグレー寄りでも「1girl, solo」は変わらないし、変わってほしくない。

したがって tag 教師で学習した latent tagger は、
**8x8 ブロックノイズやグレー化を「見ないように」訓練された特徴**を持つ。
\*1 の検出器としては目的関数が逆向き。

- REPA（大域構造・意味の収束加速） → latent tagger ◎
- \*1 の低次アーティファクト検出 → latent tagger ✕

### 最大の落とし穴: VAE 固有

latent tagger は特定 VAE の latent 空間に完全に紐付き、**VAE を替えるたびに無効化される**。
今回の発端が VAE 差し替えである以上、「2〜3 日 × VAE の数」を毎回払う設計は苦しい。
pixel tagger は永久に VAE 非依存。

### 対案: ゼロから tag 学習せず、latent stem を蒸留する

```
latent ──[ latent stem (小、VAE ごとに再 fit) ]──▶ 既存 pixel tagger の中間特徴空間
                                                  （trunk は凍結・共有・VAE 非依存）
```

- 教師 = 既存 pixel tagger の patch 特徴。学習 = `(latent, 同じ画像)` ペアでの特徴蒸留
- **tag ラベル不要**。データはキャッシュ済み latent + 画像で足りる（tagger コーパス全体は不要）
- 収束はタグ教師のスクラッチ学習より桁で速い（回帰＋教師が濃い）。
  VAE 差し替え時の再 fit は日ではなく**時間オーダー**
- REPA のターゲット空間が変わらないので `repa_weight` / `repa_align_depth` と projector の意味が保存され、
  **差し替えリスクが小さい**

### 見落としやすいコスト

フル tag 教師で学習する場合、**計算より先にストレージが効く**。
tagger コーパスは拡散 finetune データセットより桁で大きく、数百万枚を 16ch latent でキャッシュすると
JPEG より遥かに嵩む。オンザフライ VAE encode に逃げると学習コストが増える。
蒸留案はデータ量が小さくて済むので、ここでも有利。

---

## 7. コスト見積もり（概算・未実測）

| 項目 | 概算 |
|---|---|
| critic 学習データ生成 | 実 latent は `LatentCache` 流用可。生成側は 1 sample = 1 forward。20 万 sample で単一 GPU 数時間 |
| critic 本体（`LatentCNN` 級 50K〜10M params） | 数時間。`aesthetic_trainer.py` 流用 |
| 訓練時オーバーヘッド（案 0/1/2） | 数 % |
| 訓練時オーバーヘッド（案 1'） | tiny crop の decoder fwd+bwd 分 |
| 訓練時オーバーヘッド（案 5） | rollout 分だけ数倍 → **実質不可** |
| latent stem 蒸留 | 時間オーダー、VAE ごとに再 fit |
| latent tagger フル tag 学習 | 2〜3 日 + 大量ストレージ、VAE ごとに再実施 |

**真のコストは critic の有効期間**である。案 0/2/3 は critic が「学習対象モデルの近傍」でしか有効でなく、
訓練中の定期再生成が必要。案 1 はラベルが固定されるためこれが軽く、案 1' では消滅する。

---

## 8. 推奨ロードマップ

### Stage 0 — 新規モデル不要・即効

1. **latent 正規化の確認（最優先）**。VAE を差し替えたなら、新 latent 空間のチャネル別 shift/scale が
   旧値のままだと収束は壊滅的に遅くなる。**他の何より先に確認すること。**
2. **案 4 の rollout メトリクスを `log_extra_metric()` に追加**。
   「長く流すか捨てるか」の判断材料が最短で手に入る = 今回の一番の痛みに直接効く。

### Stage 1 — 診断のみ、loss にしない

判別モデルを作るなら **まず loss ではなくメトリクスとして**投入。
予測値を t ごとの曲線としてログすれば、ハックのリスクゼロで診断価値だけ取れる。

### Stage 2 — loss 化

- **第一候補: 案 1'（小クロップ decode）**。ブラックボックスも蒸留誤差もない
- 次点: 案 1（paired critic、教師は透明な物理量ベクトル）
- 案 0 を採るなら必ず **案 2 の形（patch-wise）** で。追加コストほぼゼロの上位互換
- 適用は**中〜低 t に限定**。高 t では x̂₀ から復元可能な情報が数 % しかなく信号にならない
  （`scaled-head-init-verdict` の知見）
- 重みは diffusion loss の 1/10〜1/100 から warmup

### Stage 3 — REPA の latent 化（Stage 0〜2 と独立に進行可）

latent stem 蒸留により REPA を latent 入力化。収束加速と既存コスト削減。

---

## 9. 事前登録ゲート

| ID | 対象 | 条件 |
|---|---|---|
| **G-1** | 案 1 の成立 | 保留データで critic 予測値と教師値の順位相関 ≥ 0.9。特に「latent L2 が同程度でブロック有無が違うペア」の部分集合で有意に分離すること。**通らなければ Q2 が否定され、本実装前に破棄** |
| **G-A** | latent stem 蒸留 | 保留セットで pixel tagger の patch 特徴に対しコサイン ≥ 0.9。通れば「REPA の品質を落とさずコストだけ下げた」が保証される |
| **G-B** | スカッシュ歪みの実在 | アスペクト比が 1:1 から遠いバケツ（例 1152×832）で latent stem 版の REPA loss が現行より低い。差なしなら品質側の動機は消え、コスト削減のみが残る |

案 1' には G-1 相当のゲートが不要（蒸留を挟まないため）。

---

## 10. 未決事項

- 症状 A の切り分け: 純粋な exposure bias か、高 t 帯の学習不足か（timestep 分布の見直しで済む可能性）
- 案 1' のクロップサイズと decoder backward の実コスト（未実測）
- 案 4 のモーメント整合項を loss に入れるか、メトリクスに留めるか
- latent stem 蒸留を REPA 以外（案 1 の critic backbone）に流用するか
  — ただし §6 のとおり tag 由来特徴はアーティファクトに不変なので、期待薄

---

## 参照

- `backend/core/training/base_trainer.py` — `_setup_repa` (L2722), `_get_repa_pixels_for_item` (L2802)
- `backend/core/training/aesthetic_loss.py` — frozen latent critic を loss に足す既存パターン
- `backend/core/training/vae/vae_trainer.py` — LPIPS-VGG + クロップ decoder 学習
- `subapps/aesthetic_scorer/backend/core/` — `LatentCNN` / トレーナー / latent データ生成器
- `backend/core/training/metric_registry.py` — `log_extra_metric()`
