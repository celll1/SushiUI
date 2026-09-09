# 収束加速の検討: 補助損失と潜在空間 critic

**状態: 検討のみ（未実装・未決定）**  日付: 2026-09-06

本稿は候補と検証手順をまとめる。収束改善・安全性・所要時間は未実測であり、
以下の仮説を実装の保証として扱わない。

## 1. 背景と診断対象

VAE 差し替えなどで重みの一部を初期化した学習について、単一ステップの
training debug latent/pixel は改善しても、通し生成で白／グレー化やブロック状の
ノイズが残るという相談が出発点である。再現条件と定量測定は本稿には未収録。

| 症状 | 調べる仮説 | 切り分け |
|---|---|---|
| A. 通し生成の白／グレー化 | 誤差蓄積、高ノイズ帯の学習不足、正規化・予測形式・sampler/CFG の不一致 | 同一チェックポイントで単発予測と実運用の通し生成を比較 |
| B. ブロック状ノイズ | 局所予測誤差、decoder の応答、初期化した projection の未学習 | GT roundtrip と予測 latent の decode を比較し、周期性も測る |

両者が独立した原因とはまだ言えない。latent の DC ずれと画素のグレー化も同義ではない。
SDXL の空間圧縮率 8 は、欠陥が必ず 8px 周期になる証拠ではない。
既存の [VAE 実測](VAE_DECODE_BEHAVIOR.md) では 8px グリッドは再現されておらず、
今回の初期化モデルでも実画像と指標の両方で再現確認が必要。

求めるものは収束加速と、長期学習を続けるか判断する材料である。
まず診断を追加し、その指標だけで将来の収束可否を断定しない。

## 2. 既存資産と再利用の限界

| 資産 | 実装から確認できること |
|---|---|
| `backend/core/training/base_trainer.py::_setup_repa` と `repa.py` | 教師は tagger / SigLIP2 選択式。**当初 MiniT2I 専用だったのは実装のゲートであり手法の制約ではない**（2026-09-08 に横断化）。`ArchHandler.repa_tap()` を実装したアーキで有効。配線済み: MiniT2I, Anima, Lens, Krea 2, Ideogram 4, SenseNova U1.5, SD1.5, SDXL（U-Net は空間マップの tap。REPA 論文が検証したのは DiT であり、conv U-Net の mid block への適用は外挿）。Z-Image / FLUX.2 は保留（Z-Image は vendor ループの改変とテキスト prefix スライス、FLUX.2 は dual-stream と single-stream で tap の形が 2 つ。どちらも行優先の順序を実測で確かめるベースが手元に無い）。需要が出るまで未配線 |
| `base_trainer.py::_get_repa_pixels_for_item` | 正方形リサイズは存在する。サイズ S は設定 override または encoder native size 等から決まり、384 固定ではない。PIL decode/resize は LRU ミス時のみ、上限 4096 |
| `backend/core/training/aesthetic_loss.py::AestheticLoss.__call__` | frozen scorer のロード例。ただし forward が `torch.no_grad()` 内なので入力への勾配も切れ、微分可能な補助損失の成功例にはならない |
| `subapps/aesthetic_scorer/backend/core/aesthetic_model.py::LatentCNN` | 小型 CNN の素材。16ch 構成は 97,121 params。総 VRAM と速度は活性・入力サイズ・dtype・backward に依存 |
| `backend/core/training/vae/vae_trainer.py` と `VaeEpochCropSampler` | 画像クロップを encode→decode する VAE 学習。MSE / LPIPS / YCbCr loss を再利用できるが、拡散予測の latent クロップ学習とは異なる |
| `base_trainer.py::log_extra_metric` と `metric_registry.py` | DB カラム追加なしに指標を記録できる。表示用メタデータ等は既存の登録経路に合わせる |

REPA の正方形スカッシュは形状を変える。**この点は G-B で決着した**（2026-09-09、プラン 5-1）:
既定の教師 encoder 自身が正方形 384² の processor で学習されており、スカッシュは
その学習ドメイン外ではない。対照実験は不要になった（固定解像度系の教師について）。
REPA は clean-image 表現との整合を行う手法であり、低次の欠陥への感度は別途評価する。
[REPA 原論文](https://arxiv.org/abs/2410.06940)

## 3. 当初案: 単発予測から t を回帰する

提案は「完成 latent と異なる t からの単発予測を集め、t を予測する critic のスコアを
完成側へ動かす」こと。**t の向きを先に定義する必要がある**。
相談時の完成側 t=1 は、ノイズ側を t=1 とする訓練の規約と同一ではない。

予測 x̂₀ の復元は各 architecture の既存経路に合わせる。
`z_t = α x₀ + σ ε` の ε 予測なら `(z_t − σ ε̂)/α`（α≠0）だが、
v 予測や flow 予測へこの式をそのまま使ってはいけない。
例として `z_t=(1−t)x₀+tε, v=ε−x₀` の flow なら `x̂₀=z_t−t v̂`。
高ノイズ側の重み・適用帯域は t の数値だけでなく SNR と復元の安定性で決める。

主なリスクは次のとおり。

- t は品質そのものではなく、画像内容・ノイズ・モデルの学習段階にも依存する。
  同じ出力に複数のラベルがありうる回帰は条件付き平均を学習できるが、品質順序を保証しない。
- チェックポイントが変わると出力分布と t と品質の関係が変わり、再校正が必要になる。
- frozen scalar critic の最適化は代理指標を攻略するリスクがある。「必ずハックされる」とも、
  小さい重みなら安全とも言えない。オンライン更新も安定性の保証にはならない。
- 単発予測に現れない軌跡上の破綻は、その入力だけからは直接評価できない。

最初は品質ラベルとの相関を保留チェックポイントで確認し、診断用途に限定する。

## 4. 候補の比較

下表は期待する用途であり、有効性や耐性の実証ではない。

| 案 | 入力・教師 | 用途と未解決点 |
|---|---|---|
| 0. unpaired t 回帰 | 単発 x̂₀ → t | 安価な診断候補。品質との対応と分布変化が課題 |
| 1. paired critic | (x̂₀, x₀) → decode 後の距離 | GT に固定した目的を近似できる。近似誤差・勾配攻略が残る |
| 1'. クロップ decode | 予測と GT の対応領域を decode して直接比較 | 学習 critic は不要。境界・非局所性・backward コストを要検証 |
| 2. patch-wise critic | 局所予測 → 局所品質または t | 局在診断候補。画像全体の t を全 patch に付けても局所品質の教師にはならない |
| 3. online/lagged discriminator | real/fake | 現行出力に追従できるが、データ生成・更新コストと GAN の不安定性がある |
| 4. 統計モーメント | rollout の latent/画素統計 → 参照分布 | 大域異常の診断。内容と条件の違いでも変わり、品質を一意には決めない |
| 5. trajectory critic | 複数 step の状態 | 軌跡の異常を見られるが、rollout と学習データの費用が増える |
| 6. decoder-aware critic | decoder 中間特徴等 → 品質 | VAE 応答を使う候補。encoder 特徴だけでは decoder-aware とは言えない |

### 案 1: paired critic

固定 VAE・正規化・decode 設定・指標の下で、`指標(dec(a), dec(b))` はペアの関数になる。
オフラインで教師値を作り、訓練中は latent ペアから近似すれば毎回の decode を省ける。
ただし **GT を入力に加えても、学習した代理距離の抜け穴は消えない**。
VAE や評価手順を変えれば教師ラベル自体も作り直す必要がある。

複数チェックポイントの実予測と合成劣化を使い、画像・チェックポイント単位で保留集合を分離する。
値の相関だけでなく、critic を下げる更新で真の decode 指標も改善するかを検証する。
分布外入力、同程度の latent L2 で画質が異なるペア、低周波と高周波の両方を含める。

### 案 1': クロップ decode

これは全画像 decode の厳密な置換ではない。
[VAE の非局所性](VAE_DECODE_BEHAVIOR.md) により、ゼロ padding、GroupNorm の空間統計、
mid-block attention がクロップと全画像の結果を変える。
既存測定では局所境界項を消すだけでも 14〜16 latent cells の周辺文脈を要し、
8×8 latent（SDXL で 64×64px）単独の decode を有効な内側領域として扱えない。
これらの数値は既存 VAE の推論測定であり、新 VAE や勾配の一致を保証しない。

予測・GT とも対応する文脈付き領域を decode し、内側だけに loss を適用する候補を試す。
全画像 decode の同じ領域と値・勾配方向を比較し、残る非局所誤差を測る。
VAE パラメータを凍結しても、予測入力からの autograd は維持すること。
`no_grad()` や detach で予測側を囲むと拡散モデルを学習できない。

蒸留誤差はないが、選んだ画素指標の盲点は残る。LPIPS などの特徴抽出費用も含めて
実行時間とピーク VRAM を測る。既存 VAE 学習の計測条件は
[VAE_TRAINING_RESOLUTION.md](VAE_TRAINING_RESOLUTION.md) を参照する。

### 案 4: 統計モーメント診断

固定 seed・prompt・sampler・CFG・VAE・正規化で通し生成し、チャネル別 mean/std、
低周波パワー、decode 後の輝度・彩度・分散を記録する。参照データの内容・条件も揃える。
短い 4〜8 step rollout を使うなら、通常 step 数の結果と比較して solver の粗さを分離する。

統計一致だけでは、配置・意味・多様性や将来の収束を保証できない。
まずメトリクスとし、画像と併せて判断する。loss 化するなら rollout の勾配を
どこまで保持するかを明記し、メモリ・時間・条件追従性の悪化を検証する。
no-grad rollout の測定値を加算するだけでは学習信号にならない。

## 5. 教師指標

LPIPS は案 1 ではオフライン教師、案 1' では直接 loss の候補になる。
**蒸留が LPIPS の脆弱性を消す、または必ずローパスとして働く保証はない**。
元の指標と代理回帰器それぞれの盲点があり、直接最適化による抜け穴を検証する。
LPIPS 等への攻撃の実証は [Attacking Perceptual Similarity Metrics](https://arxiv.org/abs/2305.08840)
を参照。これは本稿の latent critic を直接検証した論文ではない。

| 教師候補 | 検証すること |
|---|---|
| LPIPS-VGG | 対象のアニメ/イラストでの視覚評価との整合、色ずれ・局所欠陥への感度 |
| DISTS | 対象の欠陥への感度と、許容するテクスチャ変化が目的に合うか |
| 自前 tagger 特徴 | ドメイン一致だけでなく、意味を維持したアーティファクトを検出できるか |
| 解釈可能な量のベクトル | 周波数パワー、GT 比の高周波比、mean/std、局所分散の変化を別々に評価 |

JPEG の 8×8px ブロックと SDXL の latent 1 セル相当の 8×8px が一致しても、LPIPS が今回のブロックノイズに強い証拠にはならない。
8px 周期パワーも正常な模様に反応しうるため、周期ずらし・非周期ノイズ・正常画像を対照にする。
透明な指標も不完全であり、held-out R²/順位相関だけで loss として採用しない。

## 6. REPA の latent 化

latent 入力ならオンラインの画像ロードを省ける可能性はある。
しかし **latent セルと DiT トークンは一般に 1 対 1 ではない**。
patch size、packing、圧縮率、crop/flip の対応に従って教師のグリッドを定義する必要がある。
入力要素数だけでは計算量を比較できず、encoder の層構成とトークン数も測定する。

tag 教師は意味に有用でも、低次アーティファクトを保持する保証はない。
逆に、tag 学習がそれらへの完全な不変性を保証するわけでもないので、感度は実測で判断する。
latent 表現は encoder と正規化に依存する。**decoder だけの変更なら入力 latent 空間は
必ずしも変わらない**が、decode 品質を教師にする critic のラベルは変わりうる。

候補は `(latent, 対応画像)` から pixel tagger の patch 特徴を蒸留するモデル。
タグラベルは不要だが、必要なデータ量・収束時間・特徴精度は未測定。
凍結 trunk を共有するなら、どの層へ stem を接続し、位置埋め込みとグリッドをどう扱うかを定める。
歪んだ教師をそのまま蒸留するだけでは前処理問題を解決したことにはならない。
同じ出力次元でも REPA loss の分布と勾配は変わるので、重み等の再評価が必要。

## 7. 費用の測り方

「数 %」「数時間」「日から時間へ短縮」という根拠のない見積もりは採用しない。
GPU、モデル、解像度、batch、dtype、オフロード、サンプル数を固定して測る。

- データ生成: 予測 forward、必要なテキスト処理・decode・教師指標・I/O を含める。
- loss: critic / decoder / 特徴抽出の forward と入力 backward、活性保存を含める。
- rollout: 実行頻度も含む平均費用と、単回のピーク VRAM を分ける。
- latent 蒸留: キャッシュ容量、VAE encode、教師抽出、再 fit の費用を含める。

既存 `LatentCNN` の重みサイズだけから総 VRAM を上限見積もりできない。
案 1 は入力分布の更新、案 0/2/3 は教師関係や判別分布の更新費用も評価する。

## 8. 推奨ロードマップとゲート

1. 正規化、VAE encoder/decoder の整合、予測形式、sampler、CFG、キャッシュの由来を確認する。
   encoder を変更した場合は既存 latent キャッシュを再利用できるか確認する。
2. 固定条件の単発予測・通し生成・GT roundtrip を保存し、案 4 の指標を記録する。
3. critic はまず診断専用。loss 候補は案 1' の全画像との比較、または案 1 の代理勾配試験から始める。
4. loss の適用帯域は noise/SNR 規約を明記し、係数は値の比だけでなく主 loss に対する
   勾配ノルムと品質の変化を見て決める。patch-wise を無条件の上位互換とは扱わない。
5. REPA の latent 化は独立した実験とし、同一条件の pixel 教師版と比較する。

| ID | 採用前の検証 |
|---|---|
| G-1 | paired critic の保留画像・チェックポイントで順位相関（目安 ≥0.9）、誤差、欠陥別識別を確認。さらに代理最適化が真の指標と画像を改善するか確認。失敗は現モデル/データの棄却理由であり、理論上の予測可能性の否定ではない |
| G-C | crop と全画像で同じ領域の指標・入力勾配を比較し、文脈サイズごとの誤差と費用を測る。許容差を実験前に定める。蒸留なしでもこのゲートは必要 |
| G-A | latent 教師の保留 patch cosine（目安 ≥0.9）に加え、固定予算・複数 seed の学習 A/B で生成品質・収束・時間・VRAM を比較。特徴類似度だけで品質維持を保証しない |
| G-B | **不合格（2026-09-09、プラン 5-1 に記録）**。学習 A/B も GPU も不要だった: 既定で選ばれる教師 `siglip2-so400m-patch14-384` の processor 自体が 384×384 の正方形リサイズであり、squash は教師の学習ドメイン外ではない。空間対応も厳密に相殺する。結論は固定解像度系の教師に限る（NaFlex 指定時とリサイズフィルタ差はプラン 5-1 の残存条件） |

閾値は暫定案であり、合格した場合も対象条件外への一般化は未確認。
本稿だけではどの案も収束加速策として採用決定できない。

## 参照

- `backend/core/training/base_trainer.py` — `_setup_repa`, `_get_repa_pixels_for_item`, `log_extra_metric`
- `backend/core/training/repa.py` — 教師特徴とグリッド補間
- `backend/core/training/aesthetic_loss.py` — scorer のロード例（入力勾配なし）
- `backend/core/training/vae/vae_trainer.py` — 画像 crop の encode/decode 学習
- `subapps/aesthetic_scorer/backend/core/` — scorer とデータ生成の素材
- [VAE decode の実測と限界](VAE_DECODE_BEHAVIOR.md)
- [VAE 学習のクロップ形状と費用](VAE_TRAINING_RESOLUTION.md)
