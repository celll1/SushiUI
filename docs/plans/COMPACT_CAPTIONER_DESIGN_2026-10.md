# SigLIP2とFoveaTagに対応する軽量captionerの実装設計

2026-10-01。状態: **実装前の設計**。本書のモデル構成、API、数値は提案であり、
品質や速度の実測結果ではない。既存のSigLIP2 taggerを画像encoderとして共有し、
画像の空間特徴と明示的なタグ概念から、短い自然言語captionを生成する専用decoderを学習する。
蓄積済みのimage、tags、captionの組を教師データとして使う。
別ブランチで開発中のFoveaTagも共通feature providerを通して接続する。
依頼中の「ForbeTags」は、確認できたFoveaTagを指すものとして本書では扱う。

目標は、ロード済みモデルで1画像の全文出力を数秒以内、できれば1秒以内に収めること。
既存so400mの推論時間は利用者にとって許容範囲であるため、画像encoderの小型化を初期要件にしない。
汎用LLMのfine-tuningを導入せず、caption用のprojection、概念embedding、decoderを新規学習する。
実装担当者が入力契約、学習境界、成果物、検証条件を判断できるよう、初期構成を一つに固定する。

## 既存実装と設計の境界

- [TAGGER.md](../guides/TAGGER.md): legacy WDとSigLIP2の所有関係、推論、学習、ONNXの境界。
- `backend/core/tagger/siglip2_tagger_model.py`: defaultは
  `google/siglip2-so400m-patch16-naflex`。通常poolerとCustomAttentionPoolingを持つ。
  現在のforwardはlogitsのみを返す。LoRA版も通常poolerから分類する。
- `backend/core/tagger/siglip2_inference_manager.py`: full、LoRA、ONNXのロードと推論を所有する。
- `backend/core/tagger/tag_selection.py`: 表示用閾値、校正、不確かなタグの除外等を所有する。
- [DATASETS.md](../reference/DATASETS.md): dataset ID、revision、caption選択、sidecar同期の契約。
- `backend/database/models.py`: DatasetCaptionはcaption_type、is_tags_format、tag_dataを保持する。

captionerは独立した補助サブシステムとする。ModelTypeやARCH_REGISTRYへ画像生成architectureとして
登録しない。SenseNova img2txtとは別のモデル、ロード状態、学習runを持つ。
将来の共通テキスト結果表示は可能だが、画像生成pipelineにcaptionerの常駐を強制しない。

## 入力情報の役割

画像hidden stateに、外部辞書のタグ名称がそのまま符号化されているとは仮定しない。
線形headの`z_i = W_i h + b_i`は概念への分類方向を表し、辞書は出力indexに名称を対応づける。
一方でlogitsはhから計算できるため、追加の視覚情報を増やすわけではない。
明示的なconcept tokenは、既存の分類知識と名称対応を小型decoderが利用しやすくする手段である。
その有効性はアブレーションで確認する。

| 入力 | 用途 | 制約 |
|---|---|---|
| last_hidden_state | 構図、動作、対象間の関係、局所属性 | 有効パッチmaskと空間座標を保持 |
| 実際にtag headへ入ったpooled特徴 | 画像全体の概念要約 | custom pooler時はその出力を使用 |
| 予測タグとscore | 分類済み概念の明示 | 辞書とcheckpointの対応を検証 |
| 任意の利用者tags | 修正済み概念や既知名称 | 予測scoreと利用者指定を区別 |

名称ごとのembeddingは意味を自動理解するものではない。初期版では学習可能なtag-ID embeddingを
caption教師から獲得し、tagger語彙にない利用者タグは警告付きで除外する。
caption tokenizerで名称を埋め込む方式と、head行から初期化する方式は比較候補に留める。
head行は意味embeddingそのものではなく、custom poolerでは画像tokenとも次元が異なる。

## 初期モデル構成

```text
image → checkpointと同じprocessor → SigLIP2を1回実行
                                     ├─ patch特徴 + mask + 座標
                                     │    → projection → 32 query resampler
                                     ├─ 実際のpooled特徴 → global projection
                                     └─ tag head → logits → concept selector
user tags ────────────────────────────────────→ concept selector
                                                    ↓
                                 tag-ID + score + category + source embedding
                                                    ↓
                   visual 32 tokens + global 1 token + concept ≤64 tokens
                                                    ↓
                               4層Transformer decoder → caption + EOS
```

以下は初期実験の値であり、実装時のAPI defaultは必ず`backend/api/param_defaults.py`に置く。
checkpointに学習済み構造を保存し、ロード時にAPI defaultで上書きしない。

| 項目 | 初期値 | 理由 |
|---|---|---|
| vision encoderとtag head | 選択したSigLIP2 checkpoint、全freeze | tagger性能を保持 |
| decoder hidden | 384 | 小型decoderの容量と速度を比較可能にする |
| decoder | 4層、6 heads、FFN 1536、dropout 0.1 | 全memoryへのcross-attention |
| visual resampler | 32 learned queries、6 heads、1 cross-attention層とFFN | 可変パッチ数を固定memoryに圧縮 |
| concept数 | 最大64、category filter後のTop-K | 入力長を制限 |
| tokenizer | caption corpusで学習したBPE、語彙8192 | 対象言語とドメインを教師から学ぶ |
| tokenizer特殊token | PAD、BOS、EOS、UNK、byte fallback | 未知綴りの処理を定義 |
| 最大生成長 | BOSを除きEOSを含む96 tokens | 全文レイテンシを制限 |
| decoding | greedy、beam 1 | 初期速度測定を単純にする |

word embeddingと出力headはweight tyingする。画像tokenにはprojection後に2D座標MLPの出力を加え、
resamplerが位置を識別できるようにする。memoryにはvisual、global、conceptのtype embeddingを加える。
decoderはcausal self-attentionとmemory cross-attentionを持ち、inferenceではself KVと
cross KVをcacheする。学習ではteacher forcingで全targetを並列処理する。
2層GRUとattentionは速度比較用の第二候補とし、両方式を初期実装で同時に製品化しない。

追加モデルの規模は10M～30Mを目安とするが、tag語彙Vでconcept embeddingだけでも`384 × V`増える。
構築時にmodule別の実parameter数、dtype別容量、peak VRAMを出す。so400mを含む総容量と追加容量を
分け、速度や容量を未測定のまま断定しない。

## 特徴取得の契約

既存`forward()`のlogits戻り値を保ち、専用`forward_features()`を追加する。
既存forwardも同じ内部処理を呼ぶ設計とし、caption要求一回につきvision encoderは一回だけ実行する。
一時hookや共有変数から特徴を拾う方式は、並行要求で取り違えるため使用しない。

返却構造は`patch_features [B,N,Dv]`、`patch_mask [B,N]`、`patch_xy [B,N,2]`、
`pooled_features [B,Dp]`、`tag_logits [B,V]`、model identityとする。
Dv、Dpはcheckpointから導出し、1152を共通の固定値にしない。
custom pooler時は通常pooler_outputと取り違えず、headへ実際に渡す特徴を返す。

NaFlexのspatial_shapesから、processorと同じflatten順で正規化2D座標を作る。
paddingはresamplerでmaskし、batch内の他画像のpatch数で結果が変わらないことを検証する。
固定解像度encoderにも同じ形式を提供する。座標生成とprocessorの対応はfixtureで固定する。
一様cropでcaptionが記述する背景・人物を失わないよう、前処理はcheckpointのprocessor設定を尊重する。

初期版はロード済みの固定PyTorch fullまたはLoRA checkpointを対象にする。
学習中のlive taggerは重みと語彙が変わるのでcaptioner共有対象にしない。
現在のONNX成果物がlogitsしか公開しない場合、caption能力なしを返す。
内部node名の探索でhiddenを取り出さず、将来のversioned feature exportで対応する。

## FoveaTagの接続設計

2026-10-01に`codex/tagger-hires-architecture-design`の`61ee6049`を読み、
`backend/core/tagger/foveatag/{encoder,model,head,inference,batch}.py`と
`docs/audits/HIGH_RES_TAGGER_ARCHITECTURE_RESEARCH_2026-09.md`を確認した。
これらは本設計のbaseには未統合であり、ここでのパスは当該ブランチの参照である。
実装開始時に変更点と採用checkpointを再確認する。作業中ブランチのファイルは本コミットへ複製しない。

FoveaTagはfrom-scratchの共有multi-view encoder、overviewに基づくrouter、
局所view、multiple-instance tag headから成る。SigLIP2重みを初期化に使わない方針を尊重する。
captionerのためにSigLIP2を追加実行したり、両architectureのencoder重みを混ぜたりしない。
タグ語彙が一致しても特徴分布が違うため、caption weightsの無調整共用は保証しない。

確認時点のencoder出力はlocal_map、local_coordinates、global_tokens、global_coordinates、
global_summary。modelはglobal_summaryをprojectionし、局所mapをviewごとのslotへpoolして
projectionする。headはoverviewと局所slotを混合し、global logitに局所scoreのsmooth maximumを加える。
`return_details=True`はmixed_evidence、local_scores、best_evidence_indicesと
evidence_view_indicesを返すが、overview空間特徴と投影後global_summaryは公開していない。
`predict_image()`はoverviewを一度encodeし、routeとlocal classifyへ再利用する。
最終結果はCPU logits/score、selected_boxes、evidence_boxes、stage timingsであり、
caption用device memoryは現在のpublic resultには含まれない。

共通providerはSigLIP2専用のpatch/pooled名称を外へ強制せず、
`spatial_tokens`、`spatial_mask`、`spatial_xy`、`global_features`、`tag_logits`、
任意`evidence_tokens/mask/boxes`、任意`tag_evidence_indices`、backend identityを返す。
providerごとのprojectionがdecoder hidden 384へ合わせる。
captioner model configには`feature_backend=siglip2|foveatag`とschema versionを保存する。

| FoveaTagの取得元 | caption側の入力 | 注意点 |
|---|---|---|
| overview.global_tokensとglobal_coordinates | 32 query resamplerへの空間入力 | 全体構図を残す |
| model.global_projectionの出力 | global token | headと同じglobal経路を使う |
| head_output.mixed_evidence | 局所evidence tokens | local_view_maskをslot数分展開 |
| selected viewのsource box | evidenceのgeometry embedding | 正規化source座標とfield of viewを保持 |
| head_output.logits | 共通concept selector | SigLIP2と別のscore分布として学習 |
| best_evidence_indicesとevidence_view_indices | conceptに付随する任意の根拠box | -1と局所根拠なしを明示 |

初期FoveaTag接続はoverview32 tokens、global1 token、最大16 evidence tokens、
最大64 conceptsをdecoderへ渡す。evidenceの上限16は初期実験値であり、
taggerのlocal view数をcaptioner都合で勝手に変えない。
slot数が上限を超える場合は全valid slotから独立resamplerで16へ圧縮し、
圧縮tokenを単一の正確な根拠boxとは扱わない。box付きconceptには元の対応を残す。
初期版はmixed_evidenceを再利用し、全局所mapの再poolや局所encoder再実行を避ける。
細部が不足する場合のみlocal map取得を追加する独立実験を行う。

tagごとのbest evidenceは局所scoreを最大にした領域であり、対象物のbox、
タグの正しさ、globalより強い根拠を保証しない。confidenceに置き換えない。
同じ人物の属性同士をboxだけで結合すると、重なるcropや複数人物で誤るため、
conceptへのgeometry補助はアブレーションを通して採用する。
局所tokenのみから全文を作らず、overviewを必ず残す。

staged inferenceに要求局所のfeature return optionを追加し、overview、routing、local encodeの
同じ実行からcaption memoryを得る。既存結果のCPU転送前に要求所有のdevice tensorsを受け渡す。
共有hook、encoder再実行、全tagのlocal_scoresを永続cacheする方式は避ける。
必要なTop-K conceptの根拠indexだけを保持し、要求完了後にmemoryを解放する。
「encoder一回」の検証はFoveaTagではoverview一回と各選択local view一回という意味にする。

初期学習ではencoder、router、pooler、evidence headをfreezeする。
予測時と同じrouter/modeから特徴を作り、学習時のrandom viewだけで作ったcaptionerを
routed inferenceへそのまま持ち込まない。view選択、NMS、overview/local size、
pyramid identity、decode cap、geometry、router重みhashをcacheとbundle契約に加える。
routerやmode更新時は語彙が同じでも互換性を再評価する。
大画像のregion-readable source契約とraw decode capを守り、
region cache構築を初回caption要求の隠れた仕事にしない。

API routeとdataset/trainerは共通とし、load/statusにfeature_backend、
対応view mode、feature export能力を追加する。backendごとのproviderが未提供なら
feature非対応を返し、SigLIP2へ黙ってfallbackしない。
速度はoverview、router、region read/local encode、head、caption decoderに分解し、
各view modeのwarm p95を別々に測定する。FoveaTagの速度をSigLIP2測定から推定しない。

採用前にoverview-only、overview+concept、overview+evidence、全入力を比較する。
複数人物への属性誤結合、小物・目・装飾など局所属性、K=0、全local mask無効、
slot/view/box対応、router更新時のcache失効を検証する。
FoveaTag本体の品質gate通過とcaptionerの品質・速度gateは別々に記録する。
decoder構造、tokenizer、concept処理コードは共有できるが、初期版の学習済みbundleはbackend別にする。
両backendを混合して学習する場合は、その比率、provider別projection、backend別評価を別実験として保存する。
provider契約は最初から共通化し、FoveaTag接続と学習は固定checkpointおよび
安定したfeature contractが用意できた時点で第二実装段階として行う。

## 概念入力の契約

予測経路ではraw logitsからsigmoidした未校正scoreを利用する。
これは真の確率やconfidence保証ではない。UIの校正、OOD、共起補正、表示閾値に連動させず、
学習と推論で同一selectorを使う。selector versionを成果物に保存する。

category allowlistはGeneralを初期候補とし、Character等はデータ品質を見て実験する。
Artist、Quality、Rating、メタタグは初期caption入力から除外する。
選択は許可語彙のscore上位64件、tieはcanonical name順とする。
表示用thresholdで低score情報を全部落とさず、score embeddingから記述要否を学習させる。
全concept欠落時もvalid maskを持つempty tokenで処理できるようにする。

conceptは`E_tag(id) + MLP(score) + E_category + E_source`で表現する。
scoreでembedding全体を単に乗算すると、低scoreの概念identityまで消すため初期方式にしない。
タグの入力位置embeddingは付けず、学習時に順序をshuffleする。
canonical nameは既存alias resolverで正規化し、checkpoint語彙のindexに解決する。

`tag_source`は次の明示modeを持つ。

- `predicted`: 画像から予測したconceptのみ。
- `provided`: 指定tagsのみ。未指定ならrequest error。空リストはimage-onlyを意味する。
- `merged`: 指定tagsを優先し、残りを予測conceptで補う。重複は統合する。

providedは「利用者指定」のsource tokenを持ち、score欠測を専用表現で示す。
1.0を校正済み確率として扱わない。上限超過時は正規化済み指定順で切り、警告と採用tagsを返す。
unknown tagsも返却警告に残す。将来のopen-vocabulary入力は別versionとする。

## Datasetと学習

dataset entryには`dataset_id`、既存filters、`input_tag_caption_types`、
`target_caption_types`を別々に指定する。同一itemのtags-format captionと自然文captionをjoinする。
targetはis_tags_format=falseかつ非空、provided入力はtags-formatを要求する。
caption column名だけで形式を推定しない。欠落、unknown、過長target、画像不在を集計する。

複数targetは独立sampleとして無制限複製せず、itemを先に抽選してからtargetを一つ選ぶ。
複数datasetはeligible item数に比例したsamplingを基本とする。
同一画像のcaption差分、完全重複、近似重複をgroup化し、group単位でtrain/val/testへ分割する。
初期split比率90/5/5、seed固定。testはモデル選択に使わない。
作品・連続画像単位の漏洩も可能ならgroupに含める。推論用taggerの学習データとの重複は別に記録する。

captionの言語、token長分布、タグ数、カテゴリ、画像形状、captionにないタグの比率を事前に調べる。
実dataset件数・分布は本書では未確認。初期モデルは選択corpusの言語を対象とし、
英語と日本語を暗黙に混合しない。混合する場合はlanguage tokenと評価groupを設ける。
tokenizerはtrain splitだけから学習する。過長targetを文中で無条件truncateせず、
coverageを報告し、96 token以内で完結する教師subsetから始める。

初期学習は全vision、pooler、tag headをevalかつrequires_grad=falseとし、caption側だけ更新する。
lossはPADを除くtarget token CE、EOSを含む。label smoothing 0.05を実験値とする。
AdamW、LR 3e-4、weight decay 0.01、5% warmup、gradient clipping 1.0を初期候補とし、
batch sizeとepoch数は実dataとVRAMを測って決める。bestはvalidationの品質で選び、CEのみで決めない。

入力sourceの初期混合はpredicted 70%、provided 20%、merged 10%。providedが存在しないitemは
predictedへ戻す。教師tagsのみの成功を本番予測経路の成功と取り違えない。
concept dropoutは各token 0.15、全concept dropoutはsample 0.1を初期候補とする。
乱数、source抽選、target抽選、shuffle、dropoutをresume stateに含める。
誤タグ混入は実taggerの誤差評価後に追加し、一律ランダム置換を最初から必須にしない。

encoder適応は後段の独立実験とする。最後の層へのLoRAを試す場合はcacheを無効にし、
tagger精度とcaption品質を共に測定する。caption lossで本番taggerを直接変更しない。

## 特徴cacheと成果物

freeze中はpatch、pooled、logitsを事前計算できる。ただし学習可能なresamplerの出力だけを
cacheするとresamplerを学習できないため、raw特徴を基本とする。
NaFlexのNが大きい場合、cacheは巨大になる。例として`N=256, Dv=1152, fp16`で
patch特徴だけ約0.56 MiB/image、10万画像で約55 GiBとなる。N=1024なら約4倍。
開始前に実N分布とlogits容量を含めた見積もりを出し、shard容量とcache上限を設定する。
容量不足ならオンラインfreeze推論を使う。固定grid圧縮は別の品質比較を通して採用する。

cache keyは画像content hash、dataset revision、encoderとpooler/headの重みhash、LoRA identity、
processor config、解像度・patch budget、dtype、feature schema versionを含む。
targetcaption編集ではfeatureを再計算せず、label snapshotとsplit manifestは更新する。
cacheへtrainable projectionを含める場合、その重みhashとfreeze条件も必須とする。

推論bundleはcaption weights、tokenizer、model config、concept vocabulary、category map、
selector version、tagger identity、processor identity、学習source mode、言語、評価summaryを保存する。
tag語彙はcanonical nameで照合する。完全一致hashを初期ロード条件とし、
語彙追加・並べ替えを自動受理しない。将来のname-based remapは検証済みversionで導入する。
tagger更新時は、同じarchitectureでもfeature分布が変わるためcaptioner互換とみなさない。
ベースtaggerはbundleに必ず複製せず、外部dependencyとしてhashで参照できる。

resume checkpointはoptimizer、scheduler、RNG、dataset snapshot、split、tokenizer hash、
sampling state、step/epochを含める。exportとresume用成果物を区別し、atomic保存する。
partial/corrupt bundle、語彙衝突、feature dimension不一致はロード前に拒否する。

## 実装所有とAPI案

新規所有候補は`backend/core/captioner/`。model、concept selection、dataset、feature cache、
trainer、artifact、inference manager、feature provider interfaceとbackend別providerを分ける。
SigLIP2特徴取得だけは既存tagger model側に置き、旧predictのresponseとselection契約を維持する。
captioner managerはtagger modelのborrow lifetimeを管理する。要求中のreload/unloadを排他し、
既存GPU調停とpause/offloadに従う。二つのmanagerが同じencoderを別々にGPUへロードしない。

以下のroute名は提案であり、現時点では存在しない。実装時は[ADD_A_PARAMETER.md](../guides/ADD_A_PARAMETER.md)
に従い、openapi.yamlを先に更新する。

| API候補 | 目的 |
|---|---|
| `POST /api/v1/captioner/load` | bundle読込、tagger一致と能力確認 |
| `GET /api/v1/captioner/status` | identity、対応mode、言語、feature能力 |
| `POST /api/v1/captioner/predict` | multipart image、tag_source、tags、max_new_tokens |
| `POST /api/v1/captioner/unload` | caption重み解放、borrowed encoderは所有者に返す |
| `/api/v1/captioner-training/*` | 開始、状態、停止、resume、sample、成果物 |
| `/api/v1/schema/captioner-defaults` | 推論設定の単一default source |
| `/api/v1/schema/captioner-training-defaults` | 学習設定の単一default source |

CAPTIONER_DEFAULTS、CAPTIONER_TRAINING_DEFAULTSをparam_defaults.pyに追加し、
request schema、frontend API型、設定保存、resumeへ同じ値を伝搬する。
構造変更と推論設定を分け、max_new_tokensはbundle上限以下を要求する。
responseはcaption、used_tags/source/score、warnings、EOS到達または長さ上限終了、model identity、
token数、phase timingsを含む。tags欄のscoreは校正有無を明示する。
unknown model、ONNX feature非対応、tagger mismatch、入力不正を異なるerror codeで返す。

Dataset/Tagger側にcaption previewとbatch実行を追加する。自然文はtags sidecarへ上書きしない。
保存は利用者操作で対象caption_typeへ行い、既存dataset caption transactionとrevision更新を使う。
batchは各画像でキャンセル可能にし、成功・失敗をitem別に残す。
captioner専用run stateを既存run persistenceとprogress方式へ接続し、
tagger-trainingのパラメータへ便宜的に混在させない。

## 評価と採用条件

同じsplit、tokenizer、caption decoder容量で以下を比較する。

| 実験 | 確認する点 |
|---|---|
| pooledのみ | 最小画像条件の基準 |
| patch + pooled | 空間特徴の寄与 |
| predicted conceptのみ | タグ文章化だけの能力 |
| patch + pooled + hard概念 | scoreを省いた場合 |
| patch + pooled + soft概念 | 本命構成 |
| patch + pooled + 正解tags | 教師入力時の参考値、保証された上限ではない |
| 正しいtags + 無関係image、正しいimage + 誤tags | 各branchへの依存と幻覚 |

soft入力の改善は仮説である。品質が改善しない場合はhard入力または画像のみを選べる。
評価は人物数、属性、服装、動作、空間関係、背景、固有名詞を含む同じheld-out集合で行う。
tagger同士の一致をcaption正解の代用品にせず、画像とcaptionを人手で照合する。
100～200件のblind paired評価を開始点とし、事実誤り、重要情報欠落、自然さ、不要なタグ列挙を記録する。
希少概念と複数人物を別集計し、必要なら評価件数を増やす。
CIDEr等の自動指標とCEは補助に留め、tagsの単語一致だけを主要指標にしない。

速度はbatch=1、モデル常駐、指定device/dtype/patch budget、実caption長で測定する。
20回warmup、最低200件の計測を開始点とする。GPU timingsは同期して計測する。
request全体のwall timeと、画像decode、processor、encoder、concept/resampler、decoderを分ける。
GPU待ち時間、cold load、初回compileは別列に残し、除外によってUXの遅さを隠さない。
既存tagger単独の同条件baselineも測定する。1秒目標はqueue待ちを除くwarm request p95で判定し、
decoderだけの時間を全体速度と呼ばない。

採用条件は、soft構成が画像のみ・conceptのみとの比較で事実性を改善し、
既存taggerのlogitsと表示タグを変更せず、目標deviceのwarm p95が1秒以内であること。
1秒を超えても3秒以内なら速度目標の部分達成として報告し、採用判断は品質との実測比較で行う。
3秒を超える場合は最大長、decoder寸法、patch budget、実行backendの順に比較する。
数十～数百msという追加速度や1秒達成は、本書では未検証である。

## 実装順序と検証

1. dataset coverage、言語、長さ、split manifestを作り、freeze tagger baselineを測る。
2. feature forwardとconcept selectorを実装する。旧forwardのlogits一致、custom pooler、LoRA、
   NaFlex padding、語彙hash拒否をCPU fixtureで確認する。
3. tokenizer、resampler、decoder、cache、trainer、resumeを実装し、小subsetをoverfitできることを確認する。
   teacher forcingとincremental KV cacheのlogits一致、EOS、空concept、maskを検証する。
4. held-out品質とbranchアブレーションを行い、構成を選ぶ。tagger update互換性は失敗を明示する。
5. OpenAPI、default source、推論と学習API、GPU調停、UI、dataset保存を接続する。
   predictがencoderを一回だけ呼ぶことと、cancel/reload/unload競合を検証する。
6. FoveaTagの固定checkpointに共通providerを接続し、overview/evidence/geometryの
   アブレーション、mode別速度、router/cache互換性を検証する。
7. full bundleで品質・warm p95・容量を測定してから、必要ならONNX/INT8を追加する。
   export後も本文品質、EOS、mask、cache、dtype差分を再検証する。

backend編集時はrepo venvで変更ファイルのpy_compileと実importを両方行う。
GPUを占有中ならAGENTS.mdのCUDA trigger stubを使い、serverは直接起動しない。
frontend build/type-checkはrepo ownerが行う。
本書のコミットは設計書のみで、モデル学習、benchmark、API変更を実施した証拠にはならない。
