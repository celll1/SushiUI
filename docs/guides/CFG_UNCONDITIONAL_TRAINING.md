# CFG と無条件枝学習の数学的根拠

## 1. 目的と監査結論

この文書は、次の主張の成立範囲を明確にする。

> 条件付き生成モデルと、それと整合する無条件生成モデルを学習すれば、
> classifier-free guidance (CFG) によって条件への追従を強められる。

この主張の核は正しい。ただし、次の区別が必要である。

- caption dropout は、**共有モデルへ無条件例を供給する一方式**であって、CFG の数学そのものではない。
- dropout 方式で null 条件の出力をデータ周辺分布と同一視するには、dropout 後の画像分布が元の画像分布と一致しなければならない。
- 各時刻の guided field は、条件付き・無条件の noisy marginal から作る power tilt の score と代数的に対応する。一方、guidance scale が 1 を超える場合、その時刻族を積分した終端分布がデータ時刻の単純な power tilt になるとは一般にいえない。
- SDXL は共有 U-Net + caption dropout の標準形だが、無条件なのは text についてだけで、size/crop micro-conditioning は残る。
- SD1.5、Z-Image、Anima、および non-distilled の FLUX.2 / Krea2 も、学習と推論の empty-text 表現が一致する共有-model 型である。
- MiniT2I は一般 caption dropout ではなく、推論と一致する mask-zero の専用 label drop を使う。
- SenseNova は共有 flow denoiser 型だが、training の empty-caption prefix と inference の uncond prefix は一致しない（既定の CFG norm も線形 blend を変更する）。整合させるには encode 時点で推論 uncond の query を組む専用 `cfg_uncond_drop_rate` を使う。
- Lens も training の empty chat template と inference の zero-feature null が一致しない。
- LTX-2.3 は text-null 表現が一致するが、現行 fine-tuning loss は video だけを直接監督する。
- MiniMax-H3 は guidance-distilled で CFG 枝を持たず、empty prompt 自体を拒否するため、この命題の適用外である。
- ACE-Step 1.5 も guidance-distilled であり、空 caption は CFG null branch ではない。MiniMax Music 3 は推論 CFG を持つが、現行 repo では training 非対応である。
- Ideogram 4 の実装は caption dropout ではない。別々の conditional / unconditional transformer を同じ全サンプルで同時に学習する asymmetric CFG であり、dropout 率の議論は適用されない。

提示された検討は、score、Bayes 分解、implicit classifier、時刻ごとの tilt、tilt と noising の非可換性については概ね正しい。一方、「velocity なら一般に厳密同値」「guidance scale は classifier posterior の温度そのもの」「両枝の実効サンプル数は常に $pN$ と $(1-p)N$」「データセット別 dropout 率は必ず不正」という表現は強すぎる。以下では、定義・仮定・導出・近似を分けて修正する。

## 2. 記号

- $x_0$: データ（実装上は VAE latent でもよい）
- $c$: caption などの条件
- $x_t = \alpha_t x_0 + \sigma_t \epsilon$, $\epsilon\sim\mathcal N(0,I)$: Gaussian affine noising path
- $p_t(x_t\mid c)$, $p_t(x_t)$: 条件付き、無条件の noisy marginal
- $s_c(x_t,t)=\nabla_{x_t}\log p_t(x_t\mid c)$
- $s_u(x_t,t)=\nabla_{x_t}\log p_t(x_t)$
- $w$: CFG scale。本リポジトリの式では $w=1$ が通常の条件付き枝、$w=0$ が無条件枝

以下の「最適解」は、母集団損失、十分なモデル容量、大域最適化を仮定した関数空間上の解である。有限データ、LoRA の制約、最適化未収束、数値積分誤差は別の誤差源になる。

## 3. noise prediction と score

### 3.1 二乗損失の最適解

noise-prediction 損失を

$$
\mathcal L_\epsilon(f)=
\mathbb E\left[\lVert f(x_t,t,c)-\epsilon\rVert^2\right]
$$

とする。条件付き期待値の直交性により、任意の $f$ について

$$
\begin{aligned}
\mathbb E[\lVert f-\epsilon\rVert^2\mid x_t,t,c]
&=\lVert f-\mathbb E[\epsilon\mid x_t,t,c]\rVert^2\\
&\quad+\mathbb E[\lVert\epsilon-\mathbb E[\epsilon\mid x_t,t,c]\rVert^2\mid x_t,t,c].
\end{aligned}
$$

第2項は $f$ に依存しないため、導出結果は

$$
f^*(x_t,t,c)=\mathbb E[\epsilon\mid x_t,t,c]
$$

である。

### 3.2 Gaussian kernel からの score 恒等式

$$
q_t(x_t\mid x_0)=\mathcal N(\alpha_t x_0,\sigma_t^2I)
$$

なので、

$$
\nabla_{x_t}\log q_t(x_t\mid x_0)
=-\frac{x_t-\alpha_t x_0}{\sigma_t^2}
=-\frac{\epsilon}{\sigma_t}.
$$

周辺化した密度を微分し、posterior で平均すると、

$$
\begin{aligned}
\nabla_{x_t}\log p_t(x_t\mid c)
&=\mathbb E[\nabla_{x_t}\log q_t(x_t\mid x_0)\mid x_t,c]\\
&=-\frac{1}{\sigma_t}\mathbb E[\epsilon\mid x_t,c].
\end{aligned}
$$

従って、noise-prediction の母集団最適解は

$$
\epsilon_c^*(x_t,t)=-\sigma_t s_c(x_t,t)
$$

である。これは denoising score matching の標準的な対応である。

## 4. null 条件が無条件分布を表す条件

ここだけは caption dropout 方式の議論である。$D=1$ を「条件を null に置換した」とする。null 入力に対する最適解は自動的に全データの無条件解になるのではなく、

$$
\epsilon_\varnothing^*(x_t,t)
=\mathbb E[\epsilon\mid x_t,D=1]
$$

であり、$p_t(x_t\mid D=1)$ の score を学ぶ。

### 4.1 十分条件と、より正確な必要条件

$D$ が $(x_0,c)$ と独立で、

$$
P(D=1\mid x_0,c)=r
$$

が一定なら、$p(x_0\mid D=1)=p(x_0)$ である。これは分かりやすい十分条件（MCAR）である。

ただし、独立性そのものは必要条件ではない。画像周辺分布について必要十分なのは、ほとんど至る所で

$$
P(D=1\mid x_0)=\mathbb E[P(D=1\mid x_0,c)\mid x_0]
$$

が一定になることである。条件への依存が平均で相殺される特殊な選択規則も理論上は許される。

一般に $\bar\pi(x_0)=P(D=1\mid x_0)$ と書くと、drop された集団は

$$
p_0^{\mathrm{drop}}(x_0)
=\frac{\bar\pi(x_0)p_0(x_0)}{\mathbb E[\bar\pi(x_0)]}
$$

になる。その noisy marginal のずれは

$$
\nabla_{x_t}\log p_t^{\mathrm{drop}}(x_t)
-\nabla_{x_t}\log p_t(x_t)
=\nabla_{x_t}\log\mathbb E[\bar\pi(x_0)\mid x_t]
$$

である。従って、データセット別に dropout 率を変えると、各データセットの画像分布が異なる通常の場合には無条件枝の混合比が変わる。しかし、分布が同一、平均選択確率が一定、または重要度補正を行う場合まで「必ず不正」とすることはできない。

元から空 caption の項目は常に null 側へ入るため、内容と空 caption の有無が相関すれば同じ選択バイアスが生じる。設定率を $r$、元から空である確率を $q$ としても、単純な「$r+q$」ではなく、独立な追加 dropout なら全体の null 率は $q+(1-q)r$ である。

## 5. CFG の代数と implicit classifier

score を直接扱う理想化の下で、

$$
s_w=(1-w)s_u+ws_c=s_u+w(s_c-s_u)
$$

とする。Bayes 則から

$$
\begin{aligned}
s_c-s_u
&=\nabla_{x_t}\log p_t(x_t\mid c)-\nabla_{x_t}\log p_t(x_t)\\
&=\nabla_{x_t}\log p_t(c\mid x_t).
\end{aligned}
$$

従って、2枝の差は noisy sample を入力とする Bayes classifier の log-likelihood gradient に一致する。外部 classifier を別に学習せず、この勾配を2本の生成枝から得るため「classifier-free」である。

また、固定した時刻 $t$ ごとには

$$
\begin{aligned}
s_w
&=\nabla_{x_t}\log\left[p_t(x_t)^{1-w}p_t(x_t\mid c)^w\right]\\
&=\nabla_{x_t}\log\left[p_t(x_t)p_t(c\mid x_t)^w\right]
\end{aligned}
$$

である。正規化可能なら、これは時刻ごとの power-tilted density

$$
q_t^{(w)}(x_t\mid c)\propto
p_t(x_t)^{1-w}p_t(x_t\mid c)^w
$$

の score である。

- $w=0$: 無条件
- $w=1$: 真の条件付き field
- $w>1$: 条件尤度を冪で強調する外挿。条件追従と fidelity を上げやすい一方、diversity と安定性を損ない得る

$p_t(c\mid x_t)^w$ を「likelihood の inverse-temperature 的な冪」と呼ぶことはできるが、$w$ を classifier posterior 全体の厳密な温度 $T=1/w$ と同一視してはいけない。class ごとの再正規化項は一般に $x_t$ に依存し、その勾配を無視できないからである。

### 5.1 時刻ごとの tilt と終端サンプル分布は別

上の score 恒等式は、各 $t$ を固定すれば厳密である。しかし一般に、

$$
\mathcal N_t\left[p_0^{1-w}p_0(\cdot\mid c)^w\right]
\ne
p_t^{1-w}p_t(\cdot\mid c)^w,
$$

ここで $\mathcal N_t$ は forward noising である。畳み込みは線形だが、積と冪は非線形なので、tilting と noising は可換ではない。

従って $w>1$ では、CFG sampler の終端を単純に
$q_0^{(w)}\propto p_0^{1-w}p_0(\cdot\mid c)^w$ からの厳密サンプルとは呼べない。連続時間・完全推定でも、実際の終端分布は guided field を積分した flow/SDE の pushforward として暗黙に定まる。$w=1$ だけは元の条件付き field に戻るため、理想条件下で $p_0(x_0\mid c)$ を再現する。

## 6. velocity / flow matching で何が成立するか

「velocity prediction でも常に同じ」は、任意の flow に無条件で成立する命題ではない。条件付き枝と無条件枝が**同じ Gaussian affine probability path と同じ parameterization**を使う場合に、velocity と score の間の branch-independent な affine 変換を導けるため、係数和が1の CFG と可換になる。

一般の

$$
x_t=a_t x_0+b_t\epsilon,
\qquad
y_t=\dot a_t x_0+\dot b_t\epsilon
$$

について、二乗回帰の最適 velocity は $v^*=\mathbb E[y_t\mid x_t,c]$ である。$a_t\ne0$, $b_t\ne0$ かつ $\dot b_t-\dot a_tb_t/a_t\ne0$ の内部時刻では、$v^*$ は $x_t$ と score の branch-independent な affine 式になる。この条件の下で

$$
v_w=(1-w)v_u+wv_c
$$

は score の同じ affine blend に対応する。端点や退化した path、枝ごとに異なる parameterization、非 Gaussian path まで含めた一般論ではない。

### 6.1 Ideogram 4 の path

本実装は clean-side time を $\tau$ として

$$
x_\tau=\tau x_0+(1-\tau)\epsilon,
\qquad y=x_0-\epsilon
$$

を使う。コードでは `sigma` に対して $\tau=1-\mathrm{sigma}$、学習 target は `latents - noise` である。母集団最適解は

$$
v_c^*(x_\tau,\tau)=\mathbb E[x_0-\epsilon\mid x_\tau,c]
$$

であり、$0<\tau<1$ では

$$
x_\tau=\tau v_c^*+\mathbb E[\epsilon\mid x_\tau,c]
$$

より

$$
s_c(x_\tau,\tau)
=\frac{\tau v_c^*(x_\tau,\tau)-x_\tau}{1-\tau}.
$$

同じ式が無条件枝にも成り立つため、

$$
s_c-s_u=\frac{\tau}{1-\tau}(v_c^*-v_u^*)
$$

であり、実装の

$$
v_w=v_u+w(v_c-v_u)
$$

は内部時刻で score-space CFG と代数的に整合する。純 noise / 純 data の端点では変換が退化するため、この score 式を端点へ機械的に延長しない。

なお、時刻ごとの velocity blend を ODE で積分した分布が、時刻ごとの power tilt $q_t^{(w)}$ を通るとは一般に限らない。これは前節の非可換性を flow matching で言い直したものである。

## 7. アーキテクチャ別の適用監査

### 7.1 結論一覧

| Architecture | 推論時の CFG | fine-tuning 時の無条件学習 | この文書の理論の適用 |
|---|---|---|---|
| SD1.5 | 共有 U-Net の2枝 | caption dropout の空 caption | text-only では標準例。ControlNet 等の保持条件があれば、その条件付き周辺分布になる |
| SDXL | 共有 U-Net を positive / negative embedding で2回評価 | dataset の caption dropout で同じ U-Net に空 caption を提示 | 標準的な適用例。ただし size/crop 条件を残した条件付き周辺分布であり、非空 negative prompt は無条件分布ではない |
| Z-Image | 共有 flow transformer の2枝 | caption dropout の空 caption | 学習・推論の空文字 encode が一致する標準的な flow CFG |
| FLUX.2 | base は2枝、distilled は guidance vector の1枝 | caption dropout。ただし reference と guidance 条件は保持 | base には適用。distilled には「CFG 枝の学習」としては非適用 |
| Anima | 共有 flow transformer の2枝 | caption dropout の空 caption | Qwen/T5 条件を同時に落とす標準的な flow CFG |
| Lens | 共有 flow transformer の2枝 | 専用 `cfg_uncond_drop_rate` が選択 row を zero feature / all-false mask へ書き換える（既定 0.0） | 専用 rewrite が推論の空 negative 経路と一致。caption dropout は空 user message を chat template で encode するため別物 |
| Krea2 | base は2枝、distilled/turbo は1枝 | caption dropout の空 caption | base には適用。distilled/turbo には非適用 |
| MiniT2I | 共有 x0 predictor の2枝 | 専用 `cfg_uncond_drop_rate`（旧 `minit2i_label_drop_rate`）が text mask をゼロ化 | 専用 label drop が推論 pure-uncond と一致。一般 caption dropout とは区別する |
| SenseNova U1.5 | 同じ generation decoder を cond / uncond prefix KV cache で評価 | 専用 `cfg_uncond_drop_rate` が item を encode する時点で推論 uncond query の prefix を作る（既定 0.0）。caption dropout は training cond prefix に空文字を入れるだけで別物 | 専用 encode-stage null が推論 uncond の query 文字列と prefix 長（image token の t 座標）に一致。既定 `cfg_norm="global"` は別途考慮。reference-conditioned run では非ゼロ rate を拒否 |
| LTX-2.3 | 共有 joint video/audio transformer の2枝 | caption dropout の空 caption | text null は整合するが、現行 loss は video のみ。audio の null field は直接学習しない |
| MiniMax-H3 | なし。guidance-distilled の1 forward | なし。空 prompt は拒否 | 現行モデルには非適用。caption dropout を有効にしても CFG 枝は学習されず、空 caption になった項目は encode 時に失敗する |
| ACE-Step 1.5 | なし。turbo / guidance-distilled の1枝 | 現行 trainer は vendor の training-only CFG dropout を迂回 | 非適用。空 caption は missing-caption 条件の学習であり、learned null 条件でも CFG 枝でもない |
| MiniMax Music 3 | AR logit CFG と flow velocity CFG | training 非対応 | 推論機構はあるが現行 repo では無条件 fine-tuning を監査・実行できない |
| Ideogram 4 | separate conditional / unconditional transformer | 同一 batch の全項目で両 transformer を同時学習 | separate-branch 版として適用。caption-dropout の標本数議論は非適用 |

### 7.2 SDXL

#### 7.2.1 学習構造

SDXL は1本の U-Net に text embedding、pooled text embedding、micro-conditioning の
`time_ids` を渡して noise または velocity target へ回帰する。`train_step` 自体に
conditional / unconditional の別 loss はなく、dataset preprocessing が caption を空文字へ
置換した項目が同じ U-Net の null-condition 学習例になる。

従って、dataset の whole-caption dropout が画像内容と独立なら、§4 の dropout 議論が直接
適用できる。空文字は「埋め込みがゼロ」なのではなく、SDXL の2本の text encoder が空文字を
tokenize して作る固定 embedding である。固定された同じ表現を学習と推論で使う限り、専用の
learnable null token である必要はない。

#### 7.2.2 完全な無条件分布ではない

SDXL の dropped-caption 項目にも、画像の original size、crop top-left、target size から作る
$m$ (`time_ids`) は残る。従って正確には、空 caption 枝が学ぶのは

$$
s_u(x_t,t;m)=\nabla_{x_t}\log p_t(x_t\mid m)
$$

であり、条件付き枝は

$$
s_c(x_t,t;c,m)=\nabla_{x_t}\log p_t(x_t\mid c,m)
$$

である。差は

$$
s_c-s_u=\nabla_{x_t}\log p_t(c\mid x_t,m)
$$

になる。これは誤りではなく、「テキストについて classifier-free、micro-conditioning については
conditional」という SDXL の契約である。dropout の独立性も、厳密には $m$ を固定した各層で
$P(D=1\mid x_0,m)$ が一定という条件になる。

#### 7.2.3 negative prompt と fine-tuning

推論の negative prompt が空文字なら上の周辺分布解釈を使える。非空の $c_-$ を baseline に
すると、実装の方向は

$$
s_{c_-}+w(s_{c_+}-s_{c_-})
$$

であり、$p_t(x_t\mid m)$ を基準にした implicit classifier ではなく、positive と negative の
2条件間の contrastive extrapolation である。

caption dropout をゼロにして conditional example だけで fine-tune した場合も CFG の tensor
演算は可能だが、空 caption での出力は新しい分布に対して直接拘束されない。Ideogram 4 のように
完全に凍結された別枝ではなく、共有 U-Net の更新によって null 出力も間接的に変化するため、
「古い無条件枝のまま」とも「新しい周辺分布を学習した」とも断定できない。大きい $w$ は、この
未拘束な baseline との差を増幅する。SDXL で caption dropout を入れる理論的理由は、この null
入力での出力を新しい training marginal に直接アンカーすることにある。

### 7.3 SenseNova U1.5

#### 7.3.1 学習構造

SenseNova の training step は1本の generation-branch flow prediction にだけ loss を与える。
reference image がある場合も `img_cond` / `uncond` は推論 CFG 用であり、training step には
現れない。caption dropout が空文字を作った場合、その項目は通常の training conditional-prefix
builder を空 caption で通り、同じ generation decoder を無条件例へ回帰する。従って
separate transformer を全項目で学習する Ideogram 4 ではなく、共有モデルへ一部の null 例を
混ぜる SDXL 側の構造に近い。

`train_text_encoder=false` では理解側 prefix decoder は frozen で、generation half のみが
空 prefix を入力として学習する。`train_text_encoder=true` では空 caption 項目の gradient が
理解側 decoder にも到達する。どちらの場合も dropout の選択分布に関する §4 の条件は同じである。

#### 7.3.2 training null prefix と inference uncond prefix の不一致

現実装には重要な不一致がある。text-only の training prefix は、空 caption でも条件付き経路と
同じ generation system message と
`<think>\n\n</think>\n\n<img>` suffix を使う。一方、推論の classic uncond branch は
negative prompt（既定は空文字）を `neo1_0` template と `<img>` suffix で encode する。
`neo1_0` の system message は空で、MPT formatter も system block 自体を出力しないため、
これは「別の system message」ではなく **system block なし**の prefix である。

従って、caption dropout が直接拘束するのは

$$
v(x_t,t\mid c=\text{empty},\ \text{conditional-template})
$$

であって、推論が baseline に使う

$$
v(x_t,t\mid c=\text{empty},\ \text{uncond-template})
$$

そのものではない。さらに `_build_t2i_image_indexes` は prefix 長を全 image token の t 座標へ
書き込む。従って差は prefix K/V だけでなく、denoise 側 image token の位置にも及ぶ。
共有 weight による一般化は期待できても、両者を同じ $v_u$ と置くことは
数学的には保証されない。この不一致を残す限り、SenseNova の caption dropout は無条件枝を
改善し得るが、§5 の exact implicit-classifier identity を fine-tuned model に対して保証する装置
にはならない。

#### 7.3.3 `cfg_uncond_drop_rate`（encode stage）

上記の不一致を揃える経路が `cfg_uncond_drop_rate` である。SenseNova は
`cfg_null_stage = "encode"` を宣言する唯一の architecture で、collated 書き換えではなく
**item を encode する時点で null prefix 自体を作る**。選ばれた item の prefix は
`transformer._build_t2i_query("", append_text="<img>")`（`system_message` kwarg なし
＝ template 自身の空 system message が効き、system block は出力されない）で作られ、
これは `sensenova_pipeline_ops.encode_prompt` の text-only uncond 枝
（`_build_t2i_query(negative_prompt, append_text="<img>")`、`negative_prompt` は既定で空文字）
と同じ文字列である。

位置整合はこの構造から自動的に付く。`SenseNovaTrainingPrefix.text_length` は
`_build_prefix_inputs` が返した indexes から導かれ、`sensenova_ops.train_step` はその値を
`_build_t2i_image_indexes(token_h, token_w, prefix.text_length, ...)` に渡す。null prefix の
長さが最初から入るので、K/V cache だけ揃えて image token の t 座標が conditional 長のまま
残る「半分だけの修正」にはならない。

Bernoulli は他 2 architecture と同じ `BaseTrainer.sample_cfg_drop_mask` の1回の抽選だが、
抽選位置は batch 組み立て**前**（`len(batch)` 個）である。SenseNova の prefix は組み立ての
最中に作られるため、組み立て後に引いたのでは prefix が既に conditional になっている。
label は caption を読む前に item 数だけ引かれ、latent サイズ filter では他の per-item list と
同様に `valid_indices` で並べ替えられる（再抽選はしない）。encode 側の呼び出し口は 2 箇所
あり、batch 組み立ての `if self.is_sensenova:` 枝と、trainable 理解側 + MNT>1 の
`_sensenova_mnt_conditioning` の再 encode で、両方が同じ label を受け取る。片方を落とすと
同じ画像が MNT 0 では null、以降は conditional になる。

null prefix は memoize しない。理解側が trainable な run では cache が無効になり、frozen な
場合も cache object が forward / checkpointing / phase eviction / cleanup を通して read-only に
留まる保証がまだ無い。

reference-conditioned な run（`use_reference_images=true`）は非ゼロ rate を**拒否**する。
出荷時の `img_cfg_scale=1` では推論の CFG baseline は `img_cond`（reference を残し text だけ
落とす枝）であって full uncond ではなく、1本の Bernoulli label で両方の周辺分布は監督できない
（`needs_uncond = needs_cfg and img_cfg_scale != 1`）。reference item を text-only null に
黙って写像することはしない。拒否は route・`train_runner` の pre-flight・trainer の 3 箇所が
共有する `api.cfg_null_resolver.resolve_and_check` で行われ、model load より前に出る。

既定値は `CFG_UNCOND_DROP_DEFAULTS_BY_ARCH["sensenova"] = 0.0`。key を省略した run の挙動は
変わらない。これは学習関数を変える変更であり、base checkpoint の学習契約が公開されていない
以上、A/B quality gate なしに既定を非ゼロにしてはいけない。

#### 7.3.4 flow CFG と `cfg_norm`

SenseNova は

$$
z_t=t x_0+(1-t)n,\qquad v=(x_0-z_t)/(1-t)
$$

の flow parameterization を使い、`cfg_norm="none"` なら

$$
v_w=v_u+w(v_c-v_u)
$$

なので §6 の affine-path の議論を適用できる。

しかし SushiUI の served default は `cfg_norm="global"` である。線形 blend 後に

$$
v_w\leftarrow v_w\,
\operatorname{clamp}\left(\frac{\lVert v_c\rVert}{\lVert v_w\rVert},0,1\right)
$$

を適用するため、既定生成 field は単純な score の affine blend ではない。係数が $x_t$ に依存し、
field 全体を rescale するので、一般には §5 の power-tilted density の score ともいえない。
`cfg_zero_star`、channel norm、CFG interval、style injection も同様に、標準 CFG の数学へ追加された
実用的ヒューリスティックとして分離して扱う必要がある。

#### 7.3.5 reference-conditioned 生成

reference image を $r$ とすると、caption dropout 後も reference は保持される。この場合に理想的に
学ぶ baseline は完全な $p_t(x_t)$ ではなく $p_t(x_t\mid r)$ であり、caption 効果は

$$
\nabla\log p_t(c\mid x_t,r)
$$

になる。これは reference-only の `img_cond` branch を baseline にする2枝 CFG と整合する考え方
である。

ただし現実装では、training の empty-caption-with-reference prefix と推論の `img_cond` prefix も
同じ template ではない。また `img_cfg_scale!=1` では cond / img_cond / uncond の3枝を

$$
v_u+w(v_c-v_r)+w_r(v_r-v_u)
$$

で合成する。これは text-only の単一 implicit classifier より広い、text guidance と reference
guidance の2方向合成である。各枝が同じ joint distribution の対応する条件周辺を表すという追加
仮定が必要になる。

### 7.4 MiniMax-H3

MiniMax-H3 はこの文書の中心命題の反例ではなく、**前提を採用していない別方式**である。
checkpoint は guidance-distilled で、推論 loop は conditional prediction を1 step 1回だけ実行する。
`guidance_scale` は sampler に渡らず、negative prompt、unconditional branch、2枝差分は存在しない。

従って、現行 H3 に caption dropout を加えても「CFG 用の無条件枝」を学習することにはならない。
さらに H3 の text encoder は空文字を token 0件として明示的に拒否するため、dataset の
`caption_dropout_rate>0` が実際に発火した項目は caption encode 時に `ValueError` になる。元から
空 caption の項目も同じである。H3 training では whole-caption dropout を 0 にし、全項目に
non-empty caption を要求する必要がある。token/tag dropout も最終的に空文字を作れば同じ制約に
当たる。

通常の conditional fine-tuning は、既に guidance が蒸留された単一 field を直接更新する。
CFG の二枝整合性を保存する問題は発生しない代わりに、fine-tuning 後に CFG scale で条件強度を
調節することもできない。無条件 anchor を新設して CFG/CGL loss を導入する案は
継続学習はローカル研究案として検討されているが、蒸留時の未知の teacher
unconditional branch を復元する保証はなく、現行 training / generation の説明に使ってはいけない。

### 7.5 Ideogram 4

#### 7.5.1 caption dropout ではない

`ideogram4_train_uncond=true` の学習 step は、同じ batch の全画像について次を同時に行う。

1. conditional transformer に `[text][image]` を入力し、$v_c$ を同じ target $x_0-\epsilon$ へ回帰する。
2. separate unconditional transformer に image-only tokens と zeroed text features を入力し、$v_u$ を**同じ** target へ回帰する。
3. $\mathcal L=\mathcal L_c+\lambda\mathcal L_u$ を最小化する。

従って、null に選ばれた $rN$ 件だけを無条件枝へ渡すのではない。両枝は同じ $N$ 件、同じ timestep、同じ noise realization、同じ training target を見る。無条件枝は caption を観測しないため、その母集団最適解は、trainer が sample する画像周辺分布に対する velocity である。必要な整合条件は、両枝が同じ画像 sampling measure を使うことであり、実装は同一 batch を再利用してこれを満たす。

`ideogram4_uncond_loss_weight=\lambda` は、枝のパラメータが分離され、母集団最適化が完全なら、任意の $\lambda>0$ で無条件枝の最適関数自体を変えない。ただし有限 step、gradient clipping、optimizer dynamics、LoRA の有限容量では収束速度と二枝の実用上のバランスを変える。$\lambda=0$ は無条件枝を学習しない。

#### 7.5.2 無条件枝を更新しない場合に何が壊れるか

事前学習済み base では conditional / unconditional の組が整合していても、LoRA で conditional branch だけを新しい画像・caption 分布へ動かすと、unconditional branch は古い画像周辺分布のまま残る。このとき

$$
v_c^{\mathrm{new}}-v_u^{\mathrm{base}}
$$

は、新しい同一 joint distribution から得られる implicit-classifier direction ではない。CFG のテンソル演算自体は実行できるが、条件効果だけでなく「新しい conditional と古い marginal の分布差」も外挿する。ずれの寄与は

$$
v_w=v_c^{\mathrm{new}}+(w-1)
\left(v_c^{\mathrm{new}}-v_u^{\mathrm{base}}\right)
$$

の $(w-1)$ 部分で増幅される。

従って、`ideogram4_train_uncond` の理論的役割は「CFG を初めて可能にする」ことではない。base の CFG は既に可能である。役割は、fine-tuning 後も conditional branch と unconditional marginal branch を同じ新しい training distribution に追随させ、CFG difference の分布的意味を保つことである。

これは品質改善の保証ではない。有限容量の LoRA で両枝を同時に適合させることは追加計算・追加パラメータを要し、データ量、rank、$\lambda$、guidance scale によっては条件付き枝だけの学習より悪化し得る。最終判断には、固定 seed で `train_uncond` の ON/OFF と CFG scale を交差させた比較が必要である。

### 7.6 SD1.5

SD1.5 は SDXL と同じく共有 U-Net に対する noise / velocity 回帰であり、caption dropout の
空文字 embedding と、推論で空の negative prompt を encode した embedding が一致する。従って
text-only の経路は §4--§6 の標準例である。SDXL の `time_ids` はないため、追加条件のない通常経路
では text-null 枝を画像周辺分布として解釈しやすい。

ただし ControlNet、reference、vision conditioning 等を保持したまま text だけを drop する場合、
baseline はそれらを条件とする周辺分布である。また非空 negative prompt は SDXL と同じく
unconditional marginal ではなく、2条件間の contrastive baseline になる。

### 7.7 Z-Image

Z-Image は共有 transformer に対する rectified-flow velocity 回帰である。training と sampling は
同じ prompt encoder を使い、空 caption と空 negative prompt の表現が一致するため、独立な caption
dropout は推論の null branch を直接拘束する。実装の target は `latents - noise` で、他実装と
velocity の符号規約が異なるが、同じ branch-independent affine 変換を両枝へ適用する限り、
`v_uncond + w * (v_cond - v_uncond)` と score CFG の対応は変わらない。

### 7.8 FLUX.2

FLUX.2 は base と distilled を分ける必要がある。base checkpoint は positive / negative prompt を
同じ transformer で評価する2枝 CFG を使う。training 用と sampling 用の Qwen chat template と
取得 hidden layers は揃っているため、空 caption の dropout は空 negative branch と整合する。

distilled checkpoint は2枝 CFG を使わず、単一 forward に guidance vector を与える。ここで空 caption
を学習しても、推論用 unconditional branch を新設したことにはならない。さらに reference latent が
ある training item では、dropout 後も reference は残るので、学ぶのは完全な marginal ではなく
$p(x\mid\text{reference})$ 側の text-null field である。distilled route では training と inference の
guidance-conditioning 値も同じ契約で評価する必要がある。

### 7.9 Anima

Anima は Qwen と T5/LLM-adapter の条件を持つ共有 flow transformer である。training は sampling と
同じ prompt encode を再利用し、caption dropout は両 text-conditioning 系を同時に空 prompt として
encode する。従って空 negative prompt を使う標準2枝 sampling とは表現上整合する。

CFG schedule、SNR rescale、dynamic threshold、style guidance を有効にすると、時刻ごとの線形 CFG に
非線形補正が加わる。その場合も生成機能としての guidance は成立するが、§5 の単純な power-tilt
解釈は補正前の field にしか直接適用できない。

### 7.10 Lens

Lens では空文字を同一視してはいけない。training の caption dropout は、空の user message を
system / user / assistant chat template に通した**条件付き** text features を作る。一方、推論で
negative prompt がすべて空なら、unconditional branch は text features をゼロ、attention mask を
all-false にする特別経路を使う。

この all-false mask は単なる「近い埋め込み」ではない。joint attention mask は常に有効な image
keys と text mask を連結し、出力 head は image stream だけを読む。従って text keys がすべて
無効なら、image output は text feature の値から構造的に切断される。一方、training の empty chat
に有効 text row が残れば通常の conditional attention graph である。image keys は常に有効で、
mask は key 軸へ適用されるため、all-masked softmax row による NaN はこの経路では発生しない。

従って現行 caption dropout が拘束する field と推論 baseline は別入力であり、SenseNova と同様に
exact implicit-classifier identity は保証されない。整合させるには、drop された training sample に
推論と同じ zero-feature / zero-mask 表現を明示的に与える architecture-specific label drop が必要で
ある。単に dropout rate を上げるだけでは解決しない。公式 checkpoint tokenizer による empty chat
の token 数と固定 offset 97 の関係は未実測であり、offset 後に text row が0件になる可能性は残る。
ただし明示的な zero-feature / all-false-mask rewrite の必要性と正しさは、この未決事項に依存しない。

この rewrite は実装済みである。`cfg_uncond_drop_rate` に明示値を与えると、
`LensArchHandler.apply_cfg_null_collated`（`cfg_null_stage = "collated"`）が選択 row の
`encoder_features` をゼロ、`encoder_mask` を all-false へ out-of-place で書き換える。これは
`lens_pipeline_ops.encode_prompt` の空 negative 経路
（`neg_features = [f.new_zeros(f.shape) for f in pos_features]`,
`neg_mask = torch.zeros_like(pos_mask, dtype=torch.bool)`）と同じ表現であり、sequence 長は
positive のものをそのまま使う（推論側も `_align_text_features` で positive 長へ揃える）。
Bernoulli は MiniT2I と同じく `BaseTrainer.sample_cfg_drop_mask` が MNT loop の外側で batch ごとに
1回引き、`LensArchHandler.train_step` が `ArchHandler.apply_cfg_null_step` 経由で適用する
（MiniT2I と同一の呼び出し階層。`lens_ops.train_step` の device/dtype 移動より前なので clone は
host 側に置かれ、値は移動後の書き換えと bitwise 一致する）。既定値は `CFG_UNCOND_DROP_DEFAULTS_BY_ARCH["lens"] = 0.0` であり、key を省略した
run の挙動は変わらない。

### 7.11 Krea2

Krea2 base は共有 flow transformer の標準2枝 CFG で、training と sampling の prompt encoding が
一致する。UI/API の `cfg_scale` に対し内部では `guidance = cfg_scale - 1` とするが、最終式は通常の
`v_uncond + cfg_scale * (v_cond - v_uncond)` である。従って caption dropout は base の null branch
を直接学習する。

distilled/turbo route は `guidance=0` の単一枝で動くため、そこで caption dropout を行っても CFG
baseline の学習とは呼べない。advanced CFG の非線形補正については Anima と同じ留保が必要である。

### 7.12 MiniT2I

MiniT2I には一般 caption dropout とは別に、正規の CFG 学習機構がある。arch 非依存の
`cfg_uncond_drop_rate`（旧称 `minit2i_label_drop_rate` は deprecated spelling）で、key を省略した
場合は `CFG_UNCOND_DROP_DEFAULTS_BY_ARCH["minit2i"] = 0.1` に解決され、明示的な `0.0` はこの
継承値を無効化する。選ばれた sample は text embedding 自体を空文字へ作り直すのではなく、
attention mask をゼロにして model の mask-token unconditional 表現を選ぶ。推論の pure-uncond
branch も conditional text tensor と zero mask の組を使うため、この表現は近似でなく一致する。
`MMJiT.forward` で mask が使われる箇所は、無効 row の context を同じ learned `mask_token` へ
置換する部分であり、zero mask なら元の text embedding の値は出力へ残らない。

Bernoulli は `BaseTrainer.sample_cfg_drop_mask` が MNT loop の外側で assembled batch ごとに1回だけ
引き、CPU boolean `[B]` として `TrainStepContext.cfg_drop_mask` に載る。OOM micro-batching では
再抽選せず slice される。書き換えは `MiniT2IArchHandler.train_step` が
`ArchHandler.apply_cfg_null_step` 経由で `apply_cfg_null_collated`
（`cfg_null_stage = "collated"`）を呼んで out-of-place で行い、`minit2i_ops.train_step` 側に
抽選も書き換えも無い。collated stage を宣言する architecture は全てこの階層で適用する。

通常の `caption_dropout_rate` は empty T5 prompt を encode する。実 checkpoint tokenizer の
ローカル probe では EOS id 1 の1 row が active mask に残ったため、専用
label drop の代替ではない。両者は別 mechanism なので、`cfg_uncond_drop_rate` を明示した run で
dataset の `caption_dropout_rate` か有効な `danbooru_aug_caption_dropout_rate` が非ゼロなら拒否する
（`cfg_null_stage` を宣言する全 architecture が対象）。拒否は `POST/PUT /training/runs` だけでなく
training 開始側にもある: `train_runner._preflight_cfg_null_caption_conflict` が dataset scan と
model load の前に datasets DB の caption 設定を読んで判定し、同じ入力を train section へ置いて
`BaseTrainer.cfg_null_drop_rate()` が再確認する。key を省略した legacy run は拒否せず warning。
MiniT2I は $x_0$ prediction を blend するが、Gaussian affine path の
内部時刻では $x_0$ predictor と score が枝に依存しない affine 関係にあるため、2枝差分の議論は
適用できる。CFG interval 外で scale を1へ戻す場合は時刻依存 $w(t)$ として扱う。

### 7.13 LTX-2.3

LTX-2.3 の caption と推論 negative prompt は、installed diffusers pipeline の同じ Gemma prompt
encoder を通るため、空文字の text-null 表現は整合する。共有 joint video/audio transformer の
線形2枝 CFG に対して、表現レベルでは標準理論を適用できる。

ただし現行 SushiUI training は dummy audio input と `isolate_modalities=true` を使い、audio prediction
を loss から捨てる video-only fine-tuning である。従って dropout が直接アンカーするのは video
velocity の null response だけで、joint audio unconditional field を学習したとはいえない。img2vid
では入力 frame も保持条件なので、text-null は frame-conditioned marginal になる。

### 7.14 ACE-Step 1.5

ACE-Step 1.5 の served model は turbo / guidance-distilled で、推論は `guidance_scale=1.0` の単一枝で
ある。vendored model の training helper には learned `null_condition_emb` へ置換する training-only
CFG dropout が存在するが、SushiUI trainer はその helper を使わず、prepared condition を
`dit.decoder` へ直接渡す。従ってその null dropout は現行 fine-tuning では実行されない。

dataset の空 caption は instruction、lyrics、timbre 等を残した missing-caption condition であり、
learned null condition でも CFG unconditional branch でもない。instrumental を表す空 lyrics も
unconditional と混同してはいけない。現行 ACE-Step に caption dropout を入れる根拠は robustness
や部分条件欠落であって、CFG の二枝整合性ではない。

### 7.15 MiniMax Music 3

MiniMax Music 3 は現行 `ARCH_REGISTRY` に含まれず、fine-tuning 非対応である。推論には2種類の
guidance がある。AR stage は prompt interior token を専用 CFG token へ置換して conditional / null
logits を blend する categorical guidance、flow stage は encoder hidden states をゼロにした枝と
conditional velocity を blend する flow CFG である。

flow stage の線形 blend には §6 の代数を適用できるが、AR logit blend は diffusion score の導出
そのものではない。また upstream training の drop 分布と null-condition 契約が公開実装から確認
できず、現行 repo には学習経路もないため、「non-conditional training を追加すれば両段の CFG が
整合する」とは結論できない。将来 flow training を実装するなら、少なくとも推論と同じ zero hidden
states を選ぶ condition dropout が必要になる。

## 8. dropout 方式にだけ適用される有限標本上の注意

共有モデルで独立な caption dropout を確率 $r$ で行う場合、期待件数は無条件側 $rN$、条件付き側 $(1-r)N$ になる。$r\to0$ では null 入力の推定が弱く、$r\to1$ では条件付き推定が弱い。

ただし、推定分散を普遍的に $O(1/(rN))$、$O(1/((1-r)N))$ と断定することはできない。反復 sampling、epoch 間の再利用、共有表現、条件の種類、自己相関、モデル misspecification が有効標本数を変える。この次数は独立同分布かつ通常の漸近条件の下でのヒューリスティックである。また、この trade-off は各画像を両枝へ入れる Ideogram 4 の separate-transformer 実装には当てはまらない。

## 9. 何が理論で、何が実証事項か

理想条件下で導出できること:

- 二乗回帰の最適 noise / velocity predictor は対応 target の条件付き期待値になる。
- Gaussian affine path では predictor と noisy marginal score の関係を導ける。
- 整合する二枝の差は implicit-classifier gradient に対応する。
- 固定時刻の score blend は power tilt の score になる。
- SD1.5、SDXL、Z-Image、Anima、および base の FLUX.2 / Krea2 は、学習と推論で同じ
  empty-text 表現を使う共有-model CFG である。
- MiniT2I の mask-zero label drop は、推論 pure-uncond の表現を直接再現する。
- Ideogram 4 の学習・推論式は、同じ affine flow parameterization 上で代数的に整合する。

理論だけでは保証できないこと:

- LoRA と有限 step で母集団最適解へ到達すること。
- $w>1$ の終端分布がデータ時刻の単純な power tilt になること。
- SenseNova の異なる training/inference null templates が同じ field を与えること（`cfg_uncond_drop_rate` を使う場合は同じ query と同じ prefix 長を組むので、この問いは発生しない）。
- Lens の empty-chat condition が推論の zero-feature baseline と同じ field を与えること。
- LTX-2.3 の video-only loss が joint audio null field も正しく更新すること。
- `cfg_norm` など線形 blend 後の補正が power-tilted score を保つこと。
- guidance-distilled model に空 caption を与えるだけで、未使用の2枝 CFG が復元されること。
- `train_uncond=true` が知覚品質、prompt adherence、diversity を必ず改善すること。
- 最適な $\lambda$、LoRA rank、guidance scale。

## 10. 実装監査箇所

- `backend/core/training/ops/sd_sdxl_ops.py`: 共有 U-Net に processed-caption embedding と SDXL `time_ids` を渡す。
- `backend/core/training/ops/zimage_ops.py`: Z-Image の共有 flow loss と sampling 時の同一 prompt encode を実装する。
- `backend/core/training/ops/flux2_ops.py`: base/distilled の guidance 条件、同一 chat template、reference latent を扱う。
- `backend/core/training/ops/anima_ops.py`: sampling と共通の Qwen/T5 prompt encode で flow loss を計算する。
- `backend/core/training/ops/lens_ops.py`: empty caption も conditional chat-template 経路で encode する。
- `backend/core/models/lens/vendor/pipeline.py`: 空 negative prompt の zero-feature / zero-mask 特別経路を実装する。
- `backend/core/training/ops/krea2_ops.py`: base の2枝と distilled/turbo の単一枝 sampling を分岐する。
- `backend/core/training/ops/minit2i_ops.py`: `apply_cfg_null_collated` で選択 row の text mask をゼロ化する。
- `backend/core/models/minit2i/minit2i_pipeline_ops.py`: pure-uncond で同じ zero mask を使い、x0 prediction を blend する。
- `backend/core/training/ops/sensenova_ops.py`: training conditional prefix だけに loss を与え、empty caption も conditional template で encode する。`_build_prefix_inputs(..., cfg_null=True)` は推論 uncond と同じ query を組み、その prefix 長が `train_step` の image indexes に入る。
- `backend/core/models/sensenova/sensenova_pipeline_ops.py`: 推論 uncond/img_cond prefix と、linear CFG 後の `cfg_norm` を実装する。
- `backend/core/training/ops/ltx2_ops.py`: 共通 Gemma prompt encoder を使う一方、dummy audio を loss から除外する。
- `backend/core/training/ops/minimax_h3_ops.py`: conditional-only の joint video/audio velocity loss を計算する。
- `backend/core/models/minimax_h3/h3_pipeline_ops.py`: empty prompt を拒否し、guidance-distilled の1 forward denoise loop を実装する。
- `backend/core/training/ops/acestep_ops.py`: vendor training helper を使わず、prepared condition を decoder へ直接渡す。
- `backend/core/pipeline_backends/acestep.py`: guidance-distilled 推論を `guidance_scale=1.0` に固定する。
- `backend/core/models/minimax_music3/`: AR token-logit CFG と flow zero-condition CFG を実装するが、training entry は持たない。
- `backend/core/training/ops/ideogram4_ops.py`: 同一 `noisy` / `v_target` を両枝へ渡し、`L_cond + weight * L_uncond` を計算する。
- `backend/core/training/adapters/ideogram4_adapter.py`: conditional と optional unconditional に別 prefix の LoRA を注入・保存する。
- `backend/core/models/ideogram4/ideogram4_pipeline_ops.py`: `v_uncond + w * (v_cond - v_uncond)` で推論する。

## 11. 一次文献

- Ho & Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- Dhariwal & Nichol, [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- Salimans & Ho, [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
