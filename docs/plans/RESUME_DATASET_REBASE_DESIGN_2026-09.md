# データセット選択変更時のエポック内再開 設計書

Status: 提案（2026-09-23）。コード未実装。

## 1. 目的と用語

中断後に学習対象のデータセットを追加・選択解除した場合、処理済みバッチを再提示せず、同じエポックの残りから学習を続ける任意モードを設ける。モデル・optimizer・`global_step` は選択した checkpoint から継続する。変更前の未処理分と新規データを、現在の batch mode の契約に従って並べ直す。これは変更前のエポックを完全再現する操作ではなく、checkpoint を境界とする**バッチ計画の改訂**である。

本書の「追加・削除」は学習対象リストへのデータセットの追加・選択解除を指す。データセット本体の削除・画像の増減・順序に関わるメタデータの変更とは区別する。「出現」は同じ画像が priority `multiplier` や concept `replay` により一エポック内で複数回予定される場合の個々の提示を指す。

## 2. 現行実装と限界

- `base_trainer.py::save_training_state()` は、checkpoint と同じ step の state JSON に epoch、次の絶対 `batch_idx`、バッチ生成前の RNG 状態、全体の dataset fingerprint、batch 数、crop fingerprint を保存する。バッチ列の中身は保存しない。
- 通常・priority path は dataset fingerprint または batch 数が変わると `global_step` と optimizer 状態を維持し、epoch の記帳を 0 からやり直す。concept path は構造変更を拒否する。concept path は計画 digest も照合するが、計画本体は持たない。
- 全体の fingerprint は dataset ID 列、画像パス集合、件数が中心で、変更が「選択リストのみ」か「既存データセット内部」かを判別できない。キャプションは意図的に画像パス fingerprint から除外されている。concept の分類 hash は別にある。priority の設定・分類結果の照合値は state にない。
- `train_runner.py::get_dataset_items_cached()` の per-dataset キャッシュは snapshot の高速化用で、任意の過去 checkpoint の不変な計画台帳ではない。キャッシュや選択解除したデータセットへのアクセスを再開の正しさの前提にしない。
- `CropPlanner.spec_for()` は seed・epoch・画像パス・寸法等から個別に決まるが、現在の crop fingerprint は全データセットの fingerprint を含むため、選択リスト変更だけでも不一致になる。

旧 checkpoint の `batch_idx` と RNG だけを現在のデータ一覧に適用してはならない。シャッフル、bucket の端数、concept 割当、priority の重複、後段の batch 分割により、同じ index が別の画像を指す。

## 3. 選択可能な再開方針

新しい設定 `resume_dataset_change_policy` を設ける。既定値は `existing` とし、現行の通常・priority path の再開始挙動、および concept path の拒否を変えない。`rebase_remaining` はユーザーが明示した場合のみ実行する。`strict` は構造不一致を全 mode で拒否する。新設定を checkpoint に保存し、途中で変えた場合は明示的な移行操作として記録する。

| 方針 | データ構成が不変 | データセットの選択のみ変更 | 既存データセット内部の順序入力が変更 |
|---|---|---|---|
| `existing` | 現行どおりの厳密再開 | 現行の mode 別挙動 | 現行の mode 別挙動 |
| `rebase_remaining` | 厳密再開 | 本書の計画改訂 | 拒否 |
| `strict` | 厳密再開 | 拒否 | 拒否 |

対象は初版では固定された画像データセットの通常順序、priority training、concept batch order とする。オンライン Danbooru 注入のように checkpoint 時点で未確定の batch がある run、動画・音声、SenseNova の task 再構成、解像度 curriculum の phase 変更は対象外として起動前に拒否する。既存 mode 自体の制約は維持する。参照画像の有無や VE 分割は最終 batch 計画に含める。

## 4. 保存する計画と同一性

### 4.1 最終バッチ計画の台帳

`rebase_remaining` を有効にしたエポックでは、実行直前の**後段処理後の最終バッチ列**を不変な sidecar に保存する。既存の concept digest は継続して検証するが、digest だけでは未処理の画像を特定できない。sidecar は checkpoint 本体に複製せず、run 出力内の版付きファイルとし、checkpoint の state JSON が版・相対パス・サイズ・SHA-256・改訂番号・カーソルを参照する。ファイルは一時名に書いて flush 後に原子的に置換し、同じ checkpoint の state を最後に確定する。参照中の版は次の checkpoint ができても削除しない。

台帳は画像本体やキャプションを保存しない。各 batch の出現を `(dataset_unique_id, stable_item_id, occurrence_id, bucket/task/reference key, mode label, source occurrence)` として記録する。`stable_item_id` は初版では snapshot に記録した画像パスの規定化表現とし、その規定化手順を台帳 version に固定する。同じ dataset 内に重複する ID があれば拒否する。別 dataset の同じパスは別物である。`occurrence_id` は同一画像の基本提示、priority の反復番号、concept の replay 番号を区別する論理 ID とし、計画改訂番号とは独立させる。同じ論理出現を改訂後に再利用しない。必要な crop spec またはその再検証可能な digest も含める。million-item run でも画像バイト列や長い caption を重複保存せず、ID 辞書と batch の整数索引を使い圧縮・サイズ計測する。

台帳の prefix は `batch_idx` 個の**処理済み batch**であり、その中の出現だけが完了済みである。同じ画像の未処理反復を画像 ID の集合差で消してはならない。チェックポイント保存は既存どおり、batch の全 MNT iteration が終わった境界で行う。計画 sidecar が欠損・破損・checkpoint と step 不一致なら改訂を拒否し、`batch_idx` から推測しない。

### 4.2 データセット manifest

選択された dataset ごとに、ID、画像 ID のソート済み集合と重複数、順序に関わる寸法・bucket・reference/task の入力、mode 別分類入力の hash、設定 version を保存する。concept では tag/caption alias と確定した concept 割当、priority では entry 順・分類・multiplier を照合する。学習文としての caption だけが変わり、mode の順序入力には影響しない場合は順序を維持できるが、raw caption と処理設定の hash で学習内容の変更を検出しログへ明示する。全 caption の内容同一性を要求する機能は本設計の `strict`（順序構造の厳密照合）とは別の課題とする。

run 全体の seed、batch size、MNT、bucket 設定、crop 設定、reference/VE 分割条件、mode 設定、計画 version は改訂中に変えられない。異なる場合は dataset の選択変更であっても拒否する。現在の設定では再現できない旧 suffix を台帳から黙って実行しない。

再開時、旧選択と新選択の**共通 dataset** の順序入力 manifest が一致することを検査する。新規選択 dataset は現在の manifest で受け入れ、選択解除 dataset は旧 manifest と台帳を参照して未処理出現を取り除く。共通 dataset 同士の選択順まで入れ替わった場合は初版では拒否する。共通 dataset の順序入力が変わっても拒否する。全体 fingerprint の一致・不一致だけでは判断しない。画像バイトの同一性は順序保証とは別であり、同じパス・寸法の画像が上書きされたことまで検出するには別途 content hash が必要となる。

選択解除 dataset の**旧画像ファイルは不要**である。旧台帳に識別子が残れば未処理出現を除ける。ただし、旧台帳を持たない checkpoint、新規選択 dataset を現在読み込めない場合、または共通 dataset の順序入力が差し替わった場合は継続できない。

## 5. 改訂アルゴリズム

1. 選択 checkpoint と同じ step の state・計画台帳を読み、hash、version、カーソル、batch 境界を照合する。モデルと optimizer をロードする前に、改訂可否と新計画のプレビューを確定する。
2. 旧計画の `[0:batch_idx]` を不変の完了履歴とする。`[batch_idx:]` から選択解除 dataset の出現を除く。batch が空なら削除し、残った batch は初版ではその順序と同質条件を保持して端数のまま残す。削除した画像を埋めるために処理済み画像を再提示しない。
3. 新規選択 dataset の基本提示を当該 epoch に一回ずつ生成する。通常順序には `(run seed, epoch, dataset ID, plan revision)` 由来の専用 RNG を使い、旧 suffix や global RNG の消費に依存させない。priority に分類された新規画像は既存 `multiplier` 回の論理出現を作る。concept では旧画像の割当数を初期値にして新規画像を安定 ID 順に割り当て、同点は専用 seed と画像 ID で解決する。元の割当は変えない。新規画像の replay はこの epoch では作らず、次 epoch から通常規則へ戻す。同じ epoch 内で一度選択解除した dataset を再選択する場合は、履歴中の同じ manifest と論理出現 ID を使い、完了済み出現を差し引いた義務だけを戻す。manifest が変わっていたら拒否する。
4. mode 別に**未処理 suffix だけ**を配置する。通常順序では同質 batch を決定論的に分散する。priority は新規 priority batch を suffix の先頭に置き、その後に旧 priority の残り、通常 batch を置く。concept は残りの focus batch と新規 concept batch を現在の `front` / `spread` 設定で配置し、完了履歴を参照して replay の source より前に replay が来ないことを保証する。旧 replay の source が選択解除 dataset だけで構成される場合はその replay も除く。`front` でも全 epoch の先頭へ時間を巻き戻さず、`spread` の分散範囲も suffix に限定する。
5. 新しい計画を `old completed prefix + revised suffix` として確定する。prefix は学習実行対象に戻さない。改訂番号、親計画 hash、追加・削除 dataset、追加・除外した基本/重複/再提示の出現数、部分 batch 数、残り batch 数を台帳とログに記録する。改訂後の cursor は旧 prefix 長から始める。再中断時は改訂版の cursor と hash を保存し、次回はその版を基準にする。複数回の追加・削除も同じ手順で連鎖できる。

旧 suffix の batch 構成と concept 集中の境界を全面的に作り直す最適化は初版の対象外とする。削除により小さくなった batch と新規 dataset の端数は追加 step・実効 batch size に影響し得るため、プレビューで件数を示す。固定 batch size が必須の構成では、その batch を実行せず明示的に拒否する。

## 6. step、epoch、scheduler の意味

`global_step`、optimizer update 数、勾配累積の途中状態、LR scheduler の保存済み位置は巻き戻さない。step 指定 run の停止目標は元の `steps` を維持する。epoch 指定 run は、完了済み step と改訂後の残り batch 数・後続 epoch の見積りから目標総 step を再算出し、LR schedule の到達先が変わる場合は開始前に表示する。既存 scheduler を無言で作り直したり、過去の optimizer update を再生したりしない。scheduler が途中で総 step 数変更を安全に受け付けない場合は、その組み合わせを拒否するか、明示的な再設定方針が実装されるまで対象外とする。

エポック番号はそのまま維持するが、当該 epoch は「元の dataset snapshot を一巡した epoch」ではなく改訂履歴を持つ epoch になる。UI とログは、旧集合の完了率だけでなく、現在の選択集合に対する処理済み・残り・選択解除で取り消した予定出現数を分けて表示する。新規 dataset をその epoch 内に消化できない step 制限の場合も、その未消化数を示す。

CropPlanner の個別 spec は同じ元寸法・seed・epoch なら再計算できる。全体 dataset hash を含む現行 crop fingerprint をそのまま照合すると選択変更を拒否するため、crop **設定**の hash と、共通 item の spec/dimension hash を分離する。既存 item の crop 変更は拒否し、新規 item の spec だけ作る。集計上の `batches_per_epoch` や crop の総 step 見積りは改訂後の計画から更新する。

## 7. API・UI・実装境界

- API は `openapi.yaml` を先に更新し、`backend/api/param_defaults.py` を唯一の既定値にする。`backend/api/routes.py`、training config の保存・復元、`frontend/src/utils/api.ts`、学習画面へ同じ enum を通す。既存 checkpoint に台帳がない場合、`rebase_remaining` を選べない理由を返す。
- `backend/core/training/` に計画台帳と改訂処理を独立 module として置く。`base_trainer.py` は最終 batch 列の確定点、state の保存・復元、実行する suffix の切り出しだけを接続する。共通 dataset manifest は `train_runner.py` の snapshot 生成側で確定する。
- 再開前プレビューは旧/新 dataset ID、共通 dataset の照合結果、計画改訂番号、旧 batch cursor、改訂後残り batch と MNT iteration、優先・集中・通常・replay の増減、端数 batch と実効 batch size、予定終了 step を表示する。新規画像の詳細パスを大量に UI へ送らない。
- 旧 checkpoint からこの機能を後付けする場合、旧計画を再生成でき、保存済み digest と完全一致する場合に限り台帳を一度生成できる。旧選択 dataset の snapshot がなく完全照合できなければ改訂を拒否する。`existing` での通常再開は従来どおり可能とする。

## 8. 検証項目と実装順

1. 計画台帳の版・原子的保存・checkpoint と state の対応・破損検出を実装し、選択変更なしの通常/priority/concept 再開で未処理列が完全一致することを確認する。priority の設定・分類 hash が変わった場合も検出する。
2. 通常順序の dataset 追加、選択解除、両方同時、複数回の改訂を検証する。完了 prefix の出現 ID が再提示されず、共通 dataset の未処理 ID が失われず、新規 dataset の基本提示が一回ずつになることを数える。
3. priority の `multiplier`、concept の `front` / `spread`・複数タグ割当・replay、bucket 端数、crop、reference/VE 分割を含む小さな決定論的 fixture で改訂前後の出現台帳を比較する。途中 checkpoint を二度保存・再開しても同じ suffix に到達することを検証する。
4. データセット内部の画像追加・削除、寸法変更、concept/priority 分類変更、設定変更、欠損 sidecar、異なる checkpoint step、ID 衝突、旧画像ファイル消失を検証する。前者は拒否し、選択解除した dataset の旧ファイル消失だけでは失敗しないことを確認する。
5. 実 training run で `global_step`・optimizer update・LR schedule・MNT・勾配累積、途中停止時の未消化件数を測る。million-item 規模で台帳生成時間、圧縮サイズ、再開時間、ピークメモリを計測してから既定以外への展開を判断する。

この設計は「選択変更のみ」を初版の安全な境界とする。個別データセット内部の更新を同一 epoch に反映するには、変更前の item snapshot と、追加・削除・差し替えを item 単位で判定する別の移行契約が必要である。
