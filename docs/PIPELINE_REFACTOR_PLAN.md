# pipeline.py 分割リファクタ プラン（mixin 抽出）

## 目的
`backend/core/pipeline.py`（8,641 行 / 90 メソッド）を、挙動を一切変えずにアーキテクチャ別
mixin へ機械的に抽出し、保守性を上げる。**純粋な move（verbatim コピー）であり、ロジック改変は禁止**。

## 構造（衝突回避: pipeline.py は維持、新パッケージへ mixin を置く）
```
core/pipeline.py                     # 基底クラス本体 + mixin 合成 + シングルトン（モジュール名は不変）
core/pipeline_backends/
├── __init__.py                      # 6 mixin を re-export
├── zimage.py     class ZImageMixin
├── flux2.py      class Flux2Mixin
├── anima.py      class AnimaMixin
├── lens.py       class LensMixin
├── ideogram4.py  class Ideogram4Mixin
└── minit2i.py    class MiniT2IMixin
```
`class DiffusionPipelineManager(ZImageMixin, Flux2Mixin, AnimaMixin, LensMixin, Ideogram4Mixin, MiniT2IMixin):`
- 全 mixin は同一インスタンスに合成 → `self.*` 相互参照は MRO で解決。**依存で厳密分割する必要なし。各メソッドが全体で1回だけ定義されれば正しい。**
- 外部は `from core.pipeline import pipeline_manager` のみ → 不変（routes.py, main.py, cancellation, train_runner, custom_sampling）。
- circular なし: custom_sampling の `from core.pipeline import` は関数内ローカル。mixin は core.pipeline を import しない。

## メソッド割り当て（全 90 メソッド = base 25 + arch 65）

### base（core/pipeline.py に残す, 25）
__init__, current_pipeline_kind, load_model, _setup_img2img_steps, load_vision_encoder,
unload_vision_encoder, _apply_vision_encoder, _apply_vae_tiling, _log_component_devices,
_save_last_model, _auto_load_last_model, register_extension, _build_token_weights,
_negpip_eligible, _build_negpip_weights, _apply_controlnets, _encode_prompt_chunked,
_encode_prompt_nobos_single_chunk, _custom_te_encode, _encode_prompt_with_weights,
generate_txt2img, generate_img2img, generate_inpaint, cancel_generation, reset_cancel_flag

### ZImageMixin (10)
_load_lora_zimage, _wrap_with_lora, _unload_lora_zimage, _get_zimage_scheduler,
_generate_txt2img_zimage, _generate_img2img_zimage, _generate_inpaint_zimage,
_zimage_encode_prompt, _zimage_denoising_loop, _zimage_decode_latents

### Flux2Mixin (16)
_load_lora_flux2, _get_flux2_block_name, _wrap_with_lora_flux2, _unload_lora_flux2,
_generate_txt2img_flux2, _flux2_encode_prompt, _flux2_prepare_text_ids, _flux2_prepare_latent_ids,
_flux2_pack_latents, _flux2_unpack_latents_with_ids, _flux2_patchify_latents,
_flux2_unpatchify_latents, _flux2_compute_empirical_mu, encode_flux2_image_refs,
_generate_img2img_flux2, _generate_inpaint_flux2

### AnimaMixin (8)
_anima_resolve_dtype, _load_lora_anima, _unload_lora_anima, _anima_advanced_cfg, _anima_move,
_generate_txt2img_anima, _generate_img2img_anima, _generate_inpaint_anima

### LensMixin (8)
_load_lora_lens, _unload_lora_lens, _lens_advanced_cfg, _reload_lens_text_encoder, _lens_move,
_generate_txt2img_lens, _generate_img2img_lens, _generate_inpaint_lens

### Ideogram4Mixin (13)
_load_lora_ideogram4, _unload_lora_ideogram4, _ideogram4_move, _ideogram4_advanced_cfg,
_ideogram4_common_params, _ideogram4_encode, _ideogram4_setup_block_swap,
_ideogram4_stage_transformers, _ideogram4_unstage_transformers, _ideogram4_cleanup,
_generate_txt2img_ideogram4, _generate_img2img_ideogram4, _generate_inpaint_ideogram4

### MiniT2IMixin (10)
_minit2i_move, _minit2i_common_params, _minit2i_decode, _minit2i_encode, _load_lora_minit2i,
_unload_lora_minit2i, _minit2i_cleanup, _generate_txt2img_minit2i, _generate_img2img_minit2i,
_generate_inpaint_minit2i

## 実行（チーム体制）
- **worker（抽出）**: 各 mixin ファイルを新規作成し、担当メソッドを**逐語コピー**（デコレータ含む、4スペースインデント維持）。import ヘッダ（pipeline.py の 1–26 行）を先頭に複製（過剰 import は無害、core.pipeline は import しない）。読み取りのみ・別ファイル出力＝並列安全。
- **base worker**: `core/pipeline_new.py` を新規作成（ヘッダ + mixin import + 基底クラス + 25 メソッド逐語 + シングルトン）。
- **audit（監査）**: 各ファイルが担当メソッドを**byte 一致**で含むか、欠落/余剰/改変/未解決シンボルを検査。
- **coverage audit**: base ∪ 全 mixin == 元の 90 メソッド（欠落・重複なし）を検証。
- **orchestrator（私）**: 監査合格後に `__init__.py` 作成 → `core/pipeline_new.py` を `core/pipeline.py` へ置換 → `py_compile` + 実 import で関所。

## パリティ検証チェックリスト
- [ ] 元の 90 メソッドが過不足なく1回ずつ存在
- [ ] 各メソッド本文が逐語一致（@staticmethod 等デコレータ保持）
- [ ] 各 mixin の import ヘッダが十分（未解決名なし。`self.*` は MRO で解決）
- [ ] mixin が `core.pipeline` を import しない（circular 回避）
- [ ] `pipeline_manager` / `DiffusionPipelineManager` / `LAST_MODEL_CONFIG_FILE` を base が export
- [ ] `py_compile` 全ファイル成功
- [ ] `python -c "import core.pipeline; import api.routes"` 成功

## ロールバック
ブランチ `pipeline-refactor` で作業。失敗時は `git checkout flux2` / `git reset --hard` で破棄。
