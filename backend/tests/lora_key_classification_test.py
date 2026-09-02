"""``classify_lora_keys``: 13 architectures + "unknown" from one signature table.

Every key list here comes from the REAL training adapter: LoRA injection over a
tiny CPU stub, then that adapter's own ``save_checkpoint`` into ``tmp_path``, then
the keys read back out of the safetensors file. A signature that drifts from what
a trainer actually writes therefore fails HERE rather than showing up as a
mis-labelled LoRA in AddLoRA.

The cross-product test is the point of the file: eight architectures besides
SD1.5/SDXL write ``lora_unet_*`` stems and several of those stems contain each
other as substrings, so "X is never read as Y" is the property that has to be checked
pairwise, not per architecture.

No model loads, no CUDA, no real weights. MEASURED: 10s wall / 1.65 GiB peak
RSS, ~9s and ~1 GiB of which is importing ``sd15_adapter`` for the shared
``LoRALinearLayer`` -- the same one-time cost every sibling
``*_lora_roundtrip_cheap_test.py`` pays. The classification itself is
milliseconds. Run with:
    venv/Scripts/python.exe -m pytest backend/tests/lora_key_classification_test.py -v
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors import safe_open

from lora_roundtrip_common import randomise_lora_layers  # noqa: F401  (path bootstrap)

from core.extensions.lora_manager import (  # noqa: E402
    _sort_lora_blocks, classify_lora_keys,
)
from core.training.arch import ARCH_REGISTRY  # noqa: E402

from core.training.adapters.sd15_adapter import SD15LoRAAdapter  # noqa: E402
from core.training.adapters.sdxl_adapter import SDXLLoRAAdapter  # noqa: E402
from core.training.adapters.zimage_adapter import ZImageLoRAAdapter  # noqa: E402
from core.training.adapters.flux2_adapter import FLUX2LoRAAdapter  # noqa: E402
from core.training.adapters.anima_adapter import AnimaLoRAAdapter  # noqa: E402
from core.training.adapters.lens_adapter import LensLoRAAdapter  # noqa: E402
from core.training.adapters.ideogram4_adapter import Ideogram4LoRAAdapter  # noqa: E402
from core.training.adapters.minit2i_adapter import MiniT2ILoRAAdapter  # noqa: E402
from core.training.adapters.krea2_adapter import Krea2LoRAAdapter  # noqa: E402
from core.training.adapters.ltx2_adapter import Ltx2LoRAAdapter  # noqa: E402
from core.training.adapters.acestep_adapter import AceStepLoRAAdapter  # noqa: E402
from core.training.adapters.minimax_h3_adapter import MiniMaxH3LoRAAdapter  # noqa: E402
from core.training.adapters.sensenova_adapter import SenseNovaLoRAAdapter  # noqa: E402

D = 8
RANK, ALPHA = 2, 4
N_BLOCKS = 2
SENSENOVA_LAYERS = 42  # the adapter refuses any other count


def lin(a=D, b=D):
    return nn.Linear(a, b, bias=False)


def named(cls_name, **children):
    """An nn.Module whose CLASS NAME is ``cls_name`` -- several adapters select
    their targets by ``module.__class__.__name__``."""
    module = type(cls_name, (nn.Module,), {})()
    for name, child in children.items():
        setattr(module, name, child)
    return module


def save_keys(adapter, layers, tmp_path, name):
    randomise_lora_layers(layers)
    out = tmp_path / f"{name}.safetensors"
    adapter.save_checkpoint(layers, 1, 1, out)
    with safe_open(str(out), framework="pt", device="cpu") as f:
        return list(f.keys())


# ---------------------------------------------------------------------------
# Stub trees + real-adapter drivers, one per architecture
# ---------------------------------------------------------------------------

def _clip_te(n_layers=2):
    layers = nn.ModuleList([
        named("CLIPEncoderLayer", mlp=named("CLIPMLP", fc1=lin(), fc2=lin()))
        for _ in range(n_layers)
    ])
    return named("CLIPTextModel",
                 text_model=named("CLIPTextTransformer",
                                  encoder=named("CLIPEncoder", layers=layers)))


def _unet_2d():
    """`down_blocks.<i>.attentions.<j>` / `mid_block` / `up_blocks.<i>.attentions.<j>`
    holding Transformer2DModel blocks -- the diffusers tree SD15LoRAAdapter walks."""
    def t2d():
        block = named("BasicTransformerBlock",
                      attn1=named("Attention", to_q=lin(), to_k=lin(), to_v=lin()),
                      attn2=named("Attention", to_q=lin(), to_k=lin(), to_v=lin()))
        return named("Transformer2DModel", transformer_blocks=nn.ModuleList([block]))

    def stage():
        return named("CrossAttnBlock2D", attentions=nn.ModuleList([t2d()]))

    return named("UNet2DConditionModel",
                 down_blocks=nn.ModuleList([stage()]),
                 mid_block=stage(),
                 up_blocks=nn.ModuleList([stage()]))


def keys_sd15(tmp_path):
    trainer = SimpleNamespace(unet=_unet_2d(), text_encoder=_clip_te())
    adapter = SD15LoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) > 0
    assert adapter.apply_lora_to_text_encoders(layers) > 0
    return save_keys(adapter, layers, tmp_path, "sd15")


def keys_sdxl(tmp_path):
    trainer = SimpleNamespace(unet=_unet_2d(), text_encoder=_clip_te(),
                              text_encoder_2=_clip_te())
    adapter = SDXLLoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) > 0
    assert adapter.apply_lora_to_text_encoders(layers) > 0
    return save_keys(adapter, layers, tmp_path, "sdxl")


def _zimage_stack():
    def block():
        attn = named("ZImageAttention", to_q=lin(), to_k=lin(), to_v=lin(),
                     to_out=nn.ModuleList([lin()]))
        return named("ZImageTransformerBlock", attention=attn)
    return named("ZImageTransformer2DModel",
                 noise_refiner=nn.ModuleList([block() for _ in range(N_BLOCKS)]),
                 context_refiner=nn.ModuleList([block() for _ in range(N_BLOCKS)]),
                 layers=nn.ModuleList([block() for _ in range(N_BLOCKS)]))


def keys_zimage(tmp_path):
    trainer = SimpleNamespace(transformer=_zimage_stack())
    adapter = ZImageLoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) > 0
    return save_keys(adapter, layers, tmp_path, "zimage")


def _qwen_te(n_layers=2):
    layers = nn.ModuleList([
        named("Qwen3DecoderLayer",
              mlp=named("Qwen3MLP", gate_proj=lin(), up_proj=lin(), down_proj=lin()),
              self_attn=named("Qwen3Attention", q_proj=lin(), k_proj=lin(),
                              v_proj=lin(), o_proj=lin()))
        for _ in range(n_layers)
    ])
    return named("Qwen3Model", model=named("Qwen3Body", layers=layers))


def _flux2_stack():
    dual = named("Flux2TransformerBlock",
                 attn=named("Flux2Attention", to_q=lin(), to_k=lin(), to_v=lin(),
                            to_out=nn.ModuleList([lin()]), add_q_proj=lin(),
                            add_k_proj=lin(), add_v_proj=lin(), to_add_out=lin()),
                 ff=named("Flux2FeedForward", linear_in=lin(), linear_out=lin()))
    single = named("Flux2SingleTransformerBlock",
                   attn=named("Flux2ParallelSelfAttention",
                              to_qkv_mlp_proj=lin(), to_out=lin()))
    return named("Flux2Transformer2DModel",
                 transformer_blocks=nn.ModuleList([dual]),
                 single_transformer_blocks=nn.ModuleList([single]))


def keys_flux2(tmp_path, te_only=False):
    trainer = SimpleNamespace(transformer=_flux2_stack(), text_encoder=_qwen_te(),
                              train_text_encoder=True)
    adapter = FLUX2LoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    if not te_only:
        assert adapter.apply_lora_to_unet(layers) > 0
    assert adapter.apply_lora_to_text_encoders(layers) > 0
    return save_keys(adapter, layers, tmp_path, "flux2_te" if te_only else "flux2")


def _anima_attn(cls, out_name):
    return named(cls, q_proj=lin(), k_proj=lin(), v_proj=lin(), **{out_name: lin()})


def _anima_stack():
    def block():
        return named("Block",
                     self_attn=_anima_attn("Attention", "output_proj"),
                     cross_attn=_anima_attn("Attention", "output_proj"),
                     mlp=named("GPT2FeedForward", layer1=lin(), layer2=lin()),
                     adaln_modulation_self_attn=nn.Sequential(nn.SiLU(), lin(), lin()))

    adapter_block = named("LLMAdapterTransformerBlock",
                          self_attn=_anima_attn("LLMAdapterAttention", "o_proj"),
                          cross_attn=_anima_attn("LLMAdapterAttention", "o_proj"),
                          mlp=nn.Sequential(lin(), nn.GELU(), lin()))
    return named("AnimaDiT",
                 blocks=nn.ModuleList([block() for _ in range(N_BLOCKS)]),
                 llm_adapter=named("LLMAdapter", in_proj=lin(),
                                   blocks=nn.ModuleList([adapter_block]),
                                   out_proj=lin()))


def keys_anima(tmp_path):
    trainer = SimpleNamespace(transformer=_anima_stack(), blockskip_config=None,
                              config={})
    adapter = AnimaLoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) > 0
    return save_keys(adapter, layers, tmp_path, "anima")


def _lens_stack():
    def block():
        attn = named("LensAttention", img_qkv=lin(), txt_qkv=lin(),
                     to_out=nn.ModuleList([lin()]), to_add_out=lin())
        return named("LensTransformerBlock", attn=attn,
                     img_mlp=named("GateMLP", w1=lin(), w2=lin(), w3=lin()),
                     txt_mlp=named("GateMLP", w1=lin(), w2=lin(), w3=lin()),
                     img_mod=nn.Sequential(nn.SiLU(), lin()),
                     txt_mod=nn.Sequential(nn.SiLU(), lin()))
    return named("LensTransformer2DModel",
                 transformer_blocks=nn.ModuleList([block() for _ in range(N_BLOCKS)]))


def keys_lens(tmp_path):
    trainer = SimpleNamespace(transformer=_lens_stack(), config={})
    adapter = LensLoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) > 0
    return save_keys(adapter, layers, tmp_path, "lens")


def _ideogram4_stack():
    def block():
        return named("Ideogram4Block",
                     attention=named("Ideogram4Attention", to_q=lin(), to_k=lin(),
                                     to_v=lin(), to_out=nn.ModuleList([lin()])),
                     feed_forward=named("SwiGLU", w1=lin(), w2=lin(), w3=lin()),
                     adaln_modulation=lin())
    return named("Ideogram4Transformer",
                 layers=nn.ModuleList([block() for _ in range(N_BLOCKS)]))


def keys_ideogram4(tmp_path):
    trainer = SimpleNamespace(transformer=_ideogram4_stack(),
                              transformer_uncond=_ideogram4_stack(),
                              ideogram4_train_uncond=True, config={})
    adapter = Ideogram4LoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) > 0
    return save_keys(adapter, layers, tmp_path, "ideogram4")


def _minit2i_stack():
    def block():
        return named("MMJiTBlock", img_qkv=lin(), txt_qkv=lin(),
                     img_attn_proj=lin(), txt_attn_proj=lin(),
                     img_mlp=named("GateMLP", w1=lin(), w2=lin(), w3=lin()),
                     txt_mlp=named("GateMLP", w1=lin(), w2=lin(), w3=lin()))
    net = named("MMJiT",
                double_blocks=nn.ModuleList([block() for _ in range(N_BLOCKS)]),
                txt_preamble_blocks=nn.ModuleList([block()]),
                txt_embedder=lin(), pooled_embedder=lin())
    return named("MiniT2IMMJiTModel", model=named("MiniT2IWrapper", net=net))


def _t5_te(n_blocks=2):
    def t5_block():
        return named("T5Block", layer=nn.ModuleList([
            named("T5LayerSelfAttention",
                  SelfAttention=named("T5Attention", q=lin(), k=lin(), v=lin(), o=lin())),
            named("T5LayerFF",
                  DenseReluDense=named("T5DenseGatedActDense", wi_0=lin(), wi_1=lin(), wo=lin())),
        ]))
    return named("T5EncoderModel",
                 encoder=named("T5Stack",
                               block=nn.ModuleList([t5_block() for _ in range(n_blocks)])))


def keys_minit2i(tmp_path, te_only=False):
    trainer = SimpleNamespace(transformer=_minit2i_stack(), text_encoder=_t5_te(),
                              config={}, repa_enable=False)
    adapter = MiniT2ILoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    if not te_only:
        assert adapter.apply_lora_to_unet(layers) > 0
    assert adapter.apply_lora_to_text_encoders(layers) > 0
    return save_keys(adapter, layers, tmp_path, "minit2i_te" if te_only else "minit2i")


def _krea2_stack():
    def block():
        return named("Krea2Block",
                     attn=named("Krea2Attention", to_q=lin(), to_k=lin(), to_v=lin(),
                                to_gate=lin(), to_out=nn.ModuleList([lin()])),
                     ff=named("Krea2FF", gate=lin(), up=lin(), down=lin()))
    fusion = named("Krea2TextFusion",
                   layerwise_blocks=nn.ModuleList([block()]),
                   refiner_blocks=nn.ModuleList([block()]),
                   projector=lin())
    return named("Krea2Transformer2DModel",
                 transformer_blocks=nn.ModuleList([block() for _ in range(N_BLOCKS)]),
                 text_fusion=fusion,
                 img_in=lin(), txt_in=named("Krea2TxtIn", linear_1=lin(), linear_2=lin()),
                 time_embed=named("Krea2TimeEmbed", linear_1=lin(), linear_2=lin()),
                 time_mod_proj=lin(),
                 final_layer=named("Krea2FinalLayer", linear=lin()))


def keys_krea2(tmp_path):
    trainer = SimpleNamespace(transformer=_krea2_stack(), config={},
                              krea2_is_distilled=False)
    # Every scope on, so the text_fusion and projection roots are exercised too.
    scope = {"attn": True, "mlp": True, "text_fusion": True, "proj": True}
    adapter = Krea2LoRAAdapter(trainer, RANK, ALPHA, torch.float32, scope=scope)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) > 0
    return save_keys(adapter, layers, tmp_path, "krea2")


def _ltx2_stack():
    def attn():
        return named("LTX2Attention", to_q=lin(), to_k=lin(), to_v=lin(),
                     to_out=nn.ModuleList([lin()]))

    def block():
        return named("LTX2TransformerBlock", attn1=attn(), attn2=attn(),
                     ff=named("FeedForward",
                              net=nn.ModuleList([named("GELU", proj=lin()),
                                                 nn.Identity(), lin()])))
    return named("LTX2VideoTransformer3DModel",
                 transformer_blocks=nn.ModuleList([block() for _ in range(N_BLOCKS)]))


def keys_ltx2(tmp_path):
    trainer = SimpleNamespace(transformer=_ltx2_stack(), config={})
    adapter = Ltx2LoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) > 0
    return save_keys(adapter, layers, tmp_path, "ltx2")


def _acestep_stack():
    def attn():
        return named("AceStepAttention", q_proj=lin(), k_proj=lin(),
                     v_proj=lin(), o_proj=lin())

    def layer():
        return named("AceStepDiTLayer", self_attn=attn(), cross_attn=attn(),
                     mlp=named("Qwen3MLP", gate_proj=lin(), up_proj=lin(), down_proj=lin()))
    return named("AceStepConditionGenerationModel",
                 decoder=named("AceStepDiTModel",
                               layers=nn.ModuleList([layer() for _ in range(N_BLOCKS)])))


def keys_acestep(tmp_path):
    trainer = SimpleNamespace(transformer=_acestep_stack(), config={})
    adapter = AceStepLoRAAdapter(trainer, RANK, ALPHA, torch.float32,
                                 scope={"attention": True, "mlp": True})
    layers = {}
    assert adapter.apply_lora_to_unet(layers) > 0
    return save_keys(adapter, layers, tmp_path, "acestep")


def _h3_stack():
    def block():
        return named("MiniMaxH3Block",
                     attn=named("MiniMaxH3Attention", to_q=lin(), to_k=lin(), to_v=lin(),
                                to_out=nn.ModuleList([lin()])),
                     ff=named("FeedForward",
                              net=nn.ModuleList([named("SwiGLU", proj=lin()),
                                                 nn.Identity(), lin()])))
    return named("MiniMaxH3Transformer3DModel",
                 transformer_blocks=nn.ModuleList([block() for _ in range(N_BLOCKS)]))


def keys_minimax_h3(tmp_path):
    trainer = SimpleNamespace(transformer=_h3_stack(), config={})
    adapter = MiniMaxH3LoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) > 0
    return save_keys(adapter, layers, tmp_path, "minimax_h3")


def _sensenova_stack():
    def layer():
        return named("NEODecoderLayer",
                     self_attn=named("NEOAttention",
                                     q_proj=lin(), k_proj=lin(), v_proj=lin(), o_proj=lin(),
                                     q_proj_mot_gen=lin(), k_proj_mot_gen=lin(),
                                     v_proj_mot_gen=lin(), o_proj_mot_gen=lin()),
                     mlp=named("NEOMLP", gate_proj=lin(), up_proj=lin(), down_proj=lin()),
                     mlp_mot_gen=named("NEOMLP", gate_proj=lin(), up_proj=lin(),
                                       down_proj=lin()))
    layers = nn.ModuleList([layer() for _ in range(SENSENOVA_LAYERS)])
    return named("NEOChatModel",
                 language_model=named("NEOLM", model=named("NEOBody", layers=layers)))


def keys_sensenova(tmp_path, both_branches=False):
    trainer = SimpleNamespace(transformer=_sensenova_stack(), config={},
                              train_text_encoder=both_branches)
    adapter = SenseNovaLoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) == 294
    if both_branches:
        assert adapter.apply_lora_to_text_encoders(layers) == 294
    return save_keys(adapter, layers, tmp_path, "sensenova")


BUILDERS = {
    "sd15": keys_sd15,
    "sdxl": keys_sdxl,
    "zimage": keys_zimage,
    "flux2": keys_flux2,
    "anima": keys_anima,
    "lens": keys_lens,
    "ideogram4": keys_ideogram4,
    "minit2i": keys_minit2i,
    "krea2": keys_krea2,
    "ltx2": keys_ltx2,
    "acestep": keys_acestep,
    "minimax_h3": keys_minimax_h3,
    "sensenova": keys_sensenova,
}

# The block prefix each architecture's own list must lead with, and the count of
# distinct block labels the stub's geometry implies.
EXPECTED_BLOCKS = {
    "sd15": ["IN01", "MID", "OUT00"],
    "sdxl": ["IN01", "MID", "OUT00"],
    "zimage": ["NRef0", "NRef1", "CRef0", "CRef1", "FDiT00", "FDiT01"],
    "flux2": ["DUAL00", "SING00"],
    "anima": ["DIT00", "DIT01", "LAD00", "LAPROJ"],
    "lens": ["DUAL00", "DUAL01"],
    "ideogram4": ["FDiT00", "FDiT01", "UDiT00", "UDiT01"],
    "minit2i": ["TPRE00", "MMB00", "MMB01", "EMB"],
    "krea2": ["MMB00", "MMB01", "TFL00", "TFR00", "TFP", "PROJ"],
    "ltx2": ["MMB00", "MMB01"],
    "acestep": ["L00", "L01"],
    "minimax_h3": ["MMB00", "MMB01"],
    "sensenova": [f"L{i:02d}" for i in range(SENSENOVA_LAYERS)],
}


@pytest.fixture(scope="module")
def arch_keys(tmp_path_factory):
    """One real trainer-written key list per architecture, built once."""
    tmp_path = tmp_path_factory.mktemp("lora_keys")
    return {arch: build(tmp_path) for arch, build in BUILDERS.items()}


def test_every_training_architecture_has_a_signature():
    """classify_lora_keys must cover ARCH_REGISTRY, not a subset of it."""
    assert set(BUILDERS) == set(ARCH_REGISTRY)


@pytest.mark.parametrize("arch", sorted(BUILDERS))
def test_trainer_written_keys_classify_as_their_own_architecture(arch, arch_keys):
    result = classify_lora_keys(arch_keys[arch])
    assert result["arch"] == arch, (
        f"{arch} LoRA read as {result['arch']!r}; sample keys: {arch_keys[arch][:3]}")
    assert result["blocks"] == EXPECTED_BLOCKS[arch]


def test_no_architecture_is_ever_read_as_another(arch_keys):
    """The cross-product. Ten architectures write ``lora_unet_*`` and several of
    those stems contain each other as substrings, so this is the property the
    detection ORDER exists to hold."""
    misread = {arch: classify_lora_keys(keys)["arch"]
               for arch, keys in arch_keys.items()
               if classify_lora_keys(keys)["arch"] != arch}
    assert misread == {}


def test_sd15_unet_stem_contains_the_dit_spellings_it_must_not_be_confused_with(arch_keys):
    """Why the ordering rule is not cosmetic: an SD1.5 U-Net key literally
    contains FLUX.2's and LTX-2.3's block spellings."""
    assert any("transformer_blocks_0_attn1_to_q" in key for key in arch_keys["sd15"])


def test_zimage_flattened_spelling_yields_fdit_blocks(arch_keys):
    """Defect 2: the trainer writes ``lora_transformer_layers_<n>_...`` while the
    branch only tested the dotted ``transformer.layers.``, so the FDiT half of the
    block list came back empty."""
    keys = arch_keys["zimage"]
    assert any(key.startswith("lora_transformer_layers_0_") for key in keys)
    assert [b for b in classify_lora_keys(keys)["blocks"] if b.startswith("FDiT")]


def test_zimage_dotted_spelling_still_classifies():
    """The generation loader also accepts ``transformer.<dotted>`` (1b0a192c)."""
    dotted = ["transformer.layers.0.attention.to_q.lora_down.weight",
              "transformer.layers.1.attention.to_q.lora_down.weight"]
    assert classify_lora_keys(dotted) == {"arch": "zimage",
                                          "blocks": ["FDiT00", "FDiT01"]}


def test_text_encoder_only_files_keep_their_architecture(tmp_path):
    """Defect 3: lora_trainer.py:306-323 allows train_unet=False with
    train_text_encoder=True, so FLUX.2 and MiniT2I can each save a file with no
    transformer keys at all. SD1.5/SDXL are included as the control."""
    assert classify_lora_keys(keys_flux2(tmp_path, te_only=True))["arch"] == "flux2"
    assert classify_lora_keys(keys_minit2i(tmp_path, te_only=True))["arch"] == "minit2i"

    sd15_te = ["lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"]
    assert classify_lora_keys(sd15_te) == {"arch": "sd15", "blocks": ["BASE"]}
    sdxl_te = sd15_te + ["lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"]
    assert classify_lora_keys(sdxl_te)["arch"] == "sdxl"


def test_kohya_sd15_single_text_encoder_prefix_is_not_read_as_flux2():
    """``lora_te_`` alone is kohya's SD1.5 spelling; only FLUX.2's narrower
    ``lora_te_model_layers_`` root may claim it."""
    kohya = ["lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"]
    assert classify_lora_keys(kohya)["arch"] == "sd15"


def test_sensenova_both_branches_still_classifies(tmp_path):
    """train_text_encoder adds the un-suffixed understanding half to the same
    file; the generation half's ``_mot_gen`` marker must not be the only thing
    holding the classification up."""
    result = classify_lora_keys(keys_sensenova(tmp_path, both_branches=True))
    assert result["arch"] == "sensenova"
    assert result["blocks"] == [f"L{i:02d}" for i in range(SENSENOVA_LAYERS)]


def test_minimax_h3_comfyui_layout_still_classifies():
    """The interchange layout real MiniMax-H3 LoRAs ship in, alongside the
    sd-scripts layout this repo's trainer writes."""
    comfy = ["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
             "diffusion_model.token_refiner.blocks.1.attn.qkv_proj.lora_A.weight",
             "diffusion_model.final_layer.linear.lora_A.weight"]
    assert classify_lora_keys(comfy) == {"arch": "minimax_h3",
                                         "blocks": ["TREF01", "MMB00", "FINAL"]}


def test_a_foreign_file_stays_unknown():
    """"unknown" is a first-class value: a real file whose keys match no
    signature must not be forced into the nearest architecture."""
    foreign = ["visual.transformer.resblocks.0.attn.in_proj_weight",
               "logit_scale",
               "token_embedding.weight"]
    assert classify_lora_keys(foreign) == {"arch": "unknown", "blocks": ["BASE"]}
    assert classify_lora_keys([]) == {"arch": "unknown", "blocks": ["BASE"]}


def test_every_emitted_label_has_a_sort_key(arch_keys):
    """A label with no ``_sort_lora_blocks`` branch either raises (it falls into
    ``int(block[1:])``) or lands in the (9, 0) bucket, where input order decides
    the result -- so the sort must be stable under permutation."""
    for arch, keys in arch_keys.items():
        blocks = classify_lora_keys(keys)["blocks"]
        assert _sort_lora_blocks(list(reversed(blocks))) == blocks, arch
