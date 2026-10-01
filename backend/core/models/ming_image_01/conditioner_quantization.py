"""Native packed MLLM/connector/projection preflight before any companion weight payload."""
import torch
from .quantization import _FLOATS, _convrot_layers, prepare_component_slots


def prepare_conditioner_components(artifact, dtype, attention):
    result={}
    for component in ("mllm","connector","mlp"):
        headers=artifact.tensor_headers[component]
        if all(h.dtype in _FLOATS and not key.endswith((".weight_scale", ".comfy_quant"))
               for key,h in headers.items()):
            continue
        if component=="mllm":
            from .vendor.configuration_bailingmm2 import BailingMM2Config
            from .vendor.modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
            config=BailingMM2Config.from_dict(artifact.configs[component])
            config._attn_implementation=attention
            with torch.device("meta"):
                model=BailingMM2NativeForConditionalGeneration(config).to(dtype=dtype)
        elif component=="connector":
            from transformers import Qwen2Config,Qwen2ForCausalLM
            config=Qwen2Config.from_dict(artifact.configs[component])
            config._attn_implementation="eager"
            with torch.device("meta"):
                model=Qwen2ForCausalLM(config).to(dtype=dtype)
        else:
            from .conditioning import create_projection_modules
            with torch.device("meta"):
                model=create_projection_modules(artifact.configs["mlp"],
                    artifact.configs["mllm"]["llm_config"]["hidden_size"],
                    artifact.configs["connector"]["hidden_size"]).to(dtype=dtype)
        result[component]=prepare_component_slots(model,headers,_convrot_layers(artifact,component),dtype)
    return result
