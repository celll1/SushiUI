"""Cheap source contracts for the capability-gated img2txt frontend."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_img2txt_is_capability_gated_and_never_mounted_without_support():
    page = _source("frontend/src/app/generate/page.tsx")
    assert 'text_output_modes?.[modelInfo.type]?.includes("img2txt")' in page
    assert 'activeTab === "img2txt" && !canImg2Txt' in page
    assert 'activeTab === "img2txt" && canImg2Txt && <Img2TxtPanel />' in page


def test_queue_dispatches_text_without_publishing_a_media_result():
    processor = _source(
        "frontend/src/components/generation/GenerationQueueProcessor.tsx")
    start = processor.index("const runImg2Txt")
    end = processor.index("const runImage", start)
    branch = processor[start:end]
    assert "generateImg2Txt" in branch
    assert 'kind: "text"' in branch
    assert "appendResult(" not in branch


def test_img2txt_request_freezes_file_model_and_template_version():
    panel = _source("frontend/src/components/generation/Img2TxtPanel.tsx")
    context = _source("frontend/src/contexts/GenerationQueueContext.tsx")
    api = _source("frontend/src/utils/api.ts")
    assert 'modelIdentity: identity ? { type: identity.type, source: identity.source }' in panel
    assert 'params: Img2TxtParams;' in context
    assert 'formData.append("images", params.image' in api
    sender = api[api.index("export const generateImg2Txt"):api.index(
        "export interface StudioRenderUpload")]
    assert 'postGenerationRequest("/generate/img2txt", formData' in sender
    assert '"Content-Type": "multipart/form-data"' in sender
    assert 'formData.append("prompt_template_version"' in api
    assert 'formData.append("loras", JSON.stringify(params.loras || []))' in api
    assert '<LoRASelector' in panel
    assert 'loadedArch="sensenova"' in panel
    assert 'kind: "text";' in context
    assert 'url: string;' not in context[
        context.index("export interface TextGenerationResultSnapshot"):
        context.index("export type GenerationResultSnapshot")
    ]


def test_img2txt_image_input_accepts_drag_and_drop():
    panel = _source("frontend/src/components/generation/Img2TxtPanel.tsx")
    assert "onDragOver={handleDragOver}" in panel
    assert "onDragLeave={handleDragLeave}" in panel
    assert "onDrop={handleDrop}" in panel
    assert "selectImage(event.dataTransfer.files?.[0]);" in panel
    assert 'file.type.startsWith("image/")' in panel


def test_img2txt_input_can_be_cleared_and_same_file_reselected():
    panel = _source("frontend/src/components/generation/Img2TxtPanel.tsx")
    assert "const clearImage = () =>" in panel
    assert "setImage(null);" in panel
    assert "setPreview(null);" in panel
    assert 'event.currentTarget.value = "";' in panel
    assert "onClick={clearImage}>Clear</Button>" in panel


def test_img2txt_renders_both_structured_caption_and_tags():
    panel = _source("frontend/src/components/generation/Img2TxtPanel.tsx")
    assert "result.structured?.caption" in panel
    assert "result.structured.caption" in panel
    assert "result.structured?.tags" in panel
