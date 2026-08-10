from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ROOT / "frontend" / "src" / "contexts" / "GenerationQueueContext.tsx"
WEBSOCKET = ROOT / "frontend" / "src" / "utils" / "websocket.ts"
PANELS = {
    "txt2img": ROOT / "frontend" / "src" / "components" / "generation" / "Txt2ImgPanel.tsx",
    "img2img": ROOT / "frontend" / "src" / "components" / "generation" / "Img2ImgPanel.tsx",
    "inpaint": ROOT / "frontend" / "src" / "components" / "generation" / "InpaintPanel.tsx",
    "outpaint": ROOT / "frontend" / "src" / "components" / "generation" / "OutpaintPanel.tsx",
    "upscale": ROOT / "frontend" / "src" / "components" / "generation" / "UpscalePanel.tsx",
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_queue_context_keeps_active_generation_across_navigation() -> None:
    source = _source(CONTEXT)

    assert "usePathname" not in source
    assert "Page navigation detected while generating" not in source
    assert "wsClient.subscribe(handleProgress)" in source
    assert "setProgressSnapshot" in source
    assert "publishCompletedResult" in source
    assert 'currentItemValue?.status === "generating"' in source
    assert "allowedTypes.includes(item.type)" in source


def test_progress_connection_survives_panel_remounts() -> None:
    source = _source(WEBSOCKET)

    assert "readyState !== EventSource.CLOSED" in source
    assert "Already connected or connecting, skipping" in source


def test_generation_panels_restore_progress_and_completed_preview() -> None:
    for panel, path in PANELS.items():
        source = _source(path)

        assert "progressSnapshot" in source, panel
        assert f"completedResults.{panel}" in source, panel
        assert f'panel: "{panel}"' in source, panel
        assert "isGeneratingRef.current" in source, panel

    assert '["txt2img", "img2img", "txt2vid", "ref2vid", "txt2aud"]' in _source(PANELS["txt2img"])


def test_all_generation_modalities_publish_their_result() -> None:
    expected_kinds = {
        "txt2img": {"image", "video", "audio"},
        "img2img": {"image", "video", "audio"},
        "inpaint": {"image", "video"},
        "outpaint": {"image", "video", "audio"},
        "upscale": {"image"},
    }

    for panel, kinds in expected_kinds.items():
        source = _source(PANELS[panel])
        for kind in kinds:
            assert f'kind: "{kind}"' in source, f"{panel}:{kind}"
