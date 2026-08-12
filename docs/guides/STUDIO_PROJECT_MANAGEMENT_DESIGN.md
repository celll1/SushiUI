# Studio project management and timeline editing design

## Scope

This design covers Studio project settings, local project files and recents,
missing media, clip cutting, multi-selection, and drag previews. It does not
change the concurrent generate/inpaint work.

## Decisions

### Project canvas is a project setting

The project canvas is the fixed composition surface for the timeline. It is not
the model output size. The UI will expose it from the top bar's resolution
badge through a compact Project Settings popover; the generate pane will show a
read-only canvas summary and keep Width/Height for the requested model output.

Canvas width and height use direct numeric inputs plus a bounded slider. On
commit, each dimension is quantized to the existing 16-pixel multiple and
clamped to 64–8192. The quantized result is shown immediately, so the user can
see what the renderer will use. Changing canvas size never silently changes
generation Width/Height or clip source media; clips retain their per-clip
`cover`/`contain` fit mode.

### Local project files

Studio uses the `.sushistudio` extension. A project file is a JSON manifest
with a format marker and schema version. Save downloads the current manifest;
Open accepts `.sushistudio`, `.json`, and legacy Studio JSON manifests. New
creates a fresh project after the current project is autosaved.

The browser cannot expose a reliable absolute path for an imported file, so
the manifest stores a stable asset id plus source metadata (name, kind,
dimensions, duration, size, and last-modified time). Gallery assets retain
their Gallery id. Imported/generated media is not embedded by default: large
video/audio files would make project files unwieldy and browser file access
cannot be restored by path alone. A separate Portable Copy action may embed
small derived image/frame assets; the existing IndexedDB copy remains the fast
local cache for the same browser profile.

When a referenced local or server asset cannot be restored, its asset remains
in the manifest with `missing: true` and its clips remain in the timeline.
Those clips render as an explicit “Missing media” empty clip with the original
name and duration. The user can replace or relink the asset later; loading a
project must never silently drop a clip.

Recent projects store only metadata and a local manifest snapshot in
localStorage: id, name, updated time, canvas, duration, and a compact asset
summary. The list is bounded and most-recent-first. Recent entries are
recovery aids, not a second file system; opening a downloaded file remains the
portable path.

### Timeline editing model

`select` is the default tool. A clip click selects one clip; Shift-click adds
or removes a clip from the selection. Clicking an empty lane clears selection.
The selection is a UI state, not project content, and is restored only within
the current session.

`blade`/Cut splits every selected clip that contains the playhead. A split
preserves source offsets, presentation mode, fit mode, input roles, and take
metadata. The right-hand segment becomes the active selection. Locked tracks
are skipped and reported.

The Hand tool pans the timeline and does not edit clips. Selection and blade
remain distinct from range editing. Keyboard defaults are V (select), B
(blade), H (hand), Shift-click for additive selection, and Delete/Backspace for
the selected set.

### Drag preview

Dragging one or more selected clips uses a transient drag state containing the
source clip ids, initial timeline positions, candidate track, and candidate
start. The committed project is not mutated during the gesture. A translucent
ghost is rendered at the snapped candidate position and in the candidate
track. Invalid track moves are shown as rejected and never commit. On release,
the final candidate is committed as one undo entry; cancel restores the
original state without creating history.

Touch uses the same ghost state after the existing movement threshold. Pointer
capture keeps the preview alive while crossing track boundaries. The ghost is
visual-only and therefore cannot race autosave or generation planning.

## Data changes

- Add a project-file format marker and current schema version.
- Add `StudioAsset.missing` and source metadata for unresolved local assets.
- Add a session-only selected clip id set; `selectedClipId` remains as the
  compatibility primary selection for existing generation and inspector code.
- Add a transient drag-preview type; it is never serialized.
- Keep `width`/`height` as project canvas dimensions and generation dimensions
  in `StudioFormState`.

## Implementation phases

1. Project file codec, recents, New/Save/Open, missing asset normalization.
2. Project Settings canvas popover with quantized direct inputs and sliders.
3. Selection set, keyboard/tool behavior, multi-delete, and multi-cut.
4. Pointer-captured drag ghosts, cross-track validation, commit/cancel.
5. Focused unit tests for codec, missing assets, cut invariants, and drag
   candidate math; browser build/type-check remains an owner-run verification.

Each phase is committed separately. The concurrent generate/inpaint files are
not staged by Studio commits.
