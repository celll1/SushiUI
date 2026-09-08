"""Find a definition by symbol name instead of by file path.

`frontend/src/utils/api.ts` is being emptied module by module -- each
implementation moves to a focused module and api.ts keeps only a re-export --
so a test that names api.ts ends up scanning a re-export list. Ask for the
symbol instead, and cut its block structurally rather than with a character
window sized to the code that was there.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

FRONTEND_SRC = Path(__file__).resolve().parents[2] / "frontend" / "src"

_DECLARATORS = r"const|let|var|function|async\s+function|class|interface|type|enum"
_NOT_CODE = re.compile(r"//[^\n]*|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")


def _depth_delta(line: str) -> int:
    code = _NOT_CODE.sub("", line)
    return sum(code.count(c) for c in "([{") - sum(code.count(c) for c in ")]}")


def _block_from(source: str, at: int) -> str:
    """The construct starting on `at`'s line.

    That line, the rest of a header that spans lines (tracked by bracket
    depth, so a multi-line signature does not cut the body off), then every
    line indented deeper than the header.
    """
    line_start = source.rfind("\n", 0, at) + 1
    lines = source[line_start:].splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    body = [lines[0]]
    depth = max(_depth_delta(lines[0]), 0)
    for line in lines[1:]:
        if depth <= 0 and line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
        if depth > 0:
            depth = max(depth + _depth_delta(line), 0)
    return "\n".join(body)


def block_at(source: str, header: str) -> str:
    """`header`'s block, cut as `_block_from` describes."""
    return _block_from(source, source.index(header))


@lru_cache(maxsize=None)
def _frontend_sources() -> tuple[tuple[Path, str], ...]:
    paths = sorted(FRONTEND_SRC.rglob("*.ts")) + sorted(FRONTEND_SRC.rglob("*.tsx"))
    return tuple((p, p.read_text(encoding="utf-8")) for p in paths)


def frontend_definition(symbol: str, *, under: str | None = None) -> str:
    """The block declaring `symbol` anywhere under `frontend/src`.

    `under` narrows to a subdirectory, for a name deliberately reused across
    modules. Raises rather than returning an empty slice, which a `not in`
    assertion would sail straight through.
    """
    root = FRONTEND_SRC / under if under else FRONTEND_SRC
    pattern = re.compile(
        rf"^[ \t]*(?:export\s+)?(?:{_DECLARATORS})\s+{re.escape(symbol)}\b",
        re.MULTILINE,
    )
    hits = []
    for path, text in _frontend_sources():
        if root not in path.parents:
            continue
        match = pattern.search(text)
        if match:
            hits.append((path, text, match))
    if not hits:
        raise AssertionError(f"no declaration of `{symbol}` under {root}")
    if len(hits) > 1:
        where = ", ".join(str(p.relative_to(FRONTEND_SRC)) for p, _, _ in hits)
        raise AssertionError(
            f"`{symbol}` is declared in more than one module ({where}); "
            f"pass under= to say which one is meant")
    path, text, match = hits[0]
    block = _block_from(text, match.start())
    assert symbol in block, f"empty or misaligned slice for `{symbol}` in {path}"
    return block
