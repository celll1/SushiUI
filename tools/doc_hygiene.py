"""Check the tracked documentation against the rules in `docs/README.md`.

Run with:
    venv/Scripts/python.exe tools/doc_hygiene.py

Reports every violation it finds and exits non-zero if there is at least one.
This is a hygiene check on the CURRENT tree, not a history audit: it says
nothing about what earlier commits contain.

Checks
------
1. Every tracked Markdown file under `docs/` sits in the published taxonomy.
2. No tracked file matches `.gitignore` (a tracked-and-ignored file is invisible
   to one of the two views of the repository and drifts).
3. `docs/` holds no untracked or ignored Markdown, so the directory can be read
   as "this is what ships".
4. No tracked file records this machine's private roots.
5. Relative Markdown links resolve to something that exists.
6. Every vendored package appears in the third-party provenance ledger.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOC_DIRS = ("guides", "reference", "decisions", "audits", "legal")
DOC_ROOT_FILES = ("README.md",)

LEDGER = "docs/legal/THIRD_PARTY_PROVENANCE.md"

# Verbatim upstream license texts are reproduced as-is and are not edited to
# satisfy local rules.
EXEMPT_PREFIXES = ("docs/legal/licenses/",)

# This machine's own roots. A generic or conspicuously fictional example path
# is fine; these are the ones that identify the author's environment.
PRIVATE_PATTERNS = (
    # The project's own GitHub URL contains the owner's account name and is
    # meant to be published; a local checkout path is not.
    (re.compile(r"(?i)[a-z]:[\\/]{1,2}celll1|celll1[\\/]webui"),
     "the repository owner's local checkout path"),
    (re.compile(r"(?i)\bsatta\b"), "the repository owner's user name"),
    (re.compile(r"(?i)(?<![A-Za-z])M:[\\/]"), "the author's model/dataset drive"),
)

BINARY_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".ico", ".gif", ".pdf",
                   ".safetensors", ".gguf", ".zip", ".woff", ".woff2", ".ttf",
                   ".db", ".bin", ".pt", ".pyc", ".onnx")

LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)")


def tracked_files() -> list[str]:
    """Tracked paths that still exist, i.e. the tree as it would be committed."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    return [line for line in out.splitlines()
            if line and os.path.exists(os.path.join(REPO, line))]


def read(relative: str) -> str:
    with open(os.path.join(REPO, relative), "r", encoding="utf-8",
              errors="replace") as handle:
        return handle.read()


def check_taxonomy(files: list[str], report) -> None:
    for path in files:
        if not path.startswith("docs/") or not path.endswith(".md"):
            continue
        rest = path[len("docs/"):]
        if "/" not in rest:
            if rest not in DOC_ROOT_FILES:
                report(path, "sits at the top of docs/ instead of in "
                             + "/".join(DOC_DIRS))
            continue
        if rest.split("/", 1)[0] not in DOC_DIRS:
            report(path, "is outside the published documentation taxonomy")


def check_tracked_and_ignored(files: list[str], report) -> None:
    # Bytes, not text: text mode rewrites the separator to CRLF on Windows and
    # git then reads every entry as a name ending in a carriage return.
    result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=REPO,
                            input="\n".join(files).encode("utf-8"),
                            capture_output=True)
    for path in result.stdout.decode("utf-8", "replace").splitlines():
        if path:
            report(path, "is tracked and also matched by .gitignore")


def check_untracked_docs(report) -> None:
    out = subprocess.run(
        ["git", "status", "--porcelain", "-uall", "--ignored", "--", "docs"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        status, _, path = line.partition(" ")
        path = line[3:].strip().strip('"')
        if not path.endswith(".md"):
            continue
        if status in ("??", "!!"):
            report(path, "is under docs/ but is not tracked")


def check_private_paths(files: list[str], report) -> None:
    for path in files:
        if path.startswith(EXEMPT_PREFIXES) or path.endswith(BINARY_SUFFIXES):
            continue
        if not os.path.isfile(os.path.join(REPO, path)):
            continue
        try:
            text = read(path)
        except (OSError, UnicodeDecodeError):
            continue
        if "\x00" in text[:4096]:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for pattern, what in PRIVATE_PATTERNS:
                if pattern.search(line):
                    report(f"{path}:{number}", f"records {what}")
                    break


def check_links(files: list[str], report) -> None:
    tracked = set(files)
    for path in files:
        if not path.endswith(".md") or path.startswith(EXEMPT_PREFIXES):
            continue
        if not os.path.isfile(os.path.join(REPO, path)):
            continue  # tracked at HEAD, already deleted in the working tree
        base = os.path.dirname(path)
        for target in LINK.findall(read(path)):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            cleaned = target.split("#", 1)[0].split("?", 1)[0]
            if not cleaned:
                continue
            resolved = os.path.normpath(os.path.join(base, cleaned)).replace(os.sep, "/")
            if resolved in tracked or os.path.exists(os.path.join(REPO, resolved)):
                continue
            report(path, f"links to `{target}`, which does not exist")


def check_vendor_ledger(files: list[str], report) -> None:
    ledger = read(LEDGER)
    packages = sorted({
        path[:path.index("/vendor/") + len("/vendor/")]
        for path in files if "/vendor/" in path
    })
    for package in packages:
        if package not in ledger:
            report(LEDGER, f"does not cover the vendored package `{package}`")


def main() -> int:
    files = tracked_files()
    problems: list[str] = []

    def report(where: str, what: str) -> None:
        problems.append(f"{where}: {what}")

    check_taxonomy(files, report)
    check_tracked_and_ignored(files, report)
    check_untracked_docs(report)
    check_private_paths(files, report)
    check_links(files, report)
    check_vendor_ledger(files, report)

    for problem in problems:
        print(problem)
    print(f"\n{len(problems)} problem(s) in {len(files)} tracked files.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
