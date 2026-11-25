import re

from pr_sentinel.domain import DiffFile, ProcessedDiff

LOCK_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "cargo.lock",
    "gemfile.lock",
    "composer.lock",
    "uv.lock",
    "go.sum",
}

GENERATED_DIRS = {"vendor", "dist", "node_modules"}

BINARY_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".svg",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".ico",
    ".zip",
    ".gz",
    ".tar",
    ".jar",
    ".wasm",
    ".mp4",
    ".mp3",
    ".bin",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
)

HUNK_HEADER = re.compile(r"\+(\d+)")


class DiffProcessor:
    """Filter files that should not be reviewed and split the rest into a few chunks."""

    def __init__(self, max_file_patch_chars: int, max_patch_chars: int, max_chunks: int) -> None:
        self.max_file_patch_chars = max_file_patch_chars
        self.max_patch_chars = max_patch_chars
        self.max_chunks = max_chunks

    def process(self, files: list[dict]) -> ProcessedDiff:
        kept: list[DiffFile] = []
        skipped: list[str] = []
        for item in files:
            path = str(item.get("filename") or item.get("path") or "")
            patch = item.get("patch")
            reason = self._skip_reason(path, patch)
            if reason:
                skipped.append(f"{path} ({reason})")
                continue
            assert isinstance(patch, str)
            changed = changed_line_numbers(patch)
            if not changed:
                skipped.append(f"{path} (no added or modified lines)")
                continue
            kept.append(DiffFile(path=path, patch=patch, changed_lines=frozenset(changed)))

        chunks, unreviewed = self._chunk(kept)
        reviewed = tuple(file for chunk in chunks for file in chunk)
        return ProcessedDiff(
            reviewed_files=reviewed,
            chunks=tuple(tuple(chunk) for chunk in chunks),
            skipped_files=tuple(skipped),
            unreviewed_files=tuple(unreviewed),
        )

    def _skip_reason(self, path: str, patch: object) -> str | None:
        if not path:
            return "missing path"
        name = path.rsplit("/", 1)[-1].lower()
        parts = {part.lower() for part in path.split("/")}
        if name in LOCK_NAMES:
            return "lockfile"
        if name.endswith(".min.js") or name.endswith(".min.css") or name.endswith(".map"):
            return "generated"
        if parts & GENERATED_DIRS:
            return "generated directory"
        if name.endswith(BINARY_SUFFIXES):
            return "binary"
        if not isinstance(patch, str) or not patch.strip():
            return "no patch"
        if len(patch) > self.max_file_patch_chars:
            return "patch too large"
        return None

    def _chunk(self, files: list[DiffFile]) -> tuple[list[list[DiffFile]], list[str]]:
        chunks: list[list[DiffFile]] = []
        current: list[DiffFile] = []
        current_size = 0
        unreviewed: list[str] = []

        def close_current() -> None:
            nonlocal current, current_size
            if current:
                chunks.append(current)
                current = []
                current_size = 0

        for file in files:
            size = len(file.patch)
            if len(chunks) >= self.max_chunks:
                unreviewed.append(file.path)
                continue
            if current and current_size + size > self.max_patch_chars:
                close_current()
                if len(chunks) >= self.max_chunks:
                    unreviewed.append(file.path)
                    continue
            current.append(file)
            current_size += size

        if current:
            if len(chunks) < self.max_chunks:
                chunks.append(current)
            else:
                unreviewed.extend(file.path for file in current)
        return chunks, unreviewed


def changed_line_numbers(patch: str) -> set[int]:
    """New-file line numbers of added or modified lines (diff lines that start with +)."""
    changed: set[int] = set()
    new_line: int | None = None
    for raw in patch.splitlines():
        if raw.startswith("@@"):
            match = HUNK_HEADER.search(raw)
            new_line = int(match.group(1)) if match else None
            continue
        if new_line is None:
            continue
        if raw.startswith("+"):
            changed.add(new_line)
            new_line += 1
        elif raw.startswith("-"):
            continue
        elif raw.startswith("\\"):
            continue
        else:
            new_line += 1
    return changed
