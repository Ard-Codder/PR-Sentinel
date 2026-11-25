from dataclasses import dataclass


@dataclass(frozen=True)
class PullRequest:
    repo: str
    number: int
    title: str
    body: str
    head_sha: str
    html_url: str
    reviewers: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiffFile:
    path: str
    patch: str
    changed_lines: frozenset[int]


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    severity: str
    message: str


@dataclass(frozen=True)
class ParsedReview:
    summary: str
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class ValidatedReview:
    summary: str
    findings: tuple[Finding, ...]
    dropped: int
    rejected_all: bool


@dataclass(frozen=True)
class ProcessedDiff:
    reviewed_files: tuple[DiffFile, ...]
    chunks: tuple[tuple[DiffFile, ...], ...]
    skipped_files: tuple[str, ...]
    unreviewed_files: tuple[str, ...]
