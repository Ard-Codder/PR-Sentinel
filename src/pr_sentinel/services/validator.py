from pr_sentinel.domain import DiffFile, Finding, ParsedReview, ValidatedReview


class ResultValidator:
    def __init__(self, max_findings: int, max_message_chars: int) -> None:
        self.max_findings = max_findings
        self.max_message_chars = max_message_chars

    def validate(self, review: ParsedReview, files: tuple[DiffFile, ...] | list[DiffFile]) -> ValidatedReview:
        allowed = {file.path: file.changed_lines for file in files}
        seen: set[tuple[str, int, str]] = set()
        kept: list[Finding] = []
        dropped = 0
        for finding in review.findings:
            if len(kept) >= self.max_findings:
                dropped += 1
                continue
            key = (finding.file, finding.line, normalize_message(finding.message))
            if key in seen:
                dropped += 1
                continue
            lines = allowed.get(finding.file)
            if lines is None or finding.line not in lines:
                dropped += 1
                continue
            message = finding.message
            if len(message) > self.max_message_chars:
                message = message[: self.max_message_chars]
            seen.add(key)
            kept.append(
                Finding(
                    file=finding.file,
                    line=finding.line,
                    severity=finding.severity,
                    message=message,
                )
            )
        return ValidatedReview(
            summary=review.summary.strip(),
            findings=tuple(kept),
            dropped=dropped,
            rejected_all=bool(review.findings) and not kept,
        )


def normalize_message(message: str) -> str:
    return " ".join(message.split()).casefold()
