import logging
from typing import Protocol

from pr_sentinel.domain import ProcessedDiff, PullRequest, ValidatedReview
from pr_sentinel.integrations.github import GitHubError

logger = logging.getLogger(__name__)

MARKER = "<!-- pr-sentinel -->"
MAX_COMMENT_CHARS = 60_000


class CommentStore(Protocol):
    def get_comment_id(self, repo: str, pr_number: int) -> int | None: ...

    def set_comment_id(self, repo: str, pr_number: int, comment_id: int) -> None: ...


class CommentClient(Protocol):
    def find_marked_comment(self, repo: str, number: int, marker: str) -> int | None: ...

    def update_issue_comment(self, repo: str, comment_id: int, body: str) -> None: ...

    def create_issue_comment(self, repo: str, number: int, body: str) -> int: ...


class ReviewPublisher:
    def __init__(self, github: CommentClient, state: CommentStore) -> None:
        self.github = github
        self.state = state

    def render(
        self,
        pull: PullRequest,
        review: ValidatedReview,
        processed: ProcessedDiff,
        model: str,
    ) -> str:
        lines = [
            MARKER,
            "## PR Sentinel",
            "",
            f"- Репозиторий: `{pull.repo}` #{pull.number}",
            f"- SHA: `{pull.head_sha}`",
            f"- Модель: `{model}`",
            "",
            review.summary or "Замечаний по изменённым строкам нет.",
            "",
        ]
        if review.findings:
            lines.append("### Замечания")
            lines.append("")
            for finding in review.findings:
                lines.append(
                    f"- **{finding.severity}** `{finding.file}:{finding.line}` — {finding.message}"
                )
            lines.append("")
        if review.dropped:
            lines.append(f"Отброшено замечаний модели, не прошедших проверку: {review.dropped}.")
            lines.append("")
        if processed.skipped_files or processed.unreviewed_files:
            lines.append("### Не вошло в ревью")
            lines.append("")
            for name in processed.skipped_files:
                lines.append(f"- пропущено: `{name}`")
            for name in processed.unreviewed_files:
                lines.append(f"- не влезло в лимит diff: `{name}`")
            lines.append("")
        body = "\n".join(lines).strip() + "\n"
        if len(body) > MAX_COMMENT_CHARS:
            body = body[:MAX_COMMENT_CHARS] + "\n\n...(сообщение обрезано)\n"
        return body

    def publish(self, pull: PullRequest, body: str) -> int:
        comment_id = self.state.get_comment_id(pull.repo, pull.number)
        if comment_id is None:
            comment_id = self.github.find_marked_comment(pull.repo, pull.number, MARKER)
        if comment_id is not None:
            try:
                self.github.update_issue_comment(pull.repo, comment_id, body)
            except GitHubError as exc:
                if exc.status_code != 404:
                    raise
                logger.info("Saved comment %s is gone; creating a new one", comment_id)
                comment_id = None
        if comment_id is None:
            comment_id = self.github.create_issue_comment(pull.repo, pull.number, body)
        self.state.set_comment_id(pull.repo, pull.number, comment_id)
        return comment_id
