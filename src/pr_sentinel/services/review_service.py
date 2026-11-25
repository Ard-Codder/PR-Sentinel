import logging
import time

from pr_sentinel.domain import PullRequest
from pr_sentinel.integrations.github import GitHubError
from pr_sentinel.integrations.llm import LlmError

logger = logging.getLogger(__name__)


class ReviewService:
    def __init__(
        self,
        settings,
        github,
        state,
        processor,
        engine,
        validator,
        publisher,
        sleep=time.sleep,
    ) -> None:
        self.settings = settings
        self.github = github
        self.state = state
        self.processor = processor
        self.engine = engine
        self.validator = validator
        self.publisher = publisher
        self._sleep = sleep

    def close(self) -> None:
        for resource in (self.github, self.state, self.engine.llm):
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    def run_once(self) -> int:
        repos = self.settings.repo_list()
        pulls = self.github.list_review_requests(repos, max_repos=self.settings.max_repos)
        selected = pulls[: self.settings.max_prs_per_cycle]
        logger.info("Found %s pull request(s) assigned for review", len(selected))
        for pull in selected:
            self.review_pull_request(pull)
        return len(selected)

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except GitHubError as exc:
                logger.error("Poll failed: %s", exc)
            self._sleep(self.settings.poll_interval_seconds)

    def review_pull_request(self, pull: PullRequest) -> str:
        if not self.state.should_review(
            pull.repo,
            pull.number,
            pull.head_sha,
            dry_run=self.settings.dry_run,
        ):
            status = self.state.get_status(pull.repo, pull.number, pull.head_sha)
            logger.info(
                "Skip %s#%s sha=%s status=%s",
                pull.repo,
                pull.number,
                pull.head_sha,
                status,
            )
            return status or "skipped"

        self.state.mark(pull.repo, pull.number, pull.head_sha, "in_progress")
        try:
            return self._review_locked(pull)
        except (GitHubError, LlmError) as exc:
            logger.error("Review failed for %s#%s: %s", pull.repo, pull.number, exc)
            self.state.mark(pull.repo, pull.number, pull.head_sha, "error", str(exc))
            return "error"
        except Exception as exc:
            logger.exception("Unexpected review failure for %s#%s", pull.repo, pull.number)
            self.state.mark(pull.repo, pull.number, pull.head_sha, "error", f"unexpected: {exc}")
            return "error"

    def _review_locked(self, pull: PullRequest) -> str:
        fresh = self.github.get_pull_request(pull.repo, pull.number)
        if fresh.head_sha != pull.head_sha:
            self.state.mark(pull.repo, pull.number, pull.head_sha, "stale")
            logger.info("Skip stale %s#%s sha=%s", pull.repo, pull.number, pull.head_sha)
            return "stale"

        raw_files = self.github.list_pull_files(pull.repo, pull.number)
        processed = self.processor.process(raw_files)
        if not processed.chunks:
            self.state.mark(pull.repo, pull.number, pull.head_sha, "skipped", "no reviewable files")
            logger.info("No reviewable files in %s#%s", pull.repo, pull.number)
            return "skipped"

        parsed = self.engine.review(fresh, processed.chunks, self.settings.review_language)
        validated = self.validator.validate(parsed, processed.reviewed_files)
        if validated.rejected_all:
            self.state.mark(
                pull.repo,
                pull.number,
                pull.head_sha,
                "error",
                "all findings failed validation",
            )
            logger.error("All findings were rejected for %s#%s", pull.repo, pull.number)
            return "error"

        current = self.github.get_pull_request(pull.repo, pull.number)
        if current.head_sha != fresh.head_sha:
            self.state.mark(pull.repo, pull.number, fresh.head_sha, "stale")
            logger.info(
                "Head changed before publish for %s#%s from %s to %s",
                pull.repo,
                pull.number,
                fresh.head_sha,
                current.head_sha,
            )
            return "stale"

        body = self.publisher.render(fresh, validated, processed, self.settings.llm_model)
        if self.settings.dry_run:
            logger.info("DRY_RUN %s#%s sha=%s\n%s", fresh.repo, fresh.number, fresh.head_sha, body)
            self.state.mark(pull.repo, pull.number, pull.head_sha, "dry_run")
            return "dry_run"

        self.publisher.publish(fresh, body)
        self.state.mark(pull.repo, pull.number, pull.head_sha, "published")
        logger.info("Published review for %s#%s sha=%s", pull.repo, pull.number, pull.head_sha)
        return "published"
