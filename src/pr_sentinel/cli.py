import argparse
import logging
from datetime import timedelta

from pr_sentinel.config import Settings
from pr_sentinel.integrations.github import GitHubClient, GitHubError
from pr_sentinel.integrations.llm import LlmProvider
from pr_sentinel.services.diff_processor import DiffProcessor
from pr_sentinel.services.publisher import ReviewPublisher
from pr_sentinel.services.review_engine import ReviewEngine
from pr_sentinel.services.review_service import ReviewService
from pr_sentinel.services.validator import ResultValidator
from pr_sentinel.storage.state import StateStore


class RedactFilter(logging.Filter):
    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._secrets = [secret for secret in secrets if secret]

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        message = record.getMessage()
        redacted = message
        for secret in self._secrets:
            redacted = redacted.replace(secret, "***")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(*secrets: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    redactor = RedactFilter([secret for secret in secrets if secret])
    for handler in logging.getLogger().handlers:
        handler.addFilter(redactor)


def build_service(settings: Settings) -> ReviewService:
    github = GitHubClient(
        token=settings.github_token,
        api_url=settings.github_api_url,
        timeout=settings.github_timeout_seconds,
    )
    llm = LlmProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )
    state = StateStore(
        settings.database_path,
        stale_after=timedelta(minutes=settings.in_progress_timeout_minutes),
    )
    processor = DiffProcessor(
        max_file_patch_chars=settings.max_file_patch_chars,
        max_patch_chars=settings.max_patch_chars,
        max_chunks=settings.max_chunks,
    )
    engine = ReviewEngine(llm)
    validator = ResultValidator(
        max_findings=settings.max_findings,
        max_message_chars=settings.max_message_chars,
    )
    publisher = ReviewPublisher(github, state)
    return ReviewService(
        settings=settings,
        github=github,
        state=state,
        processor=processor,
        engine=engine,
        validator=validator,
        publisher=publisher,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pr-sentinel", description="Review GitHub pull requests with a local LLM.")
    parser.add_argument("command", nargs="?", choices=("once", "loop"), default="once")
    args = parser.parse_args(argv)
    settings = Settings()
    setup_logging(settings.github_token, settings.llm_api_key)
    if not settings.github_token.strip():
        logging.error("GITHUB_TOKEN is not set. Copy .env.example to .env and add a token.")
        return 1
    logging.info(
        "PR Sentinel starting (%s, dry_run=%s, model=%s)",
        args.command,
        settings.dry_run,
        settings.llm_model,
    )
    service = build_service(settings)
    try:
        if args.command == "once":
            service.run_once()
            return 0
        try:
            service.run_forever()
        except KeyboardInterrupt:
            logging.info("Stopped")
            return 0
        return 0
    except GitHubError as exc:
        logging.error("%s", exc)
        return 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
