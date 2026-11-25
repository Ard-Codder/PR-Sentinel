import json
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from pr_sentinel.cli import RedactFilter, build_service, main
from pr_sentinel.config import Settings
from pr_sentinel.domain import DiffFile, Finding, ParsedReview, ProcessedDiff, PullRequest, ValidatedReview
from pr_sentinel.integrations.github import GitHubClient, GitHubError
from pr_sentinel.integrations.llm import LlmError, LlmProvider
from pr_sentinel.services.diff_processor import DiffProcessor, changed_line_numbers
from pr_sentinel.services.publisher import MARKER, ReviewPublisher
from pr_sentinel.services.review_engine import ReviewEngine, UnparseableReview, parse_review_payload
from pr_sentinel.services.review_service import ReviewService
from pr_sentinel.services.validator import ResultValidator
from pr_sentinel.storage.state import StateStore

PY_PATCH = "@@ -1,2 +1,3 @@\n context\n-old\n+new\n"
PULL = PullRequest(
    repo="acme/app",
    number=7,
    title="Add feature",
    body="Please review",
    head_sha="aaa111",
    html_url="https://github.com/acme/app/pull/7",
    reviewers=("me",),
)


class Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.current


class FakeLlm:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise LlmError("no more responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeGitHub:
    def __init__(self, pull: PullRequest = PULL, files: list | None = None) -> None:
        self.pull = pull
        self.pulls: list[PullRequest] | None = None
        self.files = files if files is not None else [{"filename": "src/app.py", "patch": PY_PATCH}]
        self.shas: list[str] = []
        self.created: list[str] = []
        self.updated: list[tuple[int, str]] = []
        self.comments: list[dict] = []
        self.missing_comment_ids: set[int] = set()
        self.fail_message: str | None = None
        self.gets = 0

    def list_review_requests(self, repos: list[str], max_repos: int) -> list[PullRequest]:
        if self.pulls is not None:
            return list(self.pulls)
        return [self.pull]

    def get_pull_request(self, repo: str, number: int) -> PullRequest:
        self.gets += 1
        if self.fail_message:
            raise GitHubError(self.fail_message, status_code=500)
        source = self.pull
        for item in self.pulls or []:
            if item.repo == repo and item.number == number:
                source = item
                break
        sha = self.shas.pop(0) if self.shas else source.head_sha
        return replace(source, head_sha=sha)

    def list_pull_files(self, repo: str, number: int) -> list[dict]:
        return self.files

    def find_marked_comment(self, repo: str, number: int, marker: str) -> int | None:
        for comment in self.comments:
            if marker in comment["body"]:
                return int(comment["id"])
        return None

    def update_issue_comment(self, repo: str, comment_id: int, body: str) -> None:
        self.updated.append((comment_id, body))
        if comment_id in self.missing_comment_ids:
            raise GitHubError("missing", status_code=404)

    def create_issue_comment(self, repo: str, number: int, body: str) -> int:
        self.created.append(body)
        return 100


def review_json(summary: str = "Есть замечание", findings: list | None = None) -> str:
    return json.dumps({"summary": summary, "findings": findings or []}, ensure_ascii=False)


def valid_finding(**overrides) -> dict:
    finding = {
        "file": "src/app.py",
        "line": 2,
        "severity": "warning",
        "message": "Return value is unused",
    }
    finding.update(overrides)
    return finding


def build(tmp_path, github: FakeGitHub, llm: FakeLlm, *, dry_run: bool = True, **overrides) -> ReviewService:
    settings = Settings(
        github_token="test-token",
        dry_run=dry_run,
        database_path=str(tmp_path / "state.db"),
        llm_model="test-model",
        _env_file=None,
        **overrides,
    )
    state = StateStore(settings.database_path, stale_after=timedelta(minutes=30))
    return ReviewService(
        settings=settings,
        github=github,
        state=state,
        processor=DiffProcessor(
            settings.max_file_patch_chars,
            settings.max_patch_chars,
            settings.max_chunks,
        ),
        engine=ReviewEngine(llm),
        validator=ResultValidator(settings.max_findings, settings.max_message_chars),
        publisher=ReviewPublisher(github, state),
    )


def test_dry_run_defaults_to_true(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DRY_RUN", raising=False)
    settings = Settings(_env_file=None)
    assert settings.dry_run is True
    assert settings.repo_list() == []
    listed = Settings(github_repos=" a/b, ,c/d ", _env_file=None)
    assert listed.repo_list() == ["a/b", "c/d"]


def test_changed_lines_ignore_context_and_deletions() -> None:
    patch = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,2 +1,3 @@\n context\n-old\n+new\n"
    assert changed_line_numbers(patch) == {2}


def test_diff_processor_filters_and_chunks() -> None:
    patch = "@@ -0,0 +1 @@\n+changed\n"
    files = [
        {"filename": "src/app.py", "patch": PY_PATCH},
        {"filename": "image.png", "patch": patch},
        {"filename": "src/app.py", "patch": None},
        {"filename": "package-lock.json", "patch": patch},
        {"filename": "vendor/lib.py", "patch": patch},
        {"filename": "web/app.min.js", "patch": patch},
        {"filename": "only_delete.py", "patch": "@@ -1 +0,0 @@\n-gone\n"},
        {"filename": "huge.py", "patch": patch + ("+" * 50)},
    ]
    # The duplicate src/app.py with a patch is reviewable; the None patch is a second entry.
    processor = DiffProcessor(max_file_patch_chars=40, max_patch_chars=10_000, max_chunks=4)
    processed = processor.process(files)
    assert [item.path for item in processed.reviewed_files] == ["src/app.py"]
    assert "image.png (binary)" in processed.skipped_files
    assert "src/app.py (no patch)" in processed.skipped_files
    assert "package-lock.json (lockfile)" in processed.skipped_files
    assert "vendor/lib.py (generated directory)" in processed.skipped_files
    assert "web/app.min.js (generated)" in processed.skipped_files
    assert "only_delete.py (no added or modified lines)" in processed.skipped_files
    assert "huge.py (patch too large)" in processed.skipped_files

    same = "@@ -0,0 +1 @@\n+changed\n"
    many = [{"filename": f"f{index}.py", "patch": same} for index in range(6)]
    chunked = DiffProcessor(1000, len(same), 4).process(many)
    assert len(chunked.chunks) == 4
    assert len(chunked.reviewed_files) == 4
    assert len(chunked.unreviewed_files) == 2

    paired = DiffProcessor(1000, len(same) * 2, 4).process(many[:2])
    assert len(paired.chunks) == 1
    assert len(paired.chunks[0]) == 2


def test_validator_keeps_changed_lines_and_drops_the_rest() -> None:
    files = [DiffFile(path="src/app.py", patch=PY_PATCH, changed_lines=frozenset({2}))]
    validator = ResultValidator(max_findings=20, max_message_chars=20)
    review = ParsedReview(
        summary="Итог",
        findings=(
            Finding("src/app.py", 2, "warning", "Real issue"),
            Finding("src/app.py", 1, "warning", "Context line"),
            Finding("missing.py", 2, "info", "Invented file"),
            Finding("src/app.py", 2, "warning", " real   issue "),
            Finding("src/app.py", 2, "critical", "x" * 50),
        ),
    )
    result = validator.validate(review, files)
    assert result.rejected_all is False
    assert [item.message for item in result.findings] == ["Real issue", "x" * 20]
    assert result.dropped == 3

    empty = validator.validate(ParsedReview("Чисто", ()), files)
    assert empty.rejected_all is False
    assert empty.findings == ()

    rejected = validator.validate(ParsedReview("Плохо", (Finding("src/app.py", 1, "info", "nope"),)), files)
    assert rejected.rejected_all is True
    assert rejected.findings == ()


def test_validator_caps_findings() -> None:
    files = [DiffFile("src/app.py", PY_PATCH, frozenset({2}))]
    findings = tuple(Finding("src/app.py", 2, "info", f"note {index}") for index in range(25))
    result = ResultValidator(20, 2000).validate(ParsedReview("s", findings), files)
    assert len(result.findings) == 20
    assert result.rejected_all is False
    assert result.dropped == 5


def test_state_separates_sha_status_and_comment_id(tmp_path) -> None:
    clock = Clock()
    store = StateStore(str(tmp_path / "state.db"), now=clock, stale_after=timedelta(minutes=30))
    store.set_comment_id("acme/app", 7, 55)
    store.mark("acme/app", 7, "sha-1", "published")
    store.mark("acme/app", 7, "sha-2", "error", "llm failed")
    assert store.should_review("acme/app", 7, "sha-1", dry_run=False) is False
    assert store.should_review("acme/app", 7, "sha-2", dry_run=False) is True
    assert store.get_comment_id("acme/app", 7) == 55

    store.mark("acme/app", 7, "sha-3", "skipped", "no reviewable files")
    assert store.should_review("acme/app", 7, "sha-3", dry_run=False) is False

    store.mark("acme/app", 7, "sha-4", "dry_run")
    assert store.should_review("acme/app", 7, "sha-4", dry_run=True) is False
    assert store.should_review("acme/app", 7, "sha-4", dry_run=False) is True

    store.mark("acme/app", 7, "sha-5", "in_progress")
    assert store.should_review("acme/app", 7, "sha-5", dry_run=True) is False
    clock.current += timedelta(minutes=31)
    assert store.should_review("acme/app", 7, "sha-5", dry_run=True) is True
    store.close()


def test_parse_review_payload_accepts_fenced_and_wrapped_json() -> None:
    fenced = '```json\n{"summary":"ok","findings":[]}\n```'
    wrapped = 'note {"summary":"ok","findings":[{"file":"a.py","line":"3","severity":"Warning","message":" hi "}]}'
    hidden = '<think>secret</think>{"summary":"ok","findings":[]}'
    assert parse_review_payload(fenced).summary == "ok"
    parsed = parse_review_payload(wrapped)
    assert parsed.findings[0].line == 3
    assert parsed.findings[0].severity == "warning"
    assert parsed.findings[0].message == "hi"
    assert parse_review_payload(hidden).findings == ()
    with pytest.raises(Exception):
        parse_review_payload('{"summary":"x","findings":[{"file":"a.py","line":1,"severity":"nope","message":"x"}]}')


def test_engine_retries_invalid_json_once() -> None:
    good = review_json("Итог", [valid_finding()])
    llm = FakeLlm(["not json", good])
    file = DiffFile("src/app.py", PY_PATCH, frozenset({2}))
    parsed = ReviewEngine(llm).review(PULL, ((file,),), "ru")
    assert parsed.summary == "Итог"
    assert len(llm.calls) == 2
    assert "Previous reply" in llm.calls[1][1]

    failing = FakeLlm(["not json", "still not json"])
    with pytest.raises(UnparseableReview):
        ReviewEngine(failing).review(PULL, ((file,),), "ru")
    assert len(failing.calls) == 2


def test_llm_provider_reads_content_only() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["messages"][0]["role"] == "system"
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "{\"summary\":\"ok\",\"findings\":[]}",
                            "reasoning_content": "do not return this",
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = LlmProvider("http://127.0.0.1:1234/v1", "lm-studio", "qwen", client=client, sleep=lambda _seconds: None)
    assert provider.complete("system", "user") == '{"summary":"ok","findings":[]}'
    provider.close()


def test_github_retries_429_and_not_401() -> None:
    calls = {"count": 0}

    def retry_handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"message": "slow down"})
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(200, json={"login": "me"})

    delays: list[float] = []
    client = GitHubClient(
        "test-token",
        client=httpx.Client(transport=httpx.MockTransport(retry_handler)),
        sleep=delays.append,
    )
    assert client.get_viewer_login() == "me"
    assert calls["count"] == 2
    assert delays == [0.0]
    client.close()

    forbidden = {"count": 0}

    def deny(request: httpx.Request) -> httpx.Response:
        forbidden["count"] += 1
        return httpx.Response(401, json={"message": "bad token"})

    denied = GitHubClient(
        "test-token",
        client=httpx.Client(transport=httpx.MockTransport(deny)),
        sleep=lambda _seconds: pytest.fail("401 must not be retried"),
    )
    with pytest.raises(GitHubError) as caught:
        denied.get_viewer_login()
    assert caught.value.status_code == 401
    assert forbidden["count"] == 1
    denied.close()


def test_github_lists_assigned_pulls_and_updates_comment() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/user":
            return httpx.Response(200, json={"login": "me"})
        if path == "/user/repos":
            return httpx.Response(
                200,
                json=[{"full_name": "acme/app"}, {"full_name": "acme/other"}, {"full_name": "acme/third"}],
            )
        if path == "/repos/acme/app/pulls":
            return httpx.Response(
                200,
                json=[
                    {
                        "number": 7,
                        "title": "Keep",
                        "body": "text",
                        "html_url": "https://example.test/7",
                        "head": {"sha": "aaa"},
                        "requested_reviewers": [{"login": "me"}],
                    },
                    {
                        "number": 8,
                        "title": "Ignore",
                        "body": "",
                        "html_url": "https://example.test/8",
                        "head": {"sha": "bbb"},
                        "requested_reviewers": [{"login": "someone"}],
                    },
                ],
            )
        if path == "/repos/acme/other/pulls":
            return httpx.Response(200, json=[])
        if request.method == "PATCH" and path == "/repos/acme/app/issues/comments/9":
            return httpx.Response(200, json={"id": 9, "body": json.loads(request.content.decode())["body"]})
        if request.method == "POST" and path == "/repos/acme/app/issues/7/comments":
            return httpx.Response(201, json={"id": 15})
        return httpx.Response(404, json={"message": path})

    client = GitHubClient("test-token", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _s: None)
    found = client.list_review_requests(["acme/app"], max_repos=20)
    assert [item.number for item in found] == [7]
    assert found[0].head_sha == "aaa"

    discovered = client.list_review_requests([], max_repos=1)
    assert [item.number for item in discovered] == [7]

    client.update_issue_comment("acme/app", 9, "updated")
    assert client.create_issue_comment("acme/app", 7, "new") == 15
    client.close()


def test_publisher_reuses_comment_across_missing_state_and_404(tmp_path) -> None:
    store = StateStore(str(tmp_path / "state.db"))
    github = FakeGitHub()
    publisher = ReviewPublisher(github, store)
    review = ValidatedReview("Итог", (Finding("src/app.py", 2, "warning", "Check this"),), 1, False)
    processed = ProcessedDiff(
        reviewed_files=(),
        chunks=(),
        skipped_files=("image.png (binary)",),
        unreviewed_files=("big.py",),
    )
    body = publisher.render(PULL, review, processed, "test-model")
    assert body.startswith(MARKER)
    assert "src/app.py:2" in body
    assert "image.png" in body
    assert "big.py" in body

    github.comments.append({"id": 77, "body": f"{MARKER}\nold"})
    assert publisher.publish(PULL, body) == 77
    assert github.updated[-1][0] == 77
    assert github.created == []
    assert store.get_comment_id("acme/app", 7) == 77

    github.missing_comment_ids.add(77)
    assert publisher.publish(PULL, body) == 100
    assert github.created
    assert store.get_comment_id("acme/app", 7) == 100
    store.close()


def test_dry_run_does_not_publish_and_same_sha_is_not_repeated(tmp_path) -> None:
    github = FakeGitHub()
    llm = FakeLlm([review_json("Итог", [valid_finding()])])
    service = build(tmp_path, github, llm, dry_run=True)
    assert service.review_pull_request(PULL) == "dry_run"
    assert github.created == []
    assert github.updated == []
    assert len(llm.calls) == 1
    assert service.review_pull_request(PULL) == "dry_run"
    assert len(llm.calls) == 1
    service.close()


def test_publish_once_then_update_same_comment_for_new_sha(tmp_path) -> None:
    github = FakeGitHub()
    llm = FakeLlm([review_json("Первый", [valid_finding()]), review_json("Второй", [valid_finding()])])
    service = build(tmp_path, github, llm, dry_run=False)
    assert service.review_pull_request(PULL) == "published"
    assert len(github.created) == 1
    assert MARKER in github.created[0]
    assert service.state.get_comment_id("acme/app", 7) == 100
    assert service.review_pull_request(PULL) == "published"
    assert len(llm.calls) == 1

    later = replace(PULL, head_sha="bbb222")
    github.pull = later
    assert service.review_pull_request(later) == "published"
    assert len(github.created) == 1
    assert github.updated[-1][0] == 100
    assert "bbb222" in github.updated[-1][1]
    service.close()


def test_sha_change_before_publish_does_not_write(tmp_path) -> None:
    github = FakeGitHub()
    github.shas = [PULL.head_sha, "ccc333"]
    llm = FakeLlm([review_json("Итог", [valid_finding()])])
    service = build(tmp_path, github, llm, dry_run=False)
    assert service.review_pull_request(PULL) == "stale"
    assert github.created == []
    assert github.updated == []
    assert service.state.get_status("acme/app", 7, PULL.head_sha) == "stale"
    service.close()


def test_unparseable_and_rejected_findings_stay_retryable(tmp_path) -> None:
    github = FakeGitHub()
    llm = FakeLlm(["nope", "still nope", review_json("Итог", [valid_finding()])])
    service = build(tmp_path, github, llm, dry_run=False)
    assert service.review_pull_request(PULL) == "error"
    assert github.created == []
    assert service.state.get_status("acme/app", 7, PULL.head_sha) == "error"
    assert service.review_pull_request(PULL) == "published"
    assert len(github.created) == 1
    service.close()

    rejected_github = FakeGitHub()
    rejected_llm = FakeLlm([review_json("Плохо", [valid_finding(line=1)])])
    rejected = build(tmp_path / "rejected", rejected_github, rejected_llm, dry_run=False)
    assert rejected.review_pull_request(PULL) == "error"
    assert rejected_github.created == []
    assert rejected.state.should_review("acme/app", 7, PULL.head_sha, dry_run=False) is True
    rejected.close()


def test_empty_findings_publish_and_skipped_files_do_not(tmp_path) -> None:
    github = FakeGitHub()
    llm = FakeLlm([review_json("Замечаний нет", [])])
    service = build(tmp_path, github, llm, dry_run=False)
    assert service.review_pull_request(PULL) == "published"
    assert "Замечаний нет" in github.created[0]
    service.close()

    skipped_github = FakeGitHub(files=[{"filename": "image.png", "patch": None}])
    skipped_llm = FakeLlm([])
    skipped = build(tmp_path / "skip", skipped_github, skipped_llm, dry_run=False)
    assert skipped.review_pull_request(PULL) == "skipped"
    assert skipped_llm.calls == []
    assert skipped.review_pull_request(PULL) == "skipped"
    assert skipped_github.gets == 1
    skipped.close()


def test_github_error_can_be_retried(tmp_path) -> None:
    github = FakeGitHub()
    github.fail_message = "unavailable"
    llm = FakeLlm([review_json("Итог", [valid_finding()])])
    service = build(tmp_path, github, llm, dry_run=False)
    assert service.review_pull_request(PULL) == "error"
    assert llm.calls == []
    github.fail_message = None
    assert service.review_pull_request(PULL) == "published"
    service.close()


def test_run_once_limits_pull_requests_and_loop_sleeps(tmp_path) -> None:
    pulls = [replace(PULL, number=index, head_sha=f"sha-{index}") for index in range(1, 7)]
    github = FakeGitHub()
    github.pulls = pulls
    llm = FakeLlm([review_json("ok", []) for _ in pulls])
    service = build(tmp_path, github, llm, dry_run=True, max_prs_per_cycle=5)
    assert service.run_once() == 5
    assert len(llm.calls) == 5

    class StopLoop(Exception):
        pass

    github.pulls = []
    service._sleep = lambda _seconds: (_ for _ in ()).throw(StopLoop)
    with pytest.raises(StopLoop):
        service.run_forever()
    service.close()


def test_redact_filter_hides_token() -> None:
    redactor = RedactFilter(["super-secret"])
    record = logging_record("token super-secret sent")
    assert redactor.filter(record) is True
    assert record.getMessage() == "token *** sent"


def test_main_requires_token(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert main(["once"]) == 1


def test_build_service_closes(tmp_path) -> None:
    settings = Settings(
        github_token="test-token",
        database_path=str(tmp_path / "state.db"),
        _env_file=None,
    )
    service = build_service(settings)
    service.close()


def logging_record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
