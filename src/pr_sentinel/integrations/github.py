from collections.abc import Callable

import httpx

from pr_sentinel.retry import call_with_retry


class GitHubError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GitHubClient:
    """GitHub REST client. Reads pull requests and writes one issue comment on a PR.

    Fine-grained tokens need only Pull requests: Read and write. The files API
    returns patches, so Contents permission is not used.
    """

    def __init__(
        self,
        token: str,
        api_url: str = "https://api.github.com",
        timeout: float = 30,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] | None = None,
        attempts: int = 3,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self._token = token
        self._sleep = sleep
        self.attempts = attempts
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def list_review_requests(self, repos: list[str], *, max_repos: int) -> list:
        from pr_sentinel.domain import PullRequest

        login = self.get_viewer_login()
        if repos:
            selected = repos
        else:
            selected = self.list_repo_names()[:max_repos]
        found: list[PullRequest] = []
        for repo in selected:
            for pull in self.list_open_pulls(repo):
                if login in pull.reviewers:
                    found.append(pull)
        return found

    def get_viewer_login(self) -> str:
        payload = self._request("GET", "/user").json()
        login = payload.get("login")
        if not login:
            raise GitHubError("GitHub /user response has no login")
        return str(login)

    def list_repo_names(self) -> list[str]:
        rows = self._paginate("/user/repos", {"sort": "updated"})
        names: list[str] = []
        for row in rows:
            full_name = row.get("full_name")
            if isinstance(full_name, str) and "/" in full_name:
                names.append(full_name)
        return names

    def list_open_pulls(self, repo: str):
        from pr_sentinel.domain import PullRequest

        rows = self._paginate(f"{self._repo_path(repo)}/pulls", {"state": "open"})
        return [self._to_pull(repo, row) for row in rows]

    def get_pull_request(self, repo: str, number: int):
        payload = self._request("GET", f"{self._repo_path(repo)}/pulls/{number}").json()
        return self._to_pull(repo, payload)

    def list_pull_files(self, repo: str, number: int) -> list[dict]:
        rows = self._paginate(f"{self._repo_path(repo)}/pulls/{number}/files")
        return [row for row in rows if isinstance(row, dict)]

    def find_marked_comment(self, repo: str, number: int, marker: str) -> int | None:
        comments = self._paginate(f"{self._repo_path(repo)}/issues/{number}/comments")
        for comment in comments:
            body = comment.get("body") or ""
            if marker in body and comment.get("id") is not None:
                return int(comment["id"])
        return None

    def create_issue_comment(self, repo: str, number: int, body: str) -> int:
        payload = self._request(
            "POST",
            f"{self._repo_path(repo)}/issues/{number}/comments",
            json_body={"body": body},
        ).json()
        comment_id = payload.get("id")
        if comment_id is None:
            raise GitHubError("GitHub did not return a comment id")
        return int(comment_id)

    def update_issue_comment(self, repo: str, comment_id: int, body: str) -> None:
        self._request(
            "PATCH",
            f"{self._repo_path(repo)}/issues/comments/{comment_id}",
            json_body={"body": body},
        )

    def _paginate(self, path: str, params: dict | None = None, *, limit_pages: int = 5) -> list:
        items: list = []
        for page in range(1, limit_pages + 1):
            query = dict(params or {})
            query["per_page"] = 100
            query["page"] = page
            batch = self._request("GET", path, params=query).json()
            if not isinstance(batch, list):
                raise GitHubError(f"Expected a list from {path}")
            items.extend(batch)
            if len(batch) < 100:
                break
        return items

    def _to_pull(self, repo: str, payload: dict):
        from pr_sentinel.domain import PullRequest

        reviewers = tuple(
            str(person["login"])
            for person in payload.get("requested_reviewers") or []
            if isinstance(person, dict) and person.get("login")
        )
        head = payload.get("head") or {}
        return PullRequest(
            repo=repo,
            number=int(payload["number"]),
            title=str(payload.get("title") or ""),
            body=str(payload.get("body") or ""),
            head_sha=str(head.get("sha") or ""),
            html_url=str(payload.get("html_url") or ""),
            reviewers=reviewers,
        )

    def _repo_path(self, repo: str) -> str:
        owner, name = _split_repo(repo)
        return f"/repos/{owner}/{name}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "pr-sentinel",
        }

    def _request(self, method: str, path: str, *, params: dict | None = None, json_body: dict | None = None) -> httpx.Response:
        url = f"{self.api_url}{path}"

        def send() -> httpx.Response:
            response = self._client.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=self._headers(),
            )
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"GitHub HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            return response

        try:
            return call_with_retry(send, attempts=self.attempts, sleep=self._sleep or _default_sleep)
        except httpx.HTTPStatusError as exc:
            raise GitHubError(_format_status(exc.response), status_code=exc.response.status_code) from exc
        except httpx.HTTPError as exc:
            raise GitHubError(f"GitHub request failed: {exc.__class__.__name__}") from exc


def _split_repo(repo: str) -> tuple[str, str]:
    parts = repo.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise GitHubError(f"Repository must be owner/name, got {repo!r}")
    return parts[0], parts[1]


def _format_status(response: httpx.Response) -> str:
    message = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            message = str(payload.get("message") or "")
    except ValueError:
        message = ""
    message = message[:300]
    if message:
        return f"GitHub HTTP {response.status_code}: {message}"
    return f"GitHub HTTP {response.status_code}"


def _default_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)
