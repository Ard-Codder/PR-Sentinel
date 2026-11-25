import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlite3

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"published", "skipped"})


class ReviewRecord:
    def __init__(
        self,
        repo: str,
        pr_number: int,
        head_sha: str,
        status: str,
        error: str | None,
        updated_at: datetime,
    ) -> None:
        self.repo = repo
        self.pr_number = pr_number
        self.head_sha = head_sha
        self.status = status
        self.error = error
        self.updated_at = updated_at


class StateStore:
    """SQLite store. Review rows are keyed by repo + PR + SHA. Comment ids are keyed by repo + PR."""

    def __init__(
        self,
        path: str,
        *,
        now=None,
        stale_after: timedelta = timedelta(minutes=30),
    ) -> None:
        self.path = path
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.stale_after = stale_after
        database = Path(path)
        if database.parent and str(database.parent) not in {"", "."}:
            database.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def close(self) -> None:
        self._conn.close()

    def get(self, repo: str, pr_number: int, head_sha: str) -> ReviewRecord | None:
        row = self._conn.execute(
            """
            SELECT repo, pr_number, head_sha, status, error, updated_at
            FROM reviews
            WHERE repo = ? AND pr_number = ? AND head_sha = ?
            """,
            (repo, pr_number, head_sha),
        ).fetchone()
        if row is None:
            return None
        return ReviewRecord(
            repo=row["repo"],
            pr_number=row["pr_number"],
            head_sha=row["head_sha"],
            status=row["status"],
            error=row["error"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def get_status(self, repo: str, pr_number: int, head_sha: str) -> str | None:
        record = self.get(repo, pr_number, head_sha)
        if record is None:
            return None
        return record.status

    def mark(self, repo: str, pr_number: int, head_sha: str, status: str, error: str | None = None) -> None:
        if error:
            error = error[:500]
        updated_at = self._timestamp()
        self._conn.execute(
            """
            INSERT INTO reviews (repo, pr_number, head_sha, status, error, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo, pr_number, head_sha) DO UPDATE SET
                status = excluded.status,
                error = excluded.error,
                updated_at = excluded.updated_at
            """,
            (repo, pr_number, head_sha, status, error, updated_at),
        )
        self._conn.commit()

    def should_review(self, repo: str, pr_number: int, head_sha: str, *, dry_run: bool) -> bool:
        record = self.get(repo, pr_number, head_sha)
        if record is None:
            return True
        if record.status in TERMINAL_STATUSES:
            return False
        if record.status == "dry_run" and dry_run:
            return False
        if record.status == "in_progress" and not self._expired(record.updated_at):
            return False
        return True

    def get_comment_id(self, repo: str, pr_number: int) -> int | None:
        row = self._conn.execute(
            "SELECT comment_id FROM pr_comments WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchone()
        if row is None:
            return None
        return int(row["comment_id"])

    def set_comment_id(self, repo: str, pr_number: int, comment_id: int) -> None:
        self._conn.execute(
            """
            INSERT INTO pr_comments (repo, pr_number, comment_id)
            VALUES (?, ?, ?)
            ON CONFLICT(repo, pr_number) DO UPDATE SET comment_id = excluded.comment_id
            """,
            (repo, pr_number, comment_id),
        )
        self._conn.commit()

    def _expired(self, updated_at: datetime) -> bool:
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        now = self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now - updated_at >= self.stale_after

    def _timestamp(self) -> str:
        current = self._now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc).isoformat()

    def _setup(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                repo TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                head_sha TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (repo, pr_number, head_sha)
            );

            CREATE TABLE IF NOT EXISTS pr_comments (
                repo TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                comment_id INTEGER NOT NULL,
                PRIMARY KEY (repo, pr_number)
            );
            """
        )
        self._conn.commit()
