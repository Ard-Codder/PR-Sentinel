from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    github_token: str = ""
    github_api_url: str = "https://api.github.com"
    github_repos: str = ""
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "qwen3-coder-30b-a3b-instruct"
    llm_timeout_seconds: float = 180
    github_timeout_seconds: float = 30
    poll_interval_seconds: int = 120
    max_patch_chars: int = 80_000
    max_file_patch_chars: int = 20_000
    max_chunks: int = 4
    max_prs_per_cycle: int = 5
    max_repos: int = 20
    database_path: str = "./data/state.db"
    review_language: str = "ru"
    dry_run: bool = True
    max_findings: int = 20
    max_message_chars: int = 2_000
    in_progress_timeout_minutes: int = 30

    @field_validator("github_api_url", "llm_base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    def repo_list(self) -> list[str]:
        return [item.strip() for item in self.github_repos.split(",") if item.strip()]
