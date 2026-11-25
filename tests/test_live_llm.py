import pytest

from pr_sentinel.integrations.llm import LlmProvider
from pr_sentinel.services.review_engine import parse_review_payload


@pytest.mark.integration
def test_live_lmstudio_returns_review_json() -> None:
    provider = LlmProvider(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        model="qwen3-coder-30b-a3b-instruct",
        timeout=120,
    )
    try:
        content = provider.complete(
            "Reply with one JSON object and no markdown.",
            'Return exactly {"summary":"ok","findings":[]}',
        )
        parsed = parse_review_payload(content)
    finally:
        provider.close()
    assert parsed.summary == "ok"
    assert parsed.findings == ()
