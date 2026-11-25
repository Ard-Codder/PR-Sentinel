from collections.abc import Callable

import httpx

from pr_sentinel.retry import call_with_retry


class LlmError(Exception):
    pass


class LlmProvider:
    """OpenAI-compatible chat client. Only message.content is returned."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 180,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] | None = None,
        attempts: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._sleep = sleep
        self.attempts = attempts
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        url = f"{self.base_url}/chat/completions"

        def send() -> httpx.Response:
            response = self._client.post(url, json=payload, headers=self._headers())
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"LLM HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            return response

        try:
            response = call_with_retry(send, attempts=self.attempts, sleep=self._sleep or _default_sleep)
        except httpx.HTTPStatusError as exc:
            raise LlmError(f"LLM HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise LlmError(f"LLM request failed: {exc.__class__.__name__}") from exc

        try:
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LlmError("LLM response has no message content") from exc
        if not str(content).strip():
            raise LlmError("LLM response content is empty")
        return str(content)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }


def _default_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)
