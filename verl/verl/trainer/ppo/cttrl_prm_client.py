from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable
from urllib import error, request


@dataclass
class PRMExample:
    prompt: str
    response: str
    segments: list[str]


class _RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / max(1, requests_per_minute)
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed_at - now)
            scheduled_at = max(now, self._next_allowed_at)
            self._next_allowed_at = scheduled_at + self.interval

        if wait_seconds > 0:
            time.sleep(wait_seconds)


class OpenAICompatiblePRMClient:
    def __init__(
        self,
        api_base: str | None,
        model_name: str,
        endpoint_path: str = "/chat/completions",
        api_key_env: str = "CTTRL_PRM_API_KEY",
        timeout: int = 60,
        max_retries: int = 3,
        system_prompt: str | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | str | None = None,
        enable_thinking: bool | None = False,
    ):
        self.api_base = api_base.rstrip("/") if api_base else None
        self.model_name = model_name
        self.endpoint_path = endpoint_path
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.extra_body = self._normalize_extra_body(extra_body)
        self.enable_thinking = self._normalize_optional_bool(enable_thinking)
        self.system_prompt = system_prompt or (
            "You are a Process Reward Model. "
            "Evaluate the correctness of each reasoning segment for the given math problem. "
            "Return ONLY valid JSON in one of these forms: "
            '{"scores":[0.0, 1.0]} or [0.0, 1.0]. '
            "Each score must be a float between 0.0 and 1.0. "
            "Do not output any explanation."
        )
        self.debug = os.getenv("CTTRL_PRM_DEBUG", "").lower() in {"1", "true", "yes"}
        self.concurrency = max(1, int(os.getenv("CTTRL_PRM_CONCURRENCY", "8")))
        rate_limit = int(os.getenv("CTTRL_PRM_RATE_LIMIT_PER_MINUTE", "600"))
        self.rate_limiter = _RateLimiter(rate_limit)

    @property
    def enabled(self) -> bool:
        return bool(self.api_base and self.model_name)

    def score_batch(self, examples: Iterable[PRMExample]) -> list[list[float]]:
        examples = list(examples)
        if not examples:
            return []
        if self.concurrency == 1 or len(examples) == 1:
            return [self.score_example(example) for example in examples]

        max_workers = min(self.concurrency, len(examples))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.score_example, examples))

    def score_example(self, example: PRMExample) -> list[float]:
        if not self.enabled:
            return [0.0 for _ in example.segments]

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing PRM API key env: {self.api_key_env}")

        steps_text = "\n".join(
            [f"Segment {i + 1}: {segment}" for i, segment in enumerate(example.segments)]
        )
        user_content = (
            f"Problem:\n{example.prompt}\n\n"
            f"Reasoning Segments:\n{steps_text}\n\n"
            f"Score each segment independently from 0.0 to 1.0.\n"
            f"You must return exactly {len(example.segments)} scores.\n"
            f"Return ONLY JSON, for example: "
            f'{{"scores":[0.9,0.8,1.0]}}'
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        last_error = None
        use_response_format = True
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "stream": False,
                }
                if self.reasoning_effort:
                    payload["reasoning_effort"] = self.reasoning_effort
                if self.enable_thinking is not None and "thinking" not in self.extra_body:
                    payload["enable_thinking"] = self.enable_thinking
                payload.update(self.extra_body)
                if use_response_format:
                    payload["response_format"] = {"type": "json_object"}

                body = json.dumps(payload).encode("utf-8")
                if self.debug:
                    print("==== PRM REQUEST ====")
                    print(json.dumps(payload, ensure_ascii=False, indent=2))

                self.rate_limiter.wait()
                req = request.Request(
                    url=f"{self.api_base}{self.endpoint_path}",
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with request.urlopen(req, timeout=self.timeout) as resp:
                    raw_response = resp.read().decode("utf-8")

                response_data = json.loads(raw_response)
                content = response_data["choices"][0]["message"]["content"]

                if self.debug:
                    print("==== PRM RAW RESPONSE ====")
                    print(raw_response)
                    print("==== PRM CONTENT ====")
                    print(content)

                scores = self._parse_scores_from_text(content, len(example.segments))
                return self._normalize_scores(scores, len(example.segments))

            except error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = ""

                detail = f"{exc}"
                if err_body:
                    detail = f"{detail}, body={err_body[:500]}"

                if self.debug and err_body:
                    print("==== PRM ERROR BODY ====")
                    print(err_body)

                if exc.code == 400 and use_response_format and "response_format" in err_body.lower():
                    print("PRM server rejected response_format, retrying without response_format.")
                    use_response_format = False
                    last_error = RuntimeError(detail)
                    continue

                last_error = RuntimeError(detail)
                print(f"PRM attempt {attempt + 1}/{self.max_retries} failed: {detail}")
                time.sleep(min(2**attempt, 5))
            except (error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                print(f"PRM attempt {attempt + 1}/{self.max_retries} failed: {exc}")
                time.sleep(min(2**attempt, 5))

        raise RuntimeError(f"PRM API request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _normalize_extra_body(extra_body: dict[str, Any] | str | None) -> dict[str, Any]:
        if extra_body in (None, ""):
            return {}

        try:
            from omegaconf import OmegaConf

            if not isinstance(extra_body, str):
                extra_body = OmegaConf.to_container(extra_body, resolve=True)
        except Exception:
            pass

        if isinstance(extra_body, str):
            try:
                extra_body = json.loads(extra_body)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid PRM extra_body JSON: {extra_body}") from exc

        if not isinstance(extra_body, dict):
            raise TypeError(f"PRM extra_body must be a JSON object, got {type(extra_body).__name__}")

        return dict(extra_body)

    @staticmethod
    def _normalize_optional_bool(value: bool | str | None) -> bool | None:
        if value is None or isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "none", "null"}:
                return None
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False

        raise ValueError(f"Expected PRM enable_thinking to be bool or null, got {value!r}")

    def _parse_scores_from_text(self, text: str, expected_len: int) -> list[float]:
        clean_text = text.strip()

        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean_text, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict):
                if "scores" in parsed and isinstance(parsed["scores"], list):
                    return [float(x) for x in parsed["scores"]]
                if "step_rewards" in parsed and isinstance(parsed["step_rewards"], list):
                    return [float(x) for x in parsed["step_rewards"]]
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except Exception:
            pass

        start_idx = clean_text.find("[")
        end_idx = clean_text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                arr = json.loads(clean_text[start_idx : end_idx + 1])
                if isinstance(arr, list):
                    return [float(x) for x in arr]
            except Exception:
                pass

        numbers = re.findall(r"-?\d+(?:\.\d+)?", clean_text)
        if numbers:
            return [float(x) for x in numbers]

        print("Warning: Could not extract valid PRM scores, using defaults.")
        return [0.5] * expected_len

    @staticmethod
    def _normalize_scores(scores: list[float], expected_len: int) -> list[float]:
        clamped = [min(1.0, max(0.0, float(v))) for v in scores[:expected_len]]
        if len(clamped) < expected_len:
            clamped.extend([0.0] * (expected_len - len(clamped)))
        return clamped
