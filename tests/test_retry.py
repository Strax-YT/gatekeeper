from __future__ import annotations

import time

from src.graph.retry import call_with_retry


def test_transient_failure_is_retried_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("503")
        return "ok"

    result = call_with_retry(flaky, {}, retries=2, sleep=lambda _s: None)
    assert result.ok is True
    assert result.value == "ok"
    assert result.attempts == 3


def test_retries_are_exhausted_and_reported():
    def always_down():
        raise ConnectionError("503")

    result = call_with_retry(always_down, {}, retries=2, sleep=lambda _s: None)
    assert result.ok is False
    assert result.attempts == 3
    assert "ConnectionError" in result.error


def test_deterministic_error_fails_immediately():
    """Retrying a bug wastes budget, so non-transient errors stop at once."""
    calls = {"n": 0}

    def bad_args():
        calls["n"] += 1
        raise ValueError("employee is required")

    result = call_with_retry(bad_args, {}, retries=3, sleep=lambda _s: None)
    assert result.ok is False
    assert result.attempts == 1
    assert calls["n"] == 1


def test_timeout_is_enforced():
    def slow():
        time.sleep(2.0)
        return "never"

    result = call_with_retry(slow, {}, retries=0, timeout_s=0.15, sleep=lambda _s: None)
    assert result.ok is False
    assert "ToolTimeout" in result.error


def test_backoff_grows_between_attempts():
    waits: list[float] = []

    def down():
        raise ConnectionError("503")

    call_with_retry(down, {}, retries=3, backoff_base_s=0.1, sleep=waits.append)
    assert waits == [0.1, 0.2, 0.4]


def test_kwargs_are_passed_through():
    result = call_with_retry(lambda a, b: a + b, {"a": 2, "b": 3}, retries=0)
    assert result.value == 5
