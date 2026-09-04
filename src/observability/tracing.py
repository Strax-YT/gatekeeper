"""Span-based tracing.

Every node and every tool call opens a span. Spans carry duration, token
counts and an estimated cost, which is what lets the UI draw a waterfall of a
single run and lets the budget guard stop a run before it gets expensive.

Traces are appended to JSONL on disk so a finished run is inspectable without
the app running. If LANGSMITH_API_KEY is set, spans are additionally handed to
LangSmith; that path is optional and never blocks a run.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Rough public per-million-token rates, only used for the budget guard.
PRICE_PER_MTOK = {
    "gemini-2.0-flash": {"in": 0.10, "out": 0.40},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "deterministic": {"in": 0.0, "out": 0.0},
}


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price = PRICE_PER_MTOK.get(model, PRICE_PER_MTOK["deterministic"])
    return (tokens_in / 1_000_000) * price["in"] + (tokens_out / 1_000_000) * price["out"]


@dataclass
class Span:
    name: str
    kind: str  # "node" | "tool" | "llm"
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: str | None = None
    started_at: float = field(default_factory=time.time)
    duration_ms: int = 0
    status: str = "ok"
    error: str | None = None
    attempts: int = 1
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tracer:
    """Collects spans for one run."""

    def __init__(self, run_id: str, trace_dir: str | None = None) -> None:
        self.run_id = run_id
        self.spans: list[Span] = []
        self._stack: list[str] = []
        self._trace_dir = Path(trace_dir) if trace_dir else None
        self._langsmith = _try_langsmith()

    @contextmanager
    def span(self, name: str, kind: str = "node", **attributes: Any) -> Iterator[Span]:
        sp = Span(
            name=name,
            kind=kind,
            parent_id=self._stack[-1] if self._stack else None,
            attributes=attributes,
        )
        self._stack.append(sp.span_id)
        start = time.perf_counter()
        try:
            yield sp
        except Exception as exc:
            sp.status = "error"
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.duration_ms = int((time.perf_counter() - start) * 1000)
            self._stack.pop()
            self.spans.append(sp)
            self._emit(sp)

    def record_llm(self, model: str, tokens_in: int, tokens_out: int, duration_ms: int, name: str = "llm.plan") -> Span:
        sp = Span(
            name=name,
            kind="llm",
            parent_id=self._stack[-1] if self._stack else None,
            duration_ms=duration_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=estimate_cost_usd(model, tokens_in, tokens_out),
            attributes={"model": model},
        )
        self.spans.append(sp)
        self._emit(sp)
        return sp

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.spans), 6)

    @property
    def total_ms(self) -> int:
        return sum(s.duration_ms for s in self.spans if s.parent_id is None)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.spans]

    def _emit(self, sp: Span) -> None:
        if self._trace_dir is not None:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            path = self._trace_dir / f"{self.run_id}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"run_id": self.run_id, **sp.to_dict()}) + "\n")
        if self._langsmith is not None:
            try:
                self._langsmith.create_run(
                    name=sp.name,
                    run_type="chain" if sp.kind == "node" else sp.kind,
                    inputs=sp.attributes,
                    outputs={"status": sp.status},
                    extra={"run_id": self.run_id, "duration_ms": sp.duration_ms},
                )
            except Exception:  # pragma: no cover - telemetry must never break a run
                self._langsmith = None


def _try_langsmith():
    """Return a LangSmith client if configured, else None. Never raises."""
    import os

    if not os.getenv("LANGSMITH_API_KEY"):
        return None
    try:  # pragma: no cover - optional dependency
        from langsmith import Client

        return Client()
    except Exception:
        return None


def load_trace(run_id: str, trace_dir: str) -> list[dict[str, Any]]:
    path = Path(trace_dir) / f"{run_id}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
