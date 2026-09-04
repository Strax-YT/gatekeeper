"""Tool registry.

Each tool declares its own risk profile. The graph reads that metadata to
decide whether a human has to approve the call, so the approval policy lives
with the tool rather than being scattered through the agent prompts. Adding a
tool cannot accidentally bypass the gate: `sensitivity="write"` is enough.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Sensitivity = Literal["read", "write"]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    agent: str
    fn: Callable[..., Any]
    sensitivity: Sensitivity = "read"
    irreversible: bool = False
    args_schema: dict[str, str] = field(default_factory=dict)
    required_args: tuple[str, ...] = ()
    timeout_s: float | None = None
    max_retries: int | None = None
    preview: Callable[[dict[str, Any]], str] | None = None

    def spec(self) -> dict[str, Any]:
        """Compact description handed to the planner LLM."""
        return {
            "name": self.name,
            "agent": self.agent,
            "description": self.description,
            "sensitivity": self.sensitivity,
            "irreversible": self.irreversible,
            "args": self.args_schema,
            "required": list(self.required_args),
        }

    def preview_text(self, args: dict[str, Any]) -> str:
        if self.preview is not None:
            try:
                return self.preview(args)
            except Exception:
                pass
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(args.items()))
        return f"{self.name}({rendered})"

    def missing_args(self, args: dict[str, Any]) -> list[str]:
        return [a for a in self.required_args if a not in args or args[a] in (None, "")]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool: {name}") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [self._tools[n].spec() for n in self.names()]

    def by_agent(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name in self.names():
            out.setdefault(self._tools[name].agent, []).append(name)
        return out


registry = ToolRegistry()


def tool(
    *,
    name: str,
    description: str,
    agent: str,
    sensitivity: Sensitivity = "read",
    irreversible: bool = False,
    args_schema: dict[str, str] | None = None,
    required_args: tuple[str, ...] = (),
    timeout_s: float | None = None,
    max_retries: int | None = None,
    preview: Callable[[dict[str, Any]], str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a plain function as an agent tool."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        registry.register(
            Tool(
                name=name,
                description=description,
                agent=agent,
                fn=fn,
                sensitivity=sensitivity,
                irreversible=irreversible,
                args_schema=args_schema or {},
                required_args=required_args,
                timeout_s=timeout_s,
                max_retries=max_retries,
                preview=preview,
            )
        )
        return fn

    return wrap
