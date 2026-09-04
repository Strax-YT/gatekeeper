"""Guardrails sitting between the plan and any side effect.

Three independent checks, each of which can stop a step:

1. `approval_decision`  — does a human have to sign this off first?
2. `budget_check`       — has the run exhausted its step, tool-call or cost ceiling?
3. `validate_args`      — are the required arguments actually present?

Keeping these out of the agent prompt matters: a prompt-injected document
cannot talk its way past a Python `if`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import settings
from ..state import Budget, Task
from ..tools.registry import Tool


@dataclass
class ApprovalDecision:
    required: bool
    reason: str


def approval_decision(tool: Tool, auto_approve_reversible: bool | None = None) -> ApprovalDecision:
    """Read-only tools run freely. Irreversible writes always need a human.
    Reversible writes are configurable, and default to auto-approved."""
    if auto_approve_reversible is None:
        auto_approve_reversible = settings.auto_approve_reversible

    if tool.sensitivity == "read":
        return ApprovalDecision(False, "read-only tool")
    if tool.irreversible:
        return ApprovalDecision(True, "irreversible side effect")
    if auto_approve_reversible:
        return ApprovalDecision(False, "reversible write, auto-approved by policy")
    return ApprovalDecision(True, "write action, approval required by policy")


@dataclass
class BudgetVerdict:
    ok: bool
    reason: str = ""


def budget_check(budget: Budget, next_is_tool_call: bool = True) -> BudgetVerdict:
    if budget["steps_used"] >= budget["max_steps"]:
        return BudgetVerdict(False, f"step ceiling reached ({budget['max_steps']})")
    if next_is_tool_call and budget["tool_calls_used"] >= budget["max_tool_calls"]:
        return BudgetVerdict(False, f"tool-call ceiling reached ({budget['max_tool_calls']})")
    if budget["usd_used"] >= budget["max_usd"]:
        return BudgetVerdict(False, f"cost ceiling reached (${budget['max_usd']:.2f})")
    return BudgetVerdict(True)


def validate_args(tool: Tool, task: Task, context: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fill blank required arguments from run context where possible, and
    report anything still missing. A step with missing arguments is skipped
    rather than called with None."""
    args = dict(task.get("args") or {})
    for key in tool.required_args:
        if args.get(key) in (None, "") and context.get(key) not in (None, ""):
            args[key] = context[key]
    return args, tool.missing_args(args)
