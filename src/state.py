"""State schema shared by every node in the workflow graph.

The whole run is one serialisable dict. That is deliberate: LangGraph
checkpoints this after each node, so a run can survive a process restart and
be resumed from the exact step where a human was asked to approve something.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

RunStatus = Literal[
    "planning",
    "running",
    "awaiting_approval",
    "completed",
    "failed",
    "halted",
]

StepStatus = Literal["ok", "failed", "skipped", "rejected"]


class Task(TypedDict):
    """One unit of work the supervisor delegated to a specialist agent."""

    id: str
    agent: str
    tool: str
    args: dict[str, Any]
    rationale: str


class StepResult(TypedDict, total=False):
    task_id: str
    agent: str
    tool: str
    status: StepStatus
    output: Any
    error: str | None
    attempts: int
    duration_ms: int
    approved_by: str | None
    approval_note: str | None


class ApprovalRequest(TypedDict):
    """Payload surfaced to the human when a task needs sign-off."""

    task_id: str
    agent: str
    tool: str
    args: dict[str, Any]
    rationale: str
    sensitivity: str
    irreversible: bool
    reason: str
    preview: str


class Budget(TypedDict):
    max_steps: int
    max_tool_calls: int
    max_usd: float
    steps_used: int
    tool_calls_used: int
    usd_used: float


class RunState(TypedDict, total=False):
    run_id: str
    request: str
    context: dict[str, Any]
    plan: list[Task]
    cursor: int
    results: list[StepResult]
    trace: list[dict[str, Any]]
    status: RunStatus
    halt_reason: str | None
    summary: str
    budget: Budget
    planner_model: str
    # Set by the approval node, consumed and cleared by the executor.
    approval: dict[str, Any] | None
    pending_approval: ApprovalRequest | None
    planner_note: str


def new_budget(
    max_steps: int, max_tool_calls: int, max_usd: float
) -> Budget:
    return Budget(
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        max_usd=max_usd,
        steps_used=0,
        tool_calls_used=0,
        usd_used=0.0,
    )


def initial_state(run_id: str, request: str, context: dict[str, Any], budget: Budget) -> RunState:
    return RunState(
        run_id=run_id,
        request=request,
        context=context or {},
        plan=[],
        cursor=0,
        results=[],
        trace=[],
        status="planning",
        halt_reason=None,
        summary="",
        budget=budget,
        planner_model="",
        approval=None,
        pending_approval=None,
        planner_note="",
    )
