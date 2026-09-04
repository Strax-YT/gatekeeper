"""Graph nodes.

Flow:

    plan -> dispatch -> [approve] -> execute -> dispatch -> ... -> finalize

`dispatch` owns all the routing decisions so the policy lives in one place.
`approve` is the only node that blocks: it calls LangGraph's `interrupt`,
which checkpoints the run and returns control to the caller until a human
resumes it with a decision.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from ..config import settings
from ..llm.provider import get_planner
from ..observability.tracing import Tracer
from ..state import RunState, StepResult, Task
from ..tools.registry import registry
from .guardrails import approval_decision, budget_check, validate_args
from .retry import call_with_retry

# Tracers hold open spans and are not serialisable, so they live beside the
# graph state, keyed by run id, rather than inside it.
_TRACERS: dict[str, Tracer] = {}


def get_tracer(run_id: str) -> Tracer:
    if run_id not in _TRACERS:
        _TRACERS[run_id] = Tracer(run_id, settings.trace_dir)
    return _TRACERS[run_id]


def _merge_trace(state: RunState, tracer: Tracer) -> list[dict[str, Any]]:
    """Snapshot the tracer into serialisable state."""
    return tracer.as_dicts()


# --------------------------------------------------------------------------- #

def plan_node(state: RunState) -> dict[str, Any]:
    tracer = get_tracer(state["run_id"])
    planner = get_planner()
    with tracer.span("supervisor.plan", kind="node", request=state["request"][:200]):
        result = planner.plan(state["request"], registry.specs(), state.get("context", {}))
        tracer.record_llm(
            result.model,
            result.tokens_in,
            result.tokens_out,
            result.duration_ms,
            name="supervisor.llm",
        )

    tasks: list[Task] = [
        Task(
            id=f"t{i + 1}",
            agent=t["agent"],
            tool=t["tool"],
            args=t.get("args", {}),
            rationale=t.get("rationale", ""),
        )
        for i, t in enumerate(result.tasks)
    ]

    budget = dict(state["budget"])
    budget["usd_used"] = round(budget["usd_used"] + tracer.total_cost_usd, 6)

    return {
        "plan": tasks,
        "cursor": 0,
        "status": "running" if tasks else "failed",
        "halt_reason": None if tasks else "planner produced no executable steps",
        "planner_model": result.model,
        "planner_note": result.note,
        "budget": budget,
        "trace": _merge_trace(state, tracer),
    }


def dispatch_node(state: RunState) -> dict[str, Any]:
    """Bookkeeping only. The routing itself happens in `route_from_dispatch`,
    which reads the state this node just updated."""
    budget = dict(state["budget"])
    budget["steps_used"] = budget["steps_used"] + 1

    plan = state.get("plan", [])
    cursor = state.get("cursor", 0)
    updates: dict[str, Any] = {"budget": budget}

    if cursor >= len(plan):
        return updates

    verdict = budget_check(budget)
    if not verdict.ok:
        updates["status"] = "halted"
        updates["halt_reason"] = verdict.reason
        return updates

    task = plan[cursor]
    if not registry.has(task["tool"]):
        # Defence in depth: the planner already filters unknown tools.
        return updates

    tool = registry.get(task["tool"])
    args, missing = validate_args(tool, task, state.get("context", {}))
    if missing:
        return updates

    decision = approval_decision(tool)
    if decision.required:
        updates["pending_approval"] = {
            "task_id": task["id"],
            "agent": task["agent"],
            "tool": task["tool"],
            "args": args,
            "rationale": task.get("rationale", ""),
            "sensitivity": tool.sensitivity,
            "irreversible": tool.irreversible,
            "reason": decision.reason,
            "preview": tool.preview_text(args),
        }
        updates["status"] = "awaiting_approval"
    else:
        updates["pending_approval"] = None
        updates["status"] = "running"
    return updates


def route_from_dispatch(state: RunState) -> str:
    plan = state.get("plan", [])
    cursor = state.get("cursor", 0)

    if state.get("status") == "halted":
        return "finalize"
    if cursor >= len(plan):
        return "finalize"

    task = plan[cursor]
    if not registry.has(task["tool"]):
        return "skip"

    tool = registry.get(task["tool"])
    _, missing = validate_args(tool, task, state.get("context", {}))
    if missing:
        return "skip"

    return "approve" if state.get("pending_approval") else "execute"


def approve_node(state: RunState) -> dict[str, Any]:
    """Pause the run and wait for a human.

    `interrupt` checkpoints here and raises out of the graph. When the caller
    resumes with `Command(resume={...})`, this node re-runs from the top and
    `interrupt` returns that value instead of pausing again.
    """
    request = state.get("pending_approval") or {}
    decision = interrupt(
        {
            "type": "approval_request",
            "run_id": state.get("run_id"),
            **request,
        }
    )

    if isinstance(decision, bool):
        decision = {"approved": decision}
    decision = decision or {}
    approved = bool(decision.get("approved"))
    approver = decision.get("approver") or "unknown"
    note = decision.get("note") or ""

    if approved:
        return {
            "approval": {"approved": True, "approver": approver, "note": note},
            "status": "running",
            "pending_approval": None,
        }

    # Rejected: record the refusal, skip the step, keep going.
    tracer = get_tracer(state["run_id"])
    plan = state["plan"]
    cursor = state.get("cursor", 0)
    task = plan[cursor]
    with tracer.span(f"{task['agent']}.{task['tool']}", kind="tool", outcome="rejected") as sp:
        sp.status = "rejected"

    result = StepResult(
        task_id=task["id"],
        agent=task["agent"],
        tool=task["tool"],
        status="rejected",
        output=None,
        error=None,
        attempts=0,
        duration_ms=0,
        approved_by=approver,
        approval_note=note or "rejected by reviewer",
    )
    return {
        "results": [*list(state.get("results", [])), result],
        "cursor": cursor + 1,
        "approval": None,
        "pending_approval": None,
        "status": "running",
        "trace": _merge_trace(state, tracer),
    }


def route_from_approval(state: RunState) -> str:
    return "execute" if (state.get("approval") or {}).get("approved") else "dispatch"


def execute_node(state: RunState) -> dict[str, Any]:
    tracer = get_tracer(state["run_id"])
    plan = state["plan"]
    cursor = state.get("cursor", 0)
    task = plan[cursor]
    tool = registry.get(task["tool"])
    args, _ = validate_args(tool, task, state.get("context", {}))
    approval = state.get("approval") or {}

    with tracer.span(
        f"{task['agent']}.{task['tool']}",
        kind="tool",
        args=args,
        sensitivity=tool.sensitivity,
        irreversible=tool.irreversible,
    ) as sp:
        attempted = call_with_retry(
            tool.fn,
            args,
            retries=tool.max_retries if tool.max_retries is not None else settings.default_retries,
            timeout_s=tool.timeout_s if tool.timeout_s is not None else settings.default_timeout_s,
            backoff_base_s=settings.backoff_base_s,
        )
        sp.attempts = attempted.attempts
        if not attempted.ok:
            sp.status = "error"
            sp.error = attempted.error

    result = StepResult(
        task_id=task["id"],
        agent=task["agent"],
        tool=task["tool"],
        status="ok" if attempted.ok else "failed",
        output=attempted.value if attempted.ok else None,
        error=attempted.error,
        attempts=attempted.attempts,
        duration_ms=attempted.duration_ms,
        approved_by=approval.get("approver"),
        approval_note=approval.get("note"),
    )

    budget = dict(state["budget"])
    budget["tool_calls_used"] = budget["tool_calls_used"] + 1

    return {
        "results": [*list(state.get("results", [])), result],
        "cursor": cursor + 1,
        "approval": None,
        "pending_approval": None,
        "budget": budget,
        "status": "running",
        "trace": _merge_trace(state, tracer),
    }


def skip_node(state: RunState) -> dict[str, Any]:
    """A step that cannot run — unknown tool or unfillable required argument.
    Recorded explicitly so it shows up in the run history instead of vanishing."""
    tracer = get_tracer(state["run_id"])
    plan = state["plan"]
    cursor = state.get("cursor", 0)
    task = plan[cursor]

    reason = "unknown tool"
    if registry.has(task["tool"]):
        tool = registry.get(task["tool"])
        _, missing = validate_args(tool, task, state.get("context", {}))
        reason = f"missing required argument(s): {', '.join(missing)}"

    result = StepResult(
        task_id=task["id"],
        agent=task["agent"],
        tool=task["tool"],
        status="skipped",
        output=None,
        error=reason,
        attempts=0,
        duration_ms=0,
    )
    return {
        "results": [*list(state.get("results", [])), result],
        "cursor": cursor + 1,
        "status": "running",
        "trace": _merge_trace(state, tracer),
    }


def finalize_node(state: RunState) -> dict[str, Any]:
    tracer = get_tracer(state["run_id"])
    results = state.get("results", [])
    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") == "failed"]
    rejected = [r for r in results if r.get("status") == "rejected"]
    skipped = [r for r in results if r.get("status") == "skipped"]

    lines = [f"{len(ok)}/{len(results)} steps completed."]
    if rejected:
        lines.append(f"{len(rejected)} rejected by a reviewer: " + ", ".join(r["tool"] for r in rejected) + ".")
    if failed:
        lines.append(f"{len(failed)} failed: " + ", ".join(f"{r['tool']} ({r.get('error')})" for r in failed) + ".")
    if skipped:
        lines.append(f"{len(skipped)} skipped for missing inputs: " + ", ".join(r["tool"] for r in skipped) + ".")

    for r in ok:
        if r["tool"] == "check_compliance" and isinstance(r.get("output"), dict):
            missing = r["output"].get("missing") or []
            lines.append(
                "Compliance: all documents on file."
                if not missing
                else f"Compliance: still missing {', '.join(missing)}."
            )
        if r["tool"] == "provision_access" and isinstance(r.get("output"), dict):
            blocked = r["output"].get("blocked") or []
            if blocked:
                lines.append(f"Access: withheld {', '.join(blocked)} — needs a data owner sign-off.")

    if state.get("status") == "halted":
        status = "halted"
        lines.insert(0, f"Run halted: {state.get('halt_reason')}.")
    elif failed:
        status = "failed"
    else:
        status = "completed"

    budget = dict(state["budget"])
    budget["usd_used"] = round(tracer.total_cost_usd, 6)

    return {
        "summary": " ".join(lines),
        "status": status,
        "budget": budget,
        "pending_approval": None,
        "trace": _merge_trace(state, tracer),
    }
