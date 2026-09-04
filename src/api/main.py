"""HTTP surface.

The approval endpoint is what makes this deployable: an interrupted run lives
in SQLite, so the reviewer can be a different person, in a different process,
minutes later.

    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import settings
from ..graph.build import WorkflowRunner
from ..tools.catalog import systems_snapshot
from ..tools.registry import registry

app = FastAPI(
    title="Agentic Workflow Automation",
    version="1.0.0",
    description=(
        "Supervisor-and-specialists agent platform with human approval gates "
        "on every irreversible action."
    ),
)

runner = WorkflowRunner()


class StartRun(BaseModel):
    request: str = Field(..., examples=["Onboard Priya Sharma as a backend engineer starting 2026-10-01."])
    context: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    approved: bool
    approver: str = "reviewer"
    note: str = ""


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "planner": settings.planner_backend,
        "tools": len(registry.names()),
        "auto_approve_reversible": settings.auto_approve_reversible,
    }


@app.get("/tools")
def list_tools() -> dict[str, Any]:
    """The tool catalogue and, for each, whether it needs human approval."""
    from ..graph.guardrails import approval_decision

    out = []
    for name in registry.names():
        tool = registry.get(name)
        decision = approval_decision(tool)
        out.append(
            {
                **tool.spec(),
                "requires_approval": decision.required,
                "approval_reason": decision.reason,
            }
        )
    return {"tools": out, "by_agent": registry.by_agent()}


@app.post("/runs", status_code=201)
def create_run(body: StartRun) -> dict[str, Any]:
    view = runner.start(body.request, body.context)
    return view.to_dict()


@app.get("/runs")
def list_runs(limit: int = 50) -> dict[str, Any]:
    return {"runs": runner.list_runs(limit)}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    view = runner.view(run_id)
    if view.status == "unknown":
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    return view.to_dict()


@app.post("/runs/{run_id}/decision")
def decide(run_id: str, body: Decision) -> dict[str, Any]:
    view = runner.view(run_id)
    if view.status == "unknown":
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    if view.pending_approval is None:
        raise HTTPException(status_code=409, detail=f"run {run_id} has no pending approval")
    return runner.resume(run_id, body.approved, body.approver, body.note).to_dict()


@app.get("/runs/{run_id}/trace")
def get_trace(run_id: str) -> dict[str, Any]:
    view = runner.view(run_id)
    spans = view.trace
    return {
        "run_id": run_id,
        "span_count": len(spans),
        "total_ms": sum(s.get("duration_ms", 0) for s in spans if not s.get("parent_id")),
        "cost_usd": view.budget.get("usd_used", 0.0),
        "spans": spans,
    }


@app.get("/runs/{run_id}/history")
def get_history(run_id: str) -> dict[str, Any]:
    return {"run_id": run_id, "checkpoints": runner.history(run_id)}


@app.get("/systems")
def get_systems() -> dict[str, Any]:
    """Inspect the simulated systems of record, to confirm which side effects
    actually landed."""
    return systems_snapshot()
