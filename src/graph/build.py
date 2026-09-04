"""Graph assembly and the run façade.

`WorkflowRunner` is the only thing the API and the UI touch. It owns the
SQLite checkpointer, so a run interrupted for approval can be resumed by a
different process — which is the whole point of the human-in-the-loop design.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ..config import settings
from ..observability.tracing import load_trace
from ..state import RunState, initial_state, new_budget
from . import nodes


def build_graph(checkpointer: Any | None = None):
    g = StateGraph(RunState)

    g.add_node("plan", nodes.plan_node)
    g.add_node("dispatch", nodes.dispatch_node)
    g.add_node("approve", nodes.approve_node)
    g.add_node("execute", nodes.execute_node)
    g.add_node("skip", nodes.skip_node)
    g.add_node("finalize", nodes.finalize_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "dispatch")
    g.add_conditional_edges(
        "dispatch",
        nodes.route_from_dispatch,
        {
            "approve": "approve",
            "execute": "execute",
            "skip": "skip",
            "finalize": "finalize",
        },
    )
    g.add_conditional_edges(
        "approve",
        nodes.route_from_approval,
        {"execute": "execute", "dispatch": "dispatch"},
    )
    g.add_edge("execute", "dispatch")
    g.add_edge("skip", "dispatch")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


@dataclass
class RunView:
    """Everything a caller needs to render a run, interrupted or finished."""

    run_id: str
    status: str
    request: str
    plan: list[dict[str, Any]]
    results: list[dict[str, Any]]
    summary: str
    pending_approval: dict[str, Any] | None
    budget: dict[str, Any]
    planner_model: str
    planner_note: str
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "request": self.request,
            "plan": self.plan,
            "results": self.results,
            "summary": self.summary,
            "pending_approval": self.pending_approval,
            "budget": self.budget,
            "planner_model": self.planner_model,
            "planner_note": self.planner_note,
            "trace": self.trace,
        }


class WorkflowRunner:
    def __init__(self, db_path: str | None = None, recursion_limit: int = 80) -> None:
        settings.ensure_dirs()
        self.db_path = db_path or settings.checkpoint_db
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._conn)
        self.graph = build_graph(self.checkpointer)
        self.recursion_limit = recursion_limit

    # -- lifecycle ---------------------------------------------------------- #

    def start(
        self,
        request: str,
        context: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> RunView:
        run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
        state = initial_state(
            run_id,
            request,
            context or {},
            new_budget(settings.max_steps, settings.max_tool_calls, settings.max_usd),
        )
        self.graph.invoke(state, config=self._config(run_id))
        return self.view(run_id)

    def resume(
        self,
        run_id: str,
        approved: bool,
        approver: str = "reviewer",
        note: str = "",
    ) -> RunView:
        """Answer the outstanding approval request and continue the run."""
        self.graph.invoke(
            Command(resume={"approved": approved, "approver": approver, "note": note}),
            config=self._config(run_id),
        )
        return self.view(run_id)

    def run_to_completion(
        self,
        request: str,
        context: dict[str, Any] | None = None,
        decide: Any | None = None,
        approver: str = "auto",
        run_id: str | None = None,
        max_gates: int = 20,
    ) -> RunView:
        """Convenience path for scripts and evals: drive a run through every
        approval gate using `decide(approval_request) -> bool`."""
        if decide is None:
            def decide(_req: dict[str, Any]) -> bool:
                return True

        view = self.start(request, context, run_id=run_id)
        gates = 0
        while view.status == "awaiting_approval" and gates < max_gates:
            approved = bool(decide(view.pending_approval or {}))
            view = self.resume(
                view.run_id,
                approved=approved,
                approver=approver,
                note="auto-decision" if approved else "auto-rejected",
            )
            gates += 1
        return view

    # -- inspection --------------------------------------------------------- #

    def view(self, run_id: str) -> RunView:
        snapshot = self.graph.get_state(self._config(run_id))
        values = snapshot.values or {}
        pending = self._pending_from(snapshot, values)
        status = values.get("status", "unknown")
        if pending and status not in {"completed", "failed", "halted"}:
            status = "awaiting_approval"
        return RunView(
            run_id=run_id,
            status=status,
            request=values.get("request", ""),
            plan=list(values.get("plan", [])),
            results=list(values.get("results", [])),
            summary=values.get("summary", ""),
            pending_approval=pending,
            budget=dict(values.get("budget", {})),
            planner_model=values.get("planner_model", ""),
            planner_note=values.get("planner_note", ""),
            trace=list(values.get("trace", [])) or load_trace(run_id, settings.trace_dir),
        )

    def history(self, run_id: str) -> list[dict[str, Any]]:
        """Every checkpoint for a run, oldest first — the audit trail."""
        out = []
        for snap in self.graph.get_state_history(self._config(run_id)):
            values = snap.values or {}
            out.append(
                {
                    "checkpoint_id": snap.config.get("configurable", {}).get("checkpoint_id"),
                    "next": list(snap.next),
                    "status": values.get("status"),
                    "cursor": values.get("cursor"),
                    "steps_done": len(values.get("results", [])),
                }
            )
        return list(reversed(out))

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC LIMIT ?", (limit,)
        )
        runs = []
        for (thread_id,) in cur.fetchall():
            try:
                v = self.view(thread_id)
            except Exception:
                continue
            runs.append(
                {
                    "run_id": v.run_id,
                    "status": v.status,
                    "request": v.request,
                    "steps": len(v.results),
                    "planned": len(v.plan),
                }
            )
        return runs

    # -- internals ---------------------------------------------------------- #

    def _config(self, run_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id},
            "recursion_limit": self.recursion_limit,
        }

    @staticmethod
    def _pending_from(snapshot: Any, values: dict[str, Any]) -> dict[str, Any] | None:
        """Prefer the live interrupt payload; fall back to state."""
        for task in getattr(snapshot, "tasks", ()) or ():
            for itr in getattr(task, "interrupts", ()) or ():
                if isinstance(getattr(itr, "value", None), dict):
                    return itr.value
        pending = values.get("pending_approval")
        return dict(pending) if pending else None

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
