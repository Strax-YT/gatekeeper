from __future__ import annotations

import pytest

from src.graph.guardrails import approval_decision, budget_check, validate_args
from src.llm.provider import RulePlanner, _parse_tasks
from src.state import Task, new_budget
from src.tools.registry import registry


def test_read_only_tools_never_gated():
    assert approval_decision(registry.get("check_compliance")).required is False
    assert approval_decision(registry.get("search_policy")).required is False


@pytest.mark.parametrize(
    "name",
    ["provision_access", "revoke_access", "schedule_orientation", "send_welcome_email"],
)
def test_irreversible_tools_always_gated(name):
    """Even with auto-approval switched on, irreversible actions need a human."""
    decision = approval_decision(registry.get(name), auto_approve_reversible=True)
    assert decision.required is True
    assert "irreversible" in decision.reason


def test_reversible_write_follows_policy():
    tool = registry.get("generate_document")
    assert approval_decision(tool, auto_approve_reversible=True).required is False
    assert approval_decision(tool, auto_approve_reversible=False).required is True


def test_every_write_tool_declares_reversibility():
    """A new tool cannot be added as a write without an explicit risk call."""
    for name in registry.names():
        tool = registry.get(name)
        if tool.sensitivity == "write":
            assert isinstance(tool.irreversible, bool)
            assert tool.description, f"{name} needs a description for the planner"


def test_budget_stops_on_each_ceiling():
    b = new_budget(max_steps=2, max_tool_calls=5, max_usd=1.0)
    b["steps_used"] = 2
    assert budget_check(b).ok is False

    b = new_budget(max_steps=10, max_tool_calls=1, max_usd=1.0)
    b["tool_calls_used"] = 1
    assert budget_check(b).ok is False
    assert budget_check(b, next_is_tool_call=False).ok is True

    b = new_budget(max_steps=10, max_tool_calls=10, max_usd=0.01)
    b["usd_used"] = 0.02
    assert budget_check(b).ok is False


def test_validate_args_fills_from_context_and_reports_gaps():
    tool = registry.get("check_compliance")
    task = Task(id="t1", agent="compliance", tool="check_compliance", args={}, rationale="")

    args, missing = validate_args(tool, task, {"employee": "Priya Sharma"})
    assert args["employee"] == "Priya Sharma"
    assert missing == []

    args, missing = validate_args(tool, task, {})
    assert missing == ["employee"]


def test_hallucinated_tools_are_dropped_not_executed():
    specs = registry.specs()
    raw = """```json
    {"tasks": [
      {"tool": "provision_access", "args": {"employee": "A", "bogus_arg": 1}, "rationale": "ok"},
      {"tool": "wire_transfer", "args": {"amount": 9999}, "rationale": "not a real tool"}
    ]}
    ```"""
    tasks = _parse_tasks(raw, specs)
    assert [t["tool"] for t in tasks] == ["provision_access"]
    assert "bogus_arg" not in tasks[0]["args"]


def test_unparseable_model_output_yields_no_plan():
    assert _parse_tasks("I'm sorry, I can't help with that.", registry.specs()) == []
    assert _parse_tasks("", registry.specs()) == []


def test_rule_planner_never_plans_writes_for_a_question():
    plan = RulePlanner().plan(
        "How many days of paid leave do I get?", registry.specs(), {}
    )
    tools = [t["tool"] for t in plan.tasks]
    assert tools == ["search_policy"]
    for name in tools:
        assert registry.get(name).sensitivity == "read"
