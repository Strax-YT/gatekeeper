from __future__ import annotations

import uuid

import pytest

from src.graph.build import WorkflowRunner
from src.tools.catalog import reset_systems, systems_snapshot


@pytest.fixture()
def runner(tmp_path):
    reset_systems()
    r = WorkflowRunner(db_path=str(tmp_path / "test.sqlite"))
    yield r
    r.close()
    reset_systems()


def rid() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


def test_run_pauses_before_the_first_irreversible_action(runner):
    view = runner.start(
        "Onboard Priya Sharma as a backend engineer starting 2026-10-01.",
        context={"employee": "Priya Sharma"},
        run_id=rid(),
    )
    assert view.status == "awaiting_approval"
    assert view.pending_approval["tool"] == "provision_access"
    assert view.pending_approval["irreversible"] is True

    # Nothing irreversible has happened yet.
    snap = systems_snapshot()
    assert snap["accounts"] == {}
    assert snap["messages"] == []


def test_rejecting_every_gate_leaves_no_side_effects(runner):
    view = runner.run_to_completion(
        "Onboard Sana Iqbal as an intern starting 2026-12-01.",
        context={"employee": "Sana Iqbal"},
        decide=lambda _req: False,
        run_id=rid(),
    )
    assert view.status in {"completed", "failed"}
    snap = systems_snapshot()
    assert snap["accounts"] == {}
    assert snap["meetings"] == []
    assert snap["messages"] == []
    rejected = [r["tool"] for r in view.results if r["status"] == "rejected"]
    assert "provision_access" in rejected


def test_rejection_is_recorded_and_the_plan_continues(runner):
    view = runner.run_to_completion(
        "Onboard Neha Kulkarni as a data analyst starting 2026-11-03.",
        context={"employee": "Neha Kulkarni"},
        decide=lambda req: req.get("tool") != "schedule_orientation",
        run_id=rid(),
    )
    by_tool = {r["tool"]: r for r in view.results}
    assert by_tool["schedule_orientation"]["status"] == "rejected"
    # Later steps still ran, so one refusal does not abandon the workflow.
    assert by_tool["send_welcome_email"]["status"] == "ok"
    assert systems_snapshot()["meetings"] == []


def test_approved_actions_record_who_approved_them(runner):
    view = runner.run_to_completion(
        "Onboard Priya Sharma as a backend engineer.",
        context={"employee": "Priya Sharma"},
        decide=lambda _req: True,
        approver="yash",
        run_id=rid(),
    )
    irreversible = [r for r in view.results if r["tool"] in
                    {"provision_access", "schedule_orientation", "send_welcome_email"}]
    assert irreversible, "expected irreversible steps in this plan"
    for r in irreversible:
        assert r["status"] == "ok"
        assert r["approved_by"] == "yash"


def test_read_only_question_never_reaches_a_gate(runner):
    view = runner.start(
        "How many days of paid leave is a full-time employee entitled to?",
        run_id=rid(),
    )
    assert view.status == "completed"
    assert view.pending_approval is None
    assert [r["tool"] for r in view.results] == ["search_policy"]


def test_least_privilege_blocks_production_access(runner):
    view = runner.run_to_completion(
        "Onboard Priya Sharma as a backend engineer.",
        context={"employee": "Priya Sharma"},
        decide=lambda _req: True,
        run_id=rid(),
    )
    # The plan asks for standard systems; ask directly for prod to prove the
    # tool itself refuses rather than relying on the planner being polite.
    from src.tools.catalog import provision_access

    out = provision_access("Priya Sharma", systems=["email", "prod_database"])
    assert "prod_database" in out["blocked"]
    assert "prod_database" not in out["granted"]
    assert view.status == "completed"


def test_a_run_is_resumable_by_a_fresh_process(tmp_path):
    """The checkpointer is the whole point: a second runner over the same DB
    must be able to answer a gate opened by the first."""
    reset_systems()
    db = str(tmp_path / "resume.sqlite")
    run_id = rid()

    first = WorkflowRunner(db_path=db)
    view = first.start(
        "Onboard Priya Sharma as a backend engineer.",
        context={"employee": "Priya Sharma"},
        run_id=run_id,
    )
    assert view.status == "awaiting_approval"
    first.close()

    second = WorkflowRunner(db_path=db)
    revived = second.view(run_id)
    assert revived.status == "awaiting_approval"
    assert revived.pending_approval["tool"] == "provision_access"

    final = second.resume(run_id, approved=True, approver="second-process")
    assert final.status in {"awaiting_approval", "completed"}
    second.close()
    reset_systems()


def test_budget_ceiling_halts_a_run(tmp_path, monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "max_tool_calls", 1)
    reset_systems()
    r = WorkflowRunner(db_path=str(tmp_path / "budget.sqlite"))
    view = r.run_to_completion(
        "Onboard Priya Sharma as a backend engineer.",
        context={"employee": "Priya Sharma"},
        decide=lambda _req: True,
        run_id=rid(),
    )
    assert view.status == "halted"
    assert "ceiling" in (view.summary or "")
    r.close()
    reset_systems()


def test_audit_trail_records_every_checkpoint(runner):
    view = runner.run_to_completion(
        "Onboard Priya Sharma as a backend engineer.",
        context={"employee": "Priya Sharma"},
        decide=lambda _req: True,
        run_id=rid(),
    )
    history = runner.history(view.run_id)
    assert len(history) > 5
    assert any(h["next"] == ["approve"] for h in history)
