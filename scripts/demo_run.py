"""End-to-end demo. Run: python -m scripts.demo_run

Shows the two behaviours that matter: the run pausing before every
irreversible action, and a rejection being recorded rather than silently
dropping the rest of the plan.
"""

from __future__ import annotations

import sys

from src.graph.build import WorkflowRunner
from src.tools.catalog import reset_systems, systems_snapshot

REQUEST = "Onboard Priya Sharma as a full-time backend engineer starting 2026-10-01."


def banner(text: str) -> None:
    print("\n" + text)
    print("-" * len(text))


def main() -> int:
    reset_systems()
    runner = WorkflowRunner()

    banner("1. Plan")
    view = runner.start(REQUEST, context={"employee": "Priya Sharma", "role": "Backend Engineer",
                                          "start_date": "2026-10-01"})
    print(f"planner: {view.planner_model}   status: {view.status}")
    for t in view.plan:
        print(f"  {t['id']}  {t['agent']:<11} {t['tool']:<20} {t['rationale']}")

    banner("2. Approval gates")
    # Reject the calendar invite, approve everything else, to show both paths.
    gate = 0
    while view.status == "awaiting_approval" and gate < 10:
        req = view.pending_approval or {}
        approve = req.get("tool") != "schedule_orientation"
        print(f"  gate: {req.get('preview')}")
        print(f"         reason={req.get('reason')}  -> {'APPROVE' if approve else 'REJECT'}")
        view = runner.resume(view.run_id, approved=approve, approver="yash",
                             note="looks right" if approve else "manager is on leave that week")
        gate += 1

    banner("3. Result")
    print(f"status: {view.status}")
    print(f"summary: {view.summary}")
    for r in view.results:
        mark = {"ok": "  ok", "failed": "fail", "rejected": "rej", "skipped": "skip"}.get(r["status"], "?")
        print(f"  [{mark}] {r['tool']:<20} attempts={r.get('attempts')} {r.get('error') or ''}")

    banner("4. Side effects actually applied")
    snap = systems_snapshot()
    print(f"  accounts created : {list(snap['accounts'])}")
    print(f"  meetings booked  : {len(snap['meetings'])}  (rejected gate means 0)")
    print(f"  emails sent      : {len(snap['messages'])}")

    banner("5. Trace")
    print(f"  spans: {len(view.trace)}   cost: ${view.budget.get('usd_used', 0):.6f}")
    for sp in view.trace:
        if sp.get("kind") == "tool":
            print(f"  {sp['duration_ms']:>5}ms  {sp['name']:<32} {sp['status']}")

    banner("6. Audit trail (checkpoints)")
    for h in runner.history(view.run_id)[:8]:
        print(f"  next={h['next']!s:<14} status={h['status']!s:<18} steps_done={h['steps_done']}")

    runner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
