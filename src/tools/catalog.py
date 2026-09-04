"""Specialist tools for employee onboarding and offboarding.

The side effects are simulated against a local JSON store rather than real HR
systems, but the interfaces are the ones a real integration would expose, and
the risk metadata is honest: anything that grants access or contacts a person
is marked irreversible and cannot run without human approval.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import date, datetime, timedelta
from typing import Any

from ..config import ROOT, settings
from .registry import tool

STORE = ROOT / "data" / "systems.json"

POLICY_CORPUS: dict[str, str] = {
    "leave": (
        "Full-time employees accrue 18 days of paid leave per year, accrued monthly. "
        "Unused leave carries over up to 12 days. Leave requests over 5 consecutive "
        "days need manager approval two weeks ahead."
    ),
    "probation": (
        "New joiners serve a 90-day probation. A written review happens at day 45 and "
        "day 90. Confirmation requires a manager sign-off recorded in the HRIS."
    ),
    "equipment": (
        "Engineering roles are issued a laptop, an external monitor and a YubiKey. "
        "Hardware is shipped to the registered address two business days before the "
        "start date. Loss must be reported to IT within 24 hours."
    ),
    "background_check": (
        "Employment verification and criminal record checks are mandatory before a "
        "start date is confirmed. Education verification is required for roles at "
        "grade L4 and above."
    ),
    "access": (
        "Access follows least privilege. Production database access requires the "
        "employee's manager and a data owner to approve, and is granted for 90 days "
        "at a time. Contractors never receive production access."
    ),
}

REQUIRED_DOCS = {
    "full_time": ["signed_offer", "id_proof", "background_check", "tax_form"],
    "contractor": ["signed_contract", "id_proof", "tax_form"],
    "intern": ["signed_offer", "id_proof", "college_noc"],
}


# --------------------------------------------------------------------------- #
# simulated system of record
# --------------------------------------------------------------------------- #

def _load() -> dict[str, Any]:
    if STORE.exists():
        try:
            return json.loads(STORE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"accounts": {}, "documents": {}, "meetings": [], "messages": []}


def _save(data: dict[str, Any]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _maybe_fail(tool_name: str) -> None:
    """Fault injection so the retry path is demonstrable, not theoretical.

    Enabled with FLAKY_TOOLS=true. Deterministic per (tool, attempt-clock) so
    a demo reliably shows one failure then a success.
    """
    if not settings.flaky_tools:
        return
    seed = int(os.getenv("FLAKY_SEED", "0")) or int(time.time() // 3)
    rng = random.Random(f"{tool_name}:{seed}")
    if rng.random() < 0.4:
        raise ConnectionError(f"{tool_name}: upstream HR system returned 503")


# --------------------------------------------------------------------------- #
# compliance agent — read only
# --------------------------------------------------------------------------- #

@tool(
    name="check_compliance",
    agent="compliance",
    description=(
        "Check which mandatory onboarding documents are still missing for a new "
        "joiner. Read-only. Run this before anything that depends on a confirmed "
        "start date."
    ),
    args_schema={
        "employee": "full name of the joiner",
        "employment_type": "one of full_time, contractor, intern",
    },
    required_args=("employee",),
)
def check_compliance(employee: str, employment_type: str = "full_time", **_: Any) -> dict[str, Any]:
    _maybe_fail("check_compliance")
    data = _load()
    required = REQUIRED_DOCS.get(employment_type, REQUIRED_DOCS["full_time"])
    on_file = data["documents"].get(employee, [])
    missing = [d for d in required if d not in on_file]
    return {
        "employee": employee,
        "employment_type": employment_type,
        "required": required,
        "on_file": on_file,
        "missing": missing,
        "clear_to_start": not missing,
    }


@tool(
    name="search_policy",
    agent="compliance",
    description=(
        "Look up the company HR policy on a topic (leave, probation, equipment, "
        "background_check, access) and return the grounding text. Read-only."
    ),
    args_schema={"topic": "policy topic to look up"},
    required_args=("topic",),
)
def search_policy(topic: str, **_: Any) -> dict[str, Any]:
    _maybe_fail("search_policy")
    key = (topic or "").strip().lower().replace(" ", "_")
    if key in POLICY_CORPUS:
        return {"topic": key, "found": True, "text": POLICY_CORPUS[key]}
    # crude keyword fallback so the agent gets a grounded answer, or an honest miss
    for cand, text in POLICY_CORPUS.items():
        if cand in key or key in text.lower():
            return {"topic": cand, "found": True, "text": text}
    return {
        "topic": key,
        "found": False,
        "text": "",
        "available_topics": sorted(POLICY_CORPUS),
    }


# --------------------------------------------------------------------------- #
# documents agent — reversible write
# --------------------------------------------------------------------------- #

@tool(
    name="generate_document",
    agent="documents",
    description=(
        "Draft an onboarding document (welcome_packet, offer_letter, "
        "probation_plan) for a joiner. Creates a draft only; nothing is sent, so "
        "this is reversible."
    ),
    sensitivity="write",
    irreversible=False,
    args_schema={
        "employee": "full name of the joiner",
        "doc_type": "welcome_packet, offer_letter or probation_plan",
        "role": "job title",
    },
    required_args=("employee", "doc_type"),
    preview=lambda a: f"Draft a {a.get('doc_type', 'document')} for {a.get('employee', 'the joiner')}",
)
def generate_document(employee: str, doc_type: str = "welcome_packet", role: str = "", **_: Any) -> dict[str, Any]:
    _maybe_fail("generate_document")
    data = _load()
    doc_id = f"DOC-{len(data['documents'].get('_drafts', [])) + 1:04d}"
    body = {
        "welcome_packet": (
            f"Welcome aboard, {employee}. Your first week covers orientation, "
            "tooling setup and a meet with your buddy."
        ),
        "offer_letter": (
            f"{employee} is offered the position of {role or 'the role'}, "
            "subject to background verification."
        ),
        "probation_plan": f"90-day plan for {employee}: day 45 written review, day 90 confirmation review.",
    }.get(doc_type, f"Document for {employee}.")
    drafts = data["documents"].setdefault("_drafts", [])
    drafts.append({"id": doc_id, "employee": employee, "doc_type": doc_type, "body": body})
    _save(data)
    return {"document_id": doc_id, "doc_type": doc_type, "employee": employee, "body": body, "state": "draft"}


# --------------------------------------------------------------------------- #
# access agent — irreversible write, always gated
# --------------------------------------------------------------------------- #

@tool(
    name="provision_access",
    agent="access",
    description=(
        "Create accounts and grant system access for a joiner. Irreversible: it "
        "creates real credentials, so it always requires human approval."
    ),
    sensitivity="write",
    irreversible=True,
    args_schema={
        "employee": "full name of the joiner",
        "systems": "list of systems, e.g. ['email','github','vpn']",
        "role": "job title, used to pick a least-privilege default",
    },
    required_args=("employee",),
    preview=lambda a: (
        f"Grant {a.get('employee', 'the joiner')} access to "
        f"{', '.join(a.get('systems') or ['email', 'slack', 'github'])}"
    ),
)
def provision_access(employee: str, systems: Any = None, role: str = "", **_: Any) -> dict[str, Any]:
    _maybe_fail("provision_access")
    if isinstance(systems, str):
        systems = [s.strip() for s in systems.split(",") if s.strip()]
    systems = systems or ["email", "slack", "github"]
    # least privilege: production access is never granted by an agent
    blocked = [s for s in systems if "prod" in s.lower() or "database" in s.lower()]
    granted = [s for s in systems if s not in blocked]
    data = _load()
    data["accounts"][employee] = {
        "granted": granted,
        "role": role,
        "granted_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save(data)
    return {
        "employee": employee,
        "granted": granted,
        "blocked": blocked,
        "note": (
            "Production and database access withheld: policy requires a data owner "
            "sign-off and cannot be granted by an agent."
            if blocked
            else "Least-privilege defaults applied."
        ),
    }


@tool(
    name="revoke_access",
    agent="access",
    description=(
        "Revoke all system access for a departing employee. Irreversible and "
        "always requires human approval."
    ),
    sensitivity="write",
    irreversible=True,
    args_schema={"employee": "full name of the departing employee"},
    required_args=("employee",),
    preview=lambda a: f"Revoke all access for {a.get('employee', 'the employee')}",
)
def revoke_access(employee: str, **_: Any) -> dict[str, Any]:
    _maybe_fail("revoke_access")
    data = _load()
    prior = data["accounts"].pop(employee, {"granted": []})
    _save(data)
    return {"employee": employee, "revoked": prior.get("granted", []), "state": "revoked"}


# --------------------------------------------------------------------------- #
# scheduling agent — irreversible write (puts time on other people's calendars)
# --------------------------------------------------------------------------- #

@tool(
    name="schedule_orientation",
    agent="scheduling",
    description=(
        "Book the orientation session for a joiner. Sends calendar invites to "
        "other people, so it requires human approval."
    ),
    sensitivity="write",
    irreversible=True,
    args_schema={
        "employee": "full name of the joiner",
        "start_date": "ISO date the joiner starts, e.g. 2026-10-01",
        "attendees": "list of additional attendees",
    },
    required_args=("employee",),
    preview=lambda a: (
        f"Send calendar invites for {a.get('employee', 'the joiner')}'s orientation"
        + (f" on {a['start_date']}" if a.get("start_date") else "")
    ),
)
def schedule_orientation(employee: str, start_date: str = "", attendees: Any = None, **_: Any) -> dict[str, Any]:
    _maybe_fail("schedule_orientation")
    if isinstance(attendees, str):
        attendees = [a.strip() for a in attendees.split(",") if a.strip()]
    attendees = attendees or ["hr@example.com", "manager@example.com"]
    try:
        when = date.fromisoformat(start_date) if start_date else date.today() + timedelta(days=7)
    except ValueError:
        when = date.today() + timedelta(days=7)
    data = _load()
    meeting = {
        "employee": employee,
        "title": f"Orientation — {employee}",
        "date": when.isoformat(),
        "time": "10:00",
        "attendees": attendees,
    }
    data["meetings"].append(meeting)
    _save(data)
    return meeting


# --------------------------------------------------------------------------- #
# comms agent — irreversible write
# --------------------------------------------------------------------------- #

@tool(
    name="send_welcome_email",
    agent="comms",
    description=(
        "Email the joiner their welcome note and first-day logistics. Cannot be "
        "unsent, so it always requires human approval."
    ),
    sensitivity="write",
    irreversible=True,
    args_schema={
        "employee": "full name of the joiner",
        "to": "recipient email address",
        "document_id": "id of a previously drafted document to attach",
    },
    required_args=("employee",),
    preview=lambda a: f"Email {a.get('to') or a.get('employee', 'the joiner')} the welcome note",
)
def send_welcome_email(employee: str, to: str = "", document_id: str = "", **_: Any) -> dict[str, Any]:
    _maybe_fail("send_welcome_email")
    data = _load()
    msg = {
        "to": to or f"{employee.split()[0].lower()}@example.com",
        "subject": f"Welcome to the team, {employee.split()[0]}",
        "attached": document_id or None,
        "sent_at": datetime.now().isoformat(timespec="seconds"),
    }
    data["messages"].append(msg)
    _save(data)
    return msg


def reset_systems() -> None:
    """Clear the simulated system of record. Used by tests and the demo script."""
    if STORE.exists():
        STORE.unlink()


def systems_snapshot() -> dict[str, Any]:
    return _load()
