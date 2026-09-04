"""Planner backends.

Two implementations behind one interface:

* `GeminiPlanner`   — asks Gemini for a JSON plan, validates it against the
                      tool registry, and falls back rather than crashing.
* `RulePlanner`     — deterministic keyword planner. Runs with no API key, and
                      is what the eval suite pins against so CI is stable.

Both return the same shape, so the graph never knows which one it got.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ..config import settings

PLANNER_SYSTEM = """You are the supervisor of an HR workflow automation platform.
Break the user's request into the smallest correct sequence of tool calls.

Rules:
- Only use tools from the provided list. Never invent a tool or an argument name.
- Read-only checks come before write actions that depend on them.
- Do not plan a step whose required arguments you cannot fill from the request.
- Prefer 3-6 steps. Do not pad the plan.

Reply with JSON only, no prose and no code fences:
{"tasks":[{"agent":"...","tool":"...","args":{...},"rationale":"one short sentence"}]}
"""


class PlanResult:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: int = 0,
        fell_back: bool = False,
        note: str = "",
    ) -> None:
        self.tasks = tasks
        self.model = model
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.duration_ms = duration_ms
        self.fell_back = fell_back
        self.note = note


# --------------------------------------------------------------------------- #
# deterministic planner
# --------------------------------------------------------------------------- #

class RulePlanner:
    """Keyword-driven planner. No network, fully reproducible."""

    model = "deterministic"

    def plan(self, request: str, tool_specs: list[dict[str, Any]], context: dict[str, Any]) -> PlanResult:
        started = time.perf_counter()
        text = (request or "").lower()
        employee = context.get("employee") or _guess_employee(request)
        role = context.get("role", "")
        start_date = context.get("start_date", "")
        etype = context.get("employment_type") or _guess_employment_type(text)
        available = {s["name"] for s in tool_specs}
        tasks: list[dict[str, Any]] = []

        def add(tool: str, args: dict[str, Any], rationale: str) -> None:
            if tool in available:
                agent = next(s["agent"] for s in tool_specs if s["name"] == tool)
                tasks.append({"agent": agent, "tool": tool, "args": args, "rationale": rationale})

        offboarding = any(w in text for w in ("offboard", "off-board", "leaving", "resign", "last day", "revoke"))
        onboarding = any(w in text for w in ("onboard", "new hire", "new joiner", "joining", "starts", "start date"))
        policy_only = any(w in text for w in ("policy", "how many days", "what is the", "entitled"))

        if offboarding:
            add("revoke_access", {"employee": employee}, "Departure requires access removal.")
            add(
                "generate_document",
                {"employee": employee, "doc_type": "probation_plan", "role": role},
                "Record the exit paperwork.",
            )
        elif onboarding:
            add(
                "check_compliance",
                {"employee": employee, "employment_type": etype},
                "Confirm mandatory documents before any write action.",
            )
            add(
                "generate_document",
                {"employee": employee, "doc_type": "welcome_packet", "role": role},
                "Draft the welcome packet for review.",
            )
            add(
                "provision_access",
                {"employee": employee, "systems": ["email", "slack", "github"], "role": role},
                "Least-privilege accounts for day one.",
            )
            add(
                "schedule_orientation",
                {"employee": employee, "start_date": start_date},
                "Book orientation with HR and the manager.",
            )
            add(
                "send_welcome_email",
                {"employee": employee},
                "Send first-day logistics once the packet is approved.",
            )
        elif policy_only:
            add("search_policy", {"topic": _guess_topic(text)}, "Answer from the policy corpus.")
        else:
            # Unrecognised request: do the safe read-only thing rather than guess
            # at write actions.
            add("search_policy", {"topic": _guess_topic(text)}, "Request unclear; start with a read-only lookup.")

        out = PlanResult(
            tasks=tasks,
            model=self.model,
            tokens_in=len(request) // 4,
            tokens_out=len(json.dumps(tasks)) // 4,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return out


def _guess_employee(request: str) -> str:
    """Pull a capitalised name out of the request, skipping the leading word."""
    words = re.findall(r"\b[A-Z][a-z]{1,20}\b", request or "")
    stop = {"Onboard", "Offboard", "Please", "Can", "New", "Hire", "Monday", "Tuesday",
            "Wednesday", "Thursday", "Friday", "January", "February", "March", "April",
            "May", "June", "July", "August", "September", "October", "November", "December"}
    names = [w for w in words if w not in stop]
    if not names:
        return "the joiner"
    return " ".join(names[:2]) if len(names) >= 2 else names[0]


def _guess_employment_type(text: str) -> str:
    if "contractor" in text or "contract" in text:
        return "contractor"
    if "intern" in text:
        return "intern"
    return "full_time"


def _guess_topic(text: str) -> str:
    for topic in ("leave", "probation", "equipment", "background_check", "access"):
        if topic.replace("_", " ") in text or topic in text:
            return topic
    if "vacation" in text or "holiday" in text:
        return "leave"
    if "laptop" in text or "hardware" in text:
        return "equipment"
    return "probation"


# --------------------------------------------------------------------------- #
# Gemini planner
# --------------------------------------------------------------------------- #

class GeminiPlanner:
    """Calls Gemini's REST endpoint directly to avoid a heavy SDK dependency."""

    def __init__(self, api_key: str, model: str, timeout_s: float = 30.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self._fallback = RulePlanner()

    def plan(self, request: str, tool_specs: list[dict[str, Any]], context: dict[str, Any]) -> PlanResult:
        prompt = (
            f"{PLANNER_SYSTEM}\n\nAvailable tools:\n{json.dumps(tool_specs, indent=2)}\n\n"
            f"Known context: {json.dumps(context)}\n\nRequest: {request}\n"
        )
        started = time.perf_counter()
        try:
            raw, tin, tout = self._call(prompt)
        except Exception as exc:
            res = self._fallback.plan(request, tool_specs, context)
            res.fell_back = True
            res.note = f"Gemini unavailable ({type(exc).__name__}); used the rule planner."
            return res

        duration_ms = int((time.perf_counter() - started) * 1000)
        tasks = _parse_tasks(raw, tool_specs)
        if not tasks:
            res = self._fallback.plan(request, tool_specs, context)
            res.fell_back = True
            res.note = "Gemini returned no usable plan; used the rule planner."
            return res
        return PlanResult(tasks, self.model, tin, tout, duration_ms)

    def _call(self, prompt: str) -> tuple[str, int, int]:
        import urllib.request

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        body = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
            }
        ).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode())
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        usage = payload.get("usageMetadata", {})
        return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)


def _parse_tasks(raw: str, tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate model output against the registry. Unknown tools are dropped,
    not executed — a hallucinated tool name must never reach the executor."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    items = parsed.get("tasks") if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        return []

    by_name = {s["name"]: s for s in tool_specs}
    tasks: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("tool")
        spec = by_name.get(name)
        if spec is None:
            continue
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        allowed = set(spec.get("args", {}))
        args = {k: v for k, v in args.items() if not allowed or k in allowed}
        tasks.append(
            {
                "agent": spec["agent"],
                "tool": name,
                "args": args,
                "rationale": str(item.get("rationale", ""))[:200],
            }
        )
    return tasks


def get_planner(force: str | None = None):
    backend = force or settings.planner_backend
    if backend == "gemini" and settings.gemini_api_key:
        return GeminiPlanner(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_s)
    return RulePlanner()
