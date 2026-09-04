"""Reviewer console.

    streamlit run src/ui/app.py

The screen is organised around the one moment that matters: a run stopped at a
gate, waiting for a person to decide. Everything else on the page is quiet so
that decision is unambiguous — what is about to happen, why it needs a human,
and what has already been done.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import settings
from src.graph.build import WorkflowRunner
from src.tools.catalog import reset_systems, systems_snapshot
from src.tools.registry import registry

st.set_page_config(page_title="Workflow approvals", page_icon="◱", layout="wide")

INK = "#14181f"
MUTED = "#5c6673"
LINE = "#dfe3e8"
PENDING = "#b4690e"
DONE = "#2f6f4f"
STOPPED = "#9c2f2f"

STYLE = f"""
<style>
  .stApp {{ background: #fbfbfa; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
  .gate {{
    border: 1px solid {PENDING}; border-left: 4px solid {PENDING};
    background: #fdf8f0; padding: 1.1rem 1.25rem; margin: 0.5rem 0 1rem;
  }}
  .gate .what {{ font-size: 1.15rem; color: {INK}; font-weight: 600; margin-bottom: .35rem; }}
  .gate .why  {{ font-size: .88rem; color: {MUTED}; }}
  .step {{
    display: flex; gap: .75rem; align-items: baseline;
    padding: .45rem 0; border-bottom: 1px solid {LINE}; font-size: .92rem;
  }}
  .step .tool {{ font-weight: 600; color: {INK}; min-width: 13rem; }}
  .step .meta {{ color: {MUTED}; font-size: .82rem; }}
  .bar {{ height: 9px; background: {INK}; opacity: .75; border-radius: 1px; }}
  .bar.err {{ background: {STOPPED}; }}
  .bar.rej {{ background: {PENDING}; }}
  .tag {{ font-size: .75rem; color: {MUTED}; border: 1px solid {LINE}; padding: .1rem .4rem; }}
  code {{ font-size: .85rem; }}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource
def get_runner() -> WorkflowRunner:
    return WorkflowRunner()


runner = get_runner()

if "run_id" not in st.session_state:
    st.session_state.run_id = None


# --------------------------------------------------------------------------- #
# sidebar: start a run, or open an existing one
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.subheader("Start a workflow")
    request = st.text_area(
        "What needs doing",
        value="Onboard Priya Sharma as a full-time backend engineer starting 2026-10-01.",
        height=90,
    )
    employee = st.text_input("Employee", value="Priya Sharma")
    col_a, col_b = st.columns(2)
    role = col_a.text_input("Role", value="Backend Engineer")
    start_date = col_b.text_input("Start date", value="2026-10-01")
    employment_type = st.selectbox("Employment type", ["full_time", "contractor", "intern"])

    if st.button("Run workflow", type="primary", use_container_width=True):
        view = runner.start(
            request,
            {
                "employee": employee,
                "role": role,
                "start_date": start_date,
                "employment_type": employment_type,
            },
        )
        st.session_state.run_id = view.run_id
        st.rerun()

    st.divider()
    st.caption(f"Planner: {settings.planner_backend}  ·  {len(registry.names())} tools registered")

    runs = runner.list_runs(limit=25)
    if runs:
        st.subheader("Recent runs")
        labels = {r["run_id"]: f"{r['status']}  ·  {r['request'][:38]}" for r in runs}
        chosen = st.radio(
            "Open a run",
            options=list(labels),
            format_func=lambda k: labels[k],
            index=0,
            label_visibility="collapsed",
        )
        if st.button("Open", use_container_width=True):
            st.session_state.run_id = chosen
            st.rerun()

    st.divider()
    if st.button("Clear simulated systems", use_container_width=True):
        reset_systems()
        st.rerun()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

st.title("Workflow approvals")

if not st.session_state.run_id:
    st.write(
        "Start a workflow from the sidebar. The supervisor plans it, specialist "
        "agents carry it out, and anything irreversible stops here for a decision."
    )
    with st.expander("Which actions need approval"):
        from src.graph.guardrails import approval_decision

        for name in registry.names():
            tool = registry.get(name)
            decision = approval_decision(tool)
            mark = "needs approval" if decision.required else "runs freely"
            st.markdown(
                f"<div class='step'><span class='tool'>{name}</span>"
                f"<span class='meta'>{tool.agent} · {mark} · {decision.reason}</span></div>",
                unsafe_allow_html=True,
            )
    st.stop()

view = runner.view(st.session_state.run_id)

head = st.columns([3, 1, 1, 1])
head[0].markdown(f"**{view.request or '—'}**")
head[1].metric("Status", view.status.replace("_", " "))
head[2].metric("Steps done", f"{len(view.results)}/{len(view.plan)}")
head[3].metric("Cost", f"${view.budget.get('usd_used', 0):.4f}")

if view.planner_note:
    st.info(view.planner_note)

# --- the gate ------------------------------------------------------------- #

if view.pending_approval:
    req = view.pending_approval
    st.markdown(
        f"""<div class="gate">
              <div class="what">{req.get('preview')}</div>
              <div class="why">
                {req.get('agent')} agent · {req.get('tool')} ·
                {'irreversible' if req.get('irreversible') else 'reversible'} ·
                {req.get('reason')}
              </div>
            </div>""",
        unsafe_allow_html=True,
    )
    with st.expander("Arguments this call will use"):
        st.json(req.get("args", {}))

    approver = st.text_input("Your name", value="reviewer", key="approver")
    note = st.text_input("Note (optional)", key="note", placeholder="Why you approved or refused")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Approve and continue", type="primary", use_container_width=True):
        runner.resume(view.run_id, True, approver, note)
        st.rerun()
    if c2.button("Refuse this step", use_container_width=True):
        runner.resume(view.run_id, False, approver, note or "refused by reviewer")
        st.rerun()

elif view.summary:
    tone = {"completed": DONE, "halted": PENDING, "failed": STOPPED}.get(view.status, MUTED)
    st.markdown(
        f"<div style='border-left:4px solid {tone};padding:.6rem 1rem;background:#fff;"
        f"border:1px solid {LINE};'>{view.summary}</div>",
        unsafe_allow_html=True,
    )

# --- plan and results ----------------------------------------------------- #

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Plan")
    done = {r["task_id"]: r for r in view.results}
    for task in view.plan:
        r = done.get(task["id"])
        state = r["status"] if r else ("in review" if view.pending_approval and
                                       view.pending_approval.get("task_id") == task["id"] else "queued")
        extra = ""
        if r and r.get("approved_by"):
            extra = f" · approved by {r['approved_by']}"
        if r and r.get("error"):
            extra = f" · {r['error'][:70]}"
        st.markdown(
            f"<div class='step'><span class='tool'>{task['tool']}</span>"
            f"<span class='meta'>{task['agent']} · {state}{extra}<br>{task.get('rationale','')}</span></div>",
            unsafe_allow_html=True,
        )

with right:
    st.subheader("Trace")
    tool_spans = [s for s in view.trace if s.get("kind") == "tool"]
    if not tool_spans:
        st.caption("No tool calls yet.")
    else:
        widest = max(max(s.get("duration_ms", 0) for s in tool_spans), 1)
        for s in tool_spans:
            ms = s.get("duration_ms", 0)
            pct = max(2, int(100 * ms / widest))
            cls = "err" if s.get("status") == "error" else ("rej" if s.get("status") == "rejected" else "")
            retries = (s.get("attempts") or 1) - 1
            st.markdown(
                f"<div style='margin:.35rem 0'>"
                f"<div class='meta'>{s['name']} · {ms}ms"
                f"{f' · {retries} retry' if retries else ''}"
                f"{' · ' + s['status'] if s.get('status') != 'ok' else ''}</div>"
                f"<div class='bar {cls}' style='width:{pct}%'></div></div>",
                unsafe_allow_html=True,
            )

    st.subheader("Systems of record")
    snap: dict[str, Any] = systems_snapshot()
    st.markdown(
        f"<span class='tag'>{len(snap['accounts'])} accounts</span> "
        f"<span class='tag'>{len(snap['meetings'])} meetings</span> "
        f"<span class='tag'>{len(snap['messages'])} emails sent</span>",
        unsafe_allow_html=True,
    )
    with st.expander("Inspect"):
        st.json(snap)

with st.expander("Audit trail — every checkpoint for this run"):
    st.dataframe(runner.history(view.run_id), use_container_width=True, hide_index=True)
