"""
handoff.py — Structured continuation handoff generator without restarting cognition.

Per V6 Specification Sections AF, AG, AH, AI:
- Continuation, not restart: Transfers ExecutionState -> HandoffState -> Full Brain continuation.
- Priorities:
  1. User Goal / Objective (represented once)
  2. Verification Failure Reason (if any)
  3. Established Verified Facts & Observations (evidence, not unquestionable truth)
  4. Task Frontier (completed, current, next, blocked, remaining)
  5. Files touched / tool results
  6. Unresolved items / Errors
  7. Continuation Directive: "Treat verified state as established evidence unless contradicted, stale, or insufficiently verified for the current task."
- Hard budget cap: < 400 tokens
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from base import count_tokens
from execution_state import ExecutionState, HandoffState


def build_handoff(
    state: ExecutionState,
    reason: str = "Muscle memory encountered complexity or uncertain result",
    suggested_next_step: str = "",
    evidence: str = "",
    procedure_info: Optional[str] = None,
) -> HandoffState:
    """Builds a HandoffState from an ExecutionState."""
    return HandoffState(
        execution_state=state,
        handoff_reason=reason,
        evidence=evidence,
        suggested_next_step=suggested_next_step,
        procedure_info=procedure_info,
    )


def format_prioritized_handoff(handoff: HandoffState, max_tokens: int = 400) -> str:
    """Renders a structured, prioritized continuation handoff block strictly budgeted
    under max_tokens (default: 400 tokens).
    
    Priority ordering:
    1. User Goal / Objective
    2. Verification Failure Reason (if present)
    3. Established Verified Facts & Observations
    4. Task Frontier
    5. Completed Steps
    6. Files Touched & Tool Results
    7. Unresolved Items & Errors
    8. Continuation Directive
    """
    state = handoff.execution_state

    # 1. User Goal / Objective
    header_lines = [
        "[CONTINUATION HANDOFF FROM MUSCLE MEMORY]",
        f"Objective: {state.objective or '(unspecified)'}",
    ]

    # 2. Verification Failure Reason
    if handoff.handoff_reason:
        header_lines.append(f"Handoff / Verification Failure Reason: {handoff.handoff_reason}")

    # 8. Continuation Directive per V6 Section AG
    footer_directive = (
        "[CONTINUATION DIRECTIVE:\n"
        "1. Continue from established execution state and Task Frontier.\n"
        "2. Treat verified state as established evidence unless contradicted, stale, or insufficiently verified for the current task.\n"
        "3. Do not repeat verified completed work or re-execute successful tools.\n"
        "4. Re-evaluate only unverified hypotheses or contradicted items.\n"
        "5. Resolve the remaining unresolved portion of the task.]"
    )

    # 3. Established Verified Facts & Observations
    facts_lines: List[str] = []
    if state.verified_facts:
        facts_lines.append("Verified Facts (ESTABLISHED - DO NOT RE-EVALUATE):")
        for vf in state.verified_facts[-6:]:
            facts_lines.append(f"  [VERIFIED] {vf}")

    unverified_obs = [o for o in state.observations if o not in state.verified_facts]
    if unverified_obs:
        facts_lines.append("Observations (Raw Telemetry):")
        for obs in unverified_obs[-4:]:
            facts_lines.append(f"  - {obs}")

    # 4. Task Frontier per V6 Section AE
    frontier_lines: List[str] = []
    if state.current_step or state.next_action or state.blocked_on:
        frontier_lines.append("Execution Frontier:")
        if state.current_step:
            frontier_lines.append(f"  > Current: {state.current_step}")
        if state.next_action:
            frontier_lines.append(f"  > Next Action: {state.next_action}")
        if state.blocked_on:
            frontier_lines.append(f"  > Blocked On: {state.blocked_on}")
        if state.remaining_steps:
            frontier_lines.append(f"  > Remaining Steps: {', '.join(state.remaining_steps[:3])}")

    # 5. Completed Steps
    steps_lines: List[str] = []
    if state.completed_steps:
        steps_lines.append("Completed Steps:")
        for step in state.completed_steps[-6:]:
            steps_lines.append(f"  - {step}")

    # 6. Files Touched & Tool Results
    tool_lines: List[str] = []
    if state.tool_state:
        tool_lines.append("Files Touched & Tool Results:")
        for k, v in list(state.tool_state.items())[-6:]:
            tool_lines.append(f"  - {k}: {v}")

    # 7. Unresolved Items & Errors
    unresolved_lines: List[str] = []
    if state.unresolved:
        unresolved_lines.append("Unresolved Items (ACTION REQUIRED):")
        for item in state.unresolved[-4:]:
            unresolved_lines.append(f"  * {item}")
    if state.errors:
        unresolved_lines.append("Errors Encountered:")
        for err in state.errors[-3:]:
            unresolved_lines.append(f"  ! {err}")

    if handoff.suggested_next_step:
        unresolved_lines.append(f"Suggested Next Action: {handoff.suggested_next_step}")

    # Assemble and enforce token budget
    fixed_text = "\n".join(header_lines) + "\n\n" + footer_directive
    budget_remaining = max_tokens - count_tokens(fixed_text) - 10

    body_sections = []
    for section_lines in [facts_lines, frontier_lines, steps_lines, tool_lines, unresolved_lines]:
        if not section_lines:
            continue
        sec_text = "\n".join(section_lines)
        sec_tokens = count_tokens(sec_text)
        if sec_tokens <= budget_remaining:
            body_sections.append(sec_text)
            budget_remaining -= sec_tokens
        else:
            truncated_lines = []
            for line in section_lines:
                line_tokens = count_tokens(line) + 1
                if line_tokens <= budget_remaining:
                    truncated_lines.append(line)
                    budget_remaining -= line_tokens
                else:
                    break
            if truncated_lines:
                body_sections.append("\n".join(truncated_lines))

    ordered_middle: List[str] = []
    for s in body_sections:
        if "Verified Facts" in s or "Observations" in s:
            if s not in ordered_middle:
                ordered_middle.append(s)
    for s in body_sections:
        if "Execution Frontier:" in s:
            if s not in ordered_middle:
                ordered_middle.append(s)
    for s in body_sections:
        if "Completed Steps:" in s:
            if s not in ordered_middle:
                ordered_middle.append(s)
    for s in body_sections:
        if "Files Touched" in s:
            if s not in ordered_middle:
                ordered_middle.append(s)
    for s in body_sections:
        if "Unresolved Items" in s or "Errors Encountered" in s or "Suggested Next Action" in s:
            if s not in ordered_middle:
                ordered_middle.append(s)
    for s in body_sections:
        if s not in ordered_middle:
            ordered_middle.append(s)

    parts = ["\n".join(header_lines)]
    if ordered_middle:
        parts.append("\n\n".join(ordered_middle))
    parts.append(footer_directive)

    rendered = "\n\n".join(parts)
    while count_tokens(rendered) > max_tokens and len(parts) > 2 and parts[1]:
        lines = parts[1].split("\n")
        if len(lines) > 1:
            parts[1] = "\n".join(lines[:-1])
        else:
            parts.pop(1)
        rendered = "\n\n".join(parts)

    return rendered


def render_handoff_prompt(handoff: HandoffState, max_tokens: int = 400) -> str:
    """Renders the handoff into a structured, budget-aware context preamble (< 400 tokens)
    injected at the beginning of the Full Brain continuation turn."""
    return format_prioritized_handoff(handoff, max_tokens=max_tokens)
