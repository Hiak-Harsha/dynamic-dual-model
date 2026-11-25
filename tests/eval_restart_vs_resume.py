"""
tests/eval_restart_vs_resume.py — Simulated test of Rule R3: Continuation vs. Restart.

HONESTY NOTE (read before trusting any number this script prints):
The previous version of this script hand-picked DIFFERENT input_token values
for the "restart" and "resume" branches (1200/1400/1600/1800 vs 650/700) with
no shared mechanism connecting them — the mock was simply authored to return
smaller numbers in the branch we wanted to look better, then presented as
"empirical" evidence. That's circular: of course a scenario returns whatever
its author hardcoded it to return. This version derives every branch's token
cost from ONE shared formula driven only by how much text is actually in
`history` at call time (see `_cost_from_history` below), so if resume looks
cheaper here, it's because the handoff preamble is measurably shorter than
the full re-discovery transcript it replaces — not because a human decided
the punchline in advance.

This is STILL a simulation, not evidence about how a real model behaves —
it only proves the mechanism is internally consistent. Real evidence requires
running both conditions against a live API on real tasks (see
IMPLEMENTATION_PLAN.md Phase 4) and is explicitly not provided by this file.

Core Question: Does handing off a compact, structured ExecutionState from Muscle
Memory save tokens and tool steps compared to restarting the task from scratch?

Usage:
  python tests/eval_restart_vs_resume.py
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ModelProvider, ProviderResponse, SessionStats, ToolCall
from execution_state import ExecutionState
from handoff import build_handoff, render_handoff_prompt
from tools import execute_tool, TOOLS, _resolve_in_sandbox
from dual_mode_agent import full_brain_turn
from mock_provider import MockProvider


def _cost_from_history(history) -> int:
    """The ONE formula both conditions share: ~4 chars/token over the
    string form of everything currently in history, plus a fixed per-call
    overhead for the model's own scratch-reasoning about what to do next.
    Nothing branch-specific is hardcoded — whichever condition's history is
    shorter at call time gets a proportionally lower number, automatically."""
    text = "".join(str(h) for h in history)
    return 300 + max(1, len(text) // 4)  # 300 = fixed per-call reasoning overhead


# Mock whose cost is DERIVED from history length via the shared formula
# above, not hand-picked per branch — see HONESTY NOTE at the top of this file.
class ContinuationMockProvider(MockProvider):
    def __init__(self, exploration_steps: int = 3):
        super().__init__(tool_rounds=exploration_steps)
        self.turns_executed = 0
        self.exploration_steps = exploration_steps

    def full_brain_step(self, history, tools, effort: str) -> ProviderResponse:
        self.turns_executed += 1
        input_tokens = _cost_from_history(history)

        if self.turns_executed < self.exploration_steps:
            return ProviderResponse(
                text=f"Working, step {self.turns_executed}...",
                tool_calls=[ToolCall(id=f"call_{self.turns_executed}", name="read_file", arguments={"path": "dummy.txt"})],
                stop_reason="tool_use",
                raw=None,
                input_tokens=input_tokens,
                output_tokens=40,
                native_assistant_turn={"role": "assistant", "content": f"[step {self.turns_executed} tool_use, history was {len(str(history))} chars]"}
            )
        return ProviderResponse(
            text="Task completed.",
            tool_calls=[],
            stop_reason="end_turn",
            raw=None,
            input_tokens=input_tokens,
            output_tokens=80,
            native_assistant_turn={"role": "assistant", "content": "[final answer]"}
        )


def run_experiment():
    print("\n" + "=" * 74)
    print("EXPERIMENT: RESTART VS. RESUME (Hypothesis: Rule R3)")
    print("=" * 74)

    task_prompt = "Audit repository dependencies, verify security vulnerabilities, and patch requirements.txt"

    # EXPLICIT MODELING ASSUMPTION (this is what makes the result a simulation,
    # not evidence): we assume that a model handed a compact summary of facts
    # already established needs fewer additional tool round-trips to finish
    # than a model discovering those same facts itself. That assumption is
    # plausible but UNTESTED here — it's exactly what a live-API version of
    # this experiment would need to confirm or refute on real tasks.
    RESTART_EXPLORATION_STEPS = 3   # must discover everything itself
    RESUME_EXPLORATION_STEPS = 1    # facts already given via handoff preamble

    # 1. Condition A: Restart from scratch — cost derived from actual
    # accumulated history length via _cost_from_history, not hardcoded.
    p_restart = ContinuationMockProvider(exploration_steps=RESTART_EXPLORATION_STEPS)
    state_restart = ConversationState()
    reply_a, resp_a = full_brain_turn(p_restart, state_restart, task_prompt, effort="medium")
    steps_a = p_restart.turns_executed
    total_tokens_a = sum(
        _cost_from_history(state_restart.full_brain_history[:i]) + (40 if i < steps_a else 80)
        for i in range(1, steps_a + 1)
    )

    # 2. Condition B: Muscle Memory establishes partial facts -> Handoff -> Resume.
    # Same cost formula, applied to the shorter history a compact handoff produces.
    exec_state = ExecutionState(objective=task_prompt)
    exec_state.add_observation("requirements.txt exists with 14 pinned packages")
    exec_state.add_observation("pip-audit scanned 12 packages cleanly; 2 flags found")
    exec_state.add_completed_step("Located and inspected requirements.txt")
    exec_state.add_completed_step("Executed initial vulnerability scan")
    exec_state.mark_unresolved("Apply version bumps to requests and cryptography")
    exec_state.confidence = 0.65

    handoff = build_handoff(
        exec_state,
        reason="Muscle memory confirmed vulnerability findings but requires tool writes to patch files.",
        suggested_next_step="Patch requirements.txt for requests and cryptography and verify.",
    )
    preamble = render_handoff_prompt(handoff)
    handoff_tokens = len(preamble) // 4

    p_resume = ContinuationMockProvider(exploration_steps=RESUME_EXPLORATION_STEPS)
    state_resume = ConversationState()
    reply_b, resp_b = full_brain_turn(
        p_resume, state_resume, task_prompt, effort="medium", handoff_preamble=preamble
    )
    steps_b = p_resume.turns_executed
    total_tokens_b = sum(
        _cost_from_history(state_resume.full_brain_history[:i]) + (40 if i < steps_b else 80)
        for i in range(1, steps_b + 1)
    )

    token_savings = (total_tokens_a - total_tokens_b) / total_tokens_a if total_tokens_a else 0.0

    print(f"{'Condition':<34} {'Tool Steps':<14} {'Total Tokens (simulated)':<26}")
    print("-" * 74)
    print(f"{'Condition A (Restart from Scratch)':<34} {steps_a:<14} {total_tokens_a:<26}")
    print(f"{'Condition B (Resume from Handoff)':<34} {steps_b:<14} {total_tokens_b:<26}")
    print("-" * 74)
    print(f"Tool Steps Avoided : {steps_a - steps_b} steps ({((steps_a - steps_b)/steps_a):.0%})")
    print(f"Token Savings      : {token_savings:.1%}  (handoff preamble itself: ~{handoff_tokens} tokens)")
    print("=" * 74)
    print("[SIMULATED] This demonstrates the mechanism is internally consistent GIVEN")
    print("the modeling assumption above (fewer required exploration steps when handed")
    print("established facts). It does NOT demonstrate that a real model actually needs")
    print("fewer steps in practice — that requires a live-API run on real tasks.\n")


if __name__ == "__main__":
    run_experiment()
