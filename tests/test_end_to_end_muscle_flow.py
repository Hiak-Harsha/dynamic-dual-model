"""
tests/test_end_to_end_muscle_flow.py — Comprehensive end-to-end integration tests
validating that ordinary plain-text queries, arithmetic calculations, and greetings
execute through Muscle Memory without being erroneously escalated to Full Brain,
and that tool operations populate tool_state and pass Level-2 filesystem verification.
"""

from __future__ import annotations
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ProviderResponse, ResponseCache, ToolCall
from controller import AgentController
from mock_provider import MockProvider
from tools import _resolve_in_sandbox


def test_plain_text_muscle_delivery_without_escalation():
    """Validates that plain-text Muscle responses ('hi', 'thanks', etc.)
    pass verification and are delivered directly without Full Brain escalation."""
    provider = MockProvider()
    ctrl = AgentController(provider=provider)

    test_queries = [
        "hi",
        "thanks",
        "how are you today?",
        "tell me a short joke",
    ]

    for q in test_queries:
        task = ctrl.start(q)
        ctrl.route_task(task)
        assert task.decision.pathway == "muscle", f"Query '{q}' should route to muscle, got {task.decision.pathway}"
        reply = ctrl.execute_task(task)

        # Invariant 1: Must succeed
        assert task.success is True, f"Task failed for query '{q}'"
        assert task.exec_state.status in {"completed", "completed_verified", "completed_unverified"}, f"Status not completed: {task.exec_state.status}"

        # Invariant 2: Verification must pass via consistency tier
        assert task.verification_result is not None
        assert task.verification_result.passed is True
        assert task.verification_result.tier_used == "consistency"

        # Invariant 3: Must NOT escalate to Full Brain fallback
        assert "[mock final full-brain answer]" not in reply, f"Unnecessary Full Brain escalation for '{q}'"
        assert len(reply) > 0


def test_deterministic_arithmetic_muscle_delivery():
    """Validates that deterministic arithmetic queries execute through Muscle Tier 1
    with 0 LLM tokens, pass verification, and are delivered directly."""
    provider = MockProvider()
    ctrl = AgentController(provider=provider)

    math_tests = [
        ("what is 15 * 24?", "360"),
        ("what is 2 + 2?", "4"),
        ("calculate 100 / 5", "20"),
    ]

    for q, expected in math_tests:
        task = ctrl.start(q)
        ctrl.route_task(task)
        assert task.decision.pathway == "muscle"
        reply = ctrl.execute_task(task)

        assert task.success is True
        assert task.exec_state.status in {"completed", "completed_verified", "completed_unverified"}
        assert expected in reply, "Expected " + str(expected) + " in " + str(reply)
        assert task.exec_state.usage.brain_tokens == 0, "Deterministic math must consume 0 brain tokens"
        assert "[mock final full-brain answer]" not in reply


def test_tool_execution_populates_tool_state_and_verifies_level_2():
    """Validates that write_file inside brain.py populates exec_state.tool_state
    and triggers Level-2 execution verification against actual files on disk."""
    target_file = "e2e_verification_test.txt"
    provider = MockProvider(tool_rounds=1)
    step = 0

    def custom_step(history, tools, effort):
        nonlocal step
        step += 1
        if step == 1:
            return ProviderResponse(
                text="",
                tool_calls=[ToolCall(id="call_write_1", name="write_file", arguments={"path": target_file, "content": "verified content", "overwrite": True})],
                stop_reason="tool_use",
                raw=None,
                input_tokens=500,
                output_tokens=50,
                native_assistant_turn={"role": "assistant", "content": "[writing file]"},
            )
        return ProviderResponse(
            text="File written successfully.",
            tool_calls=[],
            stop_reason="end_turn",
            raw=None,
            input_tokens=600,
            output_tokens=30,
            native_assistant_turn={"role": "assistant", "content": "File written successfully."},
        )

    provider.full_brain_step = custom_step
    ctrl = AgentController(provider=provider)

    task = ctrl.start(f"write_file path='{target_file}' content='verified content' overwrite=True")
    ctrl.route_task(task)
    assert task.decision.pathway == "full_brain"
    reply = ctrl.execute_task(task)

    assert task.success is True
    assert task.exec_state.status in {"completed", "completed_verified", "completed_unverified"}

    # Invariant 1: tool_state must record the written file
    key = f"file_written:{target_file}"
    assert key in task.exec_state.tool_state, f"Expected '{key}' in tool_state, got {task.exec_state.tool_state}"
    assert task.exec_state.tool_state[key] is True

    # Invariant 2: Level 2 execution verification must have run and passed
    assert task.verification_result is not None
    assert task.verification_result.tier_used == "execution"
    assert task.verification_result.passed is True

    # Invariant 3: The file must actually exist on disk in the sandbox
    real_path = _resolve_in_sandbox(target_file)
    assert os.path.exists(real_path)
    with open(real_path, "r", encoding="utf-8") as f:
        assert f.read() == "verified content"

    # Clean up test file
    if os.path.exists(real_path):
        os.remove(real_path)


def test_hedging_in_muscle_triggers_safe_escalation():
    """Validates that if Muscle outputs hedging or uncertain language,
    verification fails and correctly escalates to Full Brain."""
    provider = MockProvider()
    # Inject hedging response for muscle
    provider.muscle_memory_reply = lambda msg: ProviderResponse(
        text="I think the answer might be something like this, but I am not entirely sure.",
        tool_calls=[],
        stop_reason="end_turn",
        raw=None,
        input_tokens=40,
        output_tokens=20,
    )

    ctrl = AgentController(provider=provider)
    task = ctrl.start("hi")
    ctrl.route_task(task)
    assert task.decision.pathway == "muscle"

    reply = ctrl.execute_task(task)

    # Must escalate to Full Brain due to hedging in Muscle
    assert "[mock final full-brain answer]" in reply
    # Initial muscle or continuation step completed
    assert any("Muscle" in str(s) or "step" in str(s).lower() for s in task.exec_state.completed_steps)


def test_tool_requiring_read_escalates_when_unverified():
    """Validates Section 1.6 / Step 2: A tool-requiring read request ('read sample.txt')
    is routed to Muscle via Gate 8 (read-only exploration). Because no trusted skill exists,
    Muscle Tier 3 produces a bounded reply without tool execution. The unverified status
    properly triggers continuation handoff to Brain, which executes read_file."""
    import tempfile
    from procedural_memory import ProceduralMemory
    from tools import execute_tool

    test_file = "sample_test_doc.txt"
    test_path = _resolve_in_sandbox(test_file)
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("content of sample doc")

    try:
        class ToolReadMockProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.brain_called = False

            def muscle_memory_reply(self, user_message: str) -> ProviderResponse:
                # Muscle replies without tool execution
                return ProviderResponse(
                    text="I cannot inspect files in muscle mode directly.",
                    tool_calls=[],
                    stop_reason="end_turn",
                    raw=None,
                    input_tokens=40,
                    output_tokens=15,
                )

            def full_brain_step(self, history, tools, effort):
                self.brain_called = True
                hist_str = " ".join(str(h) for h in history).lower()
                if "sample_test_doc.txt" in hist_str and self._step_count == 0:
                    self._step_count += 1
                    out = execute_tool("read_file", {"path": test_file})
                    return ProviderResponse(
                        text=f"Read file result: {out}",
                        tool_calls=[ToolCall(id="rf1", name="read_file", arguments={"path": test_file})],
                        stop_reason="tool_use",
                        raw=None,
                        input_tokens=400,
                        output_tokens=30,
                    )
                return ProviderResponse(
                    text="Completed reading sample_test_doc.txt successfully.",
                    tool_calls=[],
                    stop_reason="end_turn",
                    raw=None,
                    input_tokens=450,
                    output_tokens=30,
                )

        provider = ToolReadMockProvider()
        isolated_memory = ProceduralMemory(storage_path=tempfile.mktemp(suffix="_skills.json"))
        isolated_memory.skills.clear()
        ctrl = AgentController(provider=provider, memory=isolated_memory)

        task = ctrl.start(f"read {test_file}")
        ctrl.route_task(task)

        # 1. Gate 8 routes read-only file query to muscle with tools_required=True
        assert task.decision.pathway == "muscle"
        assert task.decision.tools_required is True

        reply = ctrl.execute_task(task)

        # 2. Muscle attempt was executed via Tier 3 (no trusted skill)
        assert any("muscle" in str(s).lower() or "step" in str(s).lower() for s in task.exec_state.completed_steps)

        # 3. Escalation to Full Brain continuation occurred because tool was required but unverified in Muscle
        assert provider.brain_called is True
        assert task.exec_state.usage.brain_tokens > 0

        # 4. Final output was produced by Brain with tool execution
        assert "sample_test_doc.txt" in reply
        assert f"file_read:{test_file}" in task.exec_state.tool_state
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)

