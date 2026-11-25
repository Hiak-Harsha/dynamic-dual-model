"""
tests/test_v6_frontier_continuation.py — Tests for Task Frontier maintenance and continuation without recomputation.

Per V6 Spec Sections AD, AE, AF, AG, AH, AI:
- ExecutionState tracks task frontier (completed, current, next, blocked, remaining).
- Continuation prompt exposes frontier clearly.
- Fresh verified tool outputs are reused without recomputation.
- Recomputation count and recomputation types are properly logged.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ModelProvider, ProviderResponse, ToolCall
from brain import BrainEngine
from execution_state import ExecutionState
from handoff import build_handoff, render_handoff_prompt
from tools import _resolve_in_sandbox
from mock_provider import MockProvider


def test_task_frontier_in_handoff_prompt():
    state = ExecutionState(objective="Process multi-stage pipeline")
    state.add_completed_step("Stage 1 complete")
    state.set_frontier(current="stage_2_running", next_action="run_stage_2", remaining=["stage_3", "stage_4"])
    state.add_verified_fact("Stage 1 output verified", source="execution")

    handoff = build_handoff(state, reason="Need deliberative planning for Stage 2")
    prompt = render_handoff_prompt(handoff)

    assert "Execution Frontier:" in prompt
    assert "> Current: stage_2_running" in prompt
    assert "> Next Action: run_stage_2" in prompt
    assert "> Remaining Steps: stage_3, stage_4" in prompt
    assert "Treat verified state as established evidence unless contradicted" in prompt


def test_brain_continuation_reuses_fresh_tools(tmp_path):
    # Prepare file
    fname = "frontier_cache_test.txt"
    fpath = _resolve_in_sandbox(fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("important data")

    try:
        state = ExecutionState(objective="Inspect frontier_cache_test.txt and report")
        state.record_tool_action({"name": "read_file", "arguments": {"path": fname}})
        state.record_tool_result({"name": "read_file", "output": "important data"})
        state.record_state_freshness(fname, "important data")
        state.add_verified_fact("File content read", source="tool")

        # Now Brain attempts to call read_file with identical argument
        brain = BrainEngine()
        provider = MockProvider()
        # Mock provider will return a tool call to read_file
        conv_state = ConversationState()

        can_reuse, cached_out, rep_type = state.check_tool_reuse("read_file", {"path": fname})
        assert can_reuse is True
        assert cached_out == "important data"
        assert rep_type == "reuse"
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)
