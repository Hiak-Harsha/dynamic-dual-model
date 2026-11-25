"""
scripts/live_validation.py — Authoritative Live and Simulated Model Validation CLI.

Validates provider capabilities and system invariants:
- Providers: Anthropic, OpenAI, and Mock
- Tests: basic, deterministic, tool, handoff, restart_resume, or all
- Supports --limit N and clean sandbox isolation

Usage:
  python scripts/live_validation.py --provider mock --test all
  python scripts/live_validation.py --provider anthropic --test tool
  python scripts/live_validation.py --provider openai --test deterministic
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ModelProvider, ResponseCache, SessionStats
from brain import BrainEngine
from controller import AgentController
from execution_state import ExecutionState, COMPLETED_VERIFIED
from handoff import build_handoff
from mock_provider import MockProvider
from procedural_memory import ProceduralMemory
from success_contracts import resolve_contract
from tools import _resolve_in_sandbox, execute_tool


def test_basic(provider: ModelProvider, limit: Optional[int] = None) -> bool:
    print("\n--- Running Test: BASIC (Conversational / Greeting) ---")
    memory = ProceduralMemory(storage_path=tempfile.mktemp(suffix="_skills.json"))
    controller = AgentController(provider=provider, memory=memory)

    turns = [
        "Hello, can you help me today?",
        "Thank you very much!",
    ]
    if limit:
        turns = turns[:limit]

    for turn in turns:
        task = controller.start(turn)
        reply = controller.execute_task(task)
        print(f"User: '{turn}' -> Route: {task.decision.pathway}, Status: {task.exec_state.status}, Success: {task.success}")
        if not task.success:
            print(f"[FAIL] Basic turn '{turn}' failed!")
            return False
    print("[PASS] Basic conversational test passed.")
    return True


def test_deterministic(provider: ModelProvider, limit: Optional[int] = None) -> bool:
    print("\n--- Running Test: DETERMINISTIC (Arithmetic & JSON AST) ---")
    memory = ProceduralMemory(storage_path=tempfile.mktemp(suffix="_skills.json"))
    controller = AgentController(provider=provider, memory=memory)

    tasks = [
        ("what is 45 * 12?", "540"),
        ("calculate 1024 / 16", "64"),
        ('validate json: {"status": "success", "count": 42}', "Valid JSON"),
    ]
    if limit:
        tasks = tasks[:limit]

    for query, expected_snippet in tasks:
        task = controller.start(query)
        reply = controller.execute_task(task)
        print(f"Query: '{query}' -> Pathway: {task.decision.pathway}, Billable Tokens: {task.exec_state.usage.brain_tokens}")
        assert task.decision.pathway == "muscle", f"Expected muscle pathway for '{query}', got {task.decision.pathway}"
        assert task.exec_state.usage.brain_tokens == 0, f"Deterministic procedure must consume 0 brain tokens for '{query}'"
        assert task.success is True, f"Contract evaluation failed for '{query}'"
        assert expected_snippet in reply or "valid" in reply.lower(), f"Expected snippet '{expected_snippet}' in reply"
    print("[PASS] Deterministic test passed (all consumed 0 brain tokens).")
    return True


def test_tool(provider: ModelProvider, limit: Optional[int] = None) -> bool:
    print("\n--- Running Test: TOOL (File Write and Read via Sandbox) ---")
    test_file = "live_validation_sample.txt"
    real_path = _resolve_in_sandbox(test_file)

    if os.path.exists(real_path):
        os.remove(real_path)

    try:
        memory = ProceduralMemory(storage_path=tempfile.mktemp(suffix="_skills.json"))
        controller = AgentController(provider=provider, memory=memory)

        # 1. Write file
        if hasattr(provider, "reset"):
            provider.reset()
        if hasattr(provider, "tool_rounds"):
            provider.tool_rounds = 2
        write_cmd = f"write hello live test world to {test_file}"
        task1 = controller.start(write_cmd)
        reply1 = controller.execute_task(task1)
        print(f"Write task -> Route: {task1.decision.pathway}, Status: {task1.exec_state.status}, Success: {task1.success}")
        assert task1.success is True, "Write task must succeed"
        assert os.path.exists(real_path), f"File {test_file} was not written to disk"

        # 2. Read file
        if hasattr(provider, "reset"):
            provider.reset()
        if hasattr(provider, "tool_rounds"):
            provider.tool_rounds = 2
        read_cmd = f"read {test_file}"
        task2 = controller.start(read_cmd)
        reply2 = controller.execute_task(task2)
        print(f"Read task -> Route: {task2.decision.pathway}, Status: {task2.exec_state.status}, Success: {task2.success}")
        assert task2.success is True, "Read task must succeed"
        print("[PASS] Tool write/read validation passed.")
        return True
    finally:
        if os.path.exists(real_path):
            os.remove(real_path)


def test_handoff(provider: ModelProvider, limit: Optional[int] = None) -> bool:
    print("\n--- Running Test: HANDOFF (Muscle Frontier Continuation) ---")
    task_input = "Inspect config_handoff.json and format full report"
    state = ExecutionState(objective=task_input)
    state.add_completed_step("Read configuration header")
    state.record_tool_action({"name": "read_file", "arguments": {"path": "config_handoff.json"}})
    state.record_tool_result({"name": "read_file", "output": '{"service": "api", "port": 8080}'})
    state.set_frontier(current="header_read", next_action="format_report", remaining=["format_report"])

    checkpoint = copy.deepcopy(state)
    handoff = build_handoff(checkpoint, reason="Escalating for formatting", suggested_next_step="Format report")

    brain = BrainEngine()
    reply, last_resp, cont_state = brain.continue_from_handoff(
        provider, ConversationState(), task_input, handoff=handoff, effort="medium"
    )

    print(f"Continuation -> Completed Steps: {len(cont_state.completed_steps)}, Next Action: {cont_state.next_action}")
    assert "Full Brain resumed execution from Muscle Memory handoff" in cont_state.completed_steps
    print("[PASS] Handoff and continuation frontier preserved.")
    return True


def test_restart_resume(provider: ModelProvider, limit: Optional[int] = None) -> bool:
    print("\n--- Running Test: RESTART VS RESUME (Recomputation Avoidance) ---")
    sample_file = "sample_data.txt"
    real_sample_path = _resolve_in_sandbox(sample_file)
    with open(real_sample_path, "w", encoding="utf-8") as f:
        f.write("cached sample data")

    try:
        prior_action = 'read_file({"path": "sample_data.txt"})'

        # Condition C: Restart
        c_state = ExecutionState(objective="test task")
        c_state.record_tool_action({"name": "read_file", "arguments": {"path": "sample_data.txt"}})
        c_actions = [prior_action]

        # Condition D: Continuation from checkpoint with cached freshness
        d_state = ExecutionState(objective="test task")
        d_state.record_tool_action({"name": "read_file", "arguments": {"path": "sample_data.txt"}})
        d_state.record_tool_result({"name": "read_file", "output": "cached sample data"})
        d_state.record_state_freshness("sample_data.txt", "cached sample data")

        can_reuse, cached_out, rep_type = d_state.check_tool_reuse("read_file", {"path": "sample_data.txt"})
        print(f"Tool check_tool_reuse: can_reuse={can_reuse}, rep_type='{rep_type}'")
        assert can_reuse is True
        assert rep_type == "reuse"
        print("[PASS] Recomputation avoidance confirmed (returns 'reuse', not recomputed).")
        return True
    finally:
        if os.path.exists(real_sample_path):
            os.remove(real_sample_path)


def run_all(provider: ModelProvider, limit: Optional[int] = None) -> bool:
    is_mock = isinstance(provider, MockProvider) or getattr(provider, "name", "") == "mock"
    mode_label = "[SIMULATED / MOCK]" if is_mock else "[LIVE MODEL]"
    print("=" * 75)
    print(f"DYNAMIC DUAL MODEL V8.1 AUTHORITATIVE VALIDATION {mode_label}")
    print(f"Provider: {provider.name} | Model: {getattr(provider, 'model', 'default')}")
    print("=" * 75)

    tests = [
        ("basic", test_basic),
        ("deterministic", test_deterministic),
        ("tool", test_tool),
        ("handoff", test_handoff),
        ("restart_resume", test_restart_resume),
    ]

    results = {}
    all_passed = True
    for name, func in tests:
        if hasattr(provider, "reset"):
            provider.reset()
        if hasattr(provider, "tool_rounds"):
            provider.tool_rounds = 1
        try:
            success = func(provider, limit=limit)
        except Exception as e:
            print(f"\n[FAIL] Test suite '{name}' raised exception: {e}")
            success = False
        results[name] = {"passed": success}
        if not success:
            print(f"\n[FAIL] Test suite '{name}' FAILED!")
            all_passed = False

    # Save validation artifact
    ts = time.strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(res_dir, exist_ok=True)
    prefix = "live_smoke" if not is_mock else "mock_validation"
    out_file = os.path.join(res_dir, f"{prefix}_{provider.name}_{ts}.json")
    payload = {
        "timestamp": ts,
        "is_live": not is_mock,
        "provider": provider.name,
        "model": getattr(provider, "model", "default"),
        "all_passed": all_passed,
        "suite_results": results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved validation results artifact: {out_file}")

    print("\n" + "=" * 75)
    print(f"VALIDATION SUMMARY {mode_label}: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Authoritative Live and Simulated Model Validation CLI")
    parser.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai", "ollama", "gemini", "openrouter"])
    parser.add_argument(
        "--test",
        default="all",
        choices=["all", "basic", "deterministic", "tool", "handoff", "restart_resume"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items per test")
    parser.add_argument("--model", type=str, default=None, help="Explicit model override")
    args = parser.parse_args()

    # Pre-flight check for live providers
    if args.provider == "mock":
        provider = MockProvider(tool_rounds=1)
    else:
        from provider_registry import build_default_registry
        factory = build_default_registry().get_factory(args.provider)
        if factory is None or not factory.is_available():
            status = factory.status_label() if factory else "UNAVAILABLE"
            print(f"\n[PRE-FLIGHT ERROR] Provider '{args.provider}' is not available ({status}).\n")
            sys.exit(1)
        default_override = "claude-3-5-sonnet-20241022" if args.provider == "anthropic" else None
        provider = factory.create(model=args.model or default_override)

    test_map = {
        "basic": test_basic,
        "deterministic": test_deterministic,
        "tool": test_tool,
        "handoff": test_handoff,
        "restart_resume": test_restart_resume,
        "all": run_all,
    }

    runner = test_map[args.test]
    success = runner(provider, limit=args.limit)
    sys.exit(0 if success else 1)
