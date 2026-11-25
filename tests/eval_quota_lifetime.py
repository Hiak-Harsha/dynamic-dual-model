"""
tests/eval_quota_lifetime.py — Quota Lifetime Experiment.

Simulates or tests how many user turns fit inside a fixed token budget
(e.g. 50,000 billable tokens) comparing:
  1. Always-Full-Brain: Every single turn runs through full tools and reasoning.
  2. Same-Model Dual-Mode: Regex pre-filter + ResponseCache + low-effort Muscle Memory + Full Brain.

Per Step 8d:
Contains 3 separate workload profiles (casual, balanced, adversarial) selectable via --workload flag.
All output is explicitly labeled [SIMULATED].

Usage:
  python tests/eval_quota_lifetime.py                         # default balanced workload, 50,000 budget
  python tests/eval_quota_lifetime.py --workload casual       # casual workload
  python tests/eval_quota_lifetime.py --workload adversarial  # adversarial workload
  python tests/eval_quota_lifetime.py --workload all          # run all 3 workloads
  python tests/eval_quota_lifetime.py --budget 100000         # custom budget
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ProviderResponse, ResponseCache, SessionStats
from dual_mode_agent import full_brain_turn, is_obvious_muscle_memory
from mock_provider import MockProvider
from router import route, RouteDecision
from procedural_memory import ProceduralMemory
from muscle_engine import execute_muscle, MuscleResult
from verification import verify, VerificationResult
from handoff import build_handoff, render_handoff_prompt
from controller import AgentController


# 3 distinct workloads per Step 8d
WORKLOADS = {
    "casual": [
        ("hi", "chat"),
        ("how are you today?", "chat"),
        ("thanks for asking", "chat"),
        ("what is 2 + 2?", "chat"),
        ("tell me a short joke", "chat"),
        ("haha that was funny", "chat"),
        ("thanks", "chat"),
        ("goodbye", "chat"),
    ],
    "balanced": [
        ("hi", "chat"),
        ("can you create math_utils.py with an add function?", "task"),
        ("thanks!", "chat"),
        ("now add a multiply function to math_utils.py", "task"),
        ("ok", "chat"),
        ("read math_utils.py and verify its syntax", "task"),
        ("what is 12 * 15?", "chat"),
        ("thanks", "chat"),  # synonym/cache hit
        ("write tests in test_math.py for both functions", "task"),
        ("got it, thank you", "chat"),
        ("refactor add to support variable number of arguments", "task"),
        ("perfect!", "chat"),
    ],
    "adversarial": [
        ("My favorite programming language is Rust.", "chat"),
        ("What language do I like?", "task"),
        ("What is the latest version of Python released today?", "task"),
        ("Please delete the file temp_data.csv", "task"),
        ("continue", "task"),
        ("yes", "chat"),
        ("Overwrite config.json with empty dictionary", "task"),
        ("thanks", "chat"),
    ],
}


def run_until_budget_exhausted(dual_mode: bool, budget: int, workload_pattern: list):
    import tempfile
    provider = MockProvider(tool_rounds=1)
    state = ConversationState()
    cache = ResponseCache()
    memory = ProceduralMemory(storage_path=tempfile.mktemp(suffix="_skills.json"))
    
    # Classification heuristic for mock
    def smart_mock_classify(msg, summary):
        if any(w in msg.lower() for w in ["math", "file", "function", "read", "write", "check", "script", "refactor", "rust", "delete", "overwrite", "language"]):
            return "full_brain"
        return "muscle_memory"
    provider.classify = smart_mock_classify

    turns_completed = 0
    tasks_completed = 0
    chat_completed = 0
    total_billed = 0
    idx = 0

    ctrl = None
    if dual_mode:
        ctrl = AgentController(provider=provider, state=state, cache=cache, memory=memory)

    while True:
        user_msg, msg_type = workload_pattern[idx % len(workload_pattern)]
        idx += 1

        if not dual_mode:
            # Always Full Brain
            reply, resp = full_brain_turn(provider, state, user_msg, effort="medium")
            cost = resp.input_tokens + resp.output_tokens
            if total_billed + cost > budget:
                break
            total_billed += cost
            turns_completed += 1
            if msg_type == "task":
                tasks_completed += 1
            else:
                chat_completed += 1
        else:
            # Dual-Mode v5
            task = ctrl.start(user_msg)
            ctrl.execute_task(task)
            cost = task.exec_state.usage.total_billable_tokens
            if cost == 0:
                cost = 10  # minimal accounting for zero-token cache turn

            if total_billed + cost > budget:
                break
            total_billed += cost
            turns_completed += 1
            if msg_type == "task":
                tasks_completed += 1
            else:
                chat_completed += 1

    return {
        "turns_completed": turns_completed,
        "tasks_completed": tasks_completed,
        "chat_completed": chat_completed,
        "total_billed": total_billed,
    }


def run_workload_experiment(workload_name: str, workload: list, budget: int):
    print(f"\nQuota Lifetime Experiment: '{workload_name.upper()}' (Budget: {budget:,} tokens) [SIMULATED]")
    print("=" * 76)

    base = run_until_budget_exhausted(dual_mode=False, budget=budget, workload_pattern=workload)
    dual = run_until_budget_exhausted(dual_mode=True, budget=budget, workload_pattern=workload)

    multiplier = dual["turns_completed"] / base["turns_completed"] if base["turns_completed"] else 1.0
    task_multiplier = dual["tasks_completed"] / base["tasks_completed"] if base["tasks_completed"] else 1.0

    print(f"{'Metric':<36} {'Always-Full-Brain':<18} {'Dual-Mode v4 (simulated)':<26}")
    print("-" * 76)
    print(f"{'Total turns completed':<36} {base['turns_completed']:<18} {dual['turns_completed']}")
    print(f"{'Real productive tasks completed':<36} {base['tasks_completed']:<18} {dual['tasks_completed']}")
    print(f"{'Chat turns handled':<36} {base['chat_completed']:<18} {dual['chat_completed']}")
    print(f"{'Tokens consumed':<36} {base['total_billed']:,}{'':<11} {dual['total_billed']:,}")
    print("=" * 76)
    print(f"LIFETIME EXTENSION FACTOR : {multiplier:.2f}x more turns")
    if base["tasks_completed"] > 0:
        print(f"PRODUCTIVE TASK MULTIPLIER: {task_multiplier:.2f}x more tasks completed")
    print("=" * 76 + "\n")


def run_experiment(budget: int, workload_choice: str = "balanced"):
    if workload_choice == "all":
        for name, wl in WORKLOADS.items():
            run_workload_experiment(name, wl, budget)
    else:
        wl = WORKLOADS.get(workload_choice, WORKLOADS["balanced"])
        run_workload_experiment(workload_choice, wl, budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=50000,
                        help="Total billable token budget (default 50,000)")
    parser.add_argument("--workload", choices=["casual", "balanced", "adversarial", "all"], default="balanced",
                        help="Workload distribution to evaluate (default: balanced)")
    args = parser.parse_args()
    run_experiment(args.budget, args.workload)
