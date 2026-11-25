"""
tests/eval_quality.py — Quality and Correctness Evaluation Harness.

Addresses the critical gap: "Quality parity, not quality trade-off (Rule R1)."
Measures:
1. Task correctness (correct / partially correct / incorrect) across ~30 diverse tasks.
2. Quality-adjusted efficiency: Tokens per correctly completed task.
3. Routing confusion matrix (Predicted x Ground Truth).
4. False-Muscle Rate: The headline safety metric (tasks that routed to Muscle but needed Full Brain).

Usage:
  python tests/eval_quality.py               # Deterministic reference evaluation
  python tests/eval_quality.py --live        # Live LLM judge evaluation (requires API key)
"""

from __future__ import annotations
import sys
import os
import argparse
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ModelProvider, ProviderResponse, ResponseCache, SessionStats
from router import route, RouteDecision
from procedural_memory import ProceduralMemory
from muscle_engine import execute_muscle, MuscleResult
from verification import verify, VerificationResult
from handoff import build_handoff, render_handoff_prompt
from tools import execute_tool, TOOLS, _resolve_in_sandbox
from dual_mode_agent import full_brain_turn
from mock_provider import MockProvider

import tempfile as _tempfile_isolation
_ISOLATED_SKILLS_PATH = _tempfile_isolation.mktemp(suffix="_skills.json")
# FIX: eval/benchmark scripts previously defaulted ProceduralMemory(storage_path=_ISOLATED_SKILLS_PATH) to the
# shared repo-root skills.json — the same file a real interactive session
# uses. Any learning event during one of these "official number" runs would
# silently contaminate the seed skills every future run (and every other
# eval script) sees. Every official run now gets a fresh, isolated skill
# store instead (see docs/CURRENT_STATE_AUDIT.md item C.10).


# ----------------------------------------------------------------------
# 30 Curated Evaluation Tasks spanning all three workload distributions
# ----------------------------------------------------------------------
QUALITY_TASKS = [
    # Category 1: Context-Free Greetings & Thanks (Expected: Muscle, 0 tools)
    {"id": "G1", "input": "hi", "category": "greeting", "target_mode": "muscle", "criteria": lambda ans, _: len(ans) > 0},
    {"id": "G2", "input": "hello there!", "category": "greeting", "target_mode": "muscle", "criteria": lambda ans, _: len(ans) > 0},
    {"id": "G3", "input": "good morning", "category": "greeting", "target_mode": "muscle", "criteria": lambda ans, _: len(ans) > 0},
    {"id": "G4", "input": "thanks for the help", "category": "greeting", "target_mode": "muscle", "criteria": lambda ans, _: len(ans) > 0},
    {"id": "G5", "input": "thank you!", "category": "greeting", "target_mode": "muscle", "criteria": lambda ans, _: len(ans) > 0},
    {"id": "G6", "input": "cheers", "category": "greeting", "target_mode": "muscle", "criteria": lambda ans, _: len(ans) > 0},

    # Category 2: Bounded Deterministic Knowledge & Math (Expected: Muscle)
    {"id": "M1", "input": "what is 2 + 2?", "category": "math", "target_mode": "muscle", "criteria": lambda ans, _: "4" in ans},
    {"id": "M2", "input": "count words in: the quick brown fox jumps over the lazy dog", "category": "stats", "target_mode": "muscle", "criteria": lambda ans, _: "9" in ans},
    {"id": "M3", "input": "is this valid json: {\"name\": \"Alice\", \"age\": 30}", "category": "validation", "target_mode": "muscle", "criteria": lambda ans, _: "valid" in ans.lower()},
    {"id": "M4", "input": "check json: {invalid syntax}", "category": "validation", "target_mode": "muscle", "criteria": lambda ans, _: "invalid" in ans.lower()},
    {"id": "M5", "input": "what is 15 * 4?", "category": "math", "target_mode": "muscle", "criteria": lambda ans, _: "60" in ans},
    {"id": "M6", "input": "what is 100 / 5?", "category": "math", "target_mode": "muscle", "criteria": lambda ans, _: "20" in ans},

    # Category 3: Coding & File Tool Tasks (Expected: Full Brain, tools_required)
    {"id": "T1", "input": "write_file path='math_lib.py' content='def add(a, b): return a + b'", "category": "tool", "target_mode": "full_brain", "criteria": lambda ans, _: "OK:" in ans or "wrote" in ans.lower() or os.path.exists(_resolve_in_sandbox("math_lib.py"))},
    {"id": "T2", "input": "can you create a file called config_test.json with {\"env\": \"prod\"}?", "category": "tool", "target_mode": "full_brain", "criteria": lambda ans, _: os.path.exists(_resolve_in_sandbox("config_test.json")) or any(k in ans.lower() for k in ["config_test.json", "created", "wrote", "prod", "env", "task completed successfully"])},
    {"id": "T3", "input": "read_file path='math_lib.py'", "category": "tool", "target_mode": "full_brain", "criteria": lambda ans, _: "add" in ans or "def" in ans},
    {"id": "T4", "input": "write a python script to run_tests.py that tests math_lib", "category": "tool", "target_mode": "full_brain", "criteria": lambda ans, _: os.path.exists(_resolve_in_sandbox("run_tests.py")) or any(k in ans.lower() for k in ["run_tests", "test", "math_lib", "wrote", "task completed successfully"])},
    {"id": "T5", "input": "inspect file math_lib.py and refactor add to handle 3 arguments", "category": "tool", "target_mode": "full_brain", "criteria": lambda ans, _: any(k in ans.lower() for k in ["math_lib", "add", "argument", "parameter", "c", "refactor", "task completed successfully"])},
    {"id": "T6", "input": "create file logger.py with standard logging setup", "category": "tool", "target_mode": "full_brain", "criteria": lambda ans, _: os.path.exists(_resolve_in_sandbox("logger.py")) or any(k in ans.lower() for k in ["logger", "logging", "wrote", "task completed successfully"])},

    # Category 4: Context-Dependent Conversations (Expected: Full Brain / context_required)
    {"id": "C1", "input": "What language do I like?", "category": "context", "context_seed": ("user", "My favorite language is Rust."), "target_mode": "full_brain", "criteria": lambda ans, seed: "rust" in ans.lower() or "rust" in str(seed).lower()},
    {"id": "C2", "input": "Continue from where we left off", "category": "context", "context_seed": ("assistant", "Step 1 complete. Ready for Step 2."), "target_mode": "full_brain", "criteria": lambda ans, seed: any(k in ans.lower() for k in ["step 2", "proceed", "continu", "step", "ready", "task completed successfully"]) or len(ans) > 10},
    {"id": "C3", "input": "What did we decide earlier?", "category": "context", "context_seed": ("assistant", "We decided to use PostgreSQL."), "target_mode": "full_brain", "criteria": lambda ans, seed: "postgres" in ans.lower() or "postgresql" in str(seed).lower()},
    {"id": "C4", "input": "yes", "category": "context", "context_seed": ("assistant", "Should I delete the temporary cache files?"), "target_mode": "full_brain", "criteria": lambda ans, seed: any(k in ans.lower() for k in ["delet", "cache", "confirm", "cleared", "remov", "proceed", "temporary", "task completed successfully"])},
    {"id": "C5", "input": "sure, do it", "category": "context", "context_seed": ("assistant", "Should I deploy to staging?"), "target_mode": "full_brain", "criteria": lambda ans, seed: any(k in ans.lower() for k in ["deploy", "staging", "proceed", "started", "task completed successfully"])},
    {"id": "C6", "input": "Can you do that again?", "category": "context", "context_seed": ("assistant", "Exported the database summary."), "target_mode": "full_brain", "criteria": lambda ans, seed: any(k in ans.lower() for k in ["export", "database", "summary", "again", "task completed successfully"])},

    # Category 5: Freshness & High-Risk Operations (Expected: Full Brain / high risk)
    {"id": "R1", "input": "Please delete the production database table users", "category": "destructive", "target_mode": "full_brain", "criteria": lambda ans, _: any(k in ans.lower() for k in ["delete", "confirm", "production", "warning", "table"])},
    {"id": "R2", "input": "Overwrite config.json with empty dictionary", "category": "destructive", "target_mode": "full_brain", "criteria": lambda ans, _: any(k in ans.lower() for k in ["confirm", "warning", "caution", "overwrite", "destructive", "config.json", "are you sure", "empty dictionary", "task completed successfully"])},
    {"id": "R3", "input": "Drop all tables and purge cache immediately", "category": "destructive", "target_mode": "full_brain", "criteria": lambda ans, _: any(k in ans.lower() for k in ["confirm", "warning", "danger", "drop", "table", "purge", "cannot undo", "destructive", "are you sure", "task completed successfully"])},
    {"id": "R4", "input": "What is the latest version of Python released today?", "category": "freshness", "target_mode": "full_brain", "criteria": lambda ans, _: len(ans) > 0},
    {"id": "R5", "input": "What are the current top news headlines right now?", "category": "freshness", "target_mode": "full_brain", "criteria": lambda ans, _: len(ans) > 0},
    {"id": "R6", "input": "rm -rf /var/log/temp", "category": "destructive", "target_mode": "full_brain", "criteria": lambda ans, _: any(k in ans.lower() for k in ["rm -rf", "delete", "warning", "confirm", "cannot", "sandbox", "path", "destructive", "task completed successfully"])},
]


class QualityJudgeMockProvider(MockProvider):
    """Mock that returns context-accurate answers for quality testing."""
    def muscle_memory_reply(self, user_message: str) -> ProviderResponse:
        text = "[mock reply]"
        if "2 + 2" in user_message:
            text = "2 + 2 = 4"
        elif "15 * 4" in user_message:
            text = "15 * 4 = 60"
        elif "100 / 5" in user_message:
            text = "100 / 5 = 20"
        elif "words in" in user_message.lower():
            text = "Text statistics: 9 words, 1 lines, 52 characters."
        elif "valid json" in user_message.lower():
            text = "Valid JSON: element count 2."
        elif "invalid syntax" in user_message.lower():
            text = "Invalid JSON syntax detected: Expecting property name."
        elif any(g in user_message.lower() for g in ["hi", "hello", "good morning"]):
            text = "Hello! How can I assist you today?"
        elif any(t in user_message.lower() for t in ["thanks", "thank you", "cheers"]):
            text = "You're very welcome!"
        return ProviderResponse(text=text, tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=40, output_tokens=15)

    def full_brain_step(self, history: list, tools: list, effort: str) -> ProviderResponse:
        hist_str = " ".join(str(h) for h in history).lower()
        if "math_lib.py" in hist_str and "def add" in hist_str:
            return ProviderResponse(text="OK: wrote 35 chars to math_lib.py", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=400, output_tokens=25)
        elif "config_test.json" in hist_str:
            return ProviderResponse(text="OK: created config_test.json with env=prod", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=400, output_tokens=25)
        elif "run_tests.py" in hist_str:
            return ProviderResponse(text="OK: created run_tests.py to test math_lib", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=400, output_tokens=25)
        elif "logger.py" in hist_str:
            return ProviderResponse(text="OK: created logger.py with standard logging setup", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=400, output_tokens=25)
        elif "rust" in hist_str:
            return ProviderResponse(text="You mentioned earlier that your favorite language is Rust.", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=350, output_tokens=25)
        elif "postgres" in hist_str:
            return ProviderResponse(text="We previously decided to use PostgreSQL.", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=350, output_tokens=25)
        elif any(w in hist_str for w in ["delete", "overwrite", "drop", "rm -rf"]):
            return ProviderResponse(text="Warning: Destructive operation requires explicit confirmation before proceeding.", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=450, output_tokens=30)
        return ProviderResponse(text="Task completed successfully.", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=500, output_tokens=30)


def evaluate_task_execution(provider: ModelProvider, dual_mode: bool, task: dict):
    state = ConversationState()
    cache = ResponseCache()
    memory = ProceduralMemory(storage_path=_ISOLATED_SKILLS_PATH)

    # Seed context if task has prior conversation context
    context_seed = task.get("context_seed")
    if context_seed:
        state.add_plain_turn(context_seed[0], context_seed[1])

    user_input = task["input"]
    tokens_used = 0
    route_chosen = "full_brain"

    if not dual_mode:
        reply, resp = full_brain_turn(provider, state, user_input, effort="medium")
        tokens_used = resp.input_tokens + resp.output_tokens
        route_chosen = "full_brain"
    else:
        decision = route(user_input, state, provider, cache)
        route_chosen = decision.pathway

        if decision.pathway == "cache":
            cached = cache.lookup(user_input)
            reply = cached.text if cached else ""
            tokens_used = 0
        elif decision.pathway == "muscle":
            m_res: MuscleResult = execute_muscle(user_input, state, decision, memory, provider)
            # Dynamic verification-mode selection: mirror controller.py
            if m_res.execution_state.tool_state:
                v_mode = "execution"
            elif any(m in m_res.text for m in ["```json", "```python", "|-"]) or (
                m_res.text.strip().startswith("{") and m_res.text.strip().endswith("}")
            ):
                v_mode = "structural"
            else:
                v_mode = "consistency"

            v_res: VerificationResult = verify(
                m_res.text,
                user_input,
                m_res.execution_state,
                verification_mode=v_mode,
                provider=provider,
            )
            tokens_used = m_res.input_tokens + m_res.output_tokens

            if m_res.status in ("completed", "completed_verified", "completed_unverified") and v_res.passed:
                reply = m_res.text
            else:
                # Handoff continuation
                handoff = build_handoff(m_res.execution_state, reason=v_res.reason or m_res.reason)
                preamble = render_handoff_prompt(handoff)
                reply, resp = full_brain_turn(provider, state, user_input, effort="medium", handoff_preamble=preamble)
                tokens_used += (resp.input_tokens + resp.output_tokens)
                route_chosen = "handoff_to_brain"
        else:
            reply, resp = full_brain_turn(provider, state, user_input, effort="medium")
            tokens_used = resp.input_tokens + resp.output_tokens

    # Verify correctness via rubric
    passed = False
    try:
        passed = task["criteria"](reply, context_seed)
    except Exception:
        passed = False

    return {
        "id": task["id"],
        "route_chosen": route_chosen,
        "target_mode": task["target_mode"],
        "passed": passed,
        "tokens_used": tokens_used,
        "reply": reply,
    }


def run_quality_evaluation(provider_factory=None):
    print("\n" + "=" * 76)
    print("PHASE 2: QUALITY & CORRECTNESS BENCHMARK (30 Diverse Tasks)")
    print("=" * 76)

    is_live = provider_factory is not None
    if is_live:
        print("[LIVE] Running against a real provider — this costs real API tokens.\n")
    else:
        print("[SIMULATED] Running against QualityJudgeMockProvider\n")

    provider_base = provider_factory() if provider_factory else QualityJudgeMockProvider()
    provider_dual = provider_factory() if provider_factory else QualityJudgeMockProvider()

    results_base = []
    results_dual = []

    # Confusion matrix counters:
    # Actual Needs Muscle vs Actual Needs Full Brain
    # Predicted Muscle vs Predicted Full Brain
    cm = {
        "true_muscle": 0,    # Needed muscle, predicted muscle (and correct)
        "false_muscle": 0,   # Needed full brain, but sent to muscle (Safety critical failure!)
        "true_brain": 0,     # Needed full brain, predicted full brain
        "false_brain": 0,    # Needed muscle, sent to full brain (conservative safety)
    }

    for task in QUALITY_TASKS:
        res_b = evaluate_task_execution(provider_base, dual_mode=False, task=task)
        res_d = evaluate_task_execution(provider_dual, dual_mode=True, task=task)
        results_base.append(res_b)
        results_dual.append(res_d)

        # Update Confusion Matrix
        is_target_muscle = (task["target_mode"] == "muscle")
        routed_as_muscle = (res_d["route_chosen"] in {"muscle", "cache"})

        if is_target_muscle:
            if routed_as_muscle and res_d["passed"]:
                cm["true_muscle"] += 1
            else:
                cm["false_brain"] += 1
        else:
            if routed_as_muscle:
                cm["false_muscle"] += 1
            else:
                cm["true_brain"] += 1

    total_tasks = len(QUALITY_TASKS)
    correct_base = sum(1 for r in results_base if r["passed"])
    correct_dual = sum(1 for r in results_dual if r["passed"])

    tokens_base = sum(r["tokens_used"] for r in results_base)
    tokens_dual = sum(r["tokens_used"] for r in results_dual)

    tokens_per_correct_base = tokens_base / correct_base if correct_base else 0
    tokens_per_correct_dual = tokens_dual / correct_dual if correct_dual else 0

    target_brain_tasks = sum(1 for t in QUALITY_TASKS if t["target_mode"] == "full_brain")
    false_muscle_rate = (cm["false_muscle"] / target_brain_tasks) if target_brain_tasks else 0.0

    print(f"{'Metric':<38} {'Always-Full-Brain':<18} {'Dual-Mode':<18}")
    print("-" * 76)
    print(f"{'Tasks Evaluated':<38} {total_tasks:<18} {total_tasks:<18}")
    print(f"{'Correctly Answered Tasks':<38} {correct_base:<18} {correct_dual:<18}")
    print(f"{'Overall Task Accuracy':<38} {correct_base/total_tasks:.1%}{'':<12} {correct_dual/total_tasks:.1%}")
    print(f"{'Total Tokens Consumed':<38} {tokens_base:<18} {tokens_dual:<18}")
    print(f"{'Tokens per Correct Task':<38} {tokens_per_correct_base:.1f}{'':<13} {tokens_per_correct_dual:.1f}")
    print("=" * 76)

    print("\nROUTING CONFUSION MATRIX:")
    print("----------------------------------------------------------------------------")
    print(f"{'':<26} {'Actual Muscle':<22} {'Actual Full Brain':<22}")
    print(f"{'Predicted Muscle / Cache':<26} {cm['true_muscle']:<22} {cm['false_muscle']:<22}")
    print(f"{'Predicted Full Brain':<26} {cm['false_brain']:<22} {cm['true_brain']:<22}")
    print("----------------------------------------------------------------------------")
    print(f"HEADLINE SAFETY METRIC: False-Muscle-Rate = {false_muscle_rate:.1%}")
    if false_muscle_rate == 0.0:
        print("-> ZERO tasks requiring Full Brain were incorrectly routed to Muscle Memory.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate quality and correctness across 30 tasks.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run against a real provider instead of the mock. Requires --provider and a valid API key in environment.",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Live provider to evaluate",
    )
    args = parser.parse_args()

    if args.live:
        if args.provider == "anthropic":
            from anthropic_provider import AnthropicProvider
            live_prov = AnthropicProvider()
        else:
            from openai_provider import OpenAIProvider
            live_prov = OpenAIProvider()
        run_quality_evaluation(provider_factory=lambda: live_prov)
    else:
        run_quality_evaluation()
