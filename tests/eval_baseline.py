"""
tests/eval_baseline.py — Baseline Comparison: Always-Full-Brain vs. Same-Model Dual-Mode.

Runs a representative multi-turn conversation workload through two architectures:
  1. Baseline (Always-Full-Brain): Every turn gets full context, tools, and high effort.
  2. Dual-Mode (Same-Model): Pre-filter + response cache + low-effort Muscle Memory + Full Brain.

Produces an honest, non-hallucinated comparison table of token counts, cache hit
rates, and cost savings.

Usage:
  python tests/eval_baseline.py              # runs with deterministic mock (verifies savings math)
  python tests/eval_baseline.py --live --provider anthropic  # runs against live API
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ModelProvider, ProviderResponse, ResponseCache, SessionStats
from dual_mode_agent import full_brain_turn, is_obvious_muscle_memory
from mock_provider import MockProvider
from tools import TOOLS


# Workload Profiles per v4 Spec Phase 7
WORKLOADS = {
    "casual": [
        {"user": "hi", "type": "chat"},
        {"user": "how are you today?", "type": "chat"},
        {"user": "thanks for asking", "type": "chat"},
        {"user": "what is 2 + 2?", "type": "chat"},
        {"user": "tell me a short joke", "type": "chat"},
        {"user": "haha that was funny", "type": "chat"},
        {"user": "thanks", "type": "chat"},
        {"user": "goodbye", "type": "chat"},
    ],
    "balanced": [
        {"user": "hi", "type": "chat"},
        {"user": "how are you today?", "type": "chat"},
        {"user": "can you create a file called math_utils.py with an add function?", "type": "task"},
        {"user": "thanks!", "type": "chat"},
        {"user": "now add a multiply function to math_utils.py", "type": "task"},
        {"user": "read math_utils.py and check if there are any syntax errors", "type": "task"},
        {"user": "what is 15 * 24?", "type": "chat"},
        {"user": "thanks", "type": "chat"},
        {"user": "write a test file test_math.py that tests both add and multiply", "type": "task"},
        {"user": "bye", "type": "chat"},
    ],
    "adversarial": [
        {"user": "My favorite programming language is Rust.", "type": "chat"},
        {"user": "What language do I like?", "type": "context_task"},
        {"user": "What is the latest version of Python released today?", "type": "freshness_task"},
        {"user": "Please delete the file temp_data.csv", "type": "destructive_task"},
        {"user": "continue", "type": "context_task"},
        {"user": "yes", "type": "ambiguous_ack"},
        {"user": "Overwrite config.json with empty dictionary", "type": "destructive_task"},
        {"user": "thanks", "type": "chat"},
    ],
}


def run_baseline_architecture(provider: ModelProvider, workload: list) -> SessionStats:
    """Always-Full-Brain: Every single turn runs through full_brain_turn with tools and medium effort."""
    state = ConversationState()
    stats = SessionStats()

    for item in workload:
        user_msg = item["user"]
        reply, resp = full_brain_turn(provider, state, user_msg, effort="medium")
        stats.add("full_brain", provider.name, resp, model=provider.model, effort="medium")
        state.add_plain_turn("user", user_msg)
        state.add_plain_turn("assistant", reply)

    return stats


from controller import AgentController
from procedural_memory import ProceduralMemory


def run_dual_mode_architecture(provider: ModelProvider, workload: list) -> SessionStats:
    """Same-Model Dual-Mode v5: Central AgentController driving all lifecycle steps."""
    import tempfile
    state = ConversationState()
    stats = SessionStats()
    cache = ResponseCache()
    memory = ProceduralMemory(storage_path=tempfile.mktemp(suffix="_skills.json"))
    ctrl = AgentController(provider=provider, state=state, stats=stats, cache=cache, memory=memory)

    for item in workload:
        user_msg = item["user"]
        task = ctrl.start(user_msg)
        ctrl.execute_task(task)

    return stats


def compute_metrics(stats: SessionStats):
    user_turns = [t for t in stats.turns if t.mode != "classify"]
    muscle_turns = [t for t in stats.turns if t.mode == "muscle_memory"]
    brain_turns = [t for t in stats.turns if t.mode == "full_brain"]
    cached_turns = [t for t in stats.turns if t.cached_response]

    total_input = sum(t.input_tokens for t in stats.turns)
    total_output = sum(t.output_tokens for t in stats.turns)
    total_tokens = total_input + total_output
    total_cache_read = sum(t.cache_read for t in stats.turns)
    cache_hit_rate = (total_cache_read / total_input) if total_input else 0.0

    # Pricing: $3/1M input, $15/1M output
    est_cost = (total_input * 0.003 + total_output * 0.015) / 1000

    return {
        "user_turns": len(user_turns),
        "muscle_turns": len(muscle_turns),
        "brain_turns": len(brain_turns),
        "cached_turns": len(cached_turns),
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": total_tokens,
        "cache_read": total_cache_read,
        "cache_hit_rate": cache_hit_rate,
        "est_cost": est_cost,
    }


def compare_workload(workload_name: str, workload: list, live: bool = False, provider_name: str = None):
    mode_desc = f"Live API ({provider_name})" if live and provider_name else "[SIMULATED] Mocked Benchmark"
    print(f"\nEvaluating Workload: '{workload_name.upper()}' ({len(workload)} turns) — {mode_desc}")
    print("=" * 74)

    if live and provider_name:
        if provider_name == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                print("\n[ERROR] ANTHROPIC_API_KEY is not set.")
                sys.exit(1)
            from anthropic_provider import AnthropicProvider
            p_baseline = AnthropicProvider()
            p_dual = AnthropicProvider()
        elif provider_name == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                print("\n[ERROR] OPENAI_API_KEY is not set.")
                sys.exit(1)
            from openai_provider import OpenAIProvider
            p_baseline = OpenAIProvider()
            p_dual = OpenAIProvider()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    else:
        p_baseline = MockProvider(tool_rounds=1)
        p_dual = MockProvider(tool_rounds=1)
        def smart_mock_classify(msg, summary):
            if any(w in msg.lower() for w in ["math", "file", "function", "read", "write", "check", "script", "rust"]):
                return "full_brain"
            return "muscle_memory"
        p_dual.classify = smart_mock_classify

    stats_baseline = run_baseline_architecture(p_baseline, workload)
    stats_dual = run_dual_mode_architecture(p_dual, workload)

    m_base = compute_metrics(stats_baseline)
    m_dual = compute_metrics(stats_dual)

    sim_tag = "" if live else " (simulated)"
    token_savings = (m_base["total_tokens"] - m_dual["total_tokens"]) / m_base["total_tokens"] if m_base["total_tokens"] else 0.0
    cost_savings = (m_base["est_cost"] - m_dual["est_cost"]) / m_base["est_cost"] if m_base["est_cost"] else 0.0

    print(f"{'Metric':<30} {'Always-Full-Brain':<20} {'Dual-Mode v4':<22}")
    print("-" * 74)
    print(f"{'Total user turns':<30} {m_base['user_turns']:<20} {m_dual['user_turns']:<22}")
    print(f"{'Muscle Memory turns':<30} {m_base['muscle_turns']:<20} {m_dual['muscle_turns']} ({m_dual['muscle_turns']/m_dual['user_turns']:.0%})")
    print(f"{'Full Brain turns':<30} {m_base['brain_turns']:<20} {m_dual['brain_turns']} ({m_dual['brain_turns']/m_dual['user_turns']:.0%})")
    print(f"{'Cached 0-token turns':<30} {m_base['cached_turns']:<20} {m_dual['cached_turns']:<22}")
    print(f"{'Total prompt tokens' + sim_tag:<30} {m_base['total_input']:<20} {m_dual['total_input']:<22}")
    print(f"{'Total output tokens' + sim_tag:<30} {m_base['total_output']:<20} {m_dual['total_output']:<22}")
    print(f"{'Total billable tokens' + sim_tag:<30} {m_base['total_tokens']:<20} {m_dual['total_tokens']:<22}")
    print(f"{'Estimated total cost' + sim_tag:<30} ${m_base['est_cost']:.4f}{'':<14} ${m_dual['est_cost']:.4f}")
    print("=" * 74)
    print(f"TOKEN SAVINGS : {token_savings:.1%}")
    print(f"COST SAVINGS  : {cost_savings:.1%}")
    print("=" * 74 + "\n")


def compare(live: bool = False, provider_name: str = None, workload_choice: str = "all"):
    if workload_choice == "all":
        for name, wl in WORKLOADS.items():
            compare_workload(name, wl, live=live, provider_name=provider_name)
    else:
        wl = WORKLOADS.get(workload_choice, WORKLOADS["balanced"])
        compare_workload(workload_choice, wl, live=live, provider_name=provider_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--live", action="store_true", default=False,
                        help="Run live evaluation using provider API key")
    parser.add_argument("--workload", choices=["casual", "balanced", "adversarial", "all"], default="all",
                        help="Workload distribution to evaluate")
    args = parser.parse_args()
    compare(live=args.live, provider_name=args.provider, workload_choice=args.workload)
