"""
dual_mode_agent.py — CLI REPL Interface for Adaptive Dual-Path Agent (v5).

IMPORTANT ARCHITECTURAL DIRECTIVE:
The CANONICAL ENGINE ENTRYPOINT is `controller.AgentController` in `controller.py`.
All core routing, execution, verification, continuation handoff, and learning logic
is owned and orchestrated strictly by `AgentController`.

This module serves solely as:
1. The interactive command-line interface (`python dual_mode_agent.py`).
2. A thin backward-compatibility shim for legacy test harnesses.
"""

from __future__ import annotations
import argparse
import hashlib
import os
import re
import time
from typing import Any, Tuple, Optional, List, Dict

from base import ConversationState, ModelProvider, ProviderResponse, ResponseCache, SessionStats
from brain import BrainEngine, select_effort
from controller import AgentController, TaskContext
from execution_state import ExecutionState
from handoff import build_handoff, render_handoff_prompt
from muscle_engine import execute_muscle, MuscleResult
from procedural_memory import ProceduralMemory
from router import route, RouteDecision
from tools import TOOLS, execute_tool
from verification import verify, VerificationResult

# Global configuration constants
MAX_TOOL_ITERATIONS = 8
SUMMARY_INTERVAL_TURNS = 8

GREETING_PATTERN = re.compile(
    r"^\s*(hi|hey|hello|greetings|howdy|sup|yo|"
    r"thanks|thank you|thx|ty|cheers|"
    r"bye|goodbye|cya|see you)\s*[!.]*\s*$",
    re.IGNORECASE,
)


def is_obvious_muscle_memory(text: str) -> bool:
    return bool(GREETING_PATTERN.match(text))


# ============================================================
# Backward-compatible Helper Wrappers
# ============================================================
def compress_summary_if_needed(provider: ModelProvider, state: ConversationState, force: bool = False) -> None:
    """Backward-compatible helper delegating to controller logic."""
    controller = AgentController(provider=provider, state=state)
    controller.compress_summary_if_needed(force=force)


def full_brain_turn(
    provider: ModelProvider,
    state: ConversationState,
    user_message: str,
    effort: str = "medium",
    handoff_preamble: Optional[str] = None,
    exec_state: Optional[ExecutionState] = None,
) -> Tuple[str, ProviderResponse]:
    """Backward-compatible helper delegating to BrainEngine."""
    brain = BrainEngine()
    reply, resp, _ = brain.execute_turn(
        provider=provider,
        state=state,
        user_message=user_message,
        effort=effort,
        handoff_preamble=handoff_preamble,
        exec_state=exec_state,
    )
    return reply, resp


def _extract_trigger_patterns(query: str) -> List[str]:
    """Extracts exact-match and key content-word trigger patterns."""
    clean = re.sub(r"[?!.,;:]+$", "", query.strip().lower())
    norm = re.sub(r"\s+", " ", clean)
    p1 = r"^\s*" + re.escape(norm) + r"\s*[!.?]*\s*$"
    stop_words = {"a", "an", "the", "is", "of", "in", "for", "to", "what", "how", "can", "you", "me", "my", "do", "does", "please"}
    content_words = [w for w in norm.split() if w not in stop_words and len(w) > 2]
    p2 = r".*\b" + r"\b.*\b".join(re.escape(w) for w in content_words[:4]) + r"\b.*" if content_words else p1
    return [p1, p2]


def _distill_procedure(raw_reply: str, max_chars: int = 500) -> str:
    """Strips conversational preambles and markdown headers for reusable skill procedure."""
    reply = re.sub(r"^(sure|of course|certainly|absolutely|i can help|here'?s?)[!,.:]*\s*", "", raw_reply.strip(), flags=re.IGNORECASE)
    reply = re.sub(r"^#+\s+", "", reply, flags=re.MULTILINE)
    return reply.strip()[:max_chars]


# ============================================================
# Main Agent REPL Execution Loop
# ============================================================
def run(
    provider: ModelProvider,
    provider_label: str = "",
    runtime: Optional[UniversalModelRuntime] = None,
):
    state = ConversationState()
    stats = SessionStats()
    cache = ResponseCache()
    memory = ProceduralMemory()
    controller = AgentController(
        provider=provider,
        runtime=runtime,
        state=state,
        stats=stats,
        cache=cache,
        memory=memory,
    )

    print(f"Adaptive Dual-Path Agent v5 ready ({provider_label}). Commands: 'quit', 'report'\n")
    while True:
        # Session Quota Guard
        TOKEN_WARN = int(os.environ.get("AGENT_TOKEN_WARN_LIMIT", 40000))
        TOKEN_LIMIT = int(os.environ.get("AGENT_TOKEN_HARD_LIMIT", 90000))
        consumed = stats.total_tokens_consumed()
        if consumed >= TOKEN_LIMIT:
            print(f"[QUOTA] Hard limit reached ({consumed} tokens). Session ended.")
            break
        if consumed >= TOKEN_WARN:
            print(f"[QUOTA WARNING] {consumed} tokens consumed. Approaching limit.")

        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input.lower() == "report":
            print(stats.report())
            continue

        # Controller manages task lifecycle: start -> execute (route, muscle/brain, handoff, verify, learn) -> output
        task = controller.start(user_input)
        reply = controller.execute_task(task)
        pathway = task.decision.pathway if task.decision else "unknown"
        print(f"Agent ({pathway}): {reply}\n")

    controller.shutdown()
    print(stats.report())


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Dual-Path Agent v9 (Universal Model Runtime)")
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        help="Model provider name (default: anthropic). Use --list-providers to see all.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to pass to the provider. If omitted, uses each provider's recommended default.",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List all registered providers and exit.",
    )
    args = parser.parse_args()

    from provider_registry import build_default_registry
    from universal_runtime import UniversalModelRuntime

    registry = build_default_registry()

    if args.list_providers:
        print("Registered providers:")
        for info in registry.provider_info():
            status = info.get("status_label") or ("AVAILABLE" if info["available"] else f"NEEDS {info['requires_api_key']}")
            print(f"  {info['name']:12s} [{status}] - {info['description']}")
        import sys
        sys.exit(0)

    # Create runtime through registry (no hardcoded if/elif chains)
    try:
        runtime = registry.create_runtime(args.provider, model=args.model)
        label = f"{args.provider} / {runtime.model_id} (via Universal Runtime)"
    except ValueError as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)

    run(runtime.provider, provider_label=label, runtime=runtime)


if __name__ == "__main__":
    main()

