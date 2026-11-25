"""
restart_agent.py — Pure Restart Baseline Agent for V7 Phase 15.

When automatic/muscle execution cannot complete a task:
RestartAgent discards all completed execution state and starts the deliberative
Brain from the raw initial user prompt as if Muscle never ran.

Provides the direct scientific comparison against ContinuationAgent to evaluate
the core Dynamic Dual Model hypothesis:
"Does structured verified continuation save tokens and latency compared to restart?"
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

from base import ConversationState, ModelProvider, ResponseCache, SessionStats
from brain import BrainEngine
from execution_state import ExecutionState, TaskUsage, COMPLETED_VERIFIED
from muscle_engine import MuscleExecutor, MuscleResult
from router import route, RouteDecision
from tools import TOOLS


@dataclass
class RestartTaskResult:
    task_id: str
    user_input: str
    final_text: str
    success: bool
    pathway: str
    muscle_attempted: bool
    restarted: bool
    muscle_tokens: int
    brain_tokens: int
    total_tokens: int
    muscle_tool_calls: int
    brain_tool_calls: int
    total_tool_calls: int
    latency_ms: float
    repetition_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class RestartAgent:
    """Agent that restarts from scratch when Muscle execution does not complete."""

    def __init__(
        self,
        provider: ModelProvider,
        cache: Optional[ResponseCache] = None,
        memory: Optional[Any] = None,
    ):
        self.provider = provider
        self.cache = cache or ResponseCache()
        self.memory = memory
        self.muscle = MuscleExecutor(provider=provider, memory=memory, cache=self.cache)
        self.brain = BrainEngine(tools=TOOLS)
        self.stats = SessionStats()

    def execute(self, user_message: str) -> RestartTaskResult:
        t0 = time.perf_counter()
        initial_state = ExecutionState(objective=user_message)
        initial_conv = ConversationState()

        # Step 1: Attempt Muscle Execution
        decision = route(
            user_message=user_message,
            state=initial_conv,
            cache=self.cache,
            memory=self.memory,
            provider=self.provider,
        )

        muscle_res: Optional[MuscleResult] = None
        muscle_attempted = False
        muscle_tokens = 0
        muscle_tool_calls = 0

        if decision.pathway in {"muscle", "cache"}:
            muscle_attempted = True
            muscle_res = self.muscle.execute(
                user_message=user_message,
                exec_state=initial_state,
                decision=decision,
                conv_state=initial_conv,
            )
            muscle_tokens = initial_state.usage.total_billable_tokens
            muscle_tool_calls = len(initial_state.tool_actions)

            # If verified, return directly
            if muscle_res.status == COMPLETED_VERIFIED and muscle_res.text:
                latency_ms = (time.perf_counter() - t0) * 1000
                return RestartTaskResult(
                    task_id=initial_state.task_id,
                    user_input=user_message,
                    final_text=muscle_res.text,
                    success=True,
                    pathway="muscle",
                    muscle_attempted=True,
                    restarted=False,
                    muscle_tokens=muscle_tokens,
                    brain_tokens=0,
                    total_tokens=muscle_tokens,
                    muscle_tool_calls=muscle_tool_calls,
                    brain_tool_calls=0,
                    total_tool_calls=muscle_tool_calls,
                    latency_ms=latency_ms,
                    repetition_count=0,
                )

        # Step 2: RESTART FROM SCRATCH
        # Discard all Muscle execution state, verified facts, tool results, and frontier
        fresh_state = ExecutionState(objective=user_message)
        fresh_conv = ConversationState()

        final_text, resp, final_state = self.brain.run_turn(
            provider=self.provider,
            state=fresh_conv,
            user_message=user_message,  # Raw user prompt, NO handoff preamble
            handoff_preamble=None,
            exec_state=fresh_state,
        )

        brain_tokens = final_state.usage.brain_tokens
        brain_tool_calls = len(final_state.tool_actions)
        total_tokens = muscle_tokens + brain_tokens
        total_tool_calls = muscle_tool_calls + brain_tool_calls
        latency_ms = (time.perf_counter() - t0) * 1000

        # Repetitions: tool calls that repeat actions already executed in the discarded Muscle attempt
        prior_signatures = {a.get("name") for a in initial_state.tool_actions if isinstance(a, dict)}
        repetition_count = sum(1 for a in final_state.tool_actions if isinstance(a, dict) and a.get("name") in prior_signatures)

        return RestartTaskResult(
            task_id=final_state.task_id,
            user_input=user_message,
            final_text=final_text,
            success=final_state.status == COMPLETED_VERIFIED,
            pathway="brain_restart",
            muscle_attempted=muscle_attempted,
            restarted=True,
            muscle_tokens=muscle_tokens,
            brain_tokens=brain_tokens,
            total_tokens=total_tokens,
            muscle_tool_calls=muscle_tool_calls,
            brain_tool_calls=brain_tool_calls,
            total_tool_calls=total_tool_calls,
            latency_ms=latency_ms,
            repetition_count=repetition_count,
            metadata={"prior_muscle_actions": len(initial_state.tool_actions)},
        )
