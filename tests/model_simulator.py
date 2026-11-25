"""
model_simulator.py — Configurable Model Simulator for $0 offline stress testing.

Provides a fully controllable, deterministic ModelProvider implementation
that simulates a wide spectrum of production model behaviors and failure modes:

Modes:
- "perfect": Flawless tool use, high confidence, precise answers.
- "weak": Low confidence answers, ungrounded claims (triggers verification escalation).
- "partial": Executes one tool then stops early (triggers Muscle -> Brain handoff).
- "tool_failure": Emits tool calls with invalid or failing arguments.
- "malformed_output": Returns invalid JSON or syntax errors.
- "timeout": Raises TimeoutError on execution steps.
- "context_overflow": Simulates context length exceeded error.
- "refusal": Simulates model safety refusal.
- "slow": Injects configurable simulated latency.
"""

from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional

from base import (
    ModelProvider,
    ProviderCapabilities,
    ProviderResponse,
    ToolCall,
    ToolDef,
    count_tokens,
)


class ModelSimulator(ModelProvider):
    """Deterministic and configurable model simulator for dual-mode stress testing."""

    name: str = "simulator"

    def __init__(
        self,
        model: str = "simulator-v1",
        mode: str = "perfect",
        simulated_latency_ms: float = 0.0,
    ):
        self.model = model
        self.mode = mode
        self.simulated_latency_ms = simulated_latency_ms
        self.call_history: List[Dict[str, Any]] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_reasoning_effort=True,
            supports_prompt_cache=True,
            supports_structured_output=True,
            supports_native_tool_history=True,
            supports_reasoning_usage=True,
            supports_tool_calls=True,
        )

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _maybe_sleep_or_raise(self) -> None:
        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        if self.mode == "timeout":
            raise TimeoutError("Simulated provider call timed out after deadline.")
        elif self.mode == "context_overflow":
            raise RuntimeError("Simulated context overflow: Maximum context window of 128,000 tokens exceeded.")
        elif self.mode == "refusal":
            raise RuntimeError("Simulated model refusal: Request violates simulated policy.")

    # ------------------------------------------------------------------
    # Routing / Classification
    # ------------------------------------------------------------------
    def classify(self, user_message: str, recent_summary: str) -> str:
        self._maybe_sleep_or_raise()
        self.call_history.append({"phase": "classify", "msg": user_message})

        if self.mode == "weak":
            # Weak mode is indecisive or misroutes
            return "full_brain"

        # Deterministic simulation
        low = user_message.lower().strip()
        if any(w in low for w in ("hi", "hello", "thanks", "2+2", "what time")):
            return "muscle_memory"
        return "full_brain"

    def route_classify(self, user_message: str, recent_summary: str) -> Dict[str, Any]:
        self._maybe_sleep_or_raise()
        self.call_history.append({"phase": "route_classify", "msg": user_message})

        if self.mode == "malformed_output":
            return {"raw": "not a valid json dict"}

        c = self.classify(user_message, recent_summary)
        is_muscle = c == "muscle_memory"
        novelty = 0.1 if is_muscle else (0.4 if self.mode == "weak" else 0.8)
        confidence = 0.55 if self.mode == "weak" else 0.95

        resp = ProviderResponse(
            text=f"Simulated route {c}",
            tool_calls=[],
            stop_reason="end_turn",
            raw=None,
            input_tokens=count_tokens(user_message),
            output_tokens=15,
        )
        return {
            "pathway": "muscle" if is_muscle else "full_brain",
            "confidence": confidence,
            "context_required": not is_muscle,
            "tools_required": not is_muscle,
            "freshness_required": False,
            "novelty": novelty,
            "risk": "low" if is_muscle else "medium",
            "reasoning": f"Simulator mode '{self.mode}' routing decision.",
            "_response": resp,
        }

    # ------------------------------------------------------------------
    # Muscle Memory
    # ------------------------------------------------------------------
    def muscle_memory_reply(self, user_message: str) -> ProviderResponse:
        self._maybe_sleep_or_raise()
        self.call_history.append({"phase": "muscle", "msg": user_message})

        if self.mode == "malformed_output":
            text = "```json\n{ unclosed json: ...\n"
        elif self.mode == "weak":
            text = "I think maybe that could be true, but I'm not really sure."
        else:
            text = f"Simulated muscle response to: {user_message}"

        in_tok = count_tokens(user_message)
        out_tok = count_tokens(text)
        return ProviderResponse(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            raw=None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )

    # ------------------------------------------------------------------
    # Full Brain
    # ------------------------------------------------------------------
    def full_brain_step(
        self,
        history: List[Any],
        tools: List[ToolDef],
        effort: str,
    ) -> ProviderResponse:
        self._maybe_sleep_or_raise()
        self.call_history.append({"phase": "brain", "effort": effort, "history_len": len(history)})

        # Check if the last turn was a tool result
        last_turn_was_tool = False
        if history and isinstance(history[-1], dict) and history[-1].get("role") in ("tool", "function"):
            last_turn_was_tool = True

        tool_calls: List[ToolCall] = []
        if not last_turn_was_tool and tools:
            if self.mode == "partial":
                # Calls first tool and stops
                t = tools[0]
                tool_calls.append(ToolCall(
                    id="sim_call_partial_1",
                    name=t.name,
                    arguments={"path": "partial.txt", "content": "partial content"},
                ))
                text = "Executing first partial step."
            elif self.mode == "tool_failure":
                # Emits tool call with invalid arguments
                tool_calls.append(ToolCall(
                    id="sim_call_fail_1",
                    name="non_existent_tool_xyz",
                    arguments={},
                ))
                text = "Attempting execution."
            else:
                # Perfect mode: calls tool with valid arguments
                t = tools[0]
                tool_calls.append(ToolCall(
                    id="sim_call_1",
                    name=t.name,
                    arguments={"path": "sim_output.txt", "content": "success"},
                ))
                text = f"Calling tool {t.name} to complete task."
        else:
            text = "Task completed successfully by ModelSimulator."

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=None,
            input_tokens=150,
            output_tokens=40,
            cache_read_tokens=20,
            cache_write_tokens=10,
        )

    # ------------------------------------------------------------------
    # Summary Compression
    # ------------------------------------------------------------------
    def summarize_history(self, history_text: str) -> str:
        self._maybe_sleep_or_raise()
        self.call_history.append({"phase": "summary"})
        return f"Summary of {len(history_text)} characters by ModelSimulator."

    # ------------------------------------------------------------------
    # History Plumbing
    # ------------------------------------------------------------------
    def append_assistant_turn(self, history: List[Any], resp: ProviderResponse) -> None:
        turn: Dict[str, Any] = {"role": "assistant", "content": resp.text}
        if resp.tool_calls:
            turn["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in resp.tool_calls
            ]
        history.append(turn)

    def append_tool_results(
        self, history: List[Any], tool_calls: List[ToolCall], results: Dict[str, str]
    ) -> None:
        for tc in tool_calls:
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": results.get(tc.id, ""),
            })

    def append_user_turn(self, history: List[Any], text: str) -> None:
        history.append({"role": "user", "content": text})
