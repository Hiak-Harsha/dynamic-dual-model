"""
openai_provider.py — SAME-MODEL dual-mode adapter for OpenAI.

Every call — classifier, Muscle Memory, Full Brain, and summary compressor —
uses the SAME model ID (default "gpt-5.5"). They are differentiated only by:

  - effort    : low (classifier/muscle/summary) vs medium/high/xhigh (brain)
  - max_tokens: 16 (classifier), 120 (muscle), 300 (summary), 8000 (brain)
  - tools     : none (classifier/muscle/summary) vs full toolset (brain)
  - context   : bare message (muscle) vs full history (brain)

Current as of Sept 2026 — verify against https://developers.openai.com/api/docs:

  - Model: "gpt-5.5" (or "gpt-5.5-pro") for everything.
  - API: Responses API (`client.responses.create`) used across all paths
    so the nested `reasoning={"effort": ...}` parameter is available uniformly.
  - Prompt caching: automatic on OpenAI once static prefix >= 1,024 tokens.
    Static instructions are padded to ~1,200 tokens to ensure cache activation.
  - Cached tokens: read from `usage.input_tokens_details.cached_tokens`.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore[import-not-found, import-untyped]
except ImportError:
    OpenAI = None  # type: ignore

from base import ModelProvider, ProviderResponse, ToolCall, ToolDef

from retry_util import with_retry


# Padded static instructions exceeding OpenAI's ~1,024 token automatic cache floor
FULL_BRAIN_INSTRUCTIONS = (
    "You are an expert software engineer, systems architect, and problem "
    "solver with access to file tools. Your capabilities include reading, "
    "writing, and analyzing code files, debugging issues, implementing "
    "features, and providing technical guidance.\n\n"

    "## Core Behavior\n"
    "- PREFER using tools over describing what to do. If a user asks you "
    "to write code, actually write it to a file — don't just show it.\n"
    "- Be thorough but stay focused on the requested result.\n"
    "- When reading files, scan for the specific information needed rather "
    "than summarizing the entire file.\n"
    "- When writing files, include complete, working code — never use "
    "placeholders like '// ... rest of code ...'.\n\n"

    "## Coding Style Guide\n"
    "Follow these conventions unless the existing codebase uses different ones:\n"
    "- Use descriptive variable and function names (no single-letter names "
    "except loop counters and well-known conventions like x/y coordinates).\n"
    "- Include docstrings for public functions and classes.\n"
    "- Handle errors explicitly — never silently swallow exceptions.\n"
    "- Prefer composition over inheritance where practical.\n"
    "- Keep functions focused: one function, one responsibility.\n"
    "- Use type hints in Python; use TypeScript over plain JavaScript.\n"
    "- Constants should be UPPER_SNAKE_CASE and defined at module level.\n"
    "- Imports should be sorted: stdlib first, then third-party, then local.\n"
    "- Maximum line length: 100 characters (soft limit, readability first).\n"
    "- Use f-strings for string formatting in Python 3.6+.\n\n"

    "## Tool Usage Guidelines\n"
    "- Always read a file before modifying it, unless you just wrote it.\n"
    "- When writing files, use the full path including any subdirectories.\n"
    "- If a tool call fails, report the error clearly and suggest an "
    "alternative approach rather than retrying blindly.\n"
    "- Prefer small, incremental file changes over rewriting entire files.\n"
    "- When creating new files, include appropriate file headers and "
    "documentation.\n\n"

    "## Output Format\n"
    "- For code explanations, use fenced code blocks with language tags.\n"
    "- For multi-step plans, use numbered lists.\n"
    "- Keep prose concise — favor bullet points over paragraphs.\n"
    "- When reporting tool results, summarize the outcome rather than "
    "echoing the full output verbatim.\n"
    "- If the user's request is ambiguous, ask ONE clarifying question "
    "rather than guessing — but lean toward doing the obvious thing if "
    "it's 80%+ likely what they meant.\n\n"

    "## Error Handling & Debugging\n"
    "- When debugging, start by reading the relevant file(s) and error "
    "messages before proposing fixes.\n"
    "- Explain the root cause of bugs, not just the fix.\n"
    "- When a fix is non-obvious, add a comment explaining WHY, not just "
    "WHAT the code does.\n"
    "- Test your assumptions: if you think a file exists, read it first; "
    "if you think a function has a certain signature, verify it.\n\n"

    "## Sandboxing and Security Guarantees\n"
    "- All file read and write operations are strictly confined to the workspace sandbox root.\n"
    "- Any attempt to navigate outside the sandbox using path traversal sequences (such as parent "
    "directory references or absolute paths outside the target directory) will be rejected by the security layer.\n"
    "- Destructive actions—including file deletions, table drops, and major schema alterations—are treated as "
    "high-risk operations. Always verify paths and ensure atomic write operations using temporary replacement files.\n"
    "- Never write partial implementations or mock placeholders when modifying user code.\n\n"

    "## Quality Parity and Adaptive Execution\n"
    "- This agent operates under strict quality parity constraints: Muscle Memory is used exclusively "
    "when equivalent quality to Full Brain is mathematically or deterministically assured.\n"
    "- Full Brain mode is engaged whenever tasks require multi-step reasoning, complex file operations, "
    "live external data, or deep structural analysis.\n"
    "- The agent maintains an ExecutionState tracker recording completed steps, observations, and tool outcomes. "
    "If a task transitions between execution modes, a compact continuation handoff block (< 400 tokens) is "
    "transferred to avoid redundant re-discovery and prevent lost context.\n"
    "- Tool execution errors must be diagnosed systematically. If a tool fails consecutively, do not repeat "
    "the same call blindly; reassess parameters, verify file permissions, or propose an alternative strategy.\n"
    "- Procedural memory learns successful execution patterns from verified Full Brain completions. Maintain "
    "clean, self-contained procedures and adhere to deterministic verification standards.\n"
    "- Session statistics record turn counts, token expenditure, and prompt cache hit rates. Never degrade "
    "solution quality to conserve tokens—Rule R1 requires that efficiency is achieved strictly through intelligent "
    "routing and caching, never through degraded responses.\n\n"

    "## Operational Constraints and Invariants\n"
    "- Maintain backward compatibility across interfaces and preserve existing public signatures.\n"
    "- Adhere to the principle of least privilege in tool execution: only invoke tools when required for the task.\n"
    "- Ensure that every tool result is verified before declaring task completion.\n"
    "- When presenting results to the user, distinguish between verified facts and model inferences.\n"
    "- Preserve deterministic formatting for machine-readable outputs, including JSON payloads and structured error blocks.\n"
    "- This system prompt is intentionally structured and padded to exceed the 1,024-token prompt caching activation "
    "threshold to guarantee that prompt caching activates on every Full Brain turn and prefix tokens are not re-billed unnecessarily."
)


def _to_openai_tool(t: ToolDef) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": t.name,
        "description": t.description,
        "parameters": t.parameters,
    }


def _extract_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    parts = []
    for item in getattr(resp, "output", []):
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []):
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", ""))
    return "".join(parts)


def _safe_responses_create(client, **kwargs):
    """Executes responses.create with graceful degradation if optional kwargs like reasoning are unsupported."""
    if "max_output_tokens" in kwargs and kwargs["max_output_tokens"] < 16:
        kwargs["max_output_tokens"] = 16
    try:
        return client.responses.create(**kwargs)
    except Exception as e:
        err_str = str(e)
        err_lower = err_str.lower()
        if "reasoning" in kwargs and ("reasoning" in err_lower or "unsupported parameter" in err_lower):
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("reasoning", None)
            try:
                return client.responses.create(**fallback_kwargs)
            except Exception as e2:
                err_str2 = str(e2)
                if "credit_balance_exhausted" in err_str2 or "insufficient_quota" in err_str2:
                    raise RuntimeError(
                        f"[OPENAI QUOTA EXHAUSTED] Your OpenAI API key has no remaining credits (HTTP 429 credit_balance_exhausted). "
                        f"Please add credits at https://platform.openai.com/settings/organization/billing/ to run live validation on {kwargs.get('model', 'gpt-4o')}."
                    ) from e2
                raise
        if "credit_balance_exhausted" in err_str or "insufficient_quota" in err_str:
            raise RuntimeError(
                f"[OPENAI QUOTA EXHAUSTED] Your OpenAI API key has no remaining credits (HTTP 429 credit_balance_exhausted). "
                f"Please add credits at https://platform.openai.com/settings/organization/billing/ to run live validation on {kwargs.get('model', 'gpt-4o')}."
            ) from e
        raise


class OpenAIProvider(ModelProvider):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        client: Optional[Any] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if client is not None:
            self.client = client
        elif OpenAI is None:
            raise ImportError(
                "The 'openai' package is not installed. "
                "Install it with: pip install 'openai>=1.50.0'"
            )
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            init_kwargs: Dict[str, Any] = {"api_key": key}
            if base_url:
                init_kwargs["base_url"] = base_url
            self.client = OpenAI(**init_kwargs)
        self.model = model

    # ---------------- classification ----------------
    def classify(self, user_message: str, recent_summary: str) -> str:
        system = (
            "You are a strict single-word classifier.\n"
            "muscle_memory: greetings, acknowledgments, small talk, simple "
            "factual questions needing no tools, files, or code.\n"
            "full_brain: anything needing code, file operations, review, "
            "multi-step reasoning, or tool use.\n"
            "If genuinely unsure, answer full_brain.\n"
            "Reply with exactly one word: muscle_memory or full_brain."
        )
        input_items = []
        if recent_summary.strip():
            # Non-conversational context preamble: inject as structured prefix
            # on the user message. NEVER as a fabricated assistant turn.
            input_items.append({"role": "user", "content": (
                f"[CONTEXT FOR CLASSIFICATION — do not respond to this, "
                f"use it only to inform your classification decision]\n"
                f"{recent_summary}\n\n"
                f"[MESSAGE TO CLASSIFY]\n{user_message}"
            )})
        else:
            input_items.append({"role": "user", "content": user_message})

        resp = with_retry(lambda: _safe_responses_create(
            self.client,
            model=self.model,                 # SAME model
            instructions=system,
            input=input_items,
            reasoning={"effort": "low"},      # minimal effort
            max_output_tokens=16,
        ))
        decision = _extract_text(resp).strip().lower()
        if "muscle_memory" in decision and "full_brain" not in decision:
            return "muscle_memory"
        return "full_brain"

    def route_classify(self, user_message: str, recent_summary: str) -> Dict[str, Any]:
        """Queries the same model with effort=low requesting a structured JSON RouteDecision."""
        system = (
            "You are a strict multi-axis task routing engine. Analyze the incoming user request "
            "and classify it across orthogonal dimensions.\n\n"
            "Respond ONLY with a valid JSON object adhering strictly to this schema:\n"
            "{\n"
            '  "pathway": "muscle" | "full_brain",\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "context_required": <boolean: true if user refers to prior turns/decisions/files>,\n'
            '  "tools_required": <boolean: true if tool/file operations or live execution is needed>,\n'
            '  "freshness_required": <boolean: true if current/live external facts are required>,\n'
            '  "novelty": <float 0.0-1.0: 0.0=routine/familiar, 1.0=unprecedented/complex>,\n'
            '  "risk": "low" | "medium" | "high",\n'
            '  "reasoning": "<concise single-sentence justification>"\n'
            "}\n\n"
            "Rules:\n"
            "- pathway: 'muscle' ONLY for simple greetings, acknowledgments, basic math/trivia needing NO tools.\n"
            "- pathway: 'full_brain' for code, file ops, debugging, multi-step tasks, or high-risk actions.\n"
            "- If unsure, choose 'full_brain'. Output raw JSON only."
        )
        input_items = []
        if recent_summary.strip():
            input_items.append({"role": "user", "content": (
                f"[CONTEXT FOR ROUTING — do not respond to this, use only to inform classification]\n"
                f"{recent_summary}\n\n"
                f"[MESSAGE TO ROUTE]\n{user_message}"
            )})
        else:
            input_items.append({"role": "user", "content": user_message})

        resp = with_retry(lambda: _safe_responses_create(
            self.client,
            model=self.model,
            instructions=system,
            input=input_items,
            reasoning={"effort": "low"},
            max_output_tokens=250,
            text={"format": {"type": "json_object"}},
        ))
        raw_text = _extract_text(resp).strip()
        u = resp.usage
        details = getattr(u, "input_tokens_details", None)
        cache_read = getattr(details, "cached_tokens", 0) or 0
        provider_resp = ProviderResponse(
            text=raw_text,
            tool_calls=[],
            stop_reason="end_turn",
            raw=resp,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=cache_read,
        )

        import json
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and "pathway" in parsed:
                parsed["_response"] = provider_resp
                return parsed
        except Exception:
            pass
        fallback = super().route_classify(user_message, recent_summary)
        fallback["_response"] = provider_resp
        return fallback

    # ---------------- muscle memory ----------------
    def muscle_memory_reply(self, user_message: str) -> ProviderResponse:
        system = (
            "You are a concise assistant. Reply in 1-3 short sentences. "
            "Never write code or reference files in this mode."
        )
        # AGGRESSIVE CONTEXT REDUCTION: bare user message only
        resp = with_retry(lambda: _safe_responses_create(
            self.client,
            model=self.model,                 # SAME model
            instructions=system,
            input=[{"role": "user", "content": user_message}],
            reasoning={"effort": "low"},      # minimal effort
            max_output_tokens=120,
        ))
        text = _extract_text(resp)
        u = resp.usage
        details = getattr(u, "input_tokens_details", None)
        cache_read = getattr(details, "cached_tokens", 0) or 0
        return ProviderResponse(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            raw=resp,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            cache_write_tokens=0,
        )

    # ---------------- full brain ----------------
    def full_brain_step(self, history: List[Any], tools: List[ToolDef], effort: str) -> ProviderResponse:
        openai_tools = [_to_openai_tool(t) for t in tools]

        resp = with_retry(lambda: _safe_responses_create(
            self.client,
            model=self.model,                 # SAME model
            instructions=FULL_BRAIN_INSTRUCTIONS,
            input=history,
            tools=openai_tools,
            reasoning={"effort": effort},
            prompt_cache_key="dual-mode-agent-session",
        ))

        tool_calls = []
        for item in resp.output:
            if item.type == "function_call":
                tool_calls.append(ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=json.loads(item.arguments or "{}"),
                ))

        text = _extract_text(resp)
        u = resp.usage
        details = getattr(u, "input_tokens_details", None)
        cache_read = getattr(details, "cached_tokens", 0) or 0
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            raw=resp,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            native_assistant_turn=[
                item.model_dump()
                for item in resp.output
                if getattr(item, "type", None) in {"message", "function_call"}
            ],
        )

    # ---------------- summary compressor ----------------
    def summarize_history_step(self, history_text: str) -> ProviderResponse:
        system = (
            "Compress the following conversation excerpt into 3-5 concise "
            "sentences capturing the key topics discussed, decisions made, "
            "and any important context. Omit greetings and filler."
        )
        resp = with_retry(lambda: _safe_responses_create(
            self.client,
            model=self.model,                 # SAME model, effort=low
            instructions=system,
            input=[{"role": "user", "content": history_text}],
            reasoning={"effort": "low"},
            max_output_tokens=300,
        ))
        text = _extract_text(resp).strip()
        u = resp.usage
        details = getattr(u, "input_tokens_details", None)
        cache_read = getattr(details, "cached_tokens", 0) or 0
        return ProviderResponse(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            raw=resp,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=cache_read,
        )

    def summarize_history(self, history_text: str) -> str:
        return self.summarize_history_step(history_text).text

    def append_assistant_turn(self, history: List[Any], resp: ProviderResponse) -> None:
        if resp.native_assistant_turn:
            history.extend(resp.native_assistant_turn)
        else:
            turn: Dict[str, Any] = {"role": "assistant", "content": resp.text}
            if resp.tool_calls:
                turn["tool_calls"] = [
                    {
                        "type": "function",
                        "id": tc.id,
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in resp.tool_calls
                ]
            history.append(turn)

    def append_tool_results(self, history: List[Any], tool_calls: List[ToolCall], results: Dict[str, str]) -> None:
        for tc in tool_calls:
            history.append({
                "type": "function_call_output",
                "call_id": tc.id,
                "output": results.get(tc.id, ""),
            })

    def append_user_turn(self, history: List[Any], text: str) -> None:
        history.append({"role": "user", "content": text})


class OpenRouterProvider(OpenAIProvider):
    name = "openrouter"

    def __init__(
        self,
        model: str = "meta-llama/llama-3.3-70b-instruct",
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
    ):
        key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        super().__init__(
            model=model,
            client=client,
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        self.name = "openrouter"

