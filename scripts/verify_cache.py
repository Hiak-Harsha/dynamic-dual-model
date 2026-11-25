"""
scripts/verify_cache.py — Live prompt-caching verification script.

Makes two sequential Full Brain calls with the same static prefix (padded system
prompt + tools) and asserts that cache_read_tokens > 0 on the second turn.

Usage:
  python scripts/verify_cache.py --live --provider anthropic   # live test against Anthropic API
  python scripts/verify_cache.py --live --provider openai      # live test against OpenAI API
  python scripts/verify_cache.py --mock                        # tests harness with MockProvider
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ModelProvider
from tools import TOOLS


def verify_cache(provider: ModelProvider, is_live: bool = False):
    print(f"\nVerifying Prompt Caching on {provider.name.upper()} ({provider.model})...")
    print("=" * 65)

    history = []
    
    # ---------------- Turn 1 ----------------
    prompt1 = "Explain the function naming and docstring rules from your coding style guide."
    print(f"[Turn 1] Sending prompt: \"{prompt1}\"")
    provider.append_user_turn(history, prompt1)
    
    resp1 = provider.full_brain_step(history, TOOLS, effort="low")
    provider.append_assistant_turn(history, resp1)
    
    print(f"  -> Total input tokens   : {resp1.input_tokens}")
    print(f"  -> Cache write tokens  : {resp1.cache_write_tokens}")
    print(f"  -> Cache read tokens   : {resp1.cache_read_tokens}")
    print(f"  -> Output tokens       : {resp1.output_tokens}")

    # ---------------- Turn 2 ----------------
    prompt2 = "Now explain the error handling and debugging rules from the same guidelines."
    print(f"\n[Turn 2] Sending prompt: \"{prompt2}\"")
    provider.append_user_turn(history, prompt2)
    
    resp2 = provider.full_brain_step(history, TOOLS, effort="low")
    provider.append_assistant_turn(history, resp2)
    
    print(f"  -> Total input tokens   : {resp2.input_tokens}")
    print(f"  -> Cache write tokens  : {resp2.cache_write_tokens}")
    print(f"  -> Cache read tokens   : {resp2.cache_read_tokens}")
    print(f"  -> Output tokens       : {resp2.output_tokens}")

    cache_read = resp2.cache_read_tokens
    total_input = resp2.input_tokens
    hit_rate = (cache_read / total_input) if total_input else 0.0

    print("\n" + "=" * 65)
    print(f"Turn 2 Cache Hit Rate: {hit_rate:.1%} ({cache_read}/{total_input} tokens)")

    # Save telemetry to results/
    os.makedirs("results", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = f"results/cache_verification_{ts}.json"
    result_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider.name,
        "model": provider.model,
        "is_live": is_live,
        "turn_1": {
            "input_tokens": resp1.input_tokens,
            "cache_write_tokens": resp1.cache_write_tokens,
            "cache_read_tokens": resp1.cache_read_tokens,
            "output_tokens": resp1.output_tokens,
        },
        "turn_2": {
            "input_tokens": resp2.input_tokens,
            "cache_write_tokens": resp2.cache_write_tokens,
            "cache_read_tokens": resp2.cache_read_tokens,
            "output_tokens": resp2.output_tokens,
        },
        "cache_hit_rate_pct": round(hit_rate * 100, 2),
        "status": "passed" if cache_read > 0 else "failed",
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n[INFO] Cache verification results saved to: {out_file}")

    if cache_read > 0:
        print("[SUCCESS] Prompt caching verified! cache_read_tokens > 0 observed.")
        print(f"The static prefix (~1,200 tokens) successfully crossed the ~1,024-token cache floor.")
        return True
    else:
        print("[FAILURE] cache_read_tokens is 0 on Turn 2.")
        if is_live:
            print("Troubleshooting tips:")
            print("1. Ensure system prompt + tool schemas exceed 1,024 tokens.")
            print("2. Ensure TOOLS order is completely static (already sorted in tools.py).")
            print("3. Ensure cache_control block is placed on the last tool.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--live", action="store_true", default=False,
                        help="Run against real provider API")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="Run with deterministic mock provider")
    args = parser.parse_args()

    if args.mock or not args.live:
        from tests.mock_provider import MockProvider
        mock = MockProvider(tool_rounds=0)
        verify_cache(mock, is_live=False)
    elif args.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("\n[ERROR] ANTHROPIC_API_KEY environment variable is not set.")
            print("Set it before running --live:")
            print("  Windows CMD: set ANTHROPIC_API_KEY=sk-ant-...")
            print("  PowerShell : $env:ANTHROPIC_API_KEY=\"sk-ant-...\"")
            print("  Linux/Mac  : export ANTHROPIC_API_KEY=sk-ant-...\n")
            sys.exit(1)
        from anthropic_provider import AnthropicProvider
        provider = AnthropicProvider()
        verify_cache(provider, is_live=True)
    elif args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n[ERROR] OPENAI_API_KEY environment variable is not set.")
            print("Set it before running --live:")
            print("  Windows CMD: set OPENAI_API_KEY=sk-...")
            print("  PowerShell : $env:OPENAI_API_KEY=\"sk-...\"")
            print("  Linux/Mac  : export OPENAI_API_KEY=sk-...\n")
            sys.exit(1)
        from openai_provider import OpenAIProvider
        provider = OpenAIProvider()
        verify_cache(provider, is_live=True)
