# Adaptive Dual-Mode Agent — v4 Critical Review & Forward Plan

This supersedes the earlier IMPLEMENTATION_PLAN.md in this project. It covers three
things: (1) what I actually verified by running your code, not just reading it,
(2) the bugs I found and fixed directly in this delivered copy, and (3) a complete,
concrete prompt for everything still remaining — organized so an agent (or you) can
work through it without re-litigating settled design questions.

**Headline verdict:** this is a real, substantially-built v4 — not a rewrite of the
conceptual discussion into prose. The router, execution-state/handoff mechanism,
procedural-memory ladder, verification tiers, and adversarial test suite all exist as
actual code, and I ran every test file myself rather than trusting file names.
Ten of the previous review's specific numbered complaints are now demonstrably fixed
(see S1). But two structurally important pieces were scaffolding without wiring
(S2), and the newest benchmark script reintroduced the exact "circular mock"
problem that had already been called out once (S2.2) - I fixed both directly in
the copy you're getting back.

---

## 1. What I verified is actually fixed (ran, not assumed)

| # | Issue from the prior review | Status | Evidence |
|---|---|---|---|
| 1 | Cache would return stale answers for "yes", "continue", "ok" regardless of context | Fixed | base.py's ResponseCache now uses an explicit ALLOW_LIST/DENY_LIST instead of broad synonym collapse. test_phase0_safety.py::test_cache_safety_allow_list and test_routing_adversarial.py::test_cache_safety_negative_assertions both pass - I ran them. |
| 2 | Binary muscle/brain routing ignored context-dependency, freshness, risk | Fixed | router.py's RouteDecision carries context_required, freshness_required, risk, tools_required, novelty as independent axes, with a 0-token deterministic gate before any model call. test_routing_adversarial.py exercises all of these - passes. |
| 3 | "Yes" after "should I delete the file?" would hit cache or muscle | Fixed | test_contextual_ack_after_destructive_prompt specifically tests this exact scenario and passes. |
| 4 | Synthetic assistant: "Understood." fake turn distorted history | Fixed | full_brain_turn() now injects context as a structured preamble on the real user turn ([CONTEXT FROM PRIOR TURNS]), never a fake assistant reply. test_no_synthetic_assistant_turn passes. |
| 5 | History/context limits were item-count based, not token-based | Fixed | ConversationState now has max_recent_tokens and full_brain_context_budget, both enforced in add_plain_turn()/reset_full_brain_if_too_long(). test_conversation_state_token_budget passes. |
| 6 | Summary compression didn't actually replace/shrink old history | Fixed | apply_summary_truncation() truncates full_brain_history to a tail after compressing. test_summary_replacement_truncates_history passes. |
| 7 | write_file silently overwrote, no atomicity | Fixed | Requires explicit overwrite=True; writes via tempfile + os.replace. test_safe_and_atomic_write_file passes. |
| 8 | Token accounting double-counted cache tokens (Anthropic vs OpenAI semantics differ) | Still correct | The normalized contract from the last round (input_tokens always = total, cache fields are subsets) is intact and tested in test_stats_hit_rate_math_and_cost. |
| 9 | Benchmark used one favorable 12-turn workload and presented savings unqualified | Fixed | eval_baseline.py now runs three labeled workload distributions (casual/balanced/adversarial) and prints [SIMULATED] on every run. I ran it: 82.7% / 56.7% / 23.6% token savings respectively - the honest range, not one cherry-picked number. |
| 10 | Classifier eval didn't report the safety-relevant metric | Fixed | eval_classifier.py reports "Dangerous False Negatives" as a headline number, not just aggregate accuracy. I ran it: 0 dangerous false negatives on the 60-case set (dry-run mode). |

---

## 2. What I found still broken, and fixed directly

### 2.1 The procedural-memory learning loop was dead code
procedural_memory.py had a complete promotion/demotion ladder (reinforce()/
penalize(), candidate -> trusted -> eligible) - but register_candidate() was
NEVER called from anywhere in the codebase. I grepped to confirm: zero call sites
outside its own definition. On top of that, muscle_engine.py's only call to
retrieve() hardcoded only_trusted=True, so even a manually-registered candidate
could never be matched, tried, or promoted.

Fixed: dual_mode_agent.py now registers a candidate skill after any
Full-Brain-via-handoff turn that was low-risk and tool-free, and muscle_engine.py
now allows candidates to be tried on low-risk requests
(only_trusted=(decision.risk != "low")).

### 2.2 eval_restart_vs_resume.py reintroduced the exact "circular mock" problem
Fixed: both conditions now derive their cost from one shared formula
(_cost_from_history, ~4 chars/token over the actual accumulated history text), so
any difference between conditions comes from the history actually being shorter in
the resume case, not from a human deciding the punchline in advance. Re-run: 51.3%
savings (derived, not hand-picked).

### 2.3 Minor hygiene
- requirements.txt cleaned up.
- session_turns.jsonl source tracking: added source: "mock"|"live" field.

---

## 3. What's genuinely still open

A. Live-API validation against real models.
B. Real uncertainty detection in Tier 3 (beyond downstream verification).
C. Generalized procedure extraction from successful trajectories.
D. Quality/correctness evaluation (tokens-per-correctly-completed-task, LLM-as-a-judge, routing confusion matrix).
