# V5 Realistic Trace Examples

This document provides realistic, production traces of the Adaptive Dual-Path Procedural / Deliberative Agent (V5).

---

## 1. Procedural Memory Records

### 1.1 Candidate Skill (Unproven)
```json
{
  "id": "learned_csv_to_json_parser",
  "name": "Procedure for convert csv to json",
  "semantic_intent": "Parse tabular comma-separated values into JSON objects",
  "preconditions": ["Input contains valid comma-separated text with header row"],
  "non_applicability_conditions": ["Input is not valid CSV", "Empty input"],
  "inputs": [{"name": "raw_csv", "type": "string", "description": "Raw CSV text"}],
  "steps": [
    "Identify header line",
    "Split rows on delimiter",
    "Construct dictionary objects mapping keys to values",
    "Serialize to JSON format"
  ],
  "procedure": "deterministic_csv_parser",
  "verification": "structural",
  "required_tools": [],
  "version": "1.0",
  "status": "candidate",
  "confidence": 0.5,
  "success_count": 1,
  "verified_success_count": 1,
  "failure_count": 0,
  "consecutive_failures": 0,
  "distinct_variants": ["a1b2c3d4"],
  "distinct_variant_count": 1
}
```

### 1.2 Trusted Skill (Proven on >=3 successes)
```json
{
  "id": "deterministic_json_validator",
  "name": "JSON Structural Validator",
  "semantic_intent": "Validate and format JSON documents",
  "preconditions": ["Input contains JSON or markdown fenced JSON block"],
  "non_applicability_conditions": ["Non-JSON plain text"],
  "steps": [
    "Extract candidate JSON block using regex",
    "Parse with python json.loads",
    "Return formatted item count and schema summary"
  ],
  "procedure": "deterministic_json_validation",
  "verification": "deterministic",
  "required_tools": [],
  "version": "1.0",
  "status": "trusted",
  "confidence": 0.8,
  "success_count": 4,
  "verified_success_count": 4,
  "failure_count": 0,
  "consecutive_failures": 0,
  "distinct_variants": ["f1a2", "b3c4", "e5d6"],
  "distinct_variant_count": 3
}
```

### 1.3 Eligible Skill (Eligible for automated Tier 2 fast path)
```json
{
  "id": "deterministic_text_statistics",
  "name": "Text Statistics Counter",
  "semantic_intent": "Count words, characters, and lines in a text snippet",
  "preconditions": ["Input requests character, word, or line count"],
  "non_applicability_conditions": ["Code execution or translation requests"],
  "steps": [
    "Strip prompt prefixes",
    "Calculate character, word, and line count",
    "Format summary markdown"
  ],
  "procedure": "deterministic_text_statistics",
  "verification": "deterministic",
  "required_tools": [],
  "version": "1.0",
  "status": "eligible",
  "confidence": 1.0,
  "success_count": 12,
  "verified_success_count": 12,
  "failure_count": 0,
  "consecutive_failures": 0,
  "distinct_variants": ["0a1b", "2c3d", "4e5f", "6a7b", "8c9d"],
  "distinct_variant_count": 5
}
```

---

## 2. Muscle Success Trace (Tier 1 Deterministic Execution)

```json
{
  "task_id": "7f8b9a1c",
  "objective": "statistics of: The quick brown fox jumps over the lazy dog.",
  "pathway": "muscle",
  "skill_used": "deterministic_text_statistics",
  "terminal_state": "COMPLETED_VERIFIED",
  "result_text": "Text Statistics:\n- Words: 9\n- Characters: 44\n- Lines: 1",
  "verification_record": {
    "method": "level_0_invariant",
    "passed": true,
    "confidence": 1.0,
    "evidence": "Deterministic result verified (49 chars)",
    "level": 0
  },
  "usage": {
    "routing_tokens": 0,
    "muscle_tokens": 0,
    "brain_tokens": 0,
    "verification_tokens": 0,
    "total_billable_tokens": 0,
    "model_calls": 0,
    "source": "simulated"
  }
}
```

---

## 3. Muscle -> Brain Continuation Handoff Trace

### Step A: Muscle Encountered Uncertainty
```json
{
  "task_id": "9e2c4d8a",
  "objective": "audit system configuration and write summary to audit.log",
  "pathway": "muscle",
  "status": "HANDOFF_REQUIRED",
  "reason": "Skill 'system_audit' requires tool access; routing to Full Brain.",
  "execution_state": {
    "verified_facts": [
      "Target log file is audit.log in workspace root"
    ],
    "completed_steps": [
      "Parsed audit request parameters"
    ],
    "unresolved": [
      "Skill 'system_audit' requires tools ['read_file', 'write_file'] — unavailable in Muscle mode"
    ],
    "decisions": [
      {"decision": "Escalate to Full Brain", "why": "Tool execution required"}
    ]
  }
}
```

### Step B: Rendered Continuation Handoff Prompt
```text
[CONTINUATION HANDOFF FROM MUSCLE MEMORY]
The previous execution phase encountered a boundary condition requiring deliberative reasoning.
Continue from the established execution state below. Do NOT restart or duplicate completed work.

Objective: audit system configuration and write summary to audit.log

Verified Facts (ESTABLISHED - DO NOT RE-EVALUATE):
- [VERIFIED] Target log file is audit.log in workspace root

Completed Steps:
- Parsed audit request parameters

Unresolved Items (ACTION REQUIRED):
- Skill 'system_audit' requires tools ['read_file', 'write_file'] — unavailable in Muscle mode

CONTINUATION DIRECTIVE:
1. Treat all [VERIFIED] facts above as confirmed ground truth.
2. Do not repeat verified completed work or re-run successful tools.
3. Focus exclusively on the Unresolved Items and completing the Objective.
```

### Step C: Deliberative Brain Continuation
```json
{
  "task_id": "9e2c4d8a",
  "resumed_from": "muscle_handoff",
  "tool_calls_executed": [
    {"name": "write_file", "arguments": {"path": "audit.log", "content": "AUDIT: System configuration verified clean."}}
  ],
  "recomputation_count": 0,
  "final_text": "Completed audit and wrote report to audit.log.",
  "terminal_state": "COMPLETED_VERIFIED"
}
```

---

## 4. Brain -> Muscle Learning Trace

```json
{
  "event": "procedure_extraction",
  "source_task_id": "4b6a8d0e",
  "query": "what is the capital of australia",
  "brain_execution": {
    "effort": "low",
    "tokens": 1280,
    "reply": "The capital of Australia is Canberra."
  },
  "verification": {
    "passed": true,
    "method": "consistency",
    "level": 3
  },
  "candidate_registered": {
    "skill_id": "learned_e8b9f1a2c3d4",
    "name": "Procedure for what is the capital of australia",
    "trigger_patterns": [
      "^\\s*what is the capital of australia\\s*[!.?]*\\s*$",
      ".*\\bcapital\\b.*\\baustralia\\b.*"
    ],
    "procedure": "The capital of Australia is Canberra.",
    "status": "candidate",
    "confidence": 0.5
  },
  "lifecycle": {
    "success_1": "reinforce(input_variant='what is the capital of australia?') -> verified=1",
    "success_2": "reinforce(input_variant='what is the capital of australia') -> verified=2",
    "success_3": "reinforce(input_variant='capital of australia') -> verified=3 -> PROMOTED TO 'trusted'",
    "success_10": "reinforce(10 verified across 4 variants) -> PROMOTED TO 'eligible'"
  }
}
```
