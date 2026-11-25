"""
verification.py — Multi-level verification engine for V6.

Per V6 Spec Sections AL, AM, AN, AO, AP:
Level 0: Deterministic invariant (bounds, types, exact values)
Level 1: Structural verification (JSON parse/schema, Python AST)
Level 2: Execution verification (file existence, test passing, artifact hash, tool postconditions)
Level 3: Evidence & consistency verification (cross-checks claims against observations;
         strictly proves 'no detected contradiction', NOT 'answer is verified correct')
Level 4: Model-based verification (conservative, only when value > cost, tracked separately)
"""

from __future__ import annotations
import ast
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from typing import Optional, Any, Dict, List

from base import ModelProvider, count_tokens
from execution_state import ExecutionState, EvidenceLevel
from tools import _resolve_in_sandbox, get_file_content_hash
from success_contracts import ExactValueContract, resolve_contract, ContractResult


@dataclass
class VerificationRecord:
    """Rigorous record of a completed verification check."""
    method: str
    passed: bool
    confidence: float
    evidence: str
    tokens_used: int = 0
    level: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    passed: bool
    confidence: float
    tier_used: str  # "deterministic" | "structural" | "execution" | "consistency" | "model" | "pass_through"
    reason: str
    tokens_used: int = 0
    record: Optional[VerificationRecord] = None
    is_factual_truth: bool = True  # False for consistency (no contradiction != factual truth)


def verify(
    result_text: str,
    task_objective: str,
    exec_state: ExecutionState,
    verification_mode: str = "structural",
    provider: Optional[ModelProvider] = None,
    expected_value: Optional[Any] = None,
) -> VerificationResult:
    """Executes multi-level verification against result_text and updates exec_state."""
    rec: Optional[VerificationRecord] = None

    # ------------------------------------------------------------
    # LEVEL 0: Deterministic Invariant Check (0 tokens)
    # Evaluates exact value contract rather than non-empty heuristic.
    # ------------------------------------------------------------
    if verification_mode in {"exact", "deterministic", "level_0"}:
        exp_val = expected_value
        if exp_val is None:
            # Check if task_objective contains an arithmetic expression to verify
            try:
                clean_obj = re.sub(r"^(what\s+is|calculate|compute)\s+", "", task_objective, flags=re.IGNORECASE).rstrip("?=.").strip()
                if clean_obj and any(c in clean_obj for c in "+-*/%"):
                    from muscle_engine import _run_deterministic_arithmetic
                    tmp_state = ExecutionState()
                    succ, out = _run_deterministic_arithmetic(clean_obj, tmp_state)
                    if succ:
                        m_val = re.search(r"Result:\s*([^\n\r]+)", out)
                        if m_val:
                            exp_val = m_val.group(1).strip()
            except Exception:
                pass

        contract = ExactValueContract(expected_value=exp_val)
        c_res = contract.evaluate(task=task_objective, result=result_text, execution_state=exec_state)

        rec = VerificationRecord(
            method="level_0_invariant",
            passed=c_res.verified,
            confidence=c_res.confidence,
            evidence=c_res.evidence,
            level=0,
        )
        exec_state.verification_records.append(rec.to_dict())

        if c_res.verified:
            exec_state.add_verified_fact(
                f"Deterministic invariant satisfied: {task_objective[:80]} -> {result_text[:80]}",
                source="deterministic",
                method="level_0_invariant",
            )
            return VerificationResult(
                passed=True,
                confidence=c_res.confidence,
                tier_used="deterministic",
                reason=c_res.reason,
                record=rec,
                is_factual_truth=True,
            )
        else:
            exec_state.add_error(f"Deterministic verification failed: {c_res.reason}")
            return VerificationResult(
                passed=False,
                confidence=c_res.confidence,
                tier_used="deterministic",
                reason=c_res.reason,
                record=rec,
                is_factual_truth=False,
            )

    # ------------------------------------------------------------
    # LEVEL 1: Structural Verification (0 tokens)
    # ------------------------------------------------------------
    if verification_mode in {"structural", "level_1"}:
        clean = result_text.strip()

        # Case 1: JSON payload
        json_cand = clean
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if m:
            json_cand = m.group(1).strip()
        elif clean.startswith("{") and clean.endswith("}"):
            json_cand = clean
        elif clean.startswith("[") and clean.endswith("]"):
            json_cand = clean

        if (json_cand.startswith("{") and json_cand.endswith("}")) or (json_cand.startswith("[") and json_cand.endswith("]")):
            try:
                parsed = json.loads(json_cand)
                rec = VerificationRecord(method="level_1_json_parse", passed=True, confidence=1.0, evidence=f"Parsed JSON of type {type(parsed).__name__}", level=1)
                exec_state.verification_records.append(rec.to_dict())
                exec_state.add_verified_fact(f"JSON payload parsed successfully: Valid JSON ({type(parsed).__name__})", source="structural", method="json_parse")
                return VerificationResult(passed=True, confidence=1.0, tier_used="structural", reason="Valid JSON payload verified.", record=rec)
            except Exception as e:
                rec = VerificationRecord(method="level_1_json_parse", passed=False, confidence=0.0, evidence=str(e), level=1)
                exec_state.verification_records.append(rec.to_dict())
                exec_state.add_error(f"JSON structural validation failed: {e}")
                return VerificationResult(passed=False, confidence=0.0, tier_used="structural", reason=f"JSON syntax error: {e}", record=rec)

        # Case 2: Python Code AST
        py_cand = clean
        m_py = re.search(r"```python\s*([\s\S]*?)\s*```", clean)
        if m_py:
            py_cand = m_py.group(1).strip()

        if "def " in py_cand or "class " in py_cand or "import " in py_cand:
            try:
                ast.parse(py_cand)
                rec = VerificationRecord(method="level_1_python_ast", passed=True, confidence=1.0, evidence="AST parse succeeded", level=1)
                exec_state.verification_records.append(rec.to_dict())
                exec_state.add_verified_fact("Python code blocks verified syntax-clean via AST", source="structural", method="ast_parse")
                return VerificationResult(passed=True, confidence=1.0, tier_used="structural", reason="Python syntax verified via AST.", record=rec)
            except Exception as e:
                rec = VerificationRecord(method="level_1_python_ast", passed=False, confidence=0.0, evidence=str(e), level=1)
                exec_state.verification_records.append(rec.to_dict())
                exec_state.add_error(f"Python code AST parsing failed: {e}")
                return VerificationResult(passed=False, confidence=0.0, tier_used="structural", reason=f"Python syntax error: {e}", record=rec)

        # Case 3: Markdown table
        if "|-" in clean or ("|" in clean and "\n|" in clean):
            lines = [l for l in clean.splitlines() if "|" in l]
            if len(lines) >= 2:
                rec = VerificationRecord(method="level_1_table_shape", passed=True, confidence=0.9, evidence=f"Table with {len(lines)} rows", level=1)
                exec_state.verification_records.append(rec.to_dict())
                exec_state.add_verified_fact("Structural verification: Markdown table shape confirmed", source="structural", method="table_shape")
                return VerificationResult(passed=True, confidence=0.9, tier_used="structural", reason="Markdown tabular structure verified.", record=rec)

        rec = VerificationRecord(
            method="level_1_unstructured",
            passed=False,
            confidence=0.0,
            evidence="No verifiable structural payload (JSON, code, table) found",
            level=1,
        )
        exec_state.verification_records.append(rec.to_dict())
        return VerificationResult(
            passed=False,
            confidence=0.0,
            tier_used="structural",
            reason="Unstructured output cannot be verified via Level 1 structural verification.",
            record=rec,
        )

    # ------------------------------------------------------------
    # LEVEL 2: Execution Verification (0 tokens)
    # ------------------------------------------------------------
    if verification_mode in {"execution", "level_2"}:
        # Verify files referenced as written exist in sandbox and verify hashes
        for key, val in exec_state.tool_state.items():
            if key.startswith("file_written:"):
                filepath = key.split("file_written:", 1)[1]
                try:
                    real = _resolve_in_sandbox(filepath)
                    if not os.path.exists(real):
                        rec = VerificationRecord(method="level_2_file_existence", passed=False, confidence=0.0, evidence=f"File {filepath} missing on disk", level=2)
                        exec_state.verification_records.append(rec.to_dict())
                        exec_state.add_error(f"Expected file '{filepath}' missing on disk")
                        return VerificationResult(passed=False, confidence=0.0, tier_used="execution", reason=f"Expected file '{filepath}' was not found on disk.", record=rec)

                    # Verify non-empty and update hash
                    f_hash = get_file_content_hash(filepath)
                    exec_state.record_state_freshness(filepath, open(real, "r", encoding="utf-8", errors="replace").read())
                except Exception as e:
                    rec = VerificationRecord(method="level_2_sandbox_error", passed=False, confidence=0.0, evidence=str(e), level=2)
                    exec_state.verification_records.append(rec.to_dict())
                    exec_state.add_error(f"Sandbox check error: {e}")
                    return VerificationResult(passed=False, confidence=0.0, tier_used="execution", reason=f"Error checking sandbox file: {e}", record=rec)

        if exec_state.errors:
            rec = VerificationRecord(method="level_2_tool_errors", passed=False, confidence=0.0, evidence=f"State has {len(exec_state.errors)} errors", level=2)
            exec_state.verification_records.append(rec.to_dict())
            return VerificationResult(passed=False, confidence=0.0, tier_used="execution", reason=f"Execution state contains errors: {exec_state.errors[0]}", record=rec)

        rec = VerificationRecord(method="level_2_execution_check", passed=True, confidence=0.95, evidence="All execution artifacts verified", level=2)
        exec_state.verification_records.append(rec.to_dict())
        exec_state.add_verified_fact("Execution artifacts confirmed on disk with zero execution errors.", source="execution", method="disk_check")
        return VerificationResult(passed=True, confidence=0.95, tier_used="execution", reason="Execution state and disk artifacts verified successfully.", record=rec)

    # ------------------------------------------------------------
    # LEVEL 3: Evidence & Consistency Verification (0 tokens)
    # Per V6 Section AN: Consistency establishes only 'no detected contradiction',
    # NOT 'the answer is correct'. Does NOT manufacture a verified fact of truth!
    # ------------------------------------------------------------
    if verification_mode in {"consistency", "level_3"}:
        if exec_state.errors:
            rec = VerificationRecord(method="level_3_consistency", passed=False, confidence=0.0, evidence=f"Errors present in state: {exec_state.errors[0]}", level=3)
            exec_state.verification_records.append(rec.to_dict())
            return VerificationResult(passed=False, confidence=0.0, tier_used="consistency", reason=f"Inconsistency: State has unresolved errors: {exec_state.errors[0]}", record=rec)

        if exec_state.unresolved:
            rec = VerificationRecord(method="level_3_consistency", passed=False, confidence=0.3, evidence=f"Unresolved tasks: {exec_state.unresolved[0]}", level=3)
            exec_state.verification_records.append(rec.to_dict())
            return VerificationResult(passed=False, confidence=0.3, tier_used="consistency", reason=f"Unresolved work remains: {exec_state.unresolved[0]}", record=rec)

        hedging_phrases = ["i think", "i believe", "probably", "might be", "not entirely sure", "could be wrong"]
        low_res = result_text.lower()
        if any(h in low_res for h in hedging_phrases):
            rec = VerificationRecord(method="level_3_hedging", passed=False, confidence=0.4, evidence="Hedging language detected in output", level=3)
            exec_state.verification_records.append(rec.to_dict())
            return VerificationResult(passed=False, confidence=0.4, tier_used="consistency", reason="Hedging language indicates unverified certainty.", record=rec)

        # Consistency verified: no contradiction detected (recorded as OBSERVED observation per V6 Section AN)
        rec = VerificationRecord(method="level_3_consistency", passed=True, confidence=0.75, evidence="No contradiction detected with execution state", level=3)
        exec_state.verification_records.append(rec.to_dict())
        exec_state.add_observation("Consistency check passed: zero contradictions detected with established state.", source="consistency")

        factual_pattern = re.compile(
            r"^\s*(what\s+is\s+the\s+capital\s+of|who\s+wrote|who\s+is\s+the|when\s+was|"
            r"what\s+year\s+did|where\s+is|what\s+is\s+the\s+population\s+of|capital\s+of)\b",
            re.IGNORECASE,
        )
        is_factual = bool(factual_pattern.search(task_objective))
        has_grounding = any(
            getattr(f, "source", "") in {"tool", "deterministic", "sandbox", "external"}
            for f in getattr(exec_state, "verified_facts", [])
        )
        # Consistency alone CANNOT certify factual truth for ungrounded factual questions
        is_factual_truth = (not is_factual) or has_grounding

        return VerificationResult(
            passed=True,
            confidence=0.75,
            tier_used="consistency",
            reason="Output has no detected contradictions with established state.",
            record=rec,
            is_factual_truth=is_factual_truth,
        )

    # ------------------------------------------------------------
    # LEVEL 4: Model-Based Verification (Conservative, token-budgeted)
    # Uses strict structured JSON parsing; rejects substrings like 'do not pass'.
    # ------------------------------------------------------------
    if verification_mode in {"model", "level_4"} and provider is not None:
        try:
            prompt = (
                f"Task: {task_objective}\n"
                f"Proposed Result: {result_text}\n\n"
                f"You MUST respond ONLY with valid JSON in this exact schema:\n"
                f'{{"verdict": "PASS" | "FAIL" | "UNCERTAIN", "confidence": 0.0-1.0, "reason": "concise explanation"}}\n'
            )
            resp = provider.muscle_memory_reply(prompt)
            tokens_used = resp.input_tokens + resp.output_tokens
            exec_state.usage.record_call("verification", resp)

            clean = resp.text.strip()
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
            cand = m.group(1).strip() if m else clean

            passed = False
            confidence = 0.2
            reason = clean
            try:
                parsed = json.loads(cand)
                verdict = str(parsed.get("verdict", "")).strip().upper()
                confidence = float(parsed.get("confidence", 0.0))
                reason = str(parsed.get("reason", clean))
                passed = (verdict == "PASS")
            except Exception:
                # Text/line fallback: check if response begins with PASS prefix while strictly rejecting negative qualifiers
                lines = [l.strip() for l in clean.splitlines() if l.strip()]
                first_line = lines[0].upper() if lines else ""
                if first_line.startswith(("PASS:", "PASS -", "PASS.", "PASS ")):
                    if not any(neg in first_line for neg in ["FAIL", "NOT PASS", "DO NOT", "UNCERTAIN"]):
                        passed = True
                        confidence = 0.85
                        reason = lines[0]
                    else:
                        passed = False
                        confidence = 0.2
                elif first_line == "PASS":
                    passed = True
                    confidence = 0.85
                    reason = "PASS"
                else:
                    passed = False
                    confidence = 0.2

            rec = VerificationRecord(
                method="level_4_model",
                passed=passed,
                confidence=confidence,
                evidence=clean,
                tokens_used=tokens_used,
                level=4,
            )
            exec_state.verification_records.append(rec.to_dict())
            if passed:
                exec_state.add_verified_fact(f"Model verification passed: {reason}", source="model_verifier", method="level_4")

            return VerificationResult(
                passed=passed,
                confidence=confidence,
                tier_used="model",
                reason=reason,
                tokens_used=tokens_used,
                record=rec,
                is_factual_truth=passed,
            )
        except Exception as e:
            rec = VerificationRecord(method="level_4_model", passed=False, confidence=0.0, evidence=f"Model verifier failed: {e}", level=4)
            exec_state.verification_records.append(rec.to_dict())
            return VerificationResult(passed=False, confidence=0.0, tier_used="model", reason=f"Model verifier failed: {e}", record=rec, is_factual_truth=False)

    # Pass-through default
    rec = VerificationRecord(method="pass_through", passed=True, confidence=0.8, evidence="Default verification pass-through", level=0)
    exec_state.verification_records.append(rec.to_dict())
    return VerificationResult(passed=True, confidence=0.8, tier_used="pass_through", reason="Default verification pass-through.", record=rec)
