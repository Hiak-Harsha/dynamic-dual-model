"""
success_contracts.py — Explicit task success contracts for V7.

Replaces generic verification heuristics with rigorous contracts exposing:
evaluate(task, result, execution_state) -> ContractResult

Separates:
- execution_completed: Did execution finish without throwing an uncaught exception?
- task_satisfied: Does the output meet the task requirement?
- verified: Is there independent, untrusted proof establishing correctness?
"""

from __future__ import annotations
import ast
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from tools import _resolve_in_sandbox, get_file_content_hash


class ContractType(str, Enum):
    EXACT_VALUE = "EXACT_VALUE"
    TEXT_STATISTICS = "TEXT_STATISTICS"
    JSON_VALID = "JSON_VALID"
    JSON_INVALID = "JSON_INVALID"
    PYTHON_SYNTAX = "PYTHON_SYNTAX"
    FILE_EXISTS = "FILE_EXISTS"
    FILE_HASH = "FILE_HASH"
    FILE_CONTENT_MATCH = "FILE_CONTENT_MATCH"
    READ_OPERATION_COMPLETED = "READ_OPERATION_COMPLETED"
    WRITE_OPERATION_COMPLETED = "WRITE_OPERATION_COMPLETED"
    TOOL_POSTCONDITION = "TOOL_POSTCONDITION"
    STRUCTURAL = "STRUCTURAL"
    EVIDENCE_REQUIRED = "EVIDENCE_REQUIRED"
    MODEL_RUBRIC = "MODEL_RUBRIC"
    NO_VERIFICATION_AVAILABLE = "NO_VERIFICATION_AVAILABLE"


@dataclass
class VerificationRequest:
    """Explicit verification request structure per V8.1 Section 5."""
    prompt: str
    target_text: str
    rubric: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractResult:
    contract_type: str
    execution_completed: bool
    task_satisfied: bool
    verified: bool
    confidence: float
    evidence: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return self.verified or self.task_satisfied

    @property
    def tier_used(self) -> str:
        ct = str(self.contract_type).upper()
        if any(w in ct for w in ["FILE", "TOOL", "EXEC", "READ", "WRITE"]):
            return "execution"
        if any(w in ct for w in ["JSON", "STRUCTURAL", "SYNTAX"]):
            return "structural"
        if any(w in ct for w in ["EXACT", "TEXT", "STATISTICS"]):
            return "deterministic"
        if "RUBRIC" in ct:
            return "model"
        return "consistency"


class SuccessContract:
    contract_type: ContractType = ContractType.NO_VERIFICATION_AVAILABLE

    def evaluate(
        self,
        task: Any,
        result: str,
        execution_state: Any,
    ) -> ContractResult:
        raise NotImplementedError("Subclasses must implement evaluate()")


class ExactValueContract(SuccessContract):
    contract_type = ContractType.EXACT_VALUE

    def __init__(self, expected_value: Optional[Any] = None):
        self.expected_value = expected_value

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        if not result or not result.strip():
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=False,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence="Empty result text",
                reason="Result is empty",
            )

        clean = result.strip()
        m = re.search(r"Result:\s*([^\n\r]+)", clean, re.IGNORECASE)
        val_str = m.group(1).strip() if m else clean

        if self.expected_value is not None:
            expected_str = str(self.expected_value).strip()
            # Try numeric comparison
            try:
                if float(val_str) == float(expected_str):
                    return ContractResult(
                        contract_type=self.contract_type.value,
                        execution_completed=True,
                        task_satisfied=True,
                        verified=True,
                        confidence=1.0,
                        evidence=f"Exact numerical match: {val_str} == {expected_str}",
                        reason="Result exactly matches expected numeric value",
                    )
            except ValueError:
                pass

            if val_str.lower() == expected_str.lower():
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=True,
                    verified=True,
                    confidence=1.0,
                    evidence=f"Exact string match: '{val_str}' == '{expected_str}'",
                    reason="Result exactly matches expected value",
                )
            else:
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=False,
                    verified=False,
                    confidence=0.0,
                    evidence=f"Value mismatch: got '{val_str}', expected '{expected_str}'",
                    reason="Result does not match expected value",
                )

        # Check for verified deterministic arithmetic provenance in execution state
        has_deterministic_proof = False
        if execution_state is not None:
            completed = str(getattr(execution_state, "completed_steps", []))
            decisions = getattr(execution_state, "decisions", [])
            meta = getattr(execution_state, "metadata", {})
            if (
                "deterministic_arithmetic" in completed
                or "ast" in completed
                or meta.get("is_deterministic", False)
                or any("arithmetic" in str(d).lower() for d in decisions)
            ):
                has_deterministic_proof = True

        # If deterministic AST evaluated the expression, trust the mathematical result
        if has_deterministic_proof and m:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=True,
                verified=True,
                confidence=1.0,
                evidence=f"Deterministic evaluated value: {val_str}",
                reason="Evaluated value mathematically proven by deterministic AST engine",
            )

        # Without an expected value or deterministic proof, the presence of a value cannot certify correctness
        return ContractResult(
            contract_type=self.contract_type.value,
            execution_completed=True,
            task_satisfied=bool(val_str),
            verified=False,
            confidence=0.5,
            evidence=f"Value present: {clean[:80]} (no ground truth expected_value provided)",
            reason="Cannot certify correctness without an expected value or deterministic execution proof",
        )


class TextStatisticsContract(SuccessContract):
    contract_type = ContractType.TEXT_STATISTICS

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        if not result or not result.strip():
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=False,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence="Empty result text",
                reason="Result is empty",
            )
        # Check if output contains line/word/character statistics (both prefix 'lines: 3' and suffix '3 lines' forms)
        has_lines = bool(re.search(r"(\b\d+\s+lines?\b|lines?:\s*\d+)", result, re.IGNORECASE))
        has_words = bool(re.search(r"(\b\d+\s+words?\b|words?:\s*\d+)", result, re.IGNORECASE))
        has_chars = bool(re.search(r"(\b\d+\s+characters?\b|characters?:\s*\d+)", result, re.IGNORECASE))

        if has_lines and has_words and has_chars:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=True,
                verified=True,
                confidence=1.0,
                evidence="Text statistics schema satisfied (lines, words, characters found)",
                reason="Complete text statistics report verified",
            )

        return ContractResult(
            contract_type=self.contract_type.value,
            execution_completed=True,
            task_satisfied=False,
            verified=False,
            confidence=0.3,
            evidence=f"Incomplete statistics: lines={has_lines}, words={has_words}, chars={has_chars}",
            reason="Missing required text statistics fields",
        )


class JsonValidContract(SuccessContract):
    contract_type = ContractType.JSON_VALID

    def __init__(self, required_keys: Optional[List[str]] = None):
        self.required_keys = required_keys or []

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        clean = result.strip()
        # Extract json fence if present
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        candidate = m.group(1).strip() if m else clean

        # Also search for candidate json objects/arrays
        if not (candidate.startswith("{") or candidate.startswith("[")):
            brace_start = candidate.find("{")
            bracket_start = candidate.find("[")
            starts = [s for s in [brace_start, bracket_start] if s >= 0]
            if starts:
                candidate = candidate[min(starts):]

        # Check if deterministic validator already verified it or certified valid JSON
        facts = getattr(execution_state, "verified_facts", [])
        for fact in facts:
            if isinstance(fact, dict) and fact.get("method") == "json.loads" and "JSON parsed successfully" in fact.get("fact", ""):
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=True,
                    verified=True,
                    confidence=1.0,
                    evidence=fact.get("fact", "JSON parsed successfully by deterministic engine"),
                    reason="Deterministic JSON validation fact verified",
                )

        if "Valid JSON payload" in clean:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=True,
                verified=True,
                confidence=1.0,
                evidence="Deterministic valid JSON certification report",
                reason="Syntactically valid JSON verified by deterministic parser",
            )

        try:
            parsed = json.loads(candidate)
            if self.required_keys and isinstance(parsed, dict):
                missing = [k for k in self.required_keys if k not in parsed]
                if missing:
                    return ContractResult(
                        contract_type=self.contract_type.value,
                        execution_completed=True,
                        task_satisfied=False,
                        verified=False,
                        confidence=0.4,
                        evidence=f"Parsed JSON successfully but missing required keys: {missing}",
                        reason=f"Missing required keys: {missing}",
                    )

            key_count = len(parsed) if isinstance(parsed, (dict, list)) else 1
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=True,
                verified=True,
                confidence=1.0,
                evidence=f"JSON payload parsed successfully ({key_count} items/keys)",
                reason="Syntactically valid JSON verified",
                metadata={"parsed_type": type(parsed).__name__, "item_count": key_count},
            )
        except Exception as e:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence=f"JSON parsing error: {e}",
                reason=f"Malformed JSON: {e}",
            )


class JsonInvalidContract(SuccessContract):
    contract_type = ContractType.JSON_INVALID

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        clean = result.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        candidate = m.group(1).strip() if m else clean

        try:
            json.loads(candidate)
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence="Expected JSON syntax error, but payload was valid JSON",
                reason="Payload unexpectedly parsed as valid JSON",
            )
        except Exception as e:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=True,
                verified=True,
                confidence=1.0,
                evidence=f"Confirmed invalid JSON with syntax error: {e}",
                reason="Expected syntax error confirmed",
            )


class PythonSyntaxContract(SuccessContract):
    contract_type = ContractType.PYTHON_SYNTAX

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)\s*```", result)
        if not code_blocks:
            code_blocks = [result.strip()]

        for idx, block in enumerate(code_blocks):
            try:
                ast.parse(block)
            except SyntaxError as e:
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=False,
                    verified=False,
                    confidence=0.0,
                    evidence=f"Python syntax error in block {idx+1}: {e}",
                    reason=f"Syntax error at line {e.lineno}: {e.msg}",
                )

        return ContractResult(
            contract_type=self.contract_type.value,
            execution_completed=True,
            task_satisfied=True,
            verified=True,
            confidence=1.0,
            evidence=f"{len(code_blocks)} Python code block(s) verified syntax-clean via AST",
            reason="All Python code blocks parsed cleanly without syntax errors",
        )


class FileExistsContract(SuccessContract):
    contract_type = ContractType.FILE_EXISTS

    def __init__(self, path: str):
        self.path = path

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        resolved = _resolve_in_sandbox(self.path)
        if resolved is not None and os.path.exists(resolved) and os.path.isfile(resolved):
            size = os.path.getsize(resolved)
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=True,
                verified=True,
                confidence=1.0,
                evidence=f"Artifact exists in sandbox: {self.path} (size={size} bytes)",
                reason="Target file existence verified on physical disk",
            )
        return ContractResult(
            contract_type=self.contract_type.value,
            execution_completed=True,
            task_satisfied=False,
            verified=False,
            confidence=0.0,
            evidence=f"Artifact not found on physical disk: {self.path}",
            reason=f"File {self.path} does not exist in sandbox",
        )


class FileHashContract(SuccessContract):
    contract_type = ContractType.FILE_HASH

    def __init__(self, path: str, expected_hash: str):
        self.path = path
        self.expected_hash = expected_hash

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        resolved = _resolve_in_sandbox(self.path)
        if resolved is None or not os.path.exists(resolved):
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence=f"Artifact missing: {self.path}",
                reason="File does not exist",
            )
        cur_hash = get_file_content_hash(resolved)
        if cur_hash == self.expected_hash:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=True,
                verified=True,
                confidence=1.0,
                evidence=f"SHA-256 match on physical disk: {cur_hash}",
                reason="Artifact integrity verified via content hash",
            )
        return ContractResult(
            contract_type=self.contract_type.value,
            execution_completed=True,
            task_satisfied=False,
            verified=False,
            confidence=0.0,
            evidence=f"SHA-256 mismatch: got {cur_hash}, expected {self.expected_hash}",
            reason="Artifact content does not match expected hash",
        )


class FileContentMatchContract(SuccessContract):
    contract_type = ContractType.FILE_CONTENT_MATCH

    def __init__(self, path: str, expected_snippet: str):
        self.path = path
        self.expected_snippet = expected_snippet

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        resolved = _resolve_in_sandbox(self.path)
        if resolved is None or not os.path.exists(resolved):
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence=f"File missing: {self.path}",
                reason="File does not exist",
            )
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if self.expected_snippet in content:
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=True,
                    verified=True,
                    confidence=1.0,
                    evidence=f"File {self.path} contains expected snippet ({len(self.expected_snippet)} chars)",
                    reason="Physical file content matches expected snippet",
                )
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence=f"Expected snippet not found in {self.path}",
                reason="File content missing expected snippet",
            )
        except Exception as e:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence=f"Could not read {self.path}: {e}",
                reason=f"Read error: {e}",
            )


class ReadOperationCompletedContract(SuccessContract):
    """Enforces that a safe read operation executed successfully, the artifact was observed,
    and output is non-empty per V8.1 Section 8."""
    contract_type = ContractType.READ_OPERATION_COMPLETED

    def __init__(self, path: Optional[str] = None):
        self.path = path

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        has_tools = bool(getattr(execution_state, "tool_results", None) or getattr(execution_state, "tool_actions", None))
        has_tool_state = any(k.startswith("file_read:") or k.startswith("file_accessed:") for k in getattr(execution_state, "tool_state", {}))
        has_freshness = bool(getattr(execution_state, "state_freshness", {}))
        if not has_tools and not has_tool_state and not has_freshness:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence="No tool execution recorded for read operation",
                reason="Read operation tool was never invoked",
            )

        # Look for read_file tool results
        read_results = [
            r for r in getattr(execution_state, "tool_results", [])
            if (r.get("name") == "read_file" or r.get("tool") == "read_file")
        ]

        if not read_results:
            accessed = [k for k in getattr(execution_state, "tool_state", {}) if k.startswith("file_accessed:") or k.startswith("file_read:")]
            if not accessed and not getattr(execution_state, "state_freshness", {}):
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=False,
                    verified=False,
                    confidence=0.0,
                    evidence="No read_file execution observed in tool records",
                    reason="Missing read_file tool execution",
                )

        if read_results:
            last_r = read_results[-1]
            if last_r.get("success") is False or last_r.get("error") or last_r.get("error_code"):
                err = last_r.get("error") or last_r.get("error_message") or last_r.get("output") or "Read failed"
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=False,
                    verified=False,
                    confidence=0.0,
                    evidence=f"read_file failed: {err}",
                    reason=f"Tool error: {err}",
                )

        clean_res = result.strip() if result else ""
        if not clean_res:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence="Output is empty after read operation",
                reason="Read operation returned no content",
            )

        freshness = getattr(execution_state, "state_freshness", {})
        return ContractResult(
            contract_type=self.contract_type.value,
            execution_completed=True,
            task_satisfied=True,
            verified=True,
            confidence=1.0,
            evidence=f"Read operation completed successfully with observed artifact ({len(freshness)} artifact(s) tracked)",
            reason="Read operation completed and verified via tool output and disk observation",
        )


class WriteOperationCompletedContract(SuccessContract):
    """Enforces that a write operation succeeded physically and the artifact exists on disk per V8.1 Section 8."""
    contract_type = ContractType.WRITE_OPERATION_COMPLETED

    def __init__(self, path: Optional[str] = None):
        self.path = path

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        written_keys = [k for k in getattr(execution_state, "tool_state", {}) if k.startswith("file_written:")]
        target_path = self.path
        if not target_path and written_keys:
            target_path = written_keys[0].split(":", 1)[1]

        if not target_path:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence="No file write recorded in execution state",
                reason="Write operation missing",
            )

        from tools import _resolve_in_sandbox
        try:
            real = _resolve_in_sandbox(target_path)
            if not os.path.exists(real) or os.path.isdir(real):
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=False,
                    verified=False,
                    confidence=0.0,
                    evidence=f"File '{target_path}' does not exist on disk",
                    reason="Artifact missing after write operation",
                )
            size = os.path.getsize(real)
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=True,
                verified=True,
                confidence=1.0,
                evidence=f"File '{target_path}' successfully written to disk ({size} bytes)",
                reason="Write operation verified on physical disk",
            )
        except Exception as e:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence=f"Write verification error: {e}",
                reason=f"Write verification exception: {e}",
            )


class ToolPostconditionContract(SuccessContract):
    contract_type = ContractType.TOOL_POSTCONDITION

    def __init__(self, tool_name: str, expected_key: Optional[str] = None, expected_value: Optional[Any] = None):
        self.tool_name = tool_name
        self.expected_key = expected_key
        self.expected_value = expected_value

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        if not getattr(execution_state, "tool_results", None):
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence="No tool results recorded in execution state",
                reason="Tool execution missing",
            )

        matching = [
            r for r in execution_state.tool_results
            if (r.get("name") == self.tool_name or r.get("tool") == self.tool_name)
        ]
        if not matching:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence=f"Tool {self.tool_name} was never invoked",
                reason=f"Missing tool execution for {self.tool_name}",
            )

        last_r = matching[-1]
        if last_r.get("success") is False or last_r.get("error") or last_r.get("error_code"):
            err_msg = last_r.get("error") or last_r.get("error_message") or last_r.get("output")
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence=f"Tool {self.tool_name} returned error: {err_msg}",
                reason="Tool execution failed",
            )

        if self.expected_key is not None:
            val = last_r.get(self.expected_key)
            if self.expected_value is not None and val != self.expected_value:
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=False,
                    verified=False,
                    confidence=0.0,
                    evidence=f"Postcondition failed on key {self.expected_key}: got {val}, expected {self.expected_value}",
                    reason="Postcondition value mismatch",
                )

        return ContractResult(
            contract_type=self.contract_type.value,
            execution_completed=True,
            task_satisfied=True,
            verified=True,
            confidence=1.0,
            evidence=f"Tool {self.tool_name} executed successfully with verified postconditions",
            reason="Tool postcondition verified",
        )


class StructuralContract(SuccessContract):
    """Enforces specific structural layout requirements (e.g. JSON, markdown table, CSV)."""
    contract_type = ContractType.STRUCTURAL

    def __init__(self, expected_structure: str = "json"):
        self.expected_structure = expected_structure.lower()

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        clean = result.strip()
        if not clean:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=False,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence="Empty result text",
                reason="Result text is empty",
            )

        if self.expected_structure == "json":
            try:
                m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
                cand = m.group(1).strip() if m else clean
                json.loads(cand)
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=True,
                    verified=True,
                    confidence=1.0,
                    evidence="Valid JSON structure verified",
                    reason="Output satisfies structural JSON requirement",
                )
            except Exception as e:
                return ContractResult(
                    contract_type=self.contract_type.value,
                    execution_completed=True,
                    task_satisfied=False,
                    verified=False,
                    confidence=0.0,
                    evidence=f"JSON structure parsing failed: {e}",
                    reason=f"Structural validation error: {e}",
                )
        elif self.expected_structure in ("table", "markdown_table"):
            has_table = "|" in clean and ("|---" in clean or "|:---" in clean or "| ---" in clean)
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=has_table,
                verified=has_table,
                confidence=1.0 if has_table else 0.0,
                evidence="Markdown table structure verified" if has_table else "No valid markdown table found",
                reason="Output satisfies structural table requirement" if has_table else "Missing markdown table format",
            )

        return ContractResult(
            contract_type=self.contract_type.value,
            execution_completed=True,
            task_satisfied=bool(clean),
            verified=bool(clean),
            confidence=0.8,
            evidence="General structure verified",
            reason="Structure verified",
        )


class EvidenceRequiredContract(SuccessContract):
    """Enforces that factual answers cannot be marked verified without independent evidence."""
    contract_type = ContractType.EVIDENCE_REQUIRED

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        # Check if execution state has independent verified evidence (e.g. tool read, file, deterministic test, verified state item)
        verified_facts = getattr(execution_state, "verified_facts", [])
        items = getattr(execution_state, "items", [])
        has_grounding = False
        for item in items:
            src = getattr(item, "source", "")
            lvl = getattr(item, "level", "")
            if src in {"tool", "deterministic", "sandbox", "external", "verification"} or lvl == "VERIFIED":
                has_grounding = True
                break
        if not has_grounding and verified_facts:
            has_grounding = True

        if not has_grounding:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=True,  # Answer was generated
                verified=False,       # But NOT verified
                confidence=0.5,
                evidence="No independent external or physical evidence grounding this factual statement",
                reason="Factual statement lacks independent grounding evidence; remains unverified",
            )

        return ContractResult(
            contract_type=self.contract_type.value,
            execution_completed=True,
            task_satisfied=True,
            verified=True,
            confidence=0.9,
            evidence=f"Factual statement grounded in {len(verified_facts)} verified evidence record(s)",
            reason="Factual answer verified with independent evidence",
        )


class ModelRubricContract(SuccessContract):
    contract_type = ContractType.MODEL_RUBRIC

    def __init__(self, rubric: str, provider: Optional[Any] = None):
        self.rubric = rubric
        self.provider = provider

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        if self.provider is None:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=True,
                verified=False,
                confidence=0.5,
                evidence="No model provider configured for rubric evaluation",
                reason="Model verifier unavailable",
            )

        prompt = (
            f"Evaluate whether the following result satisfies the rubric.\n"
            f"Rubric: {self.rubric}\n"
            f"Result to evaluate: {result}\n\n"
            f"You MUST respond ONLY with valid JSON in this exact schema:\n"
            f'{{"verdict": "PASS" | "FAIL" | "UNCERTAIN", "confidence": 0.0-1.0, "reason": "concise explanation"}}\n'
        )
        try:
            resp = self.provider.muscle_memory_reply(prompt)
            # Record verifier provider call into TaskUsage per V8.1 Section 5
            if execution_state is not None:
                usage = getattr(execution_state, "usage", None)
                if usage is not None and hasattr(usage, "record_call"):
                    usage.record_call("verification", resp)

            clean = resp.text.strip()
            # Extract JSON block
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
            cand = m.group(1).strip() if m else clean
            parsed = json.loads(cand)
            verdict = str(parsed.get("verdict", "")).strip().upper()
            conf = float(parsed.get("confidence", 0.0))
            reason = str(parsed.get("reason", "Model rubric evaluated"))

            # Strict verification: ONLY exact PASS is accepted
            is_pass = (verdict == "PASS")
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=is_pass,
                verified=is_pass,
                confidence=conf if is_pass else 0.2,
                evidence=f"Model rubric verdict: {verdict} ({conf:.2f})",
                reason=reason,
                metadata={"verdict": verdict, "tokens": resp.input_tokens + resp.output_tokens},
            )
        except Exception as e:
            return ContractResult(
                contract_type=self.contract_type.value,
                execution_completed=True,
                task_satisfied=False,
                verified=False,
                confidence=0.0,
                evidence=f"Model rubric evaluation failed: {e}",
                reason=f"Verifier error: {e}",
            )


class NoVerificationAvailableContract(SuccessContract):
    contract_type = ContractType.NO_VERIFICATION_AVAILABLE

    def evaluate(self, task: Any, result: str, execution_state: Any) -> ContractResult:
        return ContractResult(
            contract_type=self.contract_type.value,
            execution_completed=True,
            task_satisfied=True,
            verified=False,
            confidence=0.5,
            evidence="No automated verification contract applicable for this open-ended task",
            reason="Task completed but no independent verification mechanism is available",
        )


def resolve_contract(
    task_objective: str,
    contract_type_str: Optional[str] = None,
    expected_value: Optional[Any] = None,
    path: Optional[str] = None,
    expected_hash: Optional[str] = None,
    provider: Optional[Any] = None,
) -> SuccessContract:
    """Factory resolving appropriate SuccessContract instance."""
    if contract_type_str:
        norm = contract_type_str.upper().strip()
        if norm in {"EXACT_VALUE", "EXACT", "DETERMINISTIC"}:
            return ExactValueContract(expected_value=expected_value)
        if norm in {"TEXT_STATISTICS", "STATISTICS", "STATS"}:
            return TextStatisticsContract()
        if norm in {"JSON_VALID", "STRUCTURAL", "JSON"}:
            return JsonValidContract()
        if norm in {"JSON_INVALID"}:
            return JsonInvalidContract()
        if norm in {"PYTHON_SYNTAX", "PYTHON", "SYNTAX"}:
            return PythonSyntaxContract()
        if norm in {"READ_OPERATION_COMPLETED", "READ", "READ_FILE"}:
            return ReadOperationCompletedContract(path=path)
        if norm in {"WRITE_OPERATION_COMPLETED", "WRITE", "WRITE_FILE"}:
            return WriteOperationCompletedContract(path=path)
        if norm in {"FILE_EXISTS"}:
            return FileExistsContract(path=path or "")
        if norm in {"FILE_HASH"}:
            return FileHashContract(path=path or "", expected_hash=expected_hash or "")
        if norm in {"FILE_CONTENT_MATCH", "CONTENT_MATCH"}:
            return FileContentMatchContract(path=path or "", expected_snippet=str(expected_value or ""))
        if norm in {"TOOL_POSTCONDITION", "TOOL"}:
            return ToolPostconditionContract(tool_name=path or "tool", expected_value=expected_value)
        if norm in {"EVIDENCE_REQUIRED", "FACTUAL"}:
            return EvidenceRequiredContract()
        if norm in {"MODEL_RUBRIC", "MODEL"}:
            return ModelRubricContract(rubric=task_objective, provider=provider)
        if norm in {"NO_VERIFICATION_AVAILABLE", "NONE"}:
            return NoVerificationAvailableContract()

    # Check for pure read operations vs open-ended analysis/diagnosis per V8.1 Section 8
    low = task_objective.lower()
    m_path = re.search(r"['\"]?([\w\-\.\/]+\.(?:py|json|txt|md|csv))['\"]?", task_objective)
    is_pure_read = bool(
        m_path
        and any(w in low for w in ["read", "inspect", "view", "show contents", "show me its contents", "examine"])
        and not any(w in low for w in ["diagnose", "fix", "repair", "modify", "rewrite", "debug", "refactor", "clean", "extract error"])
    )
    if is_pure_read:
        return ReadOperationCompletedContract(path=path or (m_path.group(1) if m_path else None))

    is_pure_write = bool(
        m_path
        and any(w in low for w in ["write", "create", "save"])
        and not any(w in low for w in ["diagnose", "fix", "explain", "analyze"])
    )
    if is_pure_write and any(w in low for w in ["exists", "created", "saved", "written", "write"]):
        return WriteOperationCompletedContract(path=path or (m_path.group(1) if m_path else None))

    # Check for file existence or file content matches before keyword heuristics
    if any(w in low for w in ["file_exists", "file exists"]) or (m_path and any(w in low for w in ["exists", "created", "saved", "written"])):
        fpath = path or (m_path.group(1) if m_path else "")
        return FileExistsContract(path=fpath)
    if m_path and any(w in low for w in ["contains", "has snippet", "content matches"]):
        fpath = path or m_path.group(1)
        return FileContentMatchContract(path=fpath, expected_snippet=str(expected_value or ""))

    if any(w in low for w in ["calculate", "compute", "what is 2+2", "+", "-", "*", "/"]) and any(c.isdigit() for c in task_objective):
        return ExactValueContract(expected_value=expected_value)
    if any(w in low for w in ["word count", "count words", "line count", "statistics"]):
        return TextStatisticsContract()
    if "json" in low and any(w in low for w in ["validate", "parse", "check", "verify"]):
        if "invalid" in low or "malformed" in low:
            return JsonInvalidContract()
        return JsonValidContract()
    if any(w in low for w in ["python", "syntax", "code block", "ast"]) and not any(w in low for w in ["capital of", "who is"]):
        return PythonSyntaxContract()
    if any(w in low for w in ["capital of", "who wrote", "president of", "population of"]):
        return EvidenceRequiredContract()

    return NoVerificationAvailableContract()
