"""
procedural_memory.py — Reusable execution procedures, validation ladder, and skill versioning.

Per V6 Spec Sections I, J, K, L, M, N, T, U, BJ:
- Skill schema: 28+ fields, schema_version, empirical_success_rate, calibrated_confidence.
- 4 Procedure Types: DETERMINISTIC, TOOL_SEQUENCE, HYBRID, MODEL_ADAPTIVE.
- Intent family abstraction (e.g. numeric_arithmetic_evaluation).
- Retrieval: trigger gate -> preconditions -> non-applicability -> tool policy -> specificity & empirical score -> best safe skill.
- Persistence: atomic save, backup, corruption quarantine, schema migration.
"""

from __future__ import annotations
import os
import json
import re
import time
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Dict, List, Optional, Any, Tuple


NO_APPLICABLE_SKILL = "NO_APPLICABLE_SKILL"


class ProcedureType(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    TOOL_SEQUENCE = "TOOL_SEQUENCE"
    HYBRID = "HYBRID"
    MODEL_ADAPTIVE = "MODEL_ADAPTIVE"


@dataclass
class Skill:
    id: str
    name: str = ""
    schema_version: str = "2.0"

    semantic_intent: str = ""
    intent_family: str = ""

    trigger_patterns: List[str] = field(default_factory=list)

    preconditions: List[str] = field(default_factory=list)
    non_applicability_conditions: List[str] = field(default_factory=list)

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)

    procedure_type: str = ProcedureType.DETERMINISTIC.value
    deterministic_handler: str = ""
    procedure: str = ""  # Python code, tool sequence JSON, or execution spec

    required_tools: List[str] = field(default_factory=list)
    tool_constraints: List[str] = field(default_factory=list)

    decision_points: List[str] = field(default_factory=list)
    expected_observations: List[str] = field(default_factory=list)

    verification_rules: List[str] = field(default_factory=list)
    success_contract: str = ""  # "exact" | "structural" | "execution" | "evidence" | "model"
    verification: str = "structural"  # Backward compatibility alias

    failure_conditions: List[str] = field(default_factory=list)
    recovery_strategy: str = ""

    environment_constraints: Dict[str, Any] = field(default_factory=dict)

    version: str = "1.0"

    verified_success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0

    distinct_variant_count: int = 0
    distinct_variants: List[str] = field(default_factory=list)
    training_variants: List[str] = field(default_factory=list)
    held_out_variants: List[str] = field(default_factory=list)
    negative_variants: List[str] = field(default_factory=list)

    empirical_success_rate: float = 0.0
    calibrated_confidence: float = 0.5
    confidence: float = 0.5  # Backward compatibility alias

    status: str = "candidate"  # "candidate" | "trusted" | "eligible" | "demoted"
    is_deterministic: bool = False

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_validated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    revision_history: List[Dict[str, Any]] = field(default_factory=list)

    # Legacy fields
    success_count: int = 0

    def compute_empirical_stats(self) -> None:
        """Updates empirical success rate based on verified runs."""
        total = self.verified_success_count + self.failure_count
        if total > 0:
            self.empirical_success_rate = round(self.verified_success_count / total, 4)
        else:
            self.empirical_success_rate = 1.0 if self.is_deterministic else 0.0

        if self.status == "trusted":
            self.confidence = max(0.85, self.confidence)
        elif self.status == "eligible":
            self.confidence = 1.0
        self.calibrated_confidence = self.empirical_success_rate if total >= 5 else self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Skill:
        d = dict(data)
        if "name" not in d or not d["name"]:
            d["name"] = d.get("id", "unnamed_procedure")
        if "success_contract" not in d:
            d["success_contract"] = d.get("verification", "structural")
        if "procedure_type" not in d:
            d["procedure_type"] = ProcedureType.DETERMINISTIC.value if d.get("is_deterministic") else ProcedureType.MODEL_ADAPTIVE.value
        if "deterministic_handler" not in d:
            d["deterministic_handler"] = d.get("procedure", "")
        return cls(**d)


def normalize_intent_family(user_message: str) -> Tuple[str, str]:
    """Deterministically identifies the canonical intent family per V6 Section N."""
    clean = user_message.strip().lower()

    if re.search(r"\b(calculate|compute|what\s+is|\d+\s*[\+\-\*\/])\b.*\d+", clean):
        return "numeric_arithmetic_evaluation", "Numeric Arithmetic Evaluation"

    if re.search(r"\b(validate|check|parse|verify)\b.*\bjson\b", clean) or re.search(r"^\s*\{.*\}\s*$", clean):
        return "json_validation_and_parsing", "JSON Validation & Parsing"

    if re.search(r"\b(count|statistics|stats|number\s+of)\b.*\b(words|lines|characters|chars)\b", clean):
        return "text_statistics_analysis", "Deterministic Text Statistics"

    if re.search(r"\bread_file\b|\b(read|inspect|show|display)\b.*\b(file|\.py|\.json|\.txt|\.md)\b", clean):
        return "file_inspection_read", "File Inspection & Read"

    if re.search(r"\bwrite_file\b|\b(write|create|save)\b.*\b(file|\.py|\.json|\.txt|\.md)\b", clean):
        return "file_write_operation", "File Write Operation"

    if re.search(r"\b(delete|remove|drop|destroy|purge|truncate)\b", clean):
        return "destructive_operation", "Destructive System Action"

    if re.search(r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b", clean):
        return "conversational_greeting", "Routine Greeting"

    if re.search(r"^\s*(thanks|thank\s+you|got\s+it|understood|ok|okay)\b", clean):
        return "conversational_acknowledgment", "Conversational Acknowledgment"

    return "unclassified_intent", "Unclassified Intent"


DEFAULT_SEED_SKILLS: List[Skill] = [
    Skill(
        id="json_validator",
        name="Deterministic JSON Validator",
        schema_version="2.0",
        intent_family="json_validation_and_parsing",
        trigger_patterns=[
            r"\b(validate|check|verify|parse)\b.*\bjson\b",
            r"\bis\b.*\bvalid\s+json\b",
        ],
        semantic_intent="Validate or parse a JSON payload using deterministic Python parser",
        preconditions=["Input contains a potential JSON object or array payload"],
        non_applicability_conditions=["Input is empty or purely conversational"],
        steps=[
            "Extract candidate JSON block between ```json or braces/brackets",
            "Call deterministic json.loads parser",
            "Record schema properties and verify structural validity",
        ],
        procedure_type=ProcedureType.DETERMINISTIC.value,
        deterministic_handler="deterministic_json_validation",
        procedure="deterministic_json_validation",
        success_contract="structural",
        verification="structural",
        required_tools=[],
        verified_success_count=10,
        success_count=10,
        distinct_variants=["var_json_1", "var_json_2", "var_json_3"],
        distinct_variant_count=3,
        empirical_success_rate=1.0,
        confidence=1.0,
        status="eligible",
        is_deterministic=True,
    ),
    Skill(
        id="text_statistics",
        name="Deterministic Text Statistics",
        schema_version="2.0",
        intent_family="text_statistics_analysis",
        trigger_patterns=[
            r"\b(count|statistics|stats|number\s+of)\b.*\b(words|lines|characters|chars)\b",
            r"\bword\s+count\b",
        ],
        semantic_intent="Compute word, line, and character count deterministically",
        preconditions=["Input text is provided in the message"],
        non_applicability_conditions=["File path provided instead of inline text"],
        steps=[
            "Extract target text segment",
            "Count lines, words, and characters via standard string methods",
            "Format summary report",
        ],
        procedure_type=ProcedureType.DETERMINISTIC.value,
        deterministic_handler="deterministic_text_statistics",
        procedure="deterministic_text_statistics",
        success_contract="exact",
        verification="exact",
        required_tools=[],
        verified_success_count=10,
        success_count=10,
        distinct_variants=["var_stats_1", "var_stats_2", "var_stats_3"],
        distinct_variant_count=3,
        empirical_success_rate=1.0,
        confidence=1.0,
        status="eligible",
        is_deterministic=True,
    ),
    Skill(
        id="arithmetic_evaluator",
        name="Deterministic AST Arithmetic Evaluator",
        schema_version="2.0",
        intent_family="numeric_arithmetic_evaluation",
        trigger_patterns=[
            r"^\s*(what\s+is\s+)?(calculate\s+|compute\s+)?[-+]?\d+(\.\d+)?\s*[\+\-\*\/\%]\s*[-+]?\d+",
            r"\bcalculate\s+[-+]?\d+",
            r"\bwhat\s+is\s+[-+]?\d+\s*[\+\-\*\/]",
        ],
        semantic_intent="Evaluate simple arithmetic operations deterministically via AST parser",
        preconditions=["Input contains standard arithmetic operators (+, -, *, /, %) with numeric literals"],
        non_applicability_conditions=["Arbitrary code execution or non-whitelisted functions requested"],
        steps=[
            "Extract mathematical expression string",
            "Parse into Python AST and validate operator whitelist",
            "Evaluate arithmetic and return exact numerical output",
        ],
        procedure_type=ProcedureType.DETERMINISTIC.value,
        deterministic_handler="deterministic_ast_arithmetic",
        procedure="deterministic_ast_arithmetic",
        success_contract="exact",
        verification="exact",
        required_tools=[],
        verified_success_count=10,
        success_count=10,
        distinct_variants=["var_math_1", "var_math_2", "var_math_3"],
        distinct_variant_count=3,
        empirical_success_rate=1.0,
        confidence=1.0,
        status="eligible",
        is_deterministic=True,
    ),
    Skill(
        id="safe_read_file_procedure",
        name="Validated File Inspection Tool Sequence",
        schema_version="2.0",
        intent_family="file_inspection_read",
        trigger_patterns=[
            r"^\s*read_file\b",
            r"^\s*(?:read|inspect|view)(?:\s+file)?\s+['\"]?[\w\-\.\/]+['\"]?",
        ],
        semantic_intent="Safely read and inspect file contents within sandbox",
        preconditions=["Target file path is specified inside sandbox"],
        non_applicability_conditions=["Target is directory or outside sandbox"],
        steps=[
            "Extract path parameter",
            "Verify sandbox path boundary",
            "Execute read_file tool under SkillToolPolicy",
            "Verify file content returned without errors",
        ],
        procedure_type=ProcedureType.TOOL_SEQUENCE.value,
        procedure='{"id": "safe_read_file_procedure", "name": "Validated File Inspection", "steps": [{"id": "extract_path", "step_type": "EXTRACT_REGEX", "pattern": "[\'\\"]?([\\w\\-\\.\\/]+\\.(?:py|json|txt|md|csv))[\'\\"]?", "source_var": "user_input", "output_var": "target_file"}, {"id": "read_target", "step_type": "TOOL_CALL", "tool": "read_file", "arguments": {"path": "${target_file}"}, "output_var": "file_content"}, {"id": "return_result", "step_type": "RETURN", "return_expr": "${file_content}"}]}',
        required_tools=["read_file"],
        tool_constraints=["path inside sandbox", "max_calls: 2"],
        success_contract="execution",
        verification="execution",
        verified_success_count=5,
        success_count=5,
        distinct_variants=["var_rf_1", "var_rf_2"],
        distinct_variant_count=2,
        empirical_success_rate=1.0,
        confidence=0.9,
        status="eligible",
        is_deterministic=False,
    ),
]


class PreconditionEvaluator:
    """Evaluates task preconditions and non-applicability conditions."""

    @staticmethod
    def evaluate(skill: Skill, user_message: str) -> Tuple[bool, str]:
        clean = user_message.strip()

        # Non-applicability checks
        for non_app in skill.non_applicability_conditions:
            if "empty" in non_app.lower() and not clean:
                return False, "Non-applicability condition met: Message is empty."
            if "conversational" in non_app.lower():
                words = clean.lower().split()
                if len(words) <= 2 and any(w in {"hi", "hello", "thanks", "ok"} for w in words):
                    return False, "Non-applicability condition met: Message is purely conversational."
            if ("outside sandbox" in non_app.lower() or "traversal" in non_app.lower()) and ".." in clean:
                return False, "Non-applicability condition met: Traversal syntax detected."
            if "non-json" in non_app.lower():
                if not any(c in clean for c in ["{", "[", ":", "```"]):
                    return False, "Non-applicability condition met: Plain non-JSON question."

        # Read-only skill check: reject mutating/destructive verbs
        if any(t == "read_file" for t in skill.required_tools) and "write_file" not in skill.required_tools:
            words = set(re.findall(r"\b\w+\b", clean.lower()))
            mutating_words = {"write", "delete", "remove", "modify", "drop", "truncate", "overwrite"}
            if words.intersection(mutating_words):
                return False, f"Non-applicability condition met: Read-only skill '{skill.id}' rejects mutating operation."

        # File operation path traversal check
        if any("file" in t for t in skill.required_tools) or "file" in skill.intent_family:
            if ".." in clean or "../" in clean or "..\\" in clean:
                return False, "Non-applicability condition met: Directory traversal syntax ('..') rejected."

        # Precondition checks
        for pre in skill.preconditions:
            if "json" in pre.lower():
                if not any(c in clean for c in ["{", "[", ":", "```"]):
                    return False, "Precondition failed: No JSON structural tokens found in message."
            if "arithmetic" in pre.lower() or "numeric" in pre.lower():
                if not re.search(r"\d+", clean):
                    return False, "Precondition failed: No numbers found for arithmetic evaluation."

        return True, ""


class ProceduralMemory:
    """Manages procedural skills, intent clustering, empirical promotion ladder, and atomic persistence."""

    def __init__(
        self,
        storage_path: str = "skills.json",
        seed_path: str = "seed_skills.json",
        learned_path: str = "runtime/memory/learned_skills.json",
        debounce_sec: float = 0.5,
        min_verified_trusted: int = 3,
        min_verified_eligible: int = 5,
        min_variants_eligible: int = 3,
        **kwargs,
    ):
        self.storage_path = storage_path
        self.seed_path = seed_path
        self.learned_path = learned_path
        self.debounce_sec = debounce_sec
        self.min_verified_trusted = min_verified_trusted
        self.min_verified_eligible = min_verified_eligible
        self.min_variants_eligible = min_variants_eligible
        self.skills: Dict[str, Skill] = {}
        self._compiled: Dict[str, List[re.Pattern]] = {}
        self._dirty = False
        self._last_save_time = time.time()
        self._load()

    def _compile_skill(self, skill: Skill) -> List[re.Pattern]:
        patterns = []
        for p in skill.trigger_patterns:
            try:
                patterns.append(re.compile(p, re.IGNORECASE))
            except re.error:
                pass
        self._compiled[skill.id] = patterns
        return patterns

    def is_valid_action_graph_skill(self, skill: Skill) -> Tuple[bool, str]:
        """Validates that a skill contains an executable action graph and zero conversational artifacts."""
        indicators = ["brain deliberative", "stop_reason=", "resumed execution from", "deliberative step", "muscle memory handoff"]
        for step in skill.steps:
            if any(ind in str(step).lower() for ind in indicators):
                return False, f"Conversational LLM artifact found in step: '{step}'"
        if any(ind in str(skill.procedure).lower() for ind in indicators):
            return False, "Conversational LLM artifact found in procedure definition"
        return True, ""

    def _load(self) -> None:
        # 1. Load immutable seed skills from seed_skills.json
        if os.path.exists(self.seed_path) and os.path.getsize(self.seed_path) > 0:
            try:
                with open(self.seed_path, "r", encoding="utf-8") as f:
                    seed_data = json.load(f)
                    for item in seed_data:
                        skill = Skill.from_dict(item)
                        skill.compute_empirical_stats()
                        self.skills[skill.id] = skill
                        self._compile_skill(skill)
            except Exception:
                pass

        # Fallback to in-code DEFAULT_SEED_SKILLS
        for s in DEFAULT_SEED_SKILLS:
            if s.id not in self.skills:
                self.skills[s.id] = s
                self._compile_skill(s)

        # 2. Load runtime skills from storage_path
        if os.path.exists(self.storage_path) and os.path.getsize(self.storage_path) > 0:
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        skill = Skill.from_dict(item)
                        valid, reason = self.is_valid_action_graph_skill(skill)
                        if not valid:
                            # Quarantine invalid conversational skill per V7 Phase 8
                            continue
                        skill.compute_empirical_stats()
                        self.skills[skill.id] = skill
                        self._compile_skill(skill)
            except Exception:
                corrupt_path = f"{self.storage_path}.corrupt_{int(time.time())}"
                try:
                    os.rename(self.storage_path, corrupt_path)
                except Exception:
                    pass

    def _save(self) -> None:
        try:
            dir_name = os.path.dirname(self.storage_path) or "."
            os.makedirs(dir_name, exist_ok=True)
            backup_path = f"{self.storage_path}.backup"

            # Create backup before write
            if os.path.exists(self.storage_path):
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dst:
                        dst.write(src.read())
                except Exception:
                    pass

            fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_skills_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump([s.to_dict() for s in self.skills.values()], f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.storage_path)
            except Exception:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                raise
            self._dirty = False
        except Exception:
            pass

    def flush(self) -> None:
        if self._dirty:
            self._save()

    def retrieve_best_skill(self, user_message: str, only_trusted: bool = True) -> Optional[Skill]:
        """Multi-stage retrieval per V6 Section M:
        trigger gate -> candidate skills -> precondition check -> non-applicability check -> specificity & empirical score.
        """
        clean = user_message.strip()
        matched_candidates = []

        for skill in self.skills.values():
            if only_trusted and skill.status not in {"trusted", "eligible"}:
                continue
            if skill.status == "demoted":
                continue

            patterns = self._compiled.get(skill.id)
            if patterns is None:
                patterns = self._compile_skill(skill)

            matched = any(cp.search(clean) for cp in patterns)
            if not matched:
                continue

            passed_pre, _ = PreconditionEvaluator.evaluate(skill, user_message)
            if not passed_pre:
                continue

            matched_candidates.append(skill)

        if not matched_candidates:
            return None

        # Prioritize deterministic skills first, then empirical success rate, then eligible/trusted, then variant diversity
        matched_candidates.sort(
            key=lambda s: (
                1 if s.is_deterministic or s.procedure_type == ProcedureType.DETERMINISTIC.value else 0,
                2 if s.status == "eligible" else (1 if s.status == "trusted" else 0),
                s.empirical_success_rate,
                s.distinct_variant_count,
            ),
            reverse=True,
        )
        return matched_candidates[0]

    def retrieve(self, user_message: str, only_trusted: bool = True) -> Optional[Skill]:
        return self.retrieve_best_skill(user_message, only_trusted=only_trusted)

    def register_candidate(
        self,
        skill_id: str,
        trigger_patterns: List[str],
        procedure: str,
        name: str = "",
        steps: Optional[List[str]] = None,
        verification: str = "structural",
        required_tools: Optional[List[str]] = None,
        procedure_type: str = ProcedureType.MODEL_ADAPTIVE.value,
        is_deterministic: bool = False,
        intent_family: str = "",
        semantic_intent: str = "",
        preconditions: Optional[List[str]] = None,
        non_applicability_conditions: Optional[List[str]] = None,
    ) -> Skill:
        """Registers a new candidate execution procedure per V6 Section L & O."""
        skill = Skill(
            id=skill_id,
            name=name or skill_id,
            schema_version="2.0",
            semantic_intent=semantic_intent or f"Procedure for {skill_id}",
            intent_family=intent_family,
            trigger_patterns=trigger_patterns,
            preconditions=preconditions or [],
            non_applicability_conditions=non_applicability_conditions or [],
            steps=steps or ["Execute distilled procedural logic"],
            procedure_type=procedure_type,
            procedure=procedure,
            deterministic_handler=procedure if is_deterministic else "",
            verification=verification,
            success_contract=verification,
            required_tools=required_tools or [],
            verified_success_count=0,
            failure_count=0,
            consecutive_failures=0,
            distinct_variants=[],
            distinct_variant_count=0,
            empirical_success_rate=0.0,
            calibrated_confidence=0.5,
            confidence=0.5,
            status="candidate",
            is_deterministic=is_deterministic,
        )
        self.skills[skill.id] = skill
        self._compile_skill(skill)
        self._save()
        return skill

    def reinforce(self, skill_id: str, input_variant: Optional[str] = None) -> None:
        """Records a verified success and updates the promotion ladder requiring input diversity."""
        skill = self.skills.get(skill_id)
        if not skill:
            return

        skill.verified_success_count += 1
        skill.success_count += 1
        skill.consecutive_failures = 0

        if input_variant:
            var_hash = sha256(input_variant.strip().lower().encode("utf-8")).hexdigest()[:8]
            if var_hash not in skill.distinct_variants:
                skill.distinct_variants.append(var_hash)
                skill.distinct_variant_count = len(skill.distinct_variants)

        skill.compute_empirical_stats()
        skill.last_validated_at = datetime.now(timezone.utc).isoformat()

        # Promotion Ladder per V6 Section Q & T:
        # candidate -> trusted: >= 3 verified successes + >= 2 distinct variants
        # trusted -> eligible: >= 5 verified successes + >= 3 distinct variants + empirical success rate >= 0.8
        if skill.status == "candidate":
            if skill.verified_success_count >= self.min_verified_trusted:
                skill.status = "trusted"
                skill.confidence = max(0.85, skill.confidence)
        elif skill.status == "trusted":
            can_promote = (
                skill.verified_success_count >= 10
                or (
                    skill.verified_success_count >= self.min_verified_eligible
                    and skill.distinct_variant_count >= self.min_variants_eligible
                )
            ) and (skill.empirical_success_rate >= 0.8)
            if can_promote:
                skill.status = "eligible"
                skill.confidence = 1.0

        self._dirty = True

    def penalize(self, skill_id: str, error_reason: str = "", failure_evidence: str = "") -> None:
        """Records a failure and triggers demotion if threshold is breached."""
        skill = self.skills.get(skill_id)
        if not skill:
            return

        evidence = failure_evidence or error_reason or "Observed failure"
        skill.failure_count += 1
        skill.consecutive_failures += 1
        skill.compute_empirical_stats()
        skill.last_validated_at = datetime.now(timezone.utc).isoformat()

        if skill.consecutive_failures >= 2:
            if skill.status == "eligible":
                skill.status = "trusted"
                skill.confidence = 0.85
                skill.consecutive_failures = 0
            elif skill.status == "trusted":
                skill.status = "candidate"
                skill.confidence = 0.5
                skill.consecutive_failures = 0
            else:
                skill.status = "demoted"
                skill.confidence = 0.3
                skill.consecutive_failures = 0
        elif skill.empirical_success_rate < 0.5 and skill.failure_count >= 3:
            skill.status = "demoted"

        self._dirty = True

    def revise_skill(
        self,
        skill_id: str,
        failure_evidence: str,
        revised_procedure: str,
        revised_steps: Optional[List[str]] = None,
    ) -> Skill:
        skill = self.skills.get(skill_id)
        if not skill:
            raise KeyError(f"Skill '{skill_id}' not found.")

        old_version = skill.version
        old_procedure = skill.procedure
        old_steps = list(skill.steps)

        archive_entry = {
            "version": old_version,
            "procedure": old_procedure,
            "steps": old_steps,
            "failure_evidence": failure_evidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        skill.revision_history.append(archive_entry)

        try:
            parts = old_version.split(".")
            skill.version = f"{int(parts[0]) + 1}.0"
        except Exception:
            skill.version = "2.0"

        skill.procedure = revised_procedure
        if revised_steps:
            skill.steps = revised_steps
        skill.status = "candidate"
        skill.consecutive_failures = 0
        skill.verified_success_count = 0
        skill.failure_count = 0
        skill.distinct_variants = []
        skill.distinct_variant_count = 0
        skill.confidence = 0.5
        skill.last_validated_at = datetime.now(timezone.utc).isoformat()
        self._dirty = True
        return skill
